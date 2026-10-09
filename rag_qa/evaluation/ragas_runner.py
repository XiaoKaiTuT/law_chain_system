"""
函数功能：把流水线结果喂给 RAGAS，计算出各项评估指标

数据流：
    eval_run_raw.json（pipeline_runner 产出）
        ↓ 过滤：只保留 hit_stage == milvus 且有上下文的样本
        ↓ 转成 RAGAS Dataset（question / answer / retrieved_contexts / ground_truth）
        ↓ evaluate(metrics=[...], llm=裁判LLM, embeddings=裁判Embedding)
    eval_ragas.json（逐样本分数 + 汇总分数）

指标分两层（这是本模块的核心设计）：
    检索层 —— 只看上下文质量，不看生成的答案
        context_precision   捞上来的材料有没有噪声（排序质量）
        context_recall      标准答案的要点有没有被漏掉（召回能力）
    生成层 —— 看答案质量
        faithfulness        答案有没有编造（幻觉检测）
        answer_relevancy    答案有没有跑题
    参考指标（仅观察，不作优化依据）
        answer_correctness  生成答案 vs 标准答案
        ⚠️ 法律答案允许多种正确表达路径（引用不同法条/案例），
           该指标对「措辞差异」高度敏感，会系统性偏低。

为什么能分离问题：
    两层指标同时出，就能判断是「检索没捞到」还是「捞到了但生成跑偏」——
    这是单纯测端到端做不到的。

⚠️ 本模块不修改任何项目文件，只读取评估产物并调用外部 LLM。
"""
import argparse
import json
import math
import os
import sys
import time

# ---- 路径注入（bootstrap 必须最先）----------------------------------------
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_QA_DIR = os.path.dirname(_EVAL_DIR)
_PROJECT_DIR = os.path.dirname(_RAG_QA_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from rag_qa.evaluation import bootstrap                                  # noqa: E402,F401
from rag_qa.evaluation import config                                     # noqa: E402
from rag_qa.evaluation.wrappers import (                                 # noqa: E402
    _get_logger,
    build_ragas_judges,
)

logger = _get_logger()


def load_run_results() -> list[dict]:
    """读取 pipeline_runner 产出的中间结果"""
    path = config.RUN_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"找不到流水线结果：{path}\n请先运行：python -m rag_qa.evaluation.pipeline_runner"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["results"]


def select_evaluable(results: list[dict]) -> tuple[list[dict], dict]:
    """
    函数功能：筛出「可参与指标计算」的样本

    只有走完整 Milvus 检索链路的样本才有 retrieved_contexts，
    其余（Redis 缓存 / BM25 查表 / 日常问题 / 出错）必须剔除 ——
    否则 RAGAS 会因为上下文为空而给出无意义的低分。

    :param results: 全部流水线结果
    :return: (可评估样本列表, 命中路径统计)
    """
    from rag_qa.evaluation.pipeline_runner import (
        STAGE_BM25, STAGE_CACHE, STAGE_ERROR, STAGE_GENERAL, STAGE_MILVUS,
    )

    stage_names = {
        STAGE_MILVUS: "Milvus 检索（有上下文，参与计算）",
        STAGE_BM25: "BM25 查表（无上下文，剔除）",
        STAGE_CACHE: "Redis 缓存（无上下文，剔除）",
        STAGE_GENERAL: "日常问题（不检索，剔除）",
        STAGE_ERROR: "执行出错（剔除）",
    }
    stats = {stage_names[k]: 0 for k in stage_names}
    evaluable = []
    for r in results:
        stats[stage_names.get(r.get("hit_stage"), "未知")] = \
            stats.get(stage_names.get(r.get("hit_stage"), "未知"), 0) + 1

        if r.get("hit_stage") != STAGE_MILVUS:
            continue
        if not r.get("contexts"):
            continue
        if not (r.get("answer") or "").strip():
            logger.warning(f"[{r.get('qa_id')}] 上下文存在但答案为空，剔除")
            continue
        evaluable.append(r)

    return evaluable, stats


def build_ragas_dataset(samples: list[dict]):
    """
    函数功能：把评估样本转成 RAGAS 需要的 Dataset

    RAGAS 字段映射（0.2.x 用这些名字）：
        user_input           -> 提问        （旧版叫 question）
        response             -> 系统生成的答案（旧版叫 answer）
        retrieved_contexts   -> 检索到的上下文列表
        reference            -> 参考答案    （旧版叫 ground_truth）

    ⚠️ 这里同时写入新旧两套字段名，兼容 RAGAS 0.2.x 两种命名，
       避免因版本差异导致 KeyError。
    :param samples: 可评估样本
    :return: datasets.Dataset
    """
    from datasets import Dataset

    data = {
        # 新命名（ragas 0.2.x 内部 EvaluationDataset 用这个）
        "user_input": [s["eval_question"] for s in samples],
        "response": [s["answer"] for s in samples],
        "retrieved_contexts": [s["contexts"] for s in samples],
        "reference": [s["ground_truth"] for s in samples],
        # 旧命名（部分指标仍按旧名取列）
        "question": [s["eval_question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }
    return Dataset.from_dict(data)


def get_metrics():
    """按配置组装要计算的指标列表"""
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    if config.ENABLE_ANSWER_CORRECTNESS:
        metrics.append(answer_correctness)
    return metrics


def _clean(value):
    """把 NaN / numpy 标量转成可 JSON 序列化的普通值"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return round(f, config.SCORE_DECIMALS)


def run_evaluation(samples: list[dict], vector_tools=None) -> dict:
    """
    函数功能：调用 RAGAS 计算指标

    :param samples: 可评估样本
    :param vector_tools: 可复用的 VectorTools（避免重复加载 BGE-M3）
    :return: {"summary": {...}, "per_sample": [...]}
    """
    from ragas import evaluate

    logger.info(f"构建 RAGAS 数据集：{len(samples)} 条")
    dataset = build_ragas_dataset(samples)

    logger.info("构造裁判 LLM 与裁判 Embedding ...")
    wrapped_llm, wrapped_emb, _raw_emb = build_ragas_judges(vector_tools=vector_tools)

    metrics = get_metrics()
    logger.info(f"开始评估，指标：{[m.name for m in metrics]}")
    logger.info("⚠️ 这一步会调用较多 LLM（answer_relevancy 每条样本要约 3 次），请耐心等待")

    t0 = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=wrapped_llm,
        embeddings=wrapped_emb,
        raise_exceptions=False,     # 单个指标失败不整体中断
        show_progress=True,
    )
    elapsed = time.time() - t0
    logger.info(f"RAGAS 评估完成，耗时 {elapsed:.1f}s")

    # ---- 提取分数 ----------------------------------------------------------
    df = result.to_pandas()
    metric_names = [m.name for m in metrics]

    summary = {}
    for name in metric_names:
        if name in df.columns:
            summary[name] = _clean(df[name].mean())

    per_sample = []
    for i, s in enumerate(samples):
        row = {"qa_id": s.get("qa_id"), "scene": s.get("scene"),
               "source": s.get("source"), "question": s["eval_question"],
               "answer_chars": len(s["answer"]),
               "context_count": len(s["contexts"])}
        for name in metric_names:
            if name in df.columns and i < len(df):
                row[name] = _clean(df[name].iloc[i])
        per_sample.append(row)

    return {"summary": summary, "per_sample": per_sample,
            "elapsed": round(elapsed, 1), "count": len(samples)}


def save_result(payload: dict) -> str:
    """把 RAGAS 结果写成 JSON"""
    config.ensure_dirs()
    config.RAGAS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"RAGAS 结果已写入：{config.RAGAS_FILE}")
    return str(config.RAGAS_FILE)


def load_ragas_result() -> dict:
    """读取已生成的 RAGAS 结果（供 reporter 使用）"""
    if not config.RAGAS_FILE.exists():
        raise FileNotFoundError(
            f"找不到 RAGAS 结果：{config.RAGAS_FILE}\n"
            "请先运行：python -m rag_qa.evaluation.ragas_runner"
        )
    return json.loads(config.RAGAS_FILE.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="用 RAGAS 计算评估指标")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（调试用）")
    args = parser.parse_args()

    print(config.describe())

    print("\n[1/3] 读取流水线结果 ...")
    results = load_run_results()
    evaluable, stats = select_evaluable(results)
    print(f"  总样本 {len(results)} 条，可评估 {len(evaluable)} 条")
    print("  命中路径分布：")
    for k, v in stats.items():
        if v:
            print(f"    {k:<34} {v:>3} 条")

    if not evaluable:
        print("  ❌ 没有可评估的样本，请检查 pipeline_runner 的输出")
        return 1
    if args.limit:
        evaluable = evaluable[:args.limit]
        print(f"  （调试模式）只评估前 {len(evaluable)} 条")

    print(f"\n[2/3] 调用 RAGAS 评估 {len(evaluable)} 条 ...")
    try:
        payload = run_evaluation(evaluable)
    except Exception as e:  # noqa: BLE001
        logger.error(f"RAGAS 评估失败：{type(e).__name__}: {e}")
        print(f"  ❌ {type(e).__name__}: {e}")
        return 1

    print(f"\n[3/3] 保存结果 ...")
    path = save_result(payload)
    print(f"  ✅ {path}")
    print(f"\n  耗时 {payload['elapsed']}s")
    print("\n  指标汇总：")
    for k, v in payload["summary"].items():
        print(f"    {k:<24} {v if v is not None else 'N/A'}")
    print("\n  下一步：python -m rag_qa.evaluation.reporter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
