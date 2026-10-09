"""
函数功能：把评估结果整理成人类可读的 Markdown 报告

报告结构（分五部分，从"宏观结论"到"逐条明细"）：
    ① 评估概况        —— 这次测了多少、用什么配置、什么时间
    ② 命中路径分布    —— 系统在各种输入下的真实行为分布
    ③ 分层指标        —— 检索层 / 生成层 / 观察指标，并给出问题定位提示
    ④ 分组对比        —— 手工样本 vs 法答网样本（回答"分数低是题目难还是系统弱"）
    ⑤ 逐条明细        —— 每条样本的各项分数，便于人工抽查

为什么要有「命中路径分布」这一节：
    只有走完整 Milvus 检索的样本才有上下文、才能算指标。
    其余样本（命中缓存 / 命中 BM25 查表 / 判定为日常问题）被剔除，
    但它们【本身就是有价值的观测数据】—— 例如
    "10 条问题在语料中已有现成答案"，说明系统对常见问题的响应更快更准。
    如果只报告一个平均分，这些信息就丢了。
"""
import argparse
import json
import os
import statistics
import sys
import time

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_QA_DIR = os.path.dirname(_EVAL_DIR)
_PROJECT_DIR = os.path.dirname(_RAG_QA_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from rag_qa.evaluation import bootstrap                                  # noqa: E402,F401
from rag_qa.evaluation import config                                     # noqa: E402
from rag_qa.evaluation.ragas_runner import (                             # noqa: E402
    load_ragas_result,
    load_run_results,
    select_evaluable,
)
from rag_qa.evaluation.wrappers import _get_logger                       # noqa: E402

logger = _get_logger()

# 指标分层与解读
LAYER_DEFS = [
    ("检索层", ["context_precision", "context_recall"],
     "只看上下文质量，不看生成的答案。分数低说明【没捞到 / 捞错了材料】。"),
    ("生成层", ["faithfulness", "answer_relevancy"],
     "看答案质量。分数低说明【捞到了材料但写偏了 / 编造了】。"),
    ("观察指标", ["answer_correctness"],
     "⚠️ 对『措辞差异』高度敏感。法律答案允许多种正确表达路径"
     "（引用不同法条或案例），该指标会系统性偏低，"
     "仅供观察，不作为优化依据。"),
]

METRIC_CN = {
    "context_precision": "上下文精准度（检索排序质量）",
    "context_recall": "上下文召回率（要点覆盖能力）",
    "faithfulness": "忠实度（有无编造）",
    "answer_relevancy": "答案相关性（有无跑题）",
    "answer_correctness": "答案正确性（vs 标准答案，仅观察）",
}

SCENE_CN = {"manual": "手工构造（生活化）", "everyday": "法答网-生活化",
            "professional": "法答网-专业", "unknown": "未标注"}


def fmt(value) -> str:
    """格式化分数，None 显示为 N/A"""
    return "N/A" if value is None else f"{value:.4f}"


def build_report() -> str:
    """生成完整报告文本"""
    run_results = load_run_results()
    evaluable, stats = select_evaluable(run_results)
    ragas = load_ragas_result()
    summary = ragas["summary"]
    per_sample = ragas["per_sample"]

    total = len(run_results)
    n_eval = len(evaluable)
    lines: list[str] = []

    # ================= ① 评估概况 =================
    lines.append("# LawChain · 律衡 —— RAG 评估报告")
    lines.append("")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 本次评估样本：**{total}** 条，其中 **{n_eval}** 条进入指标计算")
    lines.append("")
    lines.append("## 一、评估概况")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 数据集样本总数 | {total} 条 |")
    lines.append(f"| 参与指标计算 | {n_eval} 条 |")
    lines.append(f"| 随机种子 | {config.RANDOM_SEED} |")
    lines.append(f"| 上下文截断 | 每段 {config.MAX_CONTEXT_CHARS} 字，最多 {config.MAX_CONTEXT_CHUNKS} 段 |")
    lines.append(f"| 裁判 LLM | `{config.JUDGE_MODEL}`（temperature={config.JUDGE_TEMPERATURE}） |")
    lines.append(f"| 裁判 Embedding | BGE-M3（本地） |")
    lines.append(f"| RAGAS 耗时 | {ragas.get('elapsed', 'N/A')} 秒 |")
    lines.append("")

    # ================= ② 命中路径分布 =================
    lines.append("## 二、命中路径分布")
    lines.append("")
    lines.append("系统对不同输入的处理路径不同。只有走完整检索链路的样本才有上下文，"
                 "才能计算 RAGAS 指标；其余样本被剔除，但**它们本身是有价值的观测**。")
    lines.append("")
    lines.append("| 命中路径 | 条数 | 占比 | 说明 |")
    lines.append("|---------|------|------|------|")
    notes = {
        "Milvus 检索（有上下文，参与计算）": "走完整检索链路，本次评估的主体",
        "BM25 查表（无上下文，剔除）": "问题在 `law_qa` 语料中已有现成答案，直接返回，响应更快",
        "Redis 缓存（无上下文，剔除）": "命中缓存，零成本返回",
        "日常问题（不检索，剔除）": "BERT 网关正确分流，不触发检索",
        "执行出错（剔除）": "需排查的错误",
    }
    for k, v in stats.items():
        if not v:
            continue
        pct = f"{v / total * 100:.1f}%" if total else "—"
        lines.append(f"| {k} | {v} | {pct} | {notes.get(k, '')} |")
    lines.append(f"| **合计** | **{total}** | 100% | |")
    lines.append("")
    if stats.get("BM25 查表（无上下文，剔除）"):
        lines.append(f"> **注**：有 {stats['BM25 查表（无上下文，剔除）']} 条问题被 BM25 字面命中。"
                     "这**不是缺陷** —— 说明这些问题在语料中已有权威答案，"
                     "系统能直接返回而不必走一遍向量检索。"
                     "但它意味着评估覆盖面减少，后续可考虑加大样本量。")
        lines.append("")

    # ================= ③ 分层指标 =================
    lines.append("## 三、分层指标")
    lines.append("")
    lines.append("| 层级 | 指标 | 分数 |")
    lines.append("|------|------|------|")
    for layer, names, _desc in LAYER_DEFS:
        for name in names:
            if name in summary:
                lines.append(f"| {layer} | {METRIC_CN.get(name, name)} | **{fmt(summary[name])}** |")
    lines.append("")

    # 问题定位提示
    lines.append("### 问题定位")
    lines.append("")
    cp = summary.get("context_precision")
    cr = summary.get("context_recall")
    fa = summary.get("faithfulness")
    ar = summary.get("answer_relevancy")

    def low(v):
        return v is not None and v < 0.5

    def high(v):
        return v is not None and v >= 0.7

    hints = []
    if low(cr):
        hints.append("**上下文召回率偏低** → 检索没捞全标准答案里的要点，"
                     "可检查切块粒度、`top_m` 配额、子查询改写质量。")
    if low(cp):
        hints.append("**上下文精准度偏低** → 捞上来的材料里有较多无关内容，"
                     "可检查 reranker 阈值、是否缺少最低相似度门控。")
    if low(fa):
        hints.append("**忠实度偏低** → 答案里可能有检索材料之外的推断或外部知识，"
                     "可检查 prompt 是否明确要求「只依据给定材料作答」。")
    if high(ar):
        hints.append("**答案相关性良好** → 答案没有跑题，与提问匹配。")
    if not hints:
        hints.append("各项指标均未触发告警阈值（< 0.5），可结合逐条明细进一步观察。")
    for h in hints:
        lines.append(f"- {h}")
    lines.append("")

    # ================= ④ 分组对比 =================
    lines.append("## 四、分组对比（回答『分数低是题目难还是系统弱』）")
    lines.append("")
    lines.append("样本来源不同，难度差异很大。分组看分数，才能判断系统的真实水平。")
    lines.append("")
    groups: dict[str, list[dict]] = {}
    for row in per_sample:
        groups.setdefault(row.get("scene") or "unknown", []).append(row)

    metric_names = [n for _l, names, _d in LAYER_DEFS for n in names
                    if n in summary]
    if len(groups) > 1:
        header = "| 分组 | 样本数 | " + " | ".join(
            METRIC_CN.get(m, m).split("（")[0] for m in metric_names) + " |"
        lines.append(header)
        lines.append("|------|--------|" + "|".join(["------"] * len(metric_names)) + "|")
        for scene, rows in sorted(groups.items()):
            cells = []
            for m in metric_names:
                vals = [r[m] for r in rows if r.get(m) is not None]
                cells.append(f"{statistics.mean(vals):.4f}" if vals else "N/A")
            lines.append(f"| {SCENE_CN.get(scene, scene)} | {len(rows)} | "
                         + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"> **⚠️ 注意**：分组样本量较小（最大 {max(len(v) for v in groups.values())} 条），"
                     "组间差异仅供参考，不宜据此下强结论。")
    else:
        lines.append("（样本量不足，未做分组对比）")
    lines.append("")

    # ================= ⑤ 逐条明细 =================
    lines.append("## 五、逐条明细")
    lines.append("")
    lines.append("| qa_id | 分组 | 问题 | 上下文 | 答案字数 | "
                 + " | ".join(METRIC_CN.get(m, m).split("（")[0] for m in metric_names) + " |")
    lines.append("|-------|------|------|--------|----------|"
                 + "|".join(["------"] * len(metric_names)) + "|")
    for row in per_sample:
        q = row["question"]
        q_short = q[:26] + "…" if len(q) > 26 else q
        cells = [fmt(row.get(m)) for m in metric_names]
        lines.append(
            f"| {row['qa_id']} | {SCENE_CN.get(row.get('scene'), '?')} "
            f"| {q_short} | {row.get('context_count', '?')} | {row.get('answer_chars', '?')} "
            f"| " + " | ".join(cells) + " |"
        )
    lines.append("")

    # ================= 附：局限性 =================
    lines.append("## 六、本次评估的局限")
    lines.append("")
    lines.append("诚实标注局限，比给出一个漂亮但不可信的数字更有价值。")
    lines.append("")
    lines.append("1. **参考答案的来源与检索语料不同源**：`ground_truth` 取自 `law_qa` 表"
                 "（法答网答问），而检索语料是 `law_chunk`（法条 + 案例）。"
                 "两者粒度不同，`context_recall` 会因此偏低。")
    lines.append(f"2. **样本量有限**：参与计算的样本仅 {n_eval} 条，"
                 "单个样本的波动对均分影响较大。")
    lines.append("3. **`law_qa` 的场景偏差**：该表来自法答网（法官/律师提问），"
                 "部分问题不代表普通用户咨询场景，已用关键词过滤 + 手工样本对照缓解。")
    lines.append("4. **裁判模型即被评估模型**：本次裁判用的是与系统同款的 DeepSeek。"
                 "同源模型可能存在自我偏好，理想做法是换用不同厂商的模型做裁判。")
    lines.append("5. **评估链路是显式复刻的**：`pipeline_runner.py` 按 `main.py` 的 "
                 "`search()` 顺序显式调用各接口。若 `search()` 流程变更，"
                 "评估链路需同步更新。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 `rag_qa/evaluation/reporter.py` 自动生成*")
    return "\n".join(lines)


def save_report(text: str) -> str:
    """把报告写入文件"""
    config.ensure_dirs()
    config.REPORT_FILE.write_text(text, encoding="utf-8")
    logger.info(f"报告已写入：{config.REPORT_FILE}")
    return str(config.REPORT_FILE)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 RAG 评估报告")
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="同时在终端打印报告内容")
    args = parser.parse_args()

    print("[1/2] 生成报告 ...")
    try:
        text = build_report()
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return 1

    print("[2/2] 保存 ...")
    path = save_report(text)
    print(f"  ✅ {path}")

    if args.do_print:
        print()
        print(text)

    # 简要回显关键分数
    ragas = load_ragas_result()
    print("\n  指标汇总：")
    for k, v in ragas["summary"].items():
        print(f"    {METRIC_CN.get(k, k):<34} {fmt(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
