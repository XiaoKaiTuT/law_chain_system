"""
函数功能：复现完整问答链路，显式产出「检索到的上下文」与「生成的答案」

⚠️ 本模块的调用顺序是【照抄】main.py 的 LawChainClient.search() 流程的：
        main.py                                本模块
        ─────────────────────────────────────  ──────────────────────────────
        ② 首轮查 Redis 缓存（读）                step1_check_cache
        ③ resolve_query（指代消解）              step2_resolve
        ④ _is_legal_question（BERT 分类）         step2_resolve
        ⑤ bm25_search.search（BM25 查表）        step3_bm25
        ⑥ query_generate + milvus_client.search  step4_retrieve  ★上下文在这
        ⑦ _stream_and_cache → llm.generate       step5_generate

    如果以后修改了 main.py 的 search() 流程（增删步骤、换接口、换参数），
    必须同步检查本模块，否则评估的将不再是真实链路。

为什么不用 monkey-patch 拦截 search()：
    虽然拦截能保证"测的就是真实执行"，但它依赖方法签名，
    属于隐式耦合；显式调用更透明、更易调试。
    这里选择显式复刻，并在报告里标注这一取舍。

⚠️ 本模块只调用【只读】接口：
    - 刻意不调用 search() / _stream_and_cache()，因为它们内部会
      set_question / insert_data / _save_history，会污染生产数据。
    - 全程用 SideEffectGuard 包住，跑完对比 Redis key 数与 MySQL 行数，
      一旦发现写入立刻抛错。
"""
import argparse
import json
import sys
import time
from collections import Counter

# ---- 路径注入（必须在 import 项目模块之前）--------------------------------
import os

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_QA_DIR = os.path.dirname(_EVAL_DIR)
_PROJECT_DIR = os.path.dirname(_RAG_QA_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ★ 必须最先导入 bootstrap：它会在项目模块（base.config）被 import 之前
#   把 DEEPSEEK_API_KEY 注入 os.environ。
#   否则项目 LLM 客户端会拿到 base/config.py 的占位符 'YOUR_DEEPSEEK_API_KEY'，
#   调用时报 401 Authentication Fails。
from rag_qa.evaluation import bootstrap                                 # noqa: E402,F401
from rag_qa.evaluation import config                                   # noqa: E402
from rag_qa.evaluation.dataset_builder import load_dataset              # noqa: E402
from rag_qa.evaluation.wrappers import (                                # noqa: E402
    SideEffectGuard,
    _get_logger,
)

logger = _get_logger()

# 命中路径标记
STAGE_CACHE = "cache"          # Redis 缓存命中 → 无上下文
STAGE_BM25 = "bm25"            # BM25 查表命中 → 无上下文
STAGE_MILVUS = "milvus"        # 走完整检索 → 有上下文（★ 参与指标计算）
STAGE_GENERAL = "general"      # 判定为日常问题 → 不检索
STAGE_ERROR = "error"          # 执行出错


# ============================================================================
# 工具
# ============================================================================
def format_context(documents: list[dict], content_field: str) -> str:
    """
    函数功能：把检索到的文档列表拼成上下文字符串

    ⚠️ 必须与 LLMClient._format_context() 保持完全一致，
       否则我们评估的"上下文"就不是 LLM 实际看到的内容。
       原实现：
           [i] 题目：{title} 来源：{source}
           {content_field}
    :param documents: 文档列表
    :param content_field: 取哪个字段作为正文（案例用 parent_content，法条用 text_content）
    :return: 拼接后的字符串
    """
    parts = []
    for i, doc in enumerate(documents):
        part = f'[{i}] 题目：{doc.get("title", "")} 来源：{doc.get("source", "")}\n'
        part += doc.get(content_field, "")
        parts.append(part)
    return "\n\n".join(parts)


def truncate(text: str, limit: int) -> str:
    """截断过长的文本（法律条文很长，不截断会让裁判模型上下文过长而漏判）"""
    if len(text) <= limit:
        return text
    return text[:limit] + "…（已截断）"


def build_contexts(case_chunk: list, clause_chunk: list) -> tuple[list[str], list[dict]]:
    """
    函数功能：把检索到的案例与法条整理成 RAGAS 需要的上下文列表

    RAGAS 的 retrieved_contexts 是一个「字符串列表」，
    每个元素是一个独立的检索片段，指标会逐个判断其相关性。

    这里按【单个文档】而不是【整块拼接】来组织，
    这样 context_precision 才能分辨"哪几条是有用的"。

    :param case_chunk: 案例检索结果
    :param clause_chunk: 法条检索结果
    :return: (截断后的上下文列表, 供人工核查的原始明细)
    """
    contexts: list[str] = []
    raw_detail: list[dict] = []

    for label, docs, field in (
        ("案例", case_chunk or [], "parent_content"),
        ("法条", clause_chunk or [], "text_content"),
    ):
        for doc in docs:
            content = doc.get(field, "") or ""
            if not content.strip():
                continue
            piece = format_context([doc], field)
            contexts.append(truncate(piece, config.MAX_CONTEXT_CHARS))
            raw_detail.append({
                "kind": label,
                "source": doc.get("source", ""),
                "title": doc.get("title", ""),
                "rerank_score": doc.get("rerank_score"),
                "chars": len(content),
            })

    # 总数上限，防止个别样本塞入过多片段
    if len(contexts) > config.MAX_CONTEXT_CHUNKS:
        logger.warning(
            f"上下文片段 {len(contexts)} 条超过上限 {config.MAX_CONTEXT_CHUNKS}，已截断"
        )
        contexts = contexts[:config.MAX_CONTEXT_CHUNKS]
        raw_detail = raw_detail[:config.MAX_CONTEXT_CHUNKS]

    return contexts, raw_detail


class GenerateTimeoutError(RuntimeError):
    """单条样本的答案生成超时"""


# 底层 openai 客户端的原始超时，供恢复用
_ORIGINAL_LLM_TIMEOUT = None


def apply_llm_timeout(chain, timeout: int | None = None) -> str:
    """
    函数功能：把项目 LLM 客户端的超时调小，避免单条样本卡满 10 分钟

    为什么要做这件事：
        项目的 llm_client 创建 OpenAI 客户端时没传 timeout，
        于是用 openai 库默认值 —— 实测为
            Timeout(connect=5.0, read=600, write=600, pool=600)
        read=600 意味着「两个数据块之间最多等 600 秒」。
        实测有样本（海上货物运输保险）正好卡满这个值，把整个评估拖住 10 分钟。

    为什么必须改 openai 客户端的 timeout，而不是底层 httpx 客户端的：
        openai/_base_client.py 的 _build_request 里写的是
            timeout=self.timeout if isinstance(options.timeout, NotGiven) else options.timeout
        也就是它把【客户端自己的 timeout】作为【请求级参数】显式传给 httpx。
        请求级参数优先于 httpx 客户端的默认值，所以改 httpx 客户端的 timeout
        完全不起作用（已实测：改了之后依然一直阻塞，无任何输出）。
        只有改 openai 客户端的 self.timeout 才有效。

    为什么改这个对象是安全的：
        - 只影响【本进程】的这一个客户端实例，不改项目任何文件
        - 评估是离线批处理，宽松的 600 秒没有意义，快速失败更合适
        - 客户端由项目创建，无法从外部传入配置，这是唯一可行且干净的入口
        - 超时后 openai 会抛 openai.APITimeoutError（httpx 的 TimeoutException 会被它转换），
          能被正常捕获，不会让进程崩掉

    :param chain: LawChainClient 实例
    :param timeout: 期望的超时秒数，None 则用 config.GENERATE_TIMEOUT
    :return: 状态描述（便于打印）
    """
    global _ORIGINAL_LLM_TIMEOUT

    seconds = timeout or config.GENERATE_TIMEOUT

    # 逐层拿到 openai 客户端：
    #   chain.llm_client -> LLMClient
    #   .llm             -> openai.OpenAI
    llm_client = getattr(chain, "llm_client", None)
    openai_client = getattr(llm_client, "llm", None) if llm_client else None
    if openai_client is None or not hasattr(openai_client, "timeout"):
        return "⚠️ 无法定位 openai 客户端，将沿用默认超时（可能长达 600 秒）"

    if _ORIGINAL_LLM_TIMEOUT is None:
        _ORIGINAL_LLM_TIMEOUT = openai_client.timeout

    old = openai_client.timeout
    old_read = getattr(old, "read", old)

    # 用 httpx.Timeout 保留连接超时设置，只收紧读/写/连接池等待
    import httpx

    connect = getattr(old, "connect", None) or 5.0
    openai_client.timeout = httpx.Timeout(
        connect=connect, read=seconds, write=seconds, pool=seconds
    )
    # 关掉 SDK 自身的重试：默认会重试 2 次，配合 600s 超时最坏能拖到 30 分钟。
    # 评估场景宁可快速失败，重试交给 generate_answer() 统一控制。
    old_retries = getattr(openai_client, "max_retries", None)
    openai_client.max_retries = 0

    return (f"LLM 超时 read={old_read}s → {seconds}s，"
            f"SDK 重试 {old_retries} → 0")


def restore_llm_timeout(chain) -> None:
    """把 openai 客户端超时与重试次数恢复为原值（评估结束时调用）"""
    global _ORIGINAL_LLM_TIMEOUT
    if _ORIGINAL_LLM_TIMEOUT is None:
        return
    llm_client = getattr(chain, "llm_client", None)
    openai_client = getattr(llm_client, "llm", None) if llm_client else None
    if openai_client is not None:
        openai_client.timeout = _ORIGINAL_LLM_TIMEOUT
        logger.info("已恢复 LLM 客户端原始超时")
    _ORIGINAL_LLM_TIMEOUT = None


def generate_answer(chain, resolved: str, case_chunk: list,
                    clause_chunk: list, qa_id) -> str:
    """
    函数功能：调用 LLM 生成答案（带有限重试）

    超时由 apply_llm_timeout() 在 HTTP 层统一控制；
    这里只负责「失败后重试」。

    :param chain: LawChainClient
    :param resolved: 消解后的问题（与生产链路一致，作为 prompt 里的 question）
    :param case_chunk: 案例片段
    :param clause_chunk: 法条片段
    :param qa_id: 样本标识（日志用）
    :return: 拼接好的答案文本
    :raises Exception: 全部尝试都失败时抛出最后一次的异常
    """
    last_err: Exception | None = None
    for attempt in range(1, config.MAX_GENERATE_ATTEMPTS + 1):
        chunks: list[str] = []
        try:
            gen = chain.llm_client.generate(resolved, case_chunk, clause_chunk, None)
            for piece in gen:
                chunks.append(piece)
            text = "".join(chunks)
            if text.strip():
                if attempt > 1:
                    logger.info(f"[{qa_id}] 第 {attempt} 次生成成功")
                return text
            last_err = RuntimeError("生成结果为空")
            logger.warning(f"[{qa_id}] 第 {attempt} 次生成结果为空")
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(
                f"[{qa_id}] 第 {attempt}/{config.MAX_GENERATE_ATTEMPTS} 次生成失败："
                f"{type(e).__name__}: {e}"
            )
    raise last_err if last_err else RuntimeError("答案生成失败")


# ============================================================================
# 单条样本的执行
# ============================================================================
def run_one(chain, sample: dict) -> dict:
    """
    函数功能：对一条评估样本跑完整链路，产出上下文与答案

    :param chain: LawChainClient 实例（复用，避免反复加载模型）
    :param sample: 数据集里的一条样本
    :return: 结果 dict（含 hit_stage / contexts / answer）
    """
    question = sample["eval_question"]
    qa_id = sample["qa_id"]

    result = {
        **sample,
        "hit_stage": None,
        "is_legal": None,
        "resolved": None,
        "contexts": [],
        "context_detail": [],
        "answer": "",
        "error": None,
        "elapsed": 0.0,
    }
    t0 = time.time()

    try:
        # ---- step 1：查 Redis 缓存（只读，对应 search() 的 ②）------------
        #   search() 里缓存【只对首轮生效】，评估场景等价于首轮
        cached = chain.redis_client.get_answer(question)
        if cached:
            logger.info(f"[{qa_id}] 命中 Redis 缓存 → 该样本无检索上下文，将剔除")
            result.update(hit_stage=STAGE_CACHE, answer=cached)
            return result

        # ---- step 2：指代消解 + BERT 分类（对应 ③④）--------------------
        #   评估场景没有多轮历史，history=None → 消解会直接返回原问题
        resolved = chain.llm_client.resolve_query(question, None) or question
        result["resolved"] = resolved

        is_legal = chain._is_legal_question(resolved)
        result["is_legal"] = is_legal

        if not is_legal:
            logger.info(f"[{qa_id}] 判定为【日常问题】，不检索 → 该样本将剔除")
            result.update(hit_stage=STAGE_GENERAL)
            return result

        # ---- step 3：BM25 查表（对应 ⑤）--------------------------------
        #   命中说明答案来自 law_qa 表，同样没有 Milvus 上下文
        bm25_answer = chain.bm25_search.search(resolved)
        if bm25_answer:
            logger.info(f"[{qa_id}] 命中 BM25 查表 → 该样本无检索上下文，将剔除")
            result.update(hit_stage=STAGE_BM25, answer=bm25_answer)
            return result

        # ---- step 4：Milvus 混合检索 + 重排（对应 ⑥）★ 上下文来源 --------
        optimizer_query = chain.llm_client.query_generate(resolved)
        case_chunk, clause_chunk = chain.milvus_client.search(optimizer_query)

        contexts, detail = build_contexts(case_chunk, clause_chunk)
        result["contexts"] = contexts
        result["context_detail"] = detail
        result["hit_stage"] = STAGE_MILVUS
        logger.info(
            f"[{qa_id}] Milvus 检索到 案例 {len(case_chunk)} 条 / 法条 {len(clause_chunk)} 条，"
            f"整理为 {len(contexts)} 个上下文片段"
        )

        # ---- step 5：生成答案（对应 ⑦）--------------------------------
        if not contexts:
            logger.warning(f"[{qa_id}] 没有检索到任何上下文，跳过生成")
            result.update(hit_stage=STAGE_ERROR, error="检索结果为空")
            return result

        result["answer"] = generate_answer(
            chain, resolved, case_chunk, clause_chunk, qa_id
        )

    except Exception as e:  # noqa: BLE001
        logger.error(f"[{qa_id}] 执行异常：{type(e).__name__}: {e}")
        result.update(hit_stage=STAGE_ERROR, error=f"{type(e).__name__}: {e}")
    finally:
        result["elapsed"] = round(time.time() - t0, 2)

    return result


def _load_law_chain_client():
    """
    函数功能：按【文件路径】加载项目根目录的 main.py，取出 LawChainClient

    为什么不能用 `from main import LawChainClient`：
        本模块所在的包 rag_qa/evaluation/ 下也有一个 main.py（统一入口）。
        当 evaluation 已被作为包导入时，Python 会把模块名 'main'
        解析成 rag_qa/evaluation/main.py，导致
            ImportError: cannot import name 'LawChainClient' from 'main'
        —— 这是模块名撞车，与项目代码无关。

    这里改用 importlib 按绝对路径加载项目根的 main.py，
    并显式注册/恢复 sys.modules['main']，保证：
        1. 拿到的一定是项目根的 main 模块
        2. 项目其他模块若写 `import main`，也能拿到同一个模块对象
        3. 退出时把 sys.modules['main'] 恢复原状，不影响调用方
    :return: LawChainClient 类
    """
    import importlib.util

    main_py = os.path.join(_PROJECT_DIR, "main.py")
    if not os.path.exists(main_py):
        raise FileNotFoundError(f"找不到项目入口文件：{main_py}")

    spec = importlib.util.spec_from_file_location("_law_chain_main", main_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载项目入口文件：{main_py}")

    module = importlib.util.module_from_spec(spec)
    # 用两个名字注册同一个模块对象：
    #   '_law_chain_main' —— 持久命名，避免模块只被弱引用而被垃圾回收
    #   'main'            —— 让项目内部及本模块的 `import main` 拿到同一对象
    saved = sys.modules.get("main")
    sys.modules["_law_chain_main"] = module
    sys.modules["main"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # 加载失败就还原，避免留下半成品模块
        sys.modules.pop("_law_chain_main", None)
        if saved is not None:
            sys.modules["main"] = saved
        else:
            sys.modules.pop("main", None)
        raise

    if not hasattr(module, "LawChainClient"):
        raise ImportError(f"{main_py} 里没有找到 LawChainClient")
    return module.LawChainClient


# ============================================================================
# 主流程
# ============================================================================
def save_run(results: list[dict], raw: bool = False) -> str:
    """把流水线结果写成 JSON"""
    config.ensure_dirs()
    path = config.RUN_RAW_FILE if raw else config.RUN_FILE

    # 原始明细另存一份，主文件里去掉以免过大
    payload = {"meta": {
        "total": len(results),
        "with_context": sum(1 for r in results if r.get("contexts")),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, "results": results}

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"流水线结果已写入：{path}")
    return str(path)


def summarize(results: list[dict]) -> str:
    """生成命中路径分布摘要"""
    stage_names = {
        STAGE_MILVUS: "Milvus 检索（有上下文，参与指标计算）",
        STAGE_BM25: "BM25 查表（无上下文，剔除）",
        STAGE_CACHE: "Redis 缓存（无上下文，剔除）",
        STAGE_GENERAL: "日常问题（不检索，剔除）",
        STAGE_ERROR: "执行出错",
    }
    cnt = Counter(r.get("hit_stage") for r in results)
    lines = ["命中路径分布："]
    for stage, name in stage_names.items():
        if cnt.get(stage):
            lines.append(f"  {name:<34} {cnt[stage]:>3} 条")
    lines.append(f"  {'合计':<34} {len(results):>3} 条")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="复现完整问答链路，产出上下文与答案（只读，不写生产数据）",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 条（调试用，0 表示全部）")
    parser.add_argument("--only", type=str, default="",
                        help="只跑指定 qa_id，多个用逗号分隔，如 M01,M02")
    parser.add_argument("--skip-generate", action="store_true",
                        help="跳过答案生成（只测检索层指标，省 LLM 费用）")
    args = parser.parse_args()

    print(config.describe())

    # ---- 读取数据集 --------------------------------------------------------
    samples = load_dataset()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        samples = [s for s in samples if str(s["qa_id"]) in wanted]
    if args.limit:
        samples = samples[:args.limit]
    print(f"\n[1/4] 加载数据集：{len(samples)} 条")
    if not samples:
        print("  ❌ 没有要跑的样本")
        return 1

    # ---- 初始化链路（只读使用）--------------------------------------------
    print("\n[2/4] 初始化 LawChainClient（会加载 BGE-M3 与 reranker，约 2.3 GB 显存）...")
    LawChainClient = _load_law_chain_client()

    chain = LawChainClient()
    print("  ✅ 初始化完成")

    # ★ 收紧 LLM 客户端的 HTTP 读超时。
    #   项目创建 OpenAI 客户端时没传 timeout，用的是 openai 默认的 read=600s，
    #   实测有样本正好卡满这个值（10 分钟）。这里改成 config.GENERATE_TIMEOUT。
    status = apply_llm_timeout(chain)
    print(f"  ⏱  {status}")

    # ---- 逐条执行（用副作用守卫包住）--------------------------------------
    print(f"\n[3/4] 开始执行 {len(samples)} 条样本 ...")
    results: list[dict] = []
    guard = SideEffectGuard(chain.redis_client, chain.mysql_client)
    try:
        # ⚠️ 注意顺序：chain.close() 必须放在 with 之外。
        #    SideEffectGuard 靠对比"跑前/跑后"的 Redis key 数与 MySQL 行数来发现副作用，
        #    如果在 with 内部就 close()，守卫退出时连接已断、读不到计数，
        #    检查会被静默跳过（_redis_size 返回 None → 不比较 → 看起来一切正常）。
        with guard:
            for i, sample in enumerate(samples, 1):
                print(f"\n  ({i}/{len(samples)}) [{sample['qa_id']}] "
                      f"{sample['eval_question'][:50]}")
                if args.skip_generate:
                    # 只测检索：手动走 step1~step4
                    r = run_one_retrieval_only(chain, sample)
                else:
                    r = run_one(chain, sample)

                if r.get("error"):
                    print(f"      ❌ {r['error']}")
                elif r.get("hit_stage") == STAGE_MILVUS:
                    print(f"      ✅ 上下文 {len(r['contexts'])} 段，"
                          f"答案 {len(r['answer'])} 字，耗时 {r['elapsed']}s")
                else:
                    print(f"      ⏭  命中路径={r.get('hit_stage')}（无上下文，将剔除）")
                results.append(r)
                time.sleep(config.SLEEP_BETWEEN_SAMPLES)
    finally:
        try:
            restore_llm_timeout(chain)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"恢复 LLM 超时失败（可忽略）：{e}")
        try:
            chain.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"关闭链路时出错（可忽略）：{e}")

    # ---- 落盘 + 摘要 -------------------------------------------------------
    print("\n[4/4] 保存结果 ...")
    with_context = sum(1 for r in results if r.get("contexts"))
    if with_context == 0:
        print("  ⚠️ 没有任何样本产生检索上下文！")
        print("     最可能的原因：数据集的问题仍被 BM25 字面命中。")
        print("     请检查 dataset_builder 的改写是否生效。")

    path = save_run(results, raw=True)
    print(f"  ✅ {path}")
    print()
    print(summarize(results))
    print(f"\n  可参与指标计算（有上下文）: {with_context} / {len(results)} 条")
    return 0


def run_one_retrieval_only(chain, sample: dict) -> dict:
    """
    函数功能：只跑到检索阶段，不生成答案（用于只测检索层指标，省 LLM 费用）

    与 run_one 的区别：跳过最后一步 llm_client.generate()。
    """
    result = {
        **sample, "hit_stage": None, "is_legal": None, "resolved": None,
        "contexts": [], "context_detail": [], "answer": "", "error": None,
        "elapsed": 0.0,
    }
    t0 = time.time()
    try:
        question = sample["eval_question"]
        if chain.redis_client.get_answer(question):
            result["hit_stage"] = STAGE_CACHE
            return result
        resolved = chain.llm_client.resolve_query(question, None) or question
        result["resolved"] = resolved
        result["is_legal"] = chain._is_legal_question(resolved)
        if not result["is_legal"]:
            result["hit_stage"] = STAGE_GENERAL
            return result
        if chain.bm25_search.search(resolved):
            result["hit_stage"] = STAGE_BM25
            return result

        optimizer_query = chain.llm_client.query_generate(resolved)
        case_chunk, clause_chunk = chain.milvus_client.search(optimizer_query)
        contexts, detail = build_contexts(case_chunk, clause_chunk)
        result["contexts"] = contexts
        result["context_detail"] = detail
        result["hit_stage"] = STAGE_MILVUS
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{sample['qa_id']}] 执行异常：{type(e).__name__}: {e}")
        result.update(hit_stage=STAGE_ERROR, error=f"{type(e).__name__}: {e}")
    finally:
        result["elapsed"] = round(time.time() - t0, 2)
    return result


if __name__ == "__main__":
    sys.exit(main())
