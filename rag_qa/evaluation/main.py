"""
函数功能：RAG 评估的统一入口（一键跑完整流程）

四步流水线：
    ① 构建数据集  dataset_builder  —— 从 law_qa 抽样 + 改写成口语化提问
    ② 跑问答链路  pipeline_runner  —— 产出上下文与答案（只读，不写生产数据）
    ③ 计算指标    ragas_runner     —— 用 RAGAS 算分层指标
    ④ 生成报告    reporter         —— 输出 Markdown 报告

用法：
    # 一键跑完整流程（推荐）
    python -m rag_qa.evaluation.main

    # 快速试跑（3 条样本，全部步骤都缩小规模）
    python -m rag_qa.evaluation.main --quick

    # 只跑到检索，不生成答案（省 LLM 费用，只测检索层指标）
    python -m rag_qa.evaluation.main --skip-generate

    # 只想重新出报告（不重跑前面的步骤）
    python -m rag_qa.evaluation.main --only reporter

    # 从第 ③ 步开始跑（复用已有的数据集与链路结果）
    python -m rag_qa.evaluation.main --from-step ragas

    # 只看抽样结果，零成本零 LLM 调用
    python -m rag_qa.evaluation.main --only dataset --preview-only

⚠️ 前提：MySQL / Redis / Milvus 三个服务都必须在运行。
   全部代码只调用项目其他模块的只读接口，不修改任何现有文件。
"""
import argparse
import os
import sys
import time

# ---- 路径注入（bootstrap 必须最先，早于任何项目模块）----------------------
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_QA_DIR = os.path.dirname(_EVAL_DIR)
_PROJECT_DIR = os.path.dirname(_RAG_QA_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from rag_qa.evaluation import bootstrap                                  # noqa: E402,F401
from rag_qa.evaluation import config                                     # noqa: E402

# 步骤定义：名称 -> (序号, 中文标题)
STEPS = {
    "dataset": (1, "构建评估数据集"),
    "pipeline": (2, "跑问答链路（产出上下文与答案）"),
    "ragas": (3, "计算 RAGAS 指标"),
    "reporter": (4, "生成 Markdown 报告"),
}
ALL_STEPS = ["dataset", "pipeline", "ragas", "reporter"]
STEP_ALIAS = {
    "1": "dataset", "2": "pipeline", "3": "ragas", "4": "reporter",
    "dataset_builder": "dataset", "pipeline_runner": "pipeline",
    "ragas_runner": "ragas", "report": "reporter",
}


def normalize_step(name: str) -> str | None:
    """把用户输入（数字/别名）规范成步骤键"""
    key = (name or "").strip().lower()
    key = STEP_ALIAS.get(key, key)
    return key if key in STEPS else None


# ============================================================================
# 各步骤的执行函数
# ============================================================================
def run_dataset(size: int, preview_only: bool = False,
                no_rewrite: bool = False, no_manual: bool = False) -> bool:
    """步骤 ①：构建评估数据集"""
    from rag_qa.evaluation.dataset_builder import main as db_main

    argv = ["dataset_builder", "--size", str(size)]
    if preview_only:
        argv.append("--preview-only")
    if no_rewrite:
        argv.append("--no-rewrite")
    if no_manual:
        argv.append("--no-manual")

    saved, sys.argv = sys.argv, argv
    try:
        return db_main() == 0
    finally:
        sys.argv = saved


def run_pipeline(limit: int = 0, only: str = "", skip_generate: bool = False) -> bool:
    """
    步骤 ②：跑问答链路

    注意：链路执行完会 chain.close() 释放 BGE-M3，
    因此 ③ 无法复用它的向量工具，会自行重新加载（各一次，无法避免）。
    """
    from rag_qa.evaluation.pipeline_runner import main as pr_main

    argv = ["pipeline_runner"]
    if limit:
        argv += ["--limit", str(limit)]
    if only:
        argv += ["--only", only]
    if skip_generate:
        argv.append("--skip-generate")

    saved, sys.argv = sys.argv, argv
    try:
        return pr_main() == 0
    finally:
        sys.argv = saved


def run_ragas(limit: int = 0, vector_tools=None) -> bool:
    """步骤 ③：计算 RAGAS 指标"""
    from rag_qa.evaluation.ragas_runner import main as rr_main

    argv = ["ragas_runner"]
    if limit:
        argv += ["--limit", str(limit)]

    saved, sys.argv = sys.argv, argv
    try:
        return rr_main() == 0
    finally:
        sys.argv = saved


def run_reporter(do_print: bool = False) -> bool:
    """步骤 ④：生成报告"""
    from rag_qa.evaluation.reporter import main as rp_main

    argv = ["reporter"]
    if do_print:
        argv.append("--print")

    saved, sys.argv = sys.argv, argv
    try:
        return rp_main() == 0
    finally:
        sys.argv = saved


# ============================================================================
# 主流程
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rag_qa.evaluation.main",
        description="RAG 评估统一入口（一键跑完 数据集 → 链路 → 指标 → 报告）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m rag_qa.evaluation.main                  # 完整流程\n"
            "  python -m rag_qa.evaluation.main --quick          # 快速试跑 3 条\n"
            "  python -m rag_qa.evaluation.main --only reporter  # 只重出报告\n"
            "  python -m rag_qa.evaluation.main --from-step ragas\n"
            "  python -m rag_qa.evaluation.main --skip-generate  # 只测检索层\n"
        ),
    )
    p.add_argument("--size", type=int, default=config.SAMPLE_SIZE,
                   help=f"数据集抽样条数（默认 {config.SAMPLE_SIZE}）")
    p.add_argument("--only", type=str, default="",
                   help="只跑指定步骤（dataset/pipeline/ragas/reporter，或 1/2/3/4）")
    p.add_argument("--from-step", type=str, default="",
                   help="从指定步骤开始跑（含该步）")
    p.add_argument("--quick", action="store_true",
                   help="快速试跑：3 条样本，链路与指标都只跑少量")
    p.add_argument("--limit", type=int, default=0,
                   help="链路与指标各自的条数上限（默认不限）")
    p.add_argument("--pipeline-limit", type=int, default=0,
                   help="只限制【链路】步骤的条数（优先级高于 --limit）")
    p.add_argument("--ragas-limit", type=int, default=0,
                   help="只限制【指标】步骤的条数（优先级高于 --limit）")
    p.add_argument("--generate-timeout", type=int, default=0,
                   help=f"单条样本答案生成的超时秒数（默认 {config.GENERATE_TIMEOUT}）")
    p.add_argument("--only-id", type=str, default="",
                   help="链路只跑指定 qa_id，多个用逗号分隔，如 M02,M03")
    p.add_argument("--skip-generate", action="store_true",
                   help="链路跳过答案生成（只测检索层指标，省 LLM 费用）")
    p.add_argument("--preview-only", action="store_true",
                   help="数据集只抽样预览，不调 LLM 改写、不落盘")
    p.add_argument("--no-rewrite", action="store_true",
                   help="数据集不做改写（⚠️ 会导致评估被缓存/BM25 短路）")
    p.add_argument("--no-manual", action="store_true",
                   help="数据集不合并手工样本")
    p.add_argument("--print-report", action="store_true",
                   help="在终端同时打印报告正文")
    p.add_argument("--yes", "-y", action="store_true",
                   help="跳过交互确认（用于脚本/CI）")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # ---- 决定要跑哪些步骤 --------------------------------------------------
    if args.quick:
        args.size = min(args.size, 3)
        args.limit = args.limit or 3

    # 超时可临时覆盖（便于网络慢时放宽、或想更快失败时收紧）
    if args.generate_timeout > 0:
        config.GENERATE_TIMEOUT = args.generate_timeout
        print(f"  （已覆盖）生成超时 = {config.GENERATE_TIMEOUT}s")

    # 各步骤的条数上限：细粒度参数优先于 --limit
    pipeline_limit = args.pipeline_limit or args.limit
    ragas_limit = args.ragas_limit or args.limit

    if args.only:
        key = normalize_step(args.only)
        if not key:
            print(f"❌ 未知步骤：{args.only}（可选：{', '.join(ALL_STEPS)} 或 1~4）")
            return 2
        todo = [key]
    elif args.from_step:
        key = normalize_step(args.from_step)
        if not key:
            print(f"❌ 未知步骤：{args.from_step}（可选：{', '.join(ALL_STEPS)} 或 1~4）")
            return 2
        todo = ALL_STEPS[ALL_STEPS.index(key):]
    else:
        todo = list(ALL_STEPS)

    # ---- 打印计划 ----------------------------------------------------------
    print("=" * 72)
    print("RAG 评估 · 统一入口")
    print("=" * 72)
    print(f"  项目根目录 : {_PROJECT_DIR}")
    print(f"  产物目录   : {config.OUTPUT_DIR}")
    print(f"  抽样条数   : {args.size}{'（快速试跑）' if args.quick else ''}")
    print(f"  执行步骤   : {' → '.join(f'{STEPS[s][0]}.{STEPS[s][1]}' for s in todo)}")
    if args.skip_generate:
        print("  答案生成   : 跳过（只测检索层指标）")
    if args.preview_only:
        print("  数据集     : 仅预览，不调 LLM、不落盘")
    print("=" * 72)

    # ---- 前提条件提醒 ------------------------------------------------------
    if "pipeline" in todo and not args.yes:
        print()
        print("⚠️ 接下来会连接 MySQL / Redis / Milvus 并加载 BGE-M3 与 reranker")
        print("   （约 2.3 GB 显存）。请确认三个服务都已启动。")
        try:
            ans = input("   继续？[Y/n] ").strip().lower()
        except EOFError:
            ans = "y"          # 非交互环境（如 CI）默认继续
        if ans and ans not in ("y", "yes"):
            print("   已取消。")
            return 0

    # ---- 逐步执行 ----------------------------------------------------------
    results: dict[str, bool] = {}
    t_all = time.time()

    for step in todo:
        idx, title = STEPS[step]
        print()
        print("#" * 72)
        print(f"#  步骤 {idx}/4 · {title}")
        print("#" * 72)
        t0 = time.time()

        try:
            if step == "dataset":
                ok = run_dataset(
                    size=args.size,
                    preview_only=args.preview_only,
                    no_rewrite=args.no_rewrite,
                    no_manual=args.no_manual,
                )
            elif step == "pipeline":
                if args.preview_only:
                    print("  （数据集为预览模式，未落盘，跳过链路执行）")
                    ok = True
                else:
                    ok = run_pipeline(
                        limit=pipeline_limit, only=args.only_id,
                        skip_generate=args.skip_generate,
                    )
            elif step == "ragas":
                if args.preview_only:
                    print("  （数据集为预览模式，跳过指标计算）")
                    ok = True
                else:
                    ok = run_ragas(limit=ragas_limit)
            else:  # reporter
                if args.preview_only:
                    print("  （数据集为预览模式，跳过报告生成）")
                    ok = True
                else:
                    ok = run_reporter(do_print=args.print_report)
        except KeyboardInterrupt:
            print("\n  ⚠️ 被用户中断")
            results[step] = False
            break
        except Exception as e:  # noqa: BLE001
            print(f"\n  ❌ {type(e).__name__}: {e}")
            ok = False

        elapsed = time.time() - t0
        results[step] = ok
        print(f"\n  {'✅ 完成' if ok else '❌ 失败'}（{elapsed:.1f}s）")

        if not ok:
            print(f"\n  ⛔ 步骤「{title}」失败，后续步骤已跳过。")
            break

    # ---- 汇总 --------------------------------------------------------------
    print()
    print("=" * 72)
    print("执行汇总")
    print("=" * 72)
    for step in todo:
        if step not in results:
            print(f"  ⏭   {STEPS[step][0]}. {STEPS[step][1]:<34} 未执行")
        else:
            print(f"  {'✅' if results[step] else '❌'}   "
                  f"{STEPS[step][0]}. {STEPS[step][1]:<34} "
                  f"{'成功' if results[step] else '失败'}")
    print(f"\n  总耗时：{time.time() - t_all:.1f}s")

    if results.get("reporter"):
        print(f"\n  📄 报告：{config.REPORT_FILE}")
        # 顺带回显关键分数
        try:
            from rag_qa.evaluation.ragas_runner import load_ragas_result

            ragas = load_ragas_result()
            print("\n  指标汇总：")
            for k, v in ragas["summary"].items():
                print(f"    {k:<24} {v if v is not None else 'N/A'}")
        except Exception:  # noqa: BLE001
            pass

    return 0 if all(results.get(s, False) for s in todo) else 1


if __name__ == "__main__":
    sys.exit(main())
