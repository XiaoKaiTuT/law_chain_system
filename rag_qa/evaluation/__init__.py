"""
RAG 评估模块（v5）

本包用于对 LawChain · 律衡 的检索增强生成链路做量化评估。

设计要点
--------
1. **只读调用**：本包所有代码只调用项目其他模块的公开接口，
   **不修改任何现有文件**，也**不调用任何会写 Redis / MySQL 的方法**
   （刻意绕开 `LawChainClient.search()` 与 `_stream_and_cache()`，
   因为它们内部会 `set_question` / `insert_data`，会污染生产数据）。
2. **链路显式**：不使用 monkey-patch。`pipeline_runner.py` 按
   `main.py` 的 `search()` 相同的顺序显式调用 5 个接口，
   每一步的中间结果（消解后问题、分类结果、检索片段）都能直接看到。
3. **结果可复现**：固定随机种子、中间结果落盘，同一次数据集可反复跑。

⚠️ 维护提醒
-----------
`pipeline_runner.py` 的调用顺序是「照抄」`main.py` 的 `search()` 流程的。
**如果以后修改了 `main.py` 的 `search()` 流程（增删步骤、换接口），
必须同步检查本目录的 `pipeline_runner.py`**，否则评估的将不再是真实链路。

模块索引
--------
    config.py           评估参数集中配置
    dataset_builder.py  从 law_qa 抽样 + LLM 改写成口语化评估问题
    pipeline_runner.py  复现完整问答链路，显式产出上下文与答案
    wrappers.py         BGE-M3 → LangChain Embeddings 包装、裁判 LLM 构造
    ragas_runner.py     组织 RAGAS 计算各项指标
    reporter.py         输出分层指标报告 + 命中路径分布 + 逐条明细
"""

__all__ = [
    "config",
    "dataset_builder",
    "pipeline_runner",
    "wrappers",
    "ragas_runner",
    "reporter",
]
