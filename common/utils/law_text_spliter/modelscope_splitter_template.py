# -*- coding: utf-8 -*-
"""modelscope_splitter_template.py —— 基于 ModelScope 的语义分割"备用模板"。

【定位】可选模板，主流程（legal_chunker / case_chunker）**不调用**它。
把它放在这里，是为了让你以后想升级成"语义切割"时有现成的骨架可抄。

【它做什么】

    长文本 → ① 句子切分（正则切句）
           → ② 用 ModelScope 的 embedding 模型把每个句子变成向量
           → ③ 相邻句子余弦相似度 >= 阈值 就合并，
                合并到超过 max_chunk_chars 也强制断句
           → ④ 输出 common.make_chunk 定义的统一 chunk 格式（chunk_type="semantic_chunk"）

【需要安装什么】

    pip install modelscope torch          # torch 是模型推理依赖（CPU 版即可）
    # 本项目环境已装：modelscope 1.23.0 + torch 2.10.0

【需要下载什么模型】

    iic/nlp_gte_sentence-embedding_chinese-base   （中文句子向量，约 400MB）
    首次运行会自动从 ModelScope 下载到 ~/.cache/modelscope，也可以提前手动下载：

        from modelscope import snapshot_download
        snapshot_download('iic/nlp_gte_sentence-embedding_chinese-base')

【如何替换模型】

    换成其它 embedding 模型（例如更大的 iic/nlp_gte_sentence-embedding_chinese-large），
    只需要改下面 MODEL_ID 常量：

        MODEL_ID = "iic/nlp_gte_sentence-embedding_chinese-large"

    或者实例化时传入：ModelScopeSemanticSplitter(model_id="你的模型id或本地目录")

【没有模型也能跑】

    模型不可用（没网/没下载）时会自动退化成"字符二元组余弦相似度"的轻量实现，
    流程完全一样，只是语义质量差一些——这样模板本身始终可运行、可调试。

【注意】本模板是"语义粗切"，不保证法律条文的完整性；法律/案例的正式切割
仍然应该用 legal_chunker / case_chunker（结构优先）。
"""

from __future__ import annotations

import re
from collections import Counter
from math import sqrt
from pathlib import Path

from langchain_text_splitters import TextSplitter  # 继承 LangChain 的 TextSplitter

try:
    from .chunk_common import make_chunk, normalize_text, preview_chunks, save_jsonl
except ImportError:  # 允许直接 python modelscope_splitter_template.py 运行
    from common import make_chunk, normalize_text, preview_chunks, save_jsonl  # type: ignore[no-redef]

# ======================================================================
# 配置
# ======================================================================

# 默认中文句子向量模型（可在 ModelScope 上搜索替换）
MODEL_ID = "iic/nlp_gte_sentence-embedding_chinese-base"
# 句子切分：在句末标点后断开
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*")


def split_sentences(text: str, *, min_len: int = 2) -> list[str]:
    """把长文本切成句子列表（语义分割的第一步）。

    :param text: 待切分文本
    :param min_len: 过滤掉过短的碎片（如单个换行残留）
    :return: 句子列表
    """
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(normalize_text(text))]
    return [sentence for sentence in sentences if len(sentence) >= min_len]


def _ngram_vector(text: str, n: int = 2) -> Counter:
    """把文本转成"字符 n 元组词频"向量（没有模型时用的退化方案）。

    :param text: 文本
    :param n: n 元组长度
    :return: 词频统计对象
    """
    return Counter(text[i : i + n] for i in range(max(len(text) - n + 1, 1)))


def cosine_similarity(vec_a, vec_b) -> float:
    """计算两个向量的余弦相似度。

    同时兼容：

    - 稠密向量（list / numpy.ndarray，来自 embedding 模型）；
    - 稀疏词频（Counter，来自 :func:`_ngram_vector` 的退化方案）。

    :param vec_a: 向量 A
    :param vec_b: 向量 B
    :return: 0~1 的相似度（算出负数时返回 0）
    """
    if isinstance(vec_a, Counter) or isinstance(vec_b, Counter):
        keys = set(vec_a) | set(vec_b)
        dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in keys)
        norm_a = sqrt(sum(v * v for v in vec_a.values())) or 1.0
        norm_b = sqrt(sum(v * v for v in vec_b.values())) or 1.0
        return dot / (norm_a * norm_b)
    # 稠密向量：numpy 数组也支持这种写法
    dot = float(sum(float(x) * float(y) for x, y in zip(vec_a, vec_b)))
    norm_a = sqrt(sum(float(x) ** 2 for x in vec_a)) or 1.0
    norm_b = sqrt(sum(float(y) ** 2 for y in vec_b)) or 1.0
    return max(dot / (norm_a * norm_b), 0.0)


def load_embedding_pipeline(model_id: str = MODEL_ID, device: str = "cpu"):
    """加载 ModelScope 的句子 embedding pipeline。

    :param model_id: ModelScope 模型 id 或本地模型目录
    :param device: "cpu" 或 "gpu"
    :return: ModelScope pipeline 对象
    :raises RuntimeError: 依赖缺失或模型加载失败
    """
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except ImportError as exc:  # 没装 modelscope
        raise RuntimeError("需要先安装 modelscope：pip install modelscope torch") from exc
    try:
        return pipeline(Tasks.sentence_embedding, model=model_id, device=device)
    except Exception as exc:  # 模型没下载/没网/显存不足等
        raise RuntimeError(f"加载 ModelScope 模型失败：{model_id}（{exc}）") from exc


class ModelScopeSemanticSplitter(TextSplitter):
    """基于 ModelScope embedding 的语义分割器（继承 LangChain 的 TextSplitter）。

    用法::

        splitter = ModelScopeSemanticSplitter(similarity_threshold=0.62)
        chunks = splitter.split_text(long_text)   # 和其它 LangChain splitter 用法一致

    :param model_id: ModelScope embedding 模型 id
    :param similarity_threshold: 相邻句子相似度低于该值就切一刀
    :param max_chunk_chars: 单个 chunk 的最大字符数（硬上限）
    :param min_chunk_chars: 单个 chunk 的最小字符数（避免切出过碎的小块）
    :param device: "cpu" 或 "gpu"
    :param pipeline_obj: 已加载好的 ModelScope pipeline（传入可避免重复加载）
    """

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        similarity_threshold: float = 0.62,
        max_chunk_chars: int = 800,
        min_chunk_chars: int = 120,
        device: str = "gpu",
        pipeline_obj=None,
        **kwargs,
    ) -> None:
        super().__init__(chunk_size=max_chunk_chars, chunk_overlap=0, **kwargs)
        self.model_id = model_id
        self.similarity_threshold = similarity_threshold
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.device = device
        self._pipeline = pipeline_obj
        self._use_model = pipeline_obj is not None  # 没传 pipeline 就先按退化方案走
        self._load_attempted = pipeline_obj is not None  # 避免每次都重复尝试加载模型

    def _ensure_pipeline(self):
        """按需加载 ModelScope pipeline；失败则退化并打印提示。"""
        if self._pipeline is None and not self._load_attempted:
            self._load_attempted = True
            try:
                self._pipeline = load_embedding_pipeline(self.model_id, self.device)
                self._use_model = True
                print(f"[modelscope_splitter_template] 已加载模型：{self.model_id}")
            except RuntimeError as exc:
                print(f"[modelscope_splitter_template] 警告：{exc}")
                print("[modelscope_splitter_template] 将退化为字符二元组相似度（流程不变，质量下降）。")
                self._use_model = False
        return self._pipeline

    def _embed(self, sentences: list[str]) -> list:
        """把句子编码成向量；模型不可用时退化为字符二元组词频。

        :param sentences: 句子列表
        :return: 与句子一一对应的向量列表
        """
        pipeline_obj = self._ensure_pipeline()
        if not self._use_model or pipeline_obj is None:
            return [_ngram_vector(sentence) for sentence in sentences]
        result = pipeline_obj(input={"source_sentence": sentences})
        return list(result["text_embedding"])

    def split_text(self, text: str) -> list[str]:
        """语义分割主流程：切句 → 编码 → 相邻相似度贪心合并。

        :param text: 待分割文本
        :return: chunk 文本列表
        """
        sentences = split_sentences(text)
        if not sentences:
            return []
        vectors = self._embed(sentences)

        chunks: list[str] = []
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(vectors[i - 1], vectors[i])
            merged_len = sum(len(part) for part in current) + len(sentences[i])
            # 相似度够高、且没超过硬上限 → 继续合并；否则断句
            if similarity >= self.similarity_threshold and merged_len <= self.max_chunk_chars:
                current.append(sentences[i])
            else:
                chunks.append("".join(current))
                current = [sentences[i]]
        chunks.append("".join(current))

        # 兜底：过短的块并回上一块，避免切出一堆碎片
        merged: list[str] = []
        for chunk in chunks:
            if merged and len(chunk) < self.min_chunk_chars:
                merged[-1] = merged[-1] + chunk
            else:
                merged.append(chunk)
        return merged


def split_by_semantics(
    text: str,
    source: str = "",
    *,
    model_id: str = MODEL_ID,
    similarity_threshold: float = 0.62,
    max_chunk_chars: int = 800,
    device: str = "cpu",
) -> list[dict]:
    """语义切割的便捷函数：输出与 common.make_chunk 完全一致的统一格式。

    :param text: 纯文本字符串
    :param source: 来源文件名
    :param model_id: ModelScope 模型 id
    :param similarity_threshold: 相似度阈值
    :param max_chunk_chars: 单个 chunk 最大字符数
    :param device: "cpu" 或 "gpu"
    :return: chunk 列表（chunk_type="semantic_chunk"）
    """
    splitter = ModelScopeSemanticSplitter(
        model_id=model_id,
        similarity_threshold=similarity_threshold,
        max_chunk_chars=max_chunk_chars,
        device=device,
    )
    chunks = splitter.split_text(text)
    return [
        make_chunk(
            source=source,
            doc_type="law",  # 语义块不区分法律/案例，这里按调用场景自行修改
            chunk_type="semantic_chunk",
            text=chunk,
            extra={"char_len": len(chunk), "index": i, "model_id": model_id},
        )
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    # 模板自测：直接运行会拿一段内置文本试切（模型不可用时自动退化，仍能跑通）
    demo_text = (
        "彩礼返还的义务主体是否包括女方的父母。"
        "重庆市南川区人民法院经审理认为，以结婚为目的于婚前给付彩礼的行为，"
        "应当被认定为附解除条件的赠与行为。"
        "二审法院认为，何某支付彩礼是双方包括父母在内共同商议的，"
        "且是用卢某某支付宝账户予以接受的，因此卢某某应当成为返还责任人。"
        "本案的裁判要旨在于明确彩礼返还责任主体的复合性认定规则。"
    )
    demo_chunks = split_by_semantics(demo_text, source="demo.txt", max_chunk_chars=80)
    print(f"共切出 {len(demo_chunks)} 个语义 chunk")
    preview_chunks(demo_chunks, n=5, max_len=800)
    print(f"（示例输出文件：{Path('semantic_chunks.jsonl').resolve()}）")
    save_jsonl(demo_chunks, "semantic_chunks.jsonl")
