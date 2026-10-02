# -*- coding: utf-8 -*-
"""law_text_spliter 的公共工具模块。

本模块只关心"纯文本"，不读取任何具体格式的文件（不导入 law_document_loaders）：

1. 文本规范化与通用清洗（分页标记、页码行、书眉页脚、零宽字符、多余空格/空行等）；
2. 统一的 chunk 输出格式（所有切割模块都调用 make_chunk，保证字段完全一致）；
3. JSONL 保存、chunk 预览打印、格式一致性检查。

统一 chunk 格式（JSON 可序列化）::

    {
        "source": "文件名",
        "doc_type": "law" 或 "case",
        "chunk_type": "article" / "case_section" / "semantic_chunk",
        "text": "切出来的正文",
        "metadata": {
            "law_name": "民事诉讼法",
            "path": ["第一编 总则", "第一章 ..."],
            "article_no": "第一条",
            "case_no": "1",
            "case_title": "彩礼适格返还主体...",
            "case_subtitle": "——何某诉...",
            "section": "基本案情",
            "extra": {}
        }
    }
"""

from __future__ import annotations

import json
import re
import sys
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

current_dir: str = os.path.dirname(os.path.abspath(__file__))
common_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(common_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# ======================================================================
# 一、统一 chunk 格式
# ======================================================================

# 统一格式的字段清单（用于自检：所有 chunk 的字段必须完全一致）
CHUNK_KEYS: tuple[str, ...] = ("source", "doc_type", "chunk_type", "text", "metadata")
METADATA_KEYS: tuple[str, ...] = (
    "law_name",
    "path",
    "article_no",
    "case_no",
    "case_title",
    "case_subtitle",
    "section",
    "extra",
)


def make_chunk(
    *,
    source: str,
    doc_type: str,
    chunk_type: str,
    text: str,
    law_name: str = "",
    path: list[str] | None = None,
    article_no: str = "",
    case_no: str = "",
    case_title: str = "",
    case_subtitle: str = "",
    section: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按统一格式组装一个 chunk（所有切割模块都必须调用它，字段才完全一致）。

    :param source: 来源文件名
    :param doc_type: 文档类型，"law"（法律）或 "case"（案例）
    :param chunk_type: chunk 类型，"article" / "case_section" / "semantic_chunk"
    :param text: chunk 正文
    :param law_name: 法律名称（案例留空）
    :param path: 层级路径，例如 ["第一编 总则", "第一章 ..."]
    :param article_no: 条号，例如 "第一条"（案例留空）
    :param case_no: 案例编号，例如 "1"
    :param case_title: 案例主标题
    :param case_subtitle: 案例副标题（含 "——"）
    :param section: 案例小节名，例如 "基本案情"
    :param extra: 其他附加信息（自由字段）
    :return: 统一格式的 chunk 字典
    """
    return {
        "source": source,
        "doc_type": doc_type,
        "chunk_type": chunk_type,
        "text": text,
        "metadata": {
            "law_name": law_name,
            "path": list(path or []),
            "article_no": article_no,
            "case_no": case_no,
            "case_title": case_title,
            "case_subtitle": case_subtitle,
            "section": section,
            "extra": dict(extra or {}),
        },
    }


def assert_uniform_format(chunks: list[dict[str, Any]]) -> None:
    """检查所有 chunk 的字段是否完全一致（验收标准 3 的自检函数）。

    :param chunks: 待检查的 chunk 列表
    :raises ValueError: 字段缺失或多余时抛出
    """
    for i, chunk in enumerate(chunks):
        if tuple(chunk.keys()) != CHUNK_KEYS:
            raise ValueError(f"第 {i} 个 chunk 的顶层字段不符合统一格式: {list(chunk.keys())}")
        metadata = chunk.get("metadata", {})
        if tuple(metadata.keys()) != METADATA_KEYS:
            raise ValueError(f"第 {i} 个 chunk 的 metadata 字段不符合统一格式: {list(metadata.keys())}")


# ======================================================================
# 二、文本规范化与通用清洗
# ======================================================================

# 零宽字符、方向控制字符（docx 里常出现 \u200b）
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]")
# 分页标记：===== Page 1 ===== / --------Page (0)-------- / 第 3 页
PAGE_MARKER_RE = re.compile(
    r"^[\s=\-*_#.]*(?:page\s*\(?\d+\)?|第\s*\d+\s*页)[\s=\-*_#.]*$",
    re.IGNORECASE,
)
# 页码行：整行只有一个数字
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
# 脚注编号单独成行：① ② ...
FOOTNOTE_ONLY_RE = re.compile(r"^[①-⑳]\s*$")
# 书眉 / 版权页等固定噪声行（案例书特有的，可自行增删）
NOISE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^中国法院\d{4}年度案例"),  # 书眉："中国法院2026年度案例 ·婚姻家庭与继承纠纷"
    re.compile(r"^\d{4}\s*年\s*\d{1,2}\s*月$"),  # 出版日期行："2026年6月"
    re.compile(r"^目\s*录$"),  # 单独成行的"目录"
    re.compile(r"^图书在版编目"),  # 版权页
)


def normalize_text(text: str) -> str:
    """基础规范化：统一换行、去掉零宽字符、合并连续空格、压缩连续空行。

    注意：本函数不会删除任何"行"，只做字符级别整理，因此可以安全地用在
    "靠行号定位标题块"的场景（例如案例标题回溯）之前。

    :param text: 原始文本
    :return: 规范化后的文本
    """
    if not text:
        return ""
    # 1) 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2) 去掉零宽字符（docx 每个段落之间常常有一个 \u200b）
    text = ZERO_WIDTH_RE.sub("", text)
    # 3) 全角空格、制表符统一成普通空格
    text = text.replace("\u3000", " ").replace("\t", " ")
    # 4) 逐行去首尾空格并合并连续空格（PDF 抽取结果里常有 "一 、婚姻家庭纠纷" 这种碎空格）
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # 5) 连续空行压缩为一个空行
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    """判断一行是否为"非正文"行（通用清洗用，包含纯数字页码行）。

    :param line: 待判断的一行文本
    :return: 是噪声行返回 True
    """
    s = line.strip()
    if not s:
        return True
    if PAGE_MARKER_RE.match(s) or PAGE_NUMBER_RE.match(s) or FOOTNOTE_ONLY_RE.match(s):
        return True
    return any(pattern.match(s) for pattern in NOISE_LINE_PATTERNS)


def is_block_noise_line(line: str) -> bool:
    """判断一行是否为"标题块附近的噪声行"（案例标题回溯专用）。

    与 :func:`is_noise_line` 的唯一区别：**不把纯数字行当作噪声**。
    因为案例标题块里的"编号行"（一行只有一个数字，如 "1"）本身是有效信息，
    而它和页码行长得完全一样，所以回溯标题时必须保留它。

    :param line: 待判断的一行文本
    :return: 是噪声行返回 True
    """
    s = line.strip()
    if not s:
        return True
    if PAGE_MARKER_RE.match(s) or FOOTNOTE_ONLY_RE.match(s):
        return True
    return any(pattern.match(s) for pattern in NOISE_LINE_PATTERNS)


def compact_line(line: str) -> str:
    """去掉行内所有空白字符，用于"重复行"的比较。

    PDF 抽取常常给同一句书眉配上不同的空格排版（"一 、婚姻家庭纠纷" /
    "一、婚姻家庭纠纷"），去掉空白后它们才是同一个东西。

    :param line: 一行文本
    :return: 去掉空白后的紧凑文本
    """
    return re.sub(r"\s+", "", line)


def find_repeated_lines(text: str, *, min_repeat: int = 6, max_len: int = 30) -> set[str]:
    """找出全文反复出现的短行（PDF 每页的书眉/页脚）。

    为什么需要它：案例 PDF 每一页顶部都有页眉（"中国法院2026年度案例 ·婚姻家庭与继承纠纷"
    或"一 、婚姻家庭纠纷"+页码）。逐案例切片后单看一个案例只出现几次，所以必须先
    在**全文范围**统计，再把结果传给逐案例清洗。

    但"反复出现"本身不足以判定噪声——正文里也有重复的行（例如判决书里的
    "驳回上诉，维持原判。"，或者被折行截断的法条名称）。所以这里要同时满足三个条件：

    1. **短**：长度 <= max_len，且不以数字开头（排除 "1. 裁判书字号" 这类编号字段）；
    2. **不像句子**：不含中文句读标点（。，、；：！？），排除正文与"字段：取值"行；
    3. **紧挨着页码行**：上下 1 行之内有一个"整行只有数字"的行，
       这正是书眉/页脚在版面上的位置特征。

    :param text: 全文文本
    :param min_repeat: 至少重复多少次才算噪声
    :param max_len: 参与统计的最大行长（长段落天然不会重复，无需统计）
    :return: 需要删除的"紧凑行内容"集合（比较时用 :func:`compact_line` 处理待清洗的行）
    """
    lines = [line.strip() for line in text.split("\n")]
    counter: Counter[str] = Counter()
    for line in lines:
        key = compact_line(line)
        if not (0 < len(key) <= max_len):
            continue
        if key[0].isdigit():  # 编号字段（"1. 裁判书字号"）属于正文
            continue
        if re.search(r"[。，；：！？]", key):  # 像句子的行属于正文
            continue
        counter[key] += 1

    repeated = {line for line, count in counter.items() if count >= min_repeat}
    if not repeated:
        return set()

    # 再检查"是否紧挨页码行"：书眉/页脚总是和页码在同一页的顶部/底部
    result: set[str] = set()
    for i, line in enumerate(lines):
        key = compact_line(line)
        if key not in repeated:
            continue
        neighbors = lines[max(i - 1, 0) : i + 2]
        if any(PAGE_NUMBER_RE.match(neighbor) for neighbor in neighbors):
            result.add(key)
    return result


def clean_text(
    text: str,
    *,
    extra_drop_lines: set[str] | None = None,
    extra_noise_patterns: tuple[re.Pattern[str], ...] | None = None,
    drop_repeated_min: int = 0,
    max_repeated_len: int = 40,
) -> str:
    """通用清洗：分页标记 → 页码行 → 书眉/页脚 → 多余空格与连续空行。

    前置内容剥离完成之后再调用本函数。

    :param text: 待清洗文本（一般是剥离前置内容之后的正文）
    :param extra_drop_lines: 额外要删除的行内容（如 :func:`find_repeated_lines` 的结果）
    :param extra_noise_patterns: 额外要删除的行正则（调用方自定义，例如案例的脚注行）
    :param drop_repeated_min: > 0 时，对传入文本自身再做一次重复行统计并删除
    :param max_repeated_len: 重复行统计的最大行长
    :return: 清洗后的文本
    """
    lines = normalize_text(text).split("\n")
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append("")  # 空行先保留，最后统一压缩
            continue
        if is_noise_line(s):  # 分页标记 / 页码 / 脚注编号 / 书眉
            continue
        if extra_noise_patterns and any(pattern.match(s) for pattern in extra_noise_patterns):
            continue
        # 全文重复出现的书眉/页脚（比较时统一去掉行内空白）
        if extra_drop_lines and compact_line(s) in extra_drop_lines:
            continue
        kept.append(s)
    cleaned = "\n".join(kept)
    if drop_repeated_min > 0:
        repeated = find_repeated_lines(cleaned, min_repeat=drop_repeated_min, max_len=max_repeated_len)
        cleaned = "\n".join(line for line in cleaned.split("\n") if line.strip() not in repeated)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)  # 连续空行压缩
    return cleaned.strip()


def extract_law_name(text: str, *, max_scan_lines: int = 30) -> str:
    """从文本开头若干行中识别法律名称（例如"中华人民共和国民事诉讼法"）。

    :param text: 原始全文
    :param max_scan_lines: 只在开头多少行内查找
    :return: 法律名称；识别不到返回空字符串
    """
    pattern = re.compile(
        r"^(中华人民共和国[\u4e00-\u9fa5]{1,12}|[\u4e00-\u9fa5]{2,15}(?:法|条例|规定|解释|办法))(?:（[^）]*）)?$"
    )
    for line in normalize_text(text).split("\n")[:max_scan_lines]:
        match = pattern.match(line.strip())
        if match:
            return match.group(1)
    return ""


# ======================================================================
# 三、输出：JSONL 保存 + 预览打印
# ======================================================================


def save_jsonl(chunks: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """把 chunk 列表逐行写入 JSONL 文件（UTF-8，中文不转义）。

    :param chunks: chunk 列表
    :param output_path: 输出文件路径（相对路径按当前工作目录解析）
    :return: 写入文件的绝对路径
    """
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return path


def preview_chunks(chunks: list[dict[str, Any]], *, n: int = 5, max_len: int = 800) -> None:
    """格式化打印前 n 个 chunk，每个截断到 max_len 字符，方便人工检查。

    :param chunks: chunk 列表
    :param n: 打印个数
    :param max_len: 单个 chunk 的最大显示字符数
    """
    logger.info(f"\n===== 前 {min(n, len(chunks))} 个 chunk（每个最多显示 {max_len} 字符）=====")
    for i, chunk in enumerate(chunks[:n], 1):
        payload = json.dumps(chunk, ensure_ascii=False, indent=2)
        if len(payload) > max_len:
            payload = payload[:max_len] + f"\n  ... [已截断，完整内容见 JSONL 文件，原长度 {len(payload)} 字符]"
        logger.info(f"---------- chunk #{i} ----------")
        logger.info(payload)


def print_summary(chunks: list[dict[str, Any]], *, title: str = "切割结果") -> None:
    """打印切割结果概要（总数 + 类型分布），便于快速确认效果。

    :param chunks: chunk 列表
    :param title: 概要标题
    """
    counter: Counter[str] = Counter(chunk["chunk_type"] for chunk in chunks)
    logger.info(f"\n===== {title} =====")
    logger.info(f"chunk 总数：{len(chunks)}")
    for chunk_type, count in counter.items():
        logger.info(f"  - {chunk_type}: {count}")
    if chunks:
        logger.info(f"第一个 chunk 的正文开头：{chunks[0]['text'][:40]!r}")
