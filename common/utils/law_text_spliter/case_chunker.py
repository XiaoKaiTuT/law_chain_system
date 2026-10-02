# -*- coding: utf-8 -*-
"""case_chunker.py —— 法院案例切割模块（以《中国法院2026年度案例》婚姻家庭继承卷为例）。

核心函数::

    def split_cases(text: str, source: str = "") -> list[dict]

契约：

- 输入是**纯文本字符串**，输出是 JSON 可序列化的 ``list[dict]``；
- 不读文件、不写文件、不 import law_document_loaders（读文件由 loader 负责）；
- 只有 ``__main__`` 命令行测试入口才会调用 ``law_document_loaders`` 读 pdf。

切割流程::

    原始文本 → ① 找"【案件基本信息】"锚点，向前回溯案例标题块（编号 / 主标题 / 副标题）
             → ② 丢弃标题块之前的所有内容（封面 / 序 / 编委会 / 目录）
             → ③ 按"【案件基本信息】"把正文切成一个个案例
             → ④ 每个案例按 5 个 section 标记切块；过长的 section 再用
               LangChain 的 RecursiveCharacterTextSplitter 细切（保留 section 元数据）
"""

from __future__ import annotations

import argparse
import re
import os
import sys
from pathlib import Path

current_dir: str = os.path.dirname(os.path.abspath(__file__))
common_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(common_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:  # 作为包导入时：from common.law_text_spliter.case_chunker import split_cases
    from .chunk_common import (
        assert_uniform_format,
        clean_text,
        find_repeated_lines,
        is_block_noise_line,
        make_chunk,
        normalize_text,
        preview_chunks,
        print_summary,
        save_jsonl,
    )
except ImportError:  # 作为脚本直接运行时：python case_chunker.py ...
    from common import (  # type: ignore[no-redef]
        assert_uniform_format,
        clean_text,
        find_repeated_lines,
        is_block_noise_line,
        make_chunk,
        normalize_text,
        preview_chunks,
        print_summary,
        save_jsonl,
    )

# ======================================================================
# 一、正则工具
# ======================================================================

# 案例 section 标记（顺序即书中的出现顺序）
CASE_SECTION_NAMES: tuple[str, ...] = ("案件基本信息", "基本案情", "案件焦点", "法院裁判要旨", "法官后语")
# section 标记行：整行只有"【基本案情】"（允许尾部跟脚注编号 ①）
RE_SECTION = re.compile(r"^【(" + "|".join(CASE_SECTION_NAMES) + r")】\s*[①-⑳]?\s*$")
# 案例锚点：整行就是"【案件基本信息】"
RE_CASE_ANCHOR = re.compile(r"^【案件基本信息】\s*[①-⑳]?\s*$")
# 案例编号行：整行只有一个数字，例如 "1"
RE_CASE_NO = re.compile(r"^\d{1,4}$")
# 副标题行：以"——"开头，例如 "——何某诉卢某某、窦某某婚约财产案"
RE_SUBTITLE = re.compile(r"^[—\-－]{1,2}\s*\S+")
# 案由小节行："( 一 )婚约财产纠纷" / "（二）离婚纠纷"
RE_SECTION_TITLE = re.compile(r"^[（(]\s*[一二三四五六七八九十]+\s*[)）]\s*\S+")
# 章标题行："一 、婚姻家庭纠纷"（短行 + 以纠纷/争议结尾，避免误判判决主文）
RE_CHAPTER_TITLE = re.compile(r"^[一二三四五六七八九十]+\s*、\s*[\u4e00-\u9fa5、\s]{2,12}(?:纠纷|争议|其他)$")
# 主标题自带编号："1. 彩礼适格返还主体..." / "1、彩礼..."
RE_NUMBERED_TITLE = re.compile(r"^(\d{1,4})\s*[.、]\s*(.+)$")

# 案例书特有的噪声行（脚注）："① 本书【法院裁判要旨】部分适用……"、"本书【法官后语】对……"
CASE_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[①-⑳]"),  # 脚注标记开头的行
    re.compile(r"^本书【"),  # 脚注正文
)
# 不允许被"重复行清理"删除的结构标记（它们在全书重复出现，但必须保留）
RE_STRUCTURE_MARKERS = (RE_SECTION, RE_CASE_ANCHOR, RE_SUBTITLE)


def build_case_splitter(chunk_size: int = 1200, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    """构造案例 section 的细切 splitter（LangChain 组件）。

    案例是叙事文本，没有"条"这种强边界，所以按"空行 → 换行 → 句号 → 分号 → 逗号"
    从粗到细的顺序切，尽量保持语义完整。

    :param chunk_size: 单个 chunk 的最大字符数
    :param chunk_overlap: 相邻 chunk 的重叠字符数
    :return: RecursiveCharacterTextSplitter 实例
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，"],
    )


# ======================================================================
# 二、前置内容剥离：锚点 + 回溯案例标题块
# ======================================================================


def _prev_content_index(lines: list[str], index: int, *, max_skip: int = 3) -> int:
    """从 index 往上找最近的"内容行"，最多跳过 max_skip 行空行/噪声行。

    这里用 :func:`is_block_noise_line`（**不把纯数字行当噪声**），
    因为案例标题块里的"编号行"整行就是一个数字，和页码长得很像。

    :param lines: 全部行
    :param index: 起始行号（从这个位置往上找）
    :param max_skip: 最多跳过多少行
    :return: 内容行行号；跳过太多或到顶了返回 -1
    """
    skipped = 0
    while index >= 0 and skipped <= max_skip:
        if lines[index].strip() and not is_block_noise_line(lines[index]):
            return index
        index -= 1
        skipped += 1
    return -1


def _find_chapter_title(lines: list[str], start: int, *, max_lookback: int = 200) -> str:
    """向上找最近的章标题（如"一 、婚姻家庭纠纷"），作为案例 path 的第一级。

    :param lines: 全部行
    :param start: 案例起点行号
    :param max_lookback: 最多向上看多少行
    :return: 章标题（已去掉空格，如"一、婚姻家庭纠纷"）；找不到返回空字符串
    """
    for i in range(start - 1, max(start - max_lookback, -1), -1):
        line = lines[i].strip()
        if len(line) <= 20 and RE_CHAPTER_TITLE.match(line):
            return _compact(line)
    return ""


def _compact(text: str) -> str:
    """去掉标题里的所有空白字符（PDF 抽取会给出"一 、婚姻家庭纠纷""( 三 ) 离 婚 后 财 产 纠 纷"）。

    :param text: 原始标题
    :return: 紧凑标题，例如 "一、婚姻家庭纠纷"、"(三)离婚后财产纠纷"
    """
    return re.sub(r"\s+", "", text)


def is_title_continuation(line: str, *, max_len: int = 60, min_len: int = 6) -> bool:
    """判断某一行是否可能是"主标题被折行后的上一行"。

    判据：非空、不是噪声、不是编号行/案由行/section 标记，长度在 [min_len, max_len]
    之间，不含中文冒号"："（排除"编写人：某某法院"这类署名行），并且不以句末标点
    （。！？；）结尾——标题被折行时通常断在半句中间。

    :param line: 待判断的一行
    :param max_len: 标题行的最大长度
    :param min_len: 标题行的最小长度（作者姓名往往只有 2~3 个字，用来排除署名行）
    :return: 可能是标题折行返回 True
    """
    s = line.strip()
    if not s or not (min_len <= len(s) <= max_len):
        return False
    if "：" in s:  # "编写人：北京市海淀区人民法院" 这类署名行不是标题
        return False
    if is_block_noise_line(s) or RE_CASE_NO.match(s) or RE_SECTION_TITLE.match(s):
        return False
    if any(marker.match(s) for marker in (RE_SECTION, RE_CASE_ANCHOR, RE_SUBTITLE)):
        return False
    return not re.search(r"[。！？；]$", s)


def locate_case_title_block(lines: list[str], anchor: int) -> dict:
    """从"【案件基本信息】"锚点向上回溯案例标题块，并提取案例元信息。

    标题块典型结构（从上到下）::

        ( 一 )婚约财产纠纷                ← 案由小节（可能有）
        1                                ← 案例编号（可能有）
        彩礼适格返还主体复合性的司法认定    ← 主标题
        ——何某诉卢某某、窦某某婚约财产案    ← 副标题（可能有）
        【案件基本信息】                  ← 锚点

    回溯时严格"一层一层往上"，遇到不符合格式的行立即停止，因此不会跑到上一个
    案例的正文里；中间的页码/书眉等噪声行会被跳过。

    :param lines: 规范化后的全部行
    :param anchor: "【案件基本信息】"所在行号
    :return: {"case_no", "case_title", "case_subtitle", "section", "path", "start"}
    """
    info: dict = {
        "case_no": "",
        "case_title": "",
        "case_subtitle": "",
        "section": "",
        "path": [],
        "start": anchor,
    }

    # 1) 副标题：紧邻锚点上方、以 —— 开头的那一行
    index = _prev_content_index(lines, anchor - 1)
    if index >= 0 and RE_SUBTITLE.match(lines[index].strip()):
        info["case_subtitle"] = lines[index].strip()
        index = _prev_content_index(lines, index - 1)

    # 2) 主标题：副标题上方的那一行内容
    if index >= 0:
        info["case_title"] = lines[index].strip()
        info["start"] = index
        # 标题在 PDF 里常常被折成两行（"诉讼期间父母一方收入状况发生变化时" +
        # "抚养费支付标准的认定"），这里把折行的标题逐行合并回去。
        # 注意：只在"紧邻上一行"里收，不跳过空行/页码，避免越过页边界，
        # 把上一个案例末尾的正文也吞进来。
        heading_lines = [info["case_title"]]
        while len(heading_lines) < 3 and info["start"] - 1 >= 0 and is_title_continuation(lines[info["start"] - 1]):
            info["start"] -= 1
            heading_lines.insert(0, lines[info["start"]].strip())
        info["case_title"] = "".join(heading_lines)
        index = _prev_content_index(lines, info["start"] - 1)

    # 3) 案例编号：主标题上方那一行如果是纯数字
    if index >= 0 and RE_CASE_NO.match(lines[index].strip()):
        info["case_no"] = lines[index].strip()
        info["start"] = index
        index = _prev_content_index(lines, index - 1)

    # 4) 案由小节：编号上方那一行如果是"( 一 )xxx纠纷"
    if index >= 0 and RE_SECTION_TITLE.match(lines[index].strip()):
        info["section"] = _compact(lines[index].strip())
        info["start"] = index

    # 5) 兜底：有些版本标题自带编号，如 "1. 彩礼适格返还主体..."
    if not info["case_no"]:
        match = RE_NUMBERED_TITLE.match(info["case_title"])
        if match:
            info["case_no"], info["case_title"] = match.group(1), match.group(2)

    # 6) 路径 = [章标题, 案由小节]
    chapter = _find_chapter_title(lines, info["start"])
    info["path"] = [item for item in (chapter, info["section"]) if item]
    return info


def strip_case_preface(text: str) -> tuple[str, list[dict]]:
    """剥离案例文件的前置内容，返回正文文本与案例元信息列表（验收标准 2 的关键函数）。

    前置内容包括：封面（书名、出版社、腰封广告）、编委会名单、编审人员名单、
    序、"目　录"及其后所有目录条目。目录里也有"一、婚姻家庭纠纷"
    "1. 彩礼适格返还主体……"这类和正文标题相似的行，区别是目录条目末尾有
    省略号"…"和页码，正文标题没有。

    因此这里**不用"正文标题"当锚点，而是用"【案件基本信息】"当锚点**：

    1. 找到第一个"【案件基本信息】"；
    2. 从它向上回溯最近的案例标题块（编号 / 主标题 / 副标题）；
    3. 该标题块的开头就是正文起点，之前的内容（封面、序、目录、编委会）全部丢弃；
    4. 如果找不到"【案件基本信息】"，打印警告并返回空正文（不静默产出垃圾）。

    :param text: 原始全文
    :return: (从第一个案例标题块开始的正文, 每个案例的元信息列表)
    """
    norm = normalize_text(text)
    lines = norm.split("\n")

    # 1) 找到所有案例锚点（"【案件基本信息】"）
    anchors = [i for i, line in enumerate(lines) if RE_CASE_ANCHOR.match(line.strip())]
    if not anchors:
        logger.warning("全文未找到“【案件基本信息】”锚点，无法定位案例正文，已终止切割。")
        return "", []

    # 2) 每个锚点各自向上回溯标题块
    blocks = [locate_case_title_block(lines, anchor) for anchor in anchors]
    offset = blocks[0]["start"]  # 第一个案例标题块的行号 = 正文起点
    body = "\n".join(lines[offset:])
    # 把每个案例起点的行号换算成"相对于正文文本"的行号，方便后续切片
    for block in blocks:
        block["start"] = max(block["start"] - offset, 0)
    logger.info(
        f"[case_chunker] 已剥离前置内容：发现 {len(anchors)} 个“【案件基本信息】”，"
        f"正文从原文第 {offset + 1} 行开始（{lines[offset]!r}），正文长度 {len(body)} 字符。"
    )
    return body, blocks


def _monotonic_starts(blocks: list[dict], anchors: list[int], offset: int) -> list[int]:
    """把各案例的起点行号整理成严格递增的序列（异常时退回锚点位置）。

    :param blocks: locate_case_title_block 的结果列表
    :param anchors: 各案例锚点的行号（相对于原文）
    :param offset: 正文起点在原文中的行号
    :return: 严格递增的、相对于正文文本的起点行号列表
    """
    starts: list[int] = []
    previous = -1
    for block, anchor in zip(blocks, anchors):
        start = block["start"]
        if start <= previous:  # 回溯结果异常（例如上一个案例起点更靠后）：退回锚点位置
            start = max(anchor - offset, previous + 1)
        starts.append(start)
        previous = start
    return starts


def verify_case_numbers(blocks: list[dict]) -> list[dict]:
    """用"文档顺序"校验 PDF 里识别到的案例编号。

    为什么需要校验：案例的"编号行"（整行只有一个数字，如 "8"）和"页码行"长得
    完全一样，PDF 抽取有时还会把编号整行丢掉。而案例编号本身是很有用的检索
    元数据，所以这里以**案例在文档中的先后顺序**为准（第 i 个案例编号就是 i），
    把文字里识别到的数字当作校验结果：不一致（例如把页码 198 当成了编号）或
    缺失时按顺序修正，并把原始识别结果写进 ``block["printed_case_no"]``，
    之后会放进 chunk 的 ``metadata.extra`` 里，方便你人工复核。

    :param blocks: locate_case_title_block 的结果列表
    :return: 修正后的 blocks
    """
    fixed: list[tuple[int, str]] = []
    for index, block in enumerate(blocks, 1):
        printed = block["case_no"]
        if printed != str(index):
            block["printed_case_no"] = printed
            block["case_no"] = str(index)
            fixed.append((index, printed or "(缺失)"))
    if fixed:
        logger.info(f"{len(fixed)} 个案例的编号与文档顺序不一致，已按顺序修正（原文数字见 extra）：")
        for index, printed in fixed:
            logger.info(f"    - 第 {index} 个案例：PDF 里识别到 {printed} → 修正为 {index}")
    return blocks


# ======================================================================
# 三、按案例、按 section 切割
# ======================================================================


def split_case_sections(case_text: str) -> list[dict]:
    """按 5 个标记把一个案例切成 section（标题块本身不算 section，会被丢掉）。

    :param case_text: 单个案例的文本（含标题块与各 section）
    :return: [{"section": "基本案情", "text": "【基本案情】\\n..."}]
    """
    sections: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        """把当前 section 收尾并放入结果列表。"""
        nonlocal current
        if current is None:
            return
        text = "\n".join(current["lines"]).strip()
        if text:
            sections.append({"section": current["section"], "text": text})
        current = None

    for raw_line in case_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = RE_SECTION.match(line)
        if match:
            flush()  # 遇到新标记，先收尾上一个 section
            current = {"section": match.group(1), "lines": [line]}
            continue
        if current is not None:
            current["lines"].append(line)  # 归属当前 section（标题块在第一个标记之前，会被丢弃）
    flush()
    return sections


def split_cases(
    text: str,
    source: str = "",
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    drop_repeated_min: int = 6,
) -> list[dict]:
    """把法院案例全文切成适合 RAG 的 chunk（模块对外主函数）。

    :param text: 纯文本字符串（案例全书全文），由调用方用 loader 读出
    :param source: 来源文件名，写进 chunk 的 source 字段
    :param chunk_size: 单个 section chunk 的最大字符数
    :param chunk_overlap: section 细切时的重叠字符数
    :param drop_repeated_min: 全文重复出现多少次的短行算书眉/页脚（0 表示不处理）
    :return: JSON 可序列化的 list[dict]，字段见 common.make_chunk
    """
    if not text or not text.strip():
        logger.warning("输入文本为空，无法切割")
        return []

    # ① 剥离前置内容（封面 / 序 / 编委会 / 目录），并拿到每个案例的标题块信息
    body, blocks = strip_case_preface(text)
    if not body or not blocks:
        logger.warning("未定位到案例正文，已终止切割。")
        return []
    # 用文档顺序校验案例编号（PDF 里的编号行容易与页码混淆、甚至整行缺失）
    blocks = verify_case_numbers(blocks)

    # ② 全文统计"反复出现的短行"（每页书眉/页脚），供逐案例清洗时删除
    drop_lines = find_repeated_lines(body, min_repeat=drop_repeated_min) if drop_repeated_min > 0 else set()
    # 保护结构标记行：它们在全书重复出现（58 个案例 = 58 次），但绝对不能删掉，
    # 否则按标记切 section 就失效了。
    drop_lines = {
        line for line in drop_lines if not any(marker.match(line) for marker in RE_STRUCTURE_MARKERS)
    }
    if drop_lines:
        logger.info(f"统计到 {len(drop_lines)} 行反复出现的书眉/页脚，将在清洗时删除。")

    # ③ 切出每个案例的文本范围：本案例标题块起点 → 下一个案例标题块起点
    anchors = [i for i, line in enumerate(body.split("\n")) if RE_CASE_ANCHOR.match(line.strip())]
    # 此时 blocks 里的 start 和 anchors 都已经是"相对于正文文本"的行号
    starts = _monotonic_starts(blocks, anchors, 0)
    body_lines = body.split("\n")

    splitter = build_case_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[dict] = []
    for i, block in enumerate(blocks):
        end = starts[i + 1] if i + 1 < len(starts) else len(body_lines)
        raw_case = "\n".join(body_lines[starts[i] : end])
        # ④ 通用清洗（页码行、书眉、脚注编号、多余空行）
        case_text = clean_text(
            raw_case,
            extra_drop_lines=drop_lines,
            extra_noise_patterns=CASE_NOISE_PATTERNS,
        )
        sections = split_case_sections(case_text)
        if not sections:
            logger.warning(f"案例 {block['case_no'] or '?'}（{block['case_title']!r}）未切出 section。")
            continue
        for section in sections:
            chunks.extend(_section_to_chunks(section, splitter, source, block, chunk_size))

    logger.info(
        f"解析到 {len(blocks)} 个案例，共切出 {len(chunks)} 个 chunk"
        f"（chunk_size={chunk_size}）。"
    )
    return chunks


def _section_to_chunks(
    section: dict,
    splitter: RecursiveCharacterTextSplitter,
    source: str,
    block: dict,
    chunk_size: int,
) -> list[dict]:
    """把一个 section 转成 chunk；section 超过 chunk_size 时用 splitter 细切并标记 part。

    :param section: {"section": "...", "text": "..."}
    :param splitter: LangChain splitter
    :param source: 来源文件名
    :param block: 案例标题块信息（编号、主标题、副标题、路径）
    :param chunk_size: chunk 上限字符数
    :return: chunk 列表
    """
    text = section["text"]
    base_extra: dict = {"section_char_len": len(text)}
    if "printed_case_no" in block:  # 编号被修正过：保留 PDF 里识别的原始数字，便于复核
        base_extra["printed_case_no"] = block["printed_case_no"]
    common_kwargs: dict = {
        "source": source,
        "doc_type": "case",
        "chunk_type": "case_section",
        "law_name": "",  # 案例没有法律名称，统一格式里保留空字符串
        "path": block["path"],
        "case_no": block["case_no"],
        "case_title": block["case_title"],
        "case_subtitle": block["case_subtitle"],
        "section": section["section"],
    }
    if len(text) <= chunk_size:
        return [make_chunk(**common_kwargs, text=text, extra={**base_extra, "char_len": len(text)})]

    parts = [part.strip() for part in splitter.split_text(text) if part.strip()]
    return [
        make_chunk(
            **common_kwargs,
            # 第 2 块起正文是从句子中间开始的，补一个"【小节名】（续）"上下文头，
            # 让这些 chunk 单独被检索/喂给模型时也能看出它属于哪一节
            text=part if i == 1 else f"【{section['section']}】（续）\n{part}",
            extra={**base_extra, "char_len": len(part), "part": f"{i}/{len(parts)}"},
        )
        for i, part in enumerate(parts, 1)
    ]


# ======================================================================
# 四、命令行测试入口（只有这里才调用 law_document_loaders）
# ======================================================================


def resolve_input_path(input_path: str) -> Path:
    """把用户给的输入路径解析成真实存在的 Path（兼容多种工作目录）。

    :param input_path: 用户传入的路径，例如 "../test_data/1 婚姻家庭继承.pdf"
    :return: 存在的文件路径（绝对路径）
    """
    candidates = [
        Path(input_path),  # 按当前工作目录解析
        Path(__file__).resolve().parent / input_path,  # 相对本包目录
        Path(__file__).resolve().parent.parent / input_path,  # 相对 common 目录
        Path.cwd() / input_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(f"[case_chunker] 找不到输入文件：{input_path}")


def load_text_for_cli(input_path: str) -> str:
    """命令行测试专用：调用 law_document_loaders 把文件读成纯文本。

    注意：核心函数 split_cases 不依赖本函数，也不依赖任何文件格式。
    这里按后缀选择 loader：

    - ``.pdf``       → ``OCRPDFLoader(file_path=...)``，``.load()[i].page_content``
    - ``.doc/.docx`` → ``OCRDOCLoader(filepath=...)``
    - ``.ppt/.pptx`` → ``OCRPPTLoader(filepath=...)``
    - 图片           → ``OCRIMGLoader(img_path=...)``

    :param input_path: 输入文件路径
    :return: 文件纯文本
    """
    path = resolve_input_path(input_path)
    suffix = path.suffix.lower()
    # law_document_loaders 内部用的是 "from law_ocr import ..." 这种同目录导入，
    # 所以要把该目录加入 sys.path 才能直接 import 它的模块。
    loaders_dir = Path(__file__).resolve().parent.parent / "law_document_loaders"
    if str(loaders_dir) not in sys.path:
        sys.path.insert(0, str(loaders_dir))

    if suffix == ".pdf":
        from law_pdfloader import OCRPDFLoader  # type: ignore[import-not-found]

        loader = OCRPDFLoader(file_path=str(path))
    elif suffix in {".docx", ".doc"}:
        from law_docloader import OCRDOCLoader  # type: ignore[import-not-found]

        loader = OCRDOCLoader(filepath=str(path))
    elif suffix in {".ppt", ".pptx"}:
        from law_pptloader import OCRPPTLoader  # type: ignore[import-not-found]

        loader = OCRPPTLoader(filepath=str(path))
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        from law_imgloader import OCRIMGLoader  # type: ignore[import-not-found]

        loader = OCRIMGLoader(img_path=str(path))
    else:
        raise SystemExit(f"[case_chunker] 不支持的文件类型：{suffix}")

    documents = loader.load()  # LangChain Document 列表，每个都有 page_content
    return "\n".join(document.page_content for document in documents)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：读 PDF → split_cases → 打印统计与前 5 个 chunk → 写 JSONL。

    :param argv: 命令行参数（默认取 sys.argv[1:]）
    :return: 进程退出码
    """
    parser = argparse.ArgumentParser(description="法院案例切割（LangChain 实现）")
    parser.add_argument("--input", required=True, help="待切割文件（pdf...），用 law_document_loaders 读取")
    parser.add_argument("--output", default="case_chunks.jsonl", help="输出 JSONL 文件路径")
    parser.add_argument("--chunk-size", type=int, default=1200, help="单个 section chunk 的最大字符数")
    parser.add_argument("--overlap", type=int, default=100, help="细切时的重叠字符数")
    parser.add_argument("--drop-repeated", type=int, default=6, help="全文重复多少次算书眉/页脚（0 = 不处理）")
    parser.add_argument("--preview", type=int, default=5, help="打印前多少个 chunk")
    args = parser.parse_args(argv)

    input_path = resolve_input_path(args.input)
    logger.info(f"读取文件：{input_path}")
    text = load_text_for_cli(args.input)
    logger.info(f"纯文本长度：{len(text)} 字符")

    chunks = split_cases(
        text,
        source=input_path.name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        drop_repeated_min=args.drop_repeated,
    )
    if not chunks:
        logger.warning("没有切出任何 chunk，请检查输入文件内容。")
        return 1

    assert_uniform_format(chunks)  # 自检：所有 chunk 字段完全一致
    print_summary(chunks, title="案例切割概要")
    preview_chunks(chunks, n=args.preview, max_len=800)

    output_path = save_jsonl(chunks, args.output)
    logger.info(f"已写入 JSONL：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
