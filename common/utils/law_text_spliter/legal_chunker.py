# -*- coding: utf-8 -*-
"""legal_chunker.py —— 法律条文切割模块（以《民事诉讼法》为例）。

核心函数::

    def split_law(text: str, source: str = "") -> list[dict]

契约：

- 输入是**纯文本字符串**，输出是 JSON 可序列化的 ``list[dict]``；
- 不读文件、不写文件、不 import law_document_loaders（读文件由 loader 负责）；
- 只有 ``__main__`` 命令行测试入口才会调用 ``law_document_loaders`` 读 docx。

切割流程（四步）::

    原始文本 → ① 剥离前置内容（标题 / 修订历史 / 目录）
             → ② 通用清洗（分页标记、页码行、多余空格与空行）
             → ③ 解析"编-章-节-条"结构，得到每一条的完整文本
             → ④ 用 LangChain 的 RecursiveCharacterTextSplitter 成块
                （分隔符优先级：编/章/节 > 条 > 换行/句号/分号）
"""

from __future__ import annotations

import argparse
import os
import re
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
from langchain_text_splitters import TextSplitter

try:  # 作为包导入时：from common.law_text_spliter.legal_chunker import split_law
    from .chunk_common import (
        assert_uniform_format,
        clean_text,
        extract_law_name,
        make_chunk,
        normalize_text,
        preview_chunks,
        print_summary,
        save_jsonl,
    )
except ImportError:  # 作为脚本直接运行时：python legal_chunker.py ...
    from common import (  # type: ignore[no-redef]
        assert_uniform_format,
        clean_text,
        extract_law_name,
        make_chunk,
        normalize_text,
        preview_chunks,
        print_summary,
        save_jsonl,
    )

# ======================================================================
# 一、正则工具
# ======================================================================

# 中文数字（含"两、〇"），用于匹配"第X编 / 第X章 / 第X节 / 第X条"
CN_NUM = "零〇一二三四五六七八九十百千两"

# 层级标题：整行就是"第X编/章/节"（可带标题文字），例如 "第一编 总 则"
RE_LEVEL_HEADING = re.compile(rf"^第([{CN_NUM}]+)([编章节])(?:\s*(.*))?$")
# 条：整行以"第X条"开头，后面可以跟正文（"第一条 中华人民共和国..."）
RE_ARTICLE_START = re.compile(rf"^(第[{CN_NUM}]+条)(?:\s*(.*))?$")
# 在文本中查找所有条号（多行模式，只匹配行首的"第X条"）
RE_ARTICLE_FIND = re.compile(rf"(?m)^第[{CN_NUM}]+条")
# 目录条目特征：以省略号/点线结尾（后面可能跟页码）
RE_TOC_LINE = re.compile(r"[.…]{2,}\s*\d*$")
# 目录标题行："目 录" / "目　　录"（normalize_text 之后是"目 录"）
RE_TOC_TITLE = re.compile(r"^目\s*[录次]$")
# 层级深度：编 > 章 > 节，用于维护 path
LEVEL_DEPTH = {"编": 0, "章": 1, "节": 2}


def _heading_level(line: str, *, max_len: int = 40) -> str:
    """判断一行是不是"第X编/第X章/第X节"标题行，返回层级名。

    :param line: 一行文本
    :param max_len: 标题行最大长度（正文段落远长于此，用来排除误判）
    :return: "编" / "章" / "节"；不是标题行返回空字符串
    """
    s = line.strip()
    if not s or len(s) > max_len:
        return ""
    match = RE_LEVEL_HEADING.match(s)
    return match.group(2) if match else ""


# ======================================================================
# 二、前置内容剥离
# ======================================================================


def strip_law_preface(text: str) -> str:
    """剥离法律文件的前置内容，只留正文（验收标准 1 的关键函数）。

    前置内容包括：文件标题、"历次修订历史"括号说明、"目　录"及其后所有目录条目。
    难点是目录里也有"第一编/第一章/第一节"，和正文格式一模一样，所以不能用
    "第一个第一章就是正文"这种判断。这里的做法分两步：

    1. **先找目录之后第一个"真正的第一条"**：行首匹配"第X条"，并且该行后面
       跟着正文内容（不是"…… 5"这种目录省略号 + 页码）；
    2. **再从这个"第一条"向上回溯最近的"第X编"标题**，作为正文起点，
       这样正文开头的"第一编 总 则"也会被保留（它提供 chunk 的路径信息）。

    :param text: 原始全文
    :return: 从正文起点开始的文本；找不到"第一条"时打印警告并返回规范化全文
    """
    norm = normalize_text(text)
    lines = norm.split("\n")

    # 1) 定位"目 录"所在行
    toc_index = -1
    for i, line in enumerate(lines):
        if RE_TOC_TITLE.match(line):
            toc_index = i
            break
    search_from = toc_index + 1 if toc_index >= 0 else 0  # 有目录就只从目录之后找
    if toc_index < 0:
        logger.info("未找到“目 录”行，将从全文开头查找正文。")

    # 2) 找第一个"真正的第一条"
    anchor = _find_first_body_article(lines, search_from)
    if anchor < 0:
        logger.warning("未在目录之后找到带正文内容的“第一条”，无法剥离前置内容。")
        return norm

    # 3) 从"第一条"向上回溯到最近的"第X编"（或"第X章"）
    start = _backtrack_to_body_start(lines, anchor)
    if start > 0:
        logger.info(f"已剥离前置内容：正文从第 {start + 1} 行开始（{lines[start]!r}）。")
    return "\n".join(lines[start:])


def _find_first_body_article(lines: list[str], start: int) -> int:
    """在 lines[start:] 中找第一个"行首第X条 + 后面有正文"的行号。

    :param lines: 全部行
    :param start: 从哪一行开始找
    :return: 行号；找不到返回 -1
    """
    for i in range(start, len(lines)):
        match = RE_ARTICLE_START.match(lines[i])
        if not match:
            continue
        body = (match.group(2) or "").strip()
        # 目录条目形如 "第一条 …… 5"，其"正文"是省略号+页码；正文条目是真实内容
        if len(body) >= 4 and not RE_TOC_LINE.search(body):
            return i
    return -1


def _backtrack_to_body_start(lines: list[str], anchor: int, *, max_lookback: int = 40) -> int:
    """从"第一条"所在行向上回溯，找到正文起点（最近的"第X编"，其次"第X章"）。

    回溯窗口限制为 max_lookback 行，且中间不允许出现目录条目（省略号+页码），
    避免"正文没有编标题、不小心回退到目录里"的错误情况。

    :param lines: 全部行
    :param anchor: "第一条"所在行号
    :param max_lookback: 最多向上看多少行
    :return: 正文起点的行号
    """
    for wanted in ("编", "章"):
        for i in range(anchor - 1, max(anchor - max_lookback, -1), -1):
            if any(RE_TOC_LINE.search(lines[j]) for j in range(i + 1, anchor)):
                break  # 中间夹着目录条目，说明回退过头了
            if _heading_level(lines[i]) == wanted:
                return i
    return anchor  # 既没有编也没有章：就从"第一条"本身开始


# ======================================================================
# 三、结构解析：把正文拆成"每一条"单位的文本
# ======================================================================


def parse_law_structure(body: str) -> list[dict]:
    """解析正文的"编-章-节-条"结构，返回每一条的文本与所属路径。

    规则：

    - 遇到"第X编/章/节"标题行：只更新路径（path），标题行本身不进 chunk 正文；
    - 遇到"第X条"开头行：新起一条，记录条号、当前路径与行文本；
    - 其它行：属于当前这一条（条文的多款、多项常常单独成行）。

    这样能保证**每条法律条文完整**，不会从条中间被切开。

    :param body: 剥离前置内容并清洗后的正文
    :return: [{"article_no": "第一条", "path": ["第一编 总 则", ...], "text": "..."}]
    """
    units: list[dict] = []
    path: list[str] = []  # 当前"编-章-节"路径
    current: dict | None = None  # 当前正在收集的条

    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        level = _heading_level(line)
        if level:
            # 层级标题：按层级截断后追加（"编"清空下面两级），标题行不进 chunk 正文
            path = path[: LEVEL_DEPTH[level]] + [line]
            current = None
            continue
        match = RE_ARTICLE_START.match(line)
        if match:
            current = {"article_no": match.group(1), "path": list(path), "lines": [line]}
            units.append(current)
            continue
        if current is not None:
            current["lines"].append(line)  # 分款/分项，归属于当前这一条

    return [
        {"article_no": unit["article_no"], "path": unit["path"], "text": "\n".join(unit["lines"]).strip()}
        for unit in units
        if unit["lines"]
    ]


# ======================================================================
# 四、LangChain 切割器 + 对外主函数
# ======================================================================


class LawTextSplitter(TextSplitter):
    """法律文本专用分割器（继承 LangChain 的 ``TextSplitter``）。

    分隔符优先级（从高到低）：

    1. ``第X编`` / ``第X章`` / ``第X节`` —— 层级边界，最优先；
    2. ``第X条`` —— 条文边界，保证条文完整；
    3. 空行 / 换行 / 句号 / 分号 —— 条文自身过长时的兜底细切。

    前两级用"零宽前瞻断言" ``(?m)^(?=第X条)`` 描述：切点落在"第X条"之前，
    所以每一块都从"第X条"开头，条文不会被拦腰截断。

    为什么不用 ``RecursiveCharacterTextSplitter(is_separator_regex=True)``：
    它在合并小块时用 ``separator.join(...)`` 还原文本，而这里的 separator 是
    **正则表达式字符串**（``(?m)^(?=第[...]+条)``），结果会把正则本身当成正文
    拼进去（实测会产出 ``(?m)^(?=第[零〇一...]+条)第二条 ...`` 这种脏数据）。
    所以本类自己实现"按优先级切 + 贪心合并"，逻辑更透明，结果也可验证。

    注意：文本的**结构切分**（哪一行属于哪一条）由 :func:`parse_law_structure`
    完成，本类只负责"把已切好的条文小块合并成不超过 chunk_size 的 chunk"。
    """

    # 分隔符优先级（从高到低），元素都是正则表达式
    SEPARATOR_PRIORITY: tuple[str, ...] = (
        rf"(?m)^(?=第[{CN_NUM}]+编)",  # 1. 编
        rf"(?m)^(?=第[{CN_NUM}]+章)",  # 2. 章
        rf"(?m)^(?=第[{CN_NUM}]+节)",  # 3. 节
        rf"(?m)^(?=第[{CN_NUM}]+条)",  # 4. 条（核心：保证条文完整）
        "\n\n",  # 5. 空行
        "\n",  # 6. 换行
        "。",  # 7. 句号
        "；",  # 8. 分号
    )

    def split_text(self, text: str) -> list[str]:
        """按分隔符优先级切分，再把小块贪心合并到 chunk_size 以内。

        :param text: 待切分文本
        :return: chunk 文本列表
        """
        pieces = self._split_by_priority(text, list(self.SEPARATOR_PRIORITY))
        return self._merge_pieces(pieces)

    def _split_by_priority(self, text: str, separators: list[str]) -> list[str]:
        """用"当前优先级里能匹配到"的分隔符切分；仍有超大块就降一级继续切。

        :param text: 待切分文本
        :param separators: 剩余的分隔符（从高到低）
        :return: 切分结果
        """
        if not separators:
            return [text] if text.strip() else []
        separator, rest = separators[0], separators[1:]
        if not re.search(separator, text):
            return self._split_by_priority(text, rest)  # 这个优先级在本段里匹配不到
        parts = [part for part in re.split(separator, text) if part.strip()]
        if not parts:
            return []
        if max(len(part) for part in parts) <= self._chunk_size:
            return parts
        # 还有超过 chunk_size 的块：对这些块继续用更低优先级的规则切
        result: list[str] = []
        for part in parts:
            if len(part) > self._chunk_size and rest:
                result.extend(self._split_by_priority(part, rest))
            else:
                result.append(part)
        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """把小块贪心合并成尽量接近 chunk_size 的 chunk，并处理 chunk_overlap。

        :param pieces: 已按优先级切好的小块
        :return: 合并后的 chunk 列表
        """
        merged: list[str] = []
        current: list[str] = []
        total = 0
        for piece in pieces:
            if current and total + len(piece) + 1 > self._chunk_size:
                merged.append("\n".join(current).strip())
                # 按 chunk_overlap 把上一块的结尾部分带到新块里，保持上下文连续
                carry: list[str] = []
                carry_len = 0
                for old in reversed(current):
                    if carry_len + len(old) + 1 > self._chunk_overlap:
                        break
                    carry.insert(0, old)
                    carry_len += len(old) + 1
                current, total = carry, carry_len
            current.append(piece)
            total += len(piece) + 1
        if current:
            merged.append("\n".join(current).strip())
        return [chunk for chunk in merged if chunk]


def build_law_splitter(chunk_size: int = 800, chunk_overlap: int = 0) -> LawTextSplitter:
    """构造法律文本专用 splitter（自定义分隔符优先级，见 :class:`LawTextSplitter`）。

    :param chunk_size: 单个 chunk 的最大字符数
    :param chunk_overlap: 相邻 chunk 的重叠字符数
    :return: LawTextSplitter 实例
    """
    return LawTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def build_article_splitter(chunk_size: int = 800, chunk_overlap: int = 0) -> RecursiveCharacterTextSplitter:
    """构造"单条条文过长时"使用的细切 splitter（LangChain 标准组件）。

    这里用 RecursiveCharacterTextSplitter 的**纯文本**分隔符（不启用正则），
    它是安全的：合并时用 ``separator.join(...)`` 还原出来的就是原文内容。

    :param chunk_size: 单个 chunk 的最大字符数
    :param chunk_overlap: 相邻 chunk 的重叠字符数
    :return: RecursiveCharacterTextSplitter 实例
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；"],
    )


def split_law(
    text: str,
    source: str = "",
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 0,
    merge_articles: bool = False,
) -> list[dict]:
    """把法律条文全文切成适合 RAG 的 chunk（模块对外主函数）。

    :param text: 纯文本字符串（法律条文全文），由调用方用 loader 读出
    :param source: 来源文件名，写进 chunk 的 source 字段
    :param chunk_size: 单个 chunk 的最大字符数
    :param chunk_overlap: 细切时的重叠字符数
    :param merge_articles: True 时把相邻短条文合并成不超过 chunk_size 的大块；
        False（默认）时"一条 = 一个 chunk"，检索精度更高
    :return: JSON 可序列化的 list[dict]，字段见 common.make_chunk
    """
    if not text or not text.strip():
        logger.warning("输入文本为空，无法切割")
        return []

    law_name = extract_law_name(text) or (Path(source).stem if source else "")

    # ① 剥离前置内容（标题、修订历史、目录）→ ② 通用清洗
    body = strip_law_preface(text)
    body = clean_text(body)

    # ③ 解析"编-章-节-条"结构
    units = parse_law_structure(body)
    if not units:
        logger.warning("未解析到任何“第X条”，请检查前置内容剥离结果")
        return []

    # ④ 用 LangChain splitter 成块
    chunks: list[dict] = []
    if merge_articles:
        # 合并模式：LawTextSplitter 负责"按 编/章/节 > 条 的优先级"合并相邻条文
        merger = build_law_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = _merge_articles_to_chunks(units, merger, source, law_name)
    else:
        # 默认模式：一条 = 一个 chunk；只有某一条自己超过 chunk_size 时才细切
        article_splitter = build_article_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for unit in units:
            chunks.extend(_article_to_chunks(unit, article_splitter, source, law_name, chunk_size))

    logger.info(
        f"解析到 {len(units)} 条，共切出 {len(chunks)} 个 chunk"
        f"（chunk_size={chunk_size}, merge_articles={merge_articles}）。"
    )
    return chunks


def _article_to_chunks(
    unit: dict,
    splitter: RecursiveCharacterTextSplitter,
    source: str,
    law_name: str,
    chunk_size: int,
) -> list[dict]:
    """把一条法律条文转成 chunk；条文自身超过 chunk_size 时才用 splitter 细切。

    :param unit: parse_law_structure 产出的单条数据
    :param splitter: LangChain splitter
    :param source: 来源文件名
    :param law_name: 法律名称
    :param chunk_size: chunk 上限字符数
    :return: chunk 列表
    """
    text = unit["text"]
    common_kwargs: dict = {
        "source": source,
        "doc_type": "law",
        "chunk_type": "article",
        "law_name": law_name,
        "path": unit["path"],
        "article_no": unit["article_no"],
    }
    if len(text) <= chunk_size:
        return [make_chunk(**common_kwargs, text=text, extra={"char_len": len(text)})]

    # 单条太长（例如列举了很多项）：用 splitter 按 换行/句号/分号 细切，并标记 part
    parts = [part.strip() for part in splitter.split_text(text) if part.strip()]
    return [
        make_chunk(
            **common_kwargs,
            # 第 2 块起正文是从句子中间开始的，补一个"第X条（续）"上下文头，
            # 让这些 chunk 单独被检索/喂给模型时也能看出它属于哪一条
            text=part if i == 1 else f"{unit['article_no']}（续）\n{part}",
            extra={"char_len": len(part), "article_char_len": len(text), "part": f"{i}/{len(parts)}"},
        )
        for i, part in enumerate(parts, 1)
    ]


def _merge_articles_to_chunks(
    units: list[dict],
    splitter: RecursiveCharacterTextSplitter,
    source: str,
    law_name: str,
) -> list[dict]:
    """把相邻条文合并成较大的 chunk（由 LangChain splitter 负责合并与必要细切）。

    做法：把所有条文按顺序拼成一个大文本交给 splitter；因为分隔符里"第X条"
    的优先级高于换行/句号，切点会落在条文边界上，所以合并出来的 chunk 中
    每一条都是完整的。切完后按 chunk 内出现的条号回填 article_no / path；
    如果某一块里没有条号，说明它是超长条文的续块，沿用上一条的元数据。

    :param units: parse_law_structure 产出的全部条目
    :param splitter: LangChain splitter
    :param source: 来源文件名
    :param law_name: 法律名称
    :return: chunk 列表
    """
    joined = "\n".join(unit["text"] for unit in units)
    article_path = {unit["article_no"]: unit["path"] for unit in units}

    chunks: list[dict] = []
    last: tuple[str, list[str]] = ("", [])
    for piece in splitter.split_text(joined):
        piece = piece.strip()
        if not piece:
            continue
        found = RE_ARTICLE_FIND.findall(piece)  # 本块里出现的所有条号
        if found:
            article_no = found[0]
            path = article_path.get(article_no, [])
            extra: dict = {"char_len": len(piece), "article_count": len(found)}
            if len(found) > 1:
                extra["article_nos"] = found
            last = (article_no, path)
        else:
            # 超长条文被细切出来的续块：沿用上一条的条号与路径
            article_no, path = last
            extra = {"char_len": len(piece), "continued": True}
        chunks.append(
            make_chunk(
                source=source,
                doc_type="law",
                chunk_type="article",
                text=piece,
                law_name=law_name,
                path=path,
                article_no=article_no,
                extra=extra,
            )
        )
    return chunks


# ======================================================================
# 五、命令行测试入口（只有这里才调用 law_document_loaders）
# ======================================================================


def resolve_input_path(input_path: str) -> Path:
    """把用户给的输入路径解析成真实存在的 Path（兼容多种工作目录）。

    :param input_path: 用户传入的路径，例如 "../test_data/民事诉讼法.docx"
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
    raise SystemExit(f"[legal_chunker] 找不到输入文件：{input_path}")


def load_text_for_cli(input_path: str) -> str:
    """命令行测试专用：调用 law_document_loaders 把文件读成纯文本。

    注意：核心函数 split_law 不依赖本函数，也不依赖任何文件格式。
    这里按后缀选择 loader：

    - ``.doc/.docx`` → ``OCRDOCLoader(filepath=...)``，``.load()[i].page_content``
    - ``.pdf``       → ``OCRPDFLoader(file_path=...)``
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

    if suffix in {".docx", ".doc"}:
        from law_docloader import OCRDOCLoader  # type: ignore[import-not-found]

        loader = OCRDOCLoader(filepath=str(path))
    elif suffix == ".pdf":
        from law_pdfloader import OCRPDFLoader  # type: ignore[import-not-found]

        loader = OCRPDFLoader(file_path=str(path))
    elif suffix in {".ppt", ".pptx"}:
        from law_pptloader import OCRPPTLoader  # type: ignore[import-not-found]

        loader = OCRPPTLoader(filepath=str(path))
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        from law_imgloader import OCRIMGLoader  # type: ignore[import-not-found]

        loader = OCRIMGLoader(img_path=str(path))
    else:
        raise SystemExit(f"[legal_chunker] 不支持的文件类型：{suffix}")

    documents = loader.load()  # LangChain Document 列表，每个都有 page_content
    return "\n".join(document.page_content for document in documents)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：读文件 → split_law → 打印统计与前 5 个 chunk → 写 JSONL。

    :param argv: 命令行参数（默认取 sys.argv[1:]）
    :return: 进程退出码
    """
    parser = argparse.ArgumentParser(description="法律条文切割（LangChain 实现）")
    parser.add_argument("--input", required=True, help="待切割文件（docx/pdf...），用 law_document_loaders 读取")
    parser.add_argument("--output", default="law_chunks.jsonl", help="输出 JSONL 文件路径")
    parser.add_argument("--chunk-size", type=int, default=800, help="单个 chunk 的最大字符数")
    parser.add_argument("--overlap", type=int, default=0, help="细切时的重叠字符数")
    parser.add_argument("--merge-articles", action="store_true", help="把相邻短条文合并成更大的 chunk")
    parser.add_argument("--preview", type=int, default=5, help="打印前多少个 chunk")
    args = parser.parse_args(argv)

    input_path = resolve_input_path(args.input)
    logger.info(f"读取文件：{input_path}")
    text = load_text_for_cli(args.input)
    logger.info(f"纯文本长度：{len(text)} 字符")

    chunks = split_law(
        text,
        source=input_path.name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        merge_articles=args.merge_articles,
    )
    if not chunks:
        logger.warning("没有切出任何 chunk，请检查输入文件内容。")
        return 1

    assert_uniform_format(chunks)  # 自检：所有 chunk 字段完全一致
    print_summary(chunks, title="法律条文切割概要")
    preview_chunks(chunks, n=args.preview, max_len=800)

    output_path = save_jsonl(chunks, args.output)
    logger.info(f"已写入 JSONL：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
