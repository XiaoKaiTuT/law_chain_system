# -*- coding: utf-8 -*-
"""law_text_spliter —— 法律/案例文本切割包（负责"把纯文本切成 chunk"）。

本包只处理纯文本，**不读文件、不依赖任何文件格式**：

    from common.law_text_spliter import split_law, split_cases

    text = "……"                       # 先用 law_document_loaders 读出纯文本
    law_chunks = split_law(text, source="民事诉讼法.docx")
    case_chunks = split_cases(text, source="1 婚姻家庭继承.pdf")

各模块职责：

- ``common.py``                       公共函数：清洗、统一 chunk 格式、保存 JSONL
- ``legal_chunker.py``                法律条文切割（split_law）
- ``case_chunker.py``                 法院案例切割（split_cases）
- ``modelscope_splitter_template.py`` ModelScope 语义分割备用模板（可选，不参与主流程）

完整用法见同目录 README.md。
"""

from .case_chunker import split_cases
from .legal_chunker import split_law

__all__ = ["split_law", "split_cases"]
