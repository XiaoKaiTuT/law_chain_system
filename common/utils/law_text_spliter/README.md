# law_text_spliter —— 法律 / 案例文本切割包

> 只负责"把纯文本切成 chunk"。读文件是 `law_document_loaders` 的事，本包不依赖任何文件格式。

## 1. 目录与职责

```
common/
├── law_document_loaders/          # 已存在，禁止修改：各种文件 → 纯文本
│   ├── law_docloader.py           #   OCRDOCLoader(filepath=...)  → .doc/.docx
│   ├── law_pdfloader.py           #   OCRPDFLoader(file_path=...) → .pdf
│   ├── law_pptloader.py           #   OCRPPTLoader(filepath=...)  → .ppt/.pptx
│   ├── law_imgloader.py           #   OCRIMGLoader(img_path=...)  → 图片
│   └── law_ocr.py                 #   get_ocr() 内嵌使用
├── test_data/                     # 已存在，禁止修改：两个真实测试文件
└── law_text_spliter/              # ← 本包（本次新建）
    ├── __init__.py                # 暴露 split_law / split_cases
    ├── common.py                  # 清洗、统一 chunk 格式、保存 JSONL
    ├── legal_chunker.py           # 法律条文切割（split_law + 命令行入口）
    ├── case_chunker.py            # 法院案例切割（split_cases + 命令行入口）
    ├── modelscope_splitter_template.py  # ModelScope 语义分割备用模板（可选）
    └── requirements.txt
```

## 2. 数据流

```
docx/pdf                (law_document_loaders)         (law_text_spliter)
  │  原始文件  ──────────────────────────────►  纯文本 str
  │                                                   │
  │            ① 剥离前置内容（标题/修订史/目录/封面/序/编委会）
  │            ② 通用清洗（分页标记、页码、书眉、脚注、空行）
  │            ③ 结构切分（第X编/章/节/条；【基本案情】等 section）
  │            ④ LangChain 切割器成块（控长 + 合并 + 必要细切）
  │                                                   │
  │                                                   ▼
  │                                    list[dict]（统一 JSON 格式）
  │                                                   │
  ▼                                                   ▼
 需 GPU 的 OCR（仅大图/扫描件，由 loader 决定）      JSONL 文件（方便事后查看）
```

关键契约：

* `split_law(text, source="") -> list[dict]`、`split_cases(text, source="") -> list[dict]`
  只接收**纯文本**、只返回**JSON 可序列化**的 list[dict]，不读文件、不写文件。
* 只有两个模块的 `__main__` 命令行测试入口才会调用 `law_document_loaders`。

## 3. 统一输出格式

```json
{
  "source": "文件名",
  "doc_type": "law 或 case",
  "chunk_type": "article / case_section / semantic_chunk",
  "text": "切出来的正文",
  "metadata": {
    "law_name": "中华人民共和国民事诉讼法",
    "path": ["第一编 总 则", "第一章 任务、适用范围和基本原则"],
    "article_no": "第一条",
    "case_no": "1",
    "case_title": "彩礼适格返还主体复合性的司法认定",
    "case_subtitle": "——何某诉卢某某、窦某某婚约财产案",
    "section": "基本案情",
    "extra": {}
  }
}
```

两个模块的字段**完全一致**（统一由 `common.make_chunk` 生成，`common.assert_uniform_format` 会自检）。
`metadata.extra` 用来放可选信息：

| extra 键 | 出现场景 |
| --- | --- |
| `char_len` | 每个 chunk：本块字符数 |
| `section_char_len` | 案例：所属 section 的完整字符数 |
| `article_char_len` | 法律：所属条文的完整字符数 |
| `part` | 被细切的块：`"2/3"` 表示第 2 块（共 3 块） |
| `article_nos` / `article_count` | 合并模式：本块包含的条号列表与条数 |
| `printed_case_no` | 案例编号与文档顺序不一致时，保留 PDF 里识别到的原始数字 |

## 4. 命令行运行

在 `common/law_text_spliter` 目录下执行：

```powershell
# 法律条文（用 law_document_loaders.law_docloader 读 docx）
python legal_chunker.py --input "../test_data/中华人民共和国民事诉讼法（《关于修改〈中华人民共和国民事诉讼法〉的决定》第五次修正））.docx" --output law_chunks.jsonl

# 法院案例（用 law_document_loaders.law_pdfloader 读 pdf）
python case_chunker.py --input "../test_data/1 婚姻家庭继承.pdf" --output case_chunks.jsonl

# 常用参数
python legal_chunker.py --input "..." --output law_chunks.jsonl --chunk-size 800 --overlap 0 --merge-articles
python case_chunker.py  --input "..." --output case_chunks.jsonl --chunk-size 1200 --overlap 100 --drop-repeated 6
```

输出：`chunk 总数`、类型分布、前 5 个 chunk 的 JSON（每个截断到 800 字符），并把全部 chunk 写入 JSONL。

## 5. Python 里直接调用

```python
import sys
from pathlib import Path

PROJECT = Path(r"/")
# ① 为了能 import common.law_text_spliter（项目根目录必须在 sys.path 里）
sys.path.insert(0, str(PROJECT))
# ② 为了能 import law_docloader：loader 内部写的是 "from law_ocr import get_ocr"
#    这种同目录（扁平）导入，所以必须把 loader 目录也放进 sys.path
sys.path.insert(0, str(PROJECT / "common" / "law_document_loaders"))

from law_docloader import OCRDOCLoader  # doc / docx
from law_pdfloader import OCRPDFLoader  # pdf
from common.utils.law_text_spliter import split_law, split_cases

# ① 读文件 → 纯文本（这一步永远交给 law_document_loaders）
docx_loader = OCRDOCLoader(filepath=str(next((PROJECT / "common" / "test_data").glob("*民事诉讼法*.docx"))))
law_text = "\n".join(doc.page_content for doc in docx_loader.load())

pdf_loader = OCRPDFLoader(file_path=str(PROJECT / "common" / "test_data" / "1 婚姻家庭继承.pdf"))
case_text = "\n".join(doc.page_content for doc in pdf_loader.load())

# ② 纯文本 → chunk（split_* 只认字符串）
law_chunks = split_law(law_text, source="民事诉讼法.docx")
case_chunks = split_cases(case_text, source="1 婚姻家庭继承.pdf")

print(len(law_chunks), law_chunks[0])
print(len(case_chunks), case_chunks[0])
```

> 上面这段就是本包里实际跑通的调用方式（`_load_text_for_cli` 用的也是同一套逻辑）。
> 如果你更想写 `from common.law_document_loaders.law_docloader import OCRDOCLoader`，
> 那就必须先执行 `sys.path.insert(0, str(PROJECT / "common" / "law_document_loaders"))`，
> 否则 loader 内部那句 `from law_ocr import get_ocr` 会报 `ModuleNotFoundError`。

## 6. 前置内容剥离逻辑（最容易踩坑的地方）

### 6.1 法律文件（`legal_chunker.strip_law_preface`）

前置内容：文件标题 → 历次修订历史（一大段括号说明）→ `目　录` → 全部目录条目。
难点：目录里也有 `第一编 / 第一章 / 第一节`，和正文**格式完全一样**，所以不能"看到第一章就以为是正文"。

解决办法（两步锚定）：

1. 先找 `目 录` 行，只从它下面开始找；
2. 找第一个"**真正的第一条**"：行首匹配 `第X条`，且该行后面跟着真实的正文内容
   （目录条目的"正文"是 `……` 加页码，会被 `RE_TOC_LINE` 排除）；
3. 从这个"第一条"向上回溯**最近的 `第X编` 标题**（找不到再退而找 `第X章`），
   把它作为正文起点——这样正文开头的 `第一编 总 则` 会保留下来，用于生成 `metadata.path`；
   回溯窗口限制 40 行、中间不允许夹目录条目，避免退到目录里。

### 6.2 案例文件（`case_chunker.strip_case_preface`）

前置内容：封面（书名/出版社/腰封广告）→ 编委会名单 → 编审人员名单 → 序 → 通讯编辑 → `目　录` → 目录条目。
难点：目录条目（`一、婚姻家庭纠纷`、`1. 彩礼适格返还主体…… 96`）和正文标题太像。

解决办法（换个锚点 + 回溯标题块）：

1. **不用标题当锚点，用 `【案件基本信息】` 当锚点**（正文独有，全书 58 处，目录里没有）；
2. 从锚点往上回溯标题块：`副标题（——…）` → `主标题` → `编号（整行数字）` → `案由小节（(一)xx纠纷）`，
   一层一层严格匹配，遇到不符合的行立刻停止；
3. 标题块开头之前的一切（封面、序、目录、编委会）全部丢弃；
4. **找不到锚点就打印警告并返回空结果**，绝不静默产出垃圾 chunk。

两个细节（都是被真实数据"教"出来的）：

* `_prev_content_index` 跳过噪声行时**不把"纯数字行"当噪声**——因为"编号行"整行就是一个数字，
  和页码长得一模一样（这一点和通用清洗里的 `is_noise_line` 相反）。
* 标题在 PDF 里常常折行（`诉讼期间父母一方收入状况发生变化时` + `抚养费支付标准的认定`），
  所以会向上吸收"标题折行"（长度 ≥ 6、含句末标点则停止、最多 3 行，并用"含中文冒号"排除 `编写人：…` 署名行）。
* 编号还可能整行缺失或与页码混淆（例如把页码 `198` 当成编号），所以 `verify_case_numbers`
  会用**文档顺序**校验：第 i 个案例的编号就是 i，原始识别结果保存在 `extra.printed_case_no` 里备查。

## 7. 通用清洗做了什么（`common.clean_text`）

1. `normalize_text`：统一换行、删零宽字符（docx 的 `\u200b`）、全角空格转半角、合并连续空格与空行；
2. 删除分页标记：`===== Page 1 =====`、`------- Page (0) -------`、`第 3 页`；
3. 删除页码行（整行只有数字）、单独成行的脚注编号（`①`）、书眉/版权页行、案例脚注行（`① 本书【…】…`）；
4. `find_repeated_lines`：删除**整本书反复出现**的书眉页脚（`中国法院2026年度案例 ·婚姻家庭与继承纠纷`、`一 、婚姻家庭纠纷`）。
   三个条件同时满足才算噪声：短行 + 不含句读标点（`。，；：！？`）+ **紧挨着页码行**；
   这样就不会误删正文里重复出现的句子（例如判决书里的"驳回上诉，维持原判。"）。

## 8. LangChain 切割器怎么用

| 模块 | 组件 | 分隔符优先级 |
| --- | --- | --- |
| `legal_chunker.py` | 自定义 `LawTextSplitter(TextSplitter)`（合并模式） | 第X编/章/节 → 第X条 → 空行/换行/句号/分号 |
| `legal_chunker.py` | `RecursiveCharacterTextSplitter`（单条过长时细切） | 空行 → 换行 → 句号 → 分号 |
| `case_chunker.py` | `RecursiveCharacterTextSplitter`（section 过长时细切） | 空行 → 换行 → 句号 → 分号 → 逗号 |
| `modelscope_splitter_template.py` | 自定义 `ModelScopeSemanticSplitter(TextSplitter)` | 句子相似度语义分段 |

> ⚠️ 踩坑记录：`RecursiveCharacterTextSplitter(is_separator_regex=True)` 用**正则前瞻**当分隔符时，
> 合并小块会执行 `separator.join(...)`，把**正则字符串本身**拼进正文，产出
> `(?m)^(?=第[零〇一…]+条)第二条 …` 这类脏数据。所以法律模块的"编/章/节/条"优先级
> 自己实现（`LawTextSplitter`），只在纯文本分隔符的场景使用 `RecursiveCharacterTextSplitter`。

## 9. 实测结果（test_data 两个真实文件）

| 模块 | 输入 | 结果 | 第一个 chunk 正文开头 |
| --- | --- | --- | --- |
| `split_law` | 中华人民共和国民事诉讼法（第五次修正）.docx | 306 条 → **306 个 chunk** | `第一条 中华人民共和国民事诉讼法以宪法为根据…` |
| `split_cases` | 1 婚姻家庭继承.pdf（340 页） | 58 个案例 → **401 个 chunk** | `【案件基本信息】\n1. 裁判书字号…` |

验收对照：

1. 民诉法第一个 chunk 以"第一条"开头 ✅（没有标题、没有目录）
2. 案例第一个 chunk 以"【案件基本信息】"开头 ✅（没有封面/序/目录/编委会）
3. 两个模块字段完全一致 ✅（`assert_uniform_format` 自检通过）
4. `split_law` / `split_cases` 只吃文本、只吐 list[dict] ✅
5. `law_document_loaders`、`test_data` 未被修改 ✅

## 10. ModelScope 语义切割模板（可选）

```bash
pip install modelscope torch
python modelscope_splitter_template.py     # 模板自测：内置一小段文本
```

* 默认模型：`iic/nlp_gte_sentence-embedding_chinese-base`（中文句子向量）；
* 换模型：改 `MODEL_ID`，或 `ModelScopeSemanticSplitter(model_id="你的模型id或本地目录")`；
* 流程：切句 → embedding → 相邻句子相似度 >= 阈值就合并 → 输出 `chunk_type="semantic_chunk"`；
* 模型不可用（离线 / 未下载）时会自动退化成"字符二元组余弦相似度"，流程照样能跑通（质量下降）。
* 它只是**备用模板**：法律条文和案例的正式切割仍建议用 `legal_chunker` / `case_chunker`（结构优先、可验证）。
