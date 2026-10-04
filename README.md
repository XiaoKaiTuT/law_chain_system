# LawChain · 法律问答系统

> 基于 **RAG（检索增强生成）** 的中文法律智能问答系统。
> 多级检索链：Redis 缓存 → BM25 关键词 → **意图分类与问题改写** → **Milvus 混合检索 + 重排序** → **LLM 流式生成**。
> 附带一个须弥草系二次元风格的 WebSocket 聊天前端。

<p align="center">
  <img src="static/data/nahida.jpg" width="150" alt="Nahida">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v2-3ba55d" alt="version">
  <img src="https://img.shields.io/badge/python-3.10-blue" alt="python">
  <img src="https://img.shields.io/badge/Milvus-2.5.4-00a1ea" alt="milvus">
  <img src="https://img.shields.io/badge/license-学习研究-lightgrey" alt="license">
</p>

---

## 🆕 v2 更新要点

> 本次迭代覆盖 **11 个文件、+1164 / −436 行**，主线是「**提升检索质量**」与「**改善等待体验**」。

| # | 方向 | 具体内容 |
|---|------|---------|
| 1 | **流式输出改造** | LLM 由 `stream=False` 一次性返回改为 `stream=True` 逐 token 产出；检索链 `search()` 变为可按需 `yield` 的生成器；WebSocket 新增 `answer_start` / `answer_delta` / `answer_end` 三帧协议；服务端用「工作线程推队列 + 事件循环消费」桥接，既不阻塞心跳也不再直接序列化生成器 |
| 2 | **Prompt 优化** | 由 1 个模板扩展为 **5 个**：`rag_prompt` 升级为结构化专家提示（案例引用规范、编号规则、2 个 Few-shot 示例）；新增 `classification_prompt` 与 HyDE / 子查询 / 回溯三个改写模板 |
| 3 | **四模式问题检索** | 新增 LLM **意图分类**（`direct` / `hyde` / `subquery` / `recall`）→ 按类型**改写问题** → 再检索；`subquery` 支持一次拆出多个子问题并逐个检索 |
| 4 | **混合检索 + 两段排序** | 稠密（BGE-M3 dense / COSINE）+ 稀疏（BGE-M3 sparse / IP）**混合检索**，用 `WeightedRanker(1.0, 0.7)` 融合（粗排）；新增 `_de_weight()` 按内容去重；新增 **BGE-reranker-large** 交叉编码器重排序（精排） |
| 5 | **上下文组装规范化** | 新增 `_format_context()`，把命中的案例/条款整理成 `[序号] 题目：… 来源：…` 的结构化文本，替代原先直接 `str(hit列表)` 的噪声写法 |
| 6 | **检索参数配置化** | 由 `retrieval_k` / `candidate_m` 拆分为**案例路与条款路两套独立参数**（`case_*` / `clause_*`） |
| 7 | **工程修复** | 新增 `_ensure_loaded()` 解决 Milvus 重启后「collection not loaded」；集合加载检查失败不再穿透打断整个检索 |

---

## 📖 项目简介

针对中文法律领域问答场景，本项目实现了一条**完整可运行的 RAG 链路**：
把分散在法律条文（`.docx` / `.pdf`）与法院案例（`.pdf`）中的非结构化文本，经过 OCR 抽取、
结构化切块、向量化后存入 MySQL 与 Milvus；用户提问时按"**缓存 → 关键词 → 意图分类与改写
→ 混合检索与重排序 → 大模型流式生成**"召回答案，并通过 WebSocket 逐字推送到聊天页面。

**核心特点**

| 特点 | 说明 |
|------|------|
| 🧱 端到端可跑通 | 从原始 PDF/DOCX 到网页问答，全链路代码均在仓库内 |
| 🪜 多级检索降级 | 缓存命中 → 关键词命中 → 混合检索 + 重排序 + LLM 生成，逐级兜底 |
| 🎯 四模式检索 | 按问题类型自动选择 直接检索 / HyDE / 子查询 / 回溯检索，长短问题区别对待 |
| 🔀 混合检索 + 两段排序 | 稠密+稀疏融合粗排 → 内容去重 → 交叉编码器精排，兼顾召回率与精度 |
| 📚 领域化切块 | 不用通用 `RecursiveCharacterTextSplitter` 硬切，而是**按法条结构 / 案例 section 语义切块** |
| 🔍 OCR 全格式支持 | PDF / DOCX / PPTX / 图片 统一走 RapidOCR 抽取文本 |
| 🔗 父子块设计 | 案例子块建向量检索，召回后回填**父块（完整案情）**给 LLM，兼顾精度与上下文 |
| ⚡ 流式 + 阶段提示 | 逐 token 推送；等待期显示「正在检索法规与案例…」→「正在生成回答…」，首字前不留空白气泡 |
| 💬 长连接前端 | WebSocket + 心跳保活 + 断线重连 + 请求队列，二次元风格 UI |
| 🐣 零外部模型下载 | `bge-m3` 与 `bge-reranker-large` 权重本地携带，离线可用 |

---

## 🏗️ 系统架构

### 在线问答链路

```
┌───────────────────────────────────────────────────────────────────────┐
│                          浏览器 (chat.html)                            │
│   WebSocket /ws/chat · 心跳 ping/pong · 指数退避重连 · 打字机式增量渲染   │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ JSON 帧：answer_start / delta / end
┌───────────────────────────────▼───────────────────────────────────────┐
│                       FastAPI (static/app.py)                          │
│   GET /  → chat.html     GET /health → 探活     GET /static/* → 静态资源 │
│   WS /ws/chat：单一读取者循环 + 并发心跳协程                             │
│   生成器 → 工作线程推队列 → 事件循环消费 → 逐帧下发（不阻塞心跳）          │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────────┐
│                  LawChainClient (main.py) —— 统一入口                  │
│                                                                        │
│  ① Redis 缓存 ──命中──► 直接返回                                        │
│       │未命中                                                           │
│  ② BM25 关键词 ─命中──► 直接返回                                        │
│       │未命中                                                           │
│  ③ LLM 意图分类 ──► 问题改写（按类型选策略）                             │
│       │                                                                │
│  ④ Milvus 混合检索（粗排）─► 去重 ─► BGE-reranker-large（精排）          │
│       │                                                                │
│  ⑤ LLM 流式生成 ──► yield 逐块返回                                      │
│       │                                                                │
│  ⑥ 生成结束 ──► 完整答案回填 Redis + MySQL（下次命中 ①/②）              │
└───────────────────────────────────────────────────────────────────────┘
         │                    │                      │
    ┌────▼────┐         ┌─────▼─────┐          ┌─────▼──────┐
    │  Redis  │         │   MySQL   │          │   Milvus   │
    │ 问答缓存 │         │ 结构化块  │          │ 向量 + 稀疏 │
    └─────────┘         └───────────┘          └────────────┘
```

### 数据准备链路（离线，一次性）

```
原始文档                  文本抽取                    结构化切块                 入库
─────────            ──────────────            ──────────────          ──────────
 common/data/          doc_loader.py              doc_spliter.py
   ├─ law/*.docx  ──►  ┌──────────────┐  ──►   ┌───────────────┐  ──►  MySQL
   │  *.pdf            │ OCRPDFLoader │        │ split_law     │      law_chunk
   └─ case/*.pdf  ──►  │ OCRDOCLoader │        │  (编-章-节-条) │      law_qa
      qa/*.xlsx        │ OCRPPTLoader │        ├───────────────┤
                       │ OCRIMGLoader │        │ split_cases   │
                       │ TextLoader   │        │  (5 个 section)│
                       └──────────────┘        └───────┬───────┘
                                                       │
                                                       ▼
                                            MilvusClientSystem._init_data()
                                       MySQL → 父块拼接 → BGE-M3 向量化（稠密+稀疏）
                                                       │
                                                       ▼
                                            Milvus: law_cases / law_clause
```

---

## 📁 目录结构

```
law_chain_system/
├── main.py                       # ★ 统一入口：LawChainClient（多级检索链 + 流式）
├── config.ini                    # ★ 全局配置（数据库/模型/检索参数）
├── requirments.txt               # 依赖清单（注意：文件名拼写 & 编码问题，见「已知问题」）
├── 优化项.md                      # 迭代待办清单（含 BM25 归一化问题的分析）
│
├── base/                         # 基础设施
│   ├── config.py                 #   配置加载（config.ini + 环境变量覆盖）
│   └── logger.py                 #   日志：控制台 + 文件双输出
│
├── common/                       # 数据处理层
│   ├── doc_loader.py             #   按扩展名分发到不同 Loader
│   ├── doc_spliter.py            #   按 law/case 分发到不同切块器
│   ├── data/                     #   原始语料（共约 79 MB）
│   │   ├── law/                  #     法律条文（19 个 docx/pdf）
│   │   ├── case/                 #     法院案例（24 个 pdf）
│   │   └── qa/                   #     法答网问答集（1 个 xlsx）
│   └── utils/
│       ├── law_document_loaders/ #   OCR 文档加载器
│       │   ├── law_ocr.py        #     RapidOCR 工厂（GPU 优先，回落 ONNX）
│       │   ├── law_pdfloader.py  #     PDF → 文本（版面/页码处理）
│       │   ├── law_docloader.py  #     DOCX → 文本
│       │   ├── law_pptloader.py  #     PPTX → 文本
│       │   └── law_imgloader.py  #     图片 → 文本
│       └── law_text_spliter/     #   ★ 领域化切块（本项目核心亮点）
│           ├── chunk_common.py   #     公共清洗 + 统一 chunk 格式
│           ├── legal_chunker.py  #     split_law：按"编-章-节-条"切
│           ├── case_chunker.py   #     split_cases：按 5 个 section 切
│           └── README.md         #     切块模块详细文档
│
├── mysql_qa/                     # 关键词检索层
│   ├── main.py                   #   MySQLQASystem（CLI 版，见「已知问题」）
│   ├── db/mysql_client.py        #   建表 + 批量写 + 按 id 回查
│   ├── cache/redis_client.py     #   问答缓存
│   ├── retrieval/bm25_search.py  #   BM25 关键词检索
│   └── utils/preprocess.py       #   jieba 分词
│
├── rag_qa/                       # 向量 + 重排 + 生成层
│   ├── main.py                   #   RAGQAClient（CLI 版）
│   ├── db/milvus_client.py       #   建库/建集合/混合检索/去重/加载检查
│   ├── llm/llm_client.py         #   LLM 调用：意图分类 + 问题改写 + 流式生成
│   ├── prompt/template.py        #   ★ 5 个提示词模板（v2 扩充）
│   ├── utils/embedding.py        #   BGE-M3 向量化（稠密 + 稀疏）
│   ├── utils/reranker.py         #   ★ BGE-reranker-large 重排序（v2 新增）
│   └── models/                   #   本地模型（需自行下载，勿推 GitHub）
│       ├── bge-m3/               #     ~4.3 GB
│       └── bge-reranker-large/   #     ~6.4 GB（v2 新增）
│
├── static/                       # Web 层
│   ├── app.py                    #   FastAPI 应用（流式协议 + 心跳 + 静态挂载）
│   ├── data/nahida.jpg           #   页面头像
│   └── templates/chat.html       #   二次元风格聊天页（流式渲染）
│
└── logs/app.log                  # 运行日志
```

---

## 🔄 核心流程详解

### 1. 提问 → 答案（在线链路）

`main.py` 中 `LawChainClient.search()` 是整条链路的核心。注意它**有两种返回形态**：
命中缓存/关键词时返回 `str`，需要 LLM 生成时返回**生成器**（由 `app.py` 判断类型后分别处理）。

```python
def search(self, question: str):
    if not question:
        return "消息为空，请重新输入有效问题 QAQ"

    # ① Redis 缓存：命中直接返回 str
    answer = self.redis_client.get_answer(question)
    if answer:
        return answer

    # ② MySQL BM25 关键词检索：命中直接返回 str
    answer = self.bm25_search.search(question)
    if answer:
        return answer

    # ③ 意图分类 + 问题改写（v2 新增）
    optimizer_query = self.llm_client.query_generate(question)

    # ④ 混合检索 + 去重 + 重排序（v2 改造）
    case_chunk, clause_chunk = self.milvus_client.search(optimizer_query)

    # ⑤ 返回生成器：边生成边 yield，边累积
    return self._stream_and_cache(question, case_chunk, clause_chunk)


def _stream_and_cache(self, question, case_chunk, clause_chunk):
    """流式产出答案，并在结束时把完整答案回填缓存"""
    chunks = []
    try:
        for chunk in self.llm_client.generate(question, case_chunk, clause_chunk):
            chunks.append(chunk)
            yield chunk                      # ← 逐块交给上层推给前端
    finally:
        full_answer = "".join(chunks)        # ← 生成器被消费完才执行
        if full_answer:
            self.redis_client.set_question(question, full_answer)
            self.mysql_client.insert_data([{"question": question, "answer": full_answer}])
```

**降级策略的意义**：高频问题在第 ①/② 级就被拦下，只有真正"没见过"的问题才触发
分类 + 改写 + 混合检索 + 重排 + LLM 生成，显著降低 LLM 调用成本与响应延迟。

### 2. 四模式问题检索（v2 核心）

`LLMClient.query_generate()` 先用一次 LLM 调用做**意图分类**，再按类型选择改写策略：

```
用户问题
   │
   ▼
┌──────────────────────┐
│ _search_classification│  返回 direct / hyde / subquery / recall
└──────────┬───────────┘  非法标签 → 兜底 direct；调用失败 → 兜底 direct
           │
   ┌───────┼────────┬──────────────┬──────────────┐
   ▼       ▼        ▼              ▼              ▼
direct   hyde    subquery       recall        (异常)
   │       │        │              │              │
   │  生成假设性   拆成 2~3 个    去口语化、     原问题
   │  法律分析     独立子问题     术语规范化     直接返回
   │       │        │              │
   └───────┴────────┴──────────────┘
                   │
                   ▼
        返回 str（单查询）或 list[str]（多子查询）→ 交给 Milvus 检索
```

| 模式 | 适用场景 | 处理方式 | 返回值 |
|------|---------|---------|--------|
| `direct` | 术语清晰、可直检（"离婚冷静期是多久？"） | 不改写 | `str`（原问题） |
| `hyde` | 情境化提问（"甲借乙10万没打欠条能要回吗？"） | 让 LLM 先生成**假设性法律分析**，用该分析去检索 | `str` |
| `subquery` | 含多个法律维度（"婚前买房婚后还贷离婚怎么分？"） | 拆成 **2~3 个语义完整的子问题**，逐个检索后汇总 | `list[str]` |
| `recall` | 口语化、冗长（"我在公司干了好几年老板一直不签合同现在把我开了咋办"） | 简化 + 术语规范化，提升检索关键词集中性 | `str` |

> 💡 `MilvusClientSystem.search()` 的入参既接受 `str` 也接受 `list`，
> 就是为了对接 `subquery` 的子问题列表；多子查询时重排配额按 `max(top_m // num, 1)` 分配。

### 3. Milvus 混合检索 + 两段排序（v2 核心）

```
查询问题（可能来自改写）
   │
   ▼  BGE-M3 一次编码出两种向量
┌──────────────────────────────────────────────────────┐
│  dense_vector  (1024 维)        sparse_vector (词权重) │
└───────┬──────────────────────────────┬───────────────┘
        │ AnnSearchRequest             │ AnnSearchRequest
        │ metric = COSINE              │ metric = IP
        ▼                              ▼
┌──────────────────────────────────────────────────────┐
│   hybrid_search(collection, reqs=[dense, sparse],     │
│                 ranker=WeightedRanker(1.0, 0.7))      │  ← 粗排
│   对 law_cases 与 law_clause 各执行一次                 │
└───────────────────────┬──────────────────────────────┘
                        │  case_top_k=5 / clause_top_k=6
                        ▼
              _de_weight() 按 parent_content / text_content 去重
                        │
                        ▼
        BGE-reranker-large 交叉编码器精排（query, doc）对
                        │  案例按 parent_content 重排
                        │  条款按 text_content  重排
                        ▼
             case_top_m=3 条案例 + clause_top_m=4 条条款
                        │
                        ▼
        LLMClient._format_context() → [序号] 题目：… 来源：… + 正文
                        │
                        ▼
                   组装进 rag_prompt
```

**两段排序的分工**：

- **粗排（召回）** 用混合检索保证**召回率**——稠密向量管语义相似，稀疏向量管关键词精确命中
  （法条编号、专业术语等稠密向量容易漏的场景）。
- **精排（重排）** 用交叉编码器保证**精度**——reranker 把 query 与每个候选**拼接后一起编码**，
  比双塔式的向量点积更能捕捉细粒度相关性，代价是算力高，所以只对粗排的少量候选做。

### 4. WebSocket 通信协议（v2 新增流式帧）

| 方向 | 消息体 | 说明 |
|------|--------|------|
| C → S | `{"type":"question","text":"..."}` | 用户提问 |
| C → S | `{"type":"ping"}` / `{"type":"pong"}` | 心跳 |
| S → C | `{"type":"answer_start"}` | **开始回答**；前端保留等待指示器，只更新阶段文案 |
| S → C | `{"type":"answer_delta","text":"片段"}` | **增量文本**；首个 delta 到达时前端才撤指示器并建气泡 |
| S → C | `{"type":"answer_end"}` | **回答结束**；前端做一次 markdown 收尾渲染 |
| S → C | `{"type":"answer","text":"..."}` | 一次性回答（缓存/BM25 命中/空问题/无结果） |
| S → C | `{"type":"ping"}` / `{"type":"pong"}` | 心跳 |
| S → C | `{"type":"error","text":"..."}` | 异常兜底 |

> 为兼容旧客户端，**纯文本消息仍按提问处理**。

**为什么 `answer_start` 不立刻建气泡**：服务端在 `answer_start` 之前要完成意图分类、
问题改写、混合检索与重排，之后还要等 LLM 出第一个 token，实测这段可达 **3~10 秒**。
若此刻就撤掉等待指示器、开一个空气泡，用户会看到空白以为卡死。因此
**指示器保留到第一个字真正到来**，只把文案从「正在检索法规与案例…」换成「正在生成回答…」。

### 5. 心跳保活机制

长连接最容易出问题的地方是"连接看起来还在，实际已经死了"。本项目做法：

- **服务端**：`heartbeat_loop()` 每 `HEARTBEAT_INTERVAL`(30s) 主动发 `ping`，
  并检查共享的 `ConnState`；连续 `HEARTBEAT_MAX_MISS`(2) 个窗口没收到**任何**帧，
  即判定链路已死并 `close(1001)`。
- **关键设计：单一读取者**。Starlette 的 WebSocket 只允许一个协程在读，
  因此**心跳协程从不调用 `receive_text()`**，只负责发送与计时；由 `receive_loop()`
  统一读取并在收到任何帧时刷新时间戳。这样避免了心跳与业务逻辑争抢消息、
  甚至把用户提问"吃掉"的经典 bug。
- **前端**：收到 `ping` 立即回 `pong`；同时有 120s 看门狗（任何服务端帧都会重置），
  超时主动断开并**指数退避重连**（1s→2s→4s→8s，上限 15s）；离线期间的提问进队列，
  重连成功后自动补发。

### 6. 领域化切块

通用切块器会把法条从中间切断、把案例的"裁判要旨"和"基本案情"混在一起。
本项目针对两种文体分别写切块逻辑：

**法条 `split_law()` —— 四步**
```
原始文本 → ① 剥离前置内容（标题/修订历史/目录）
        → ② 通用清洗（分页标记、页码行、书眉页脚、零宽字符）
        → ③ 解析"编-章-节-条"层级，得到每条完整文本
        → ④ RecursiveCharacterTextSplitter 成块
           分隔符优先级：编/章/节 > 条 > 换行/句号/分号
```

**案例 `split_cases()` —— 四步**
```
原始文本 → ① 以"【案件基本信息】"为锚点，向前回溯案例标题块（编号/主标题/副标题）
        → ② 丢弃标题块之前的内容（封面/序/编委会/目录）
        → ③ 按锚点把正文切成一个个案例
        → ④ 每个案例按 5 个固定 section 标记切块；过长 section 再细切（保留元数据）

5 个 section（case_chunker.py 中 CASE_SECTION_NAMES，正则按"整行只有【section】"匹配）：
    案件基本信息 · 基本案情 · 案件焦点 · 法院裁判要旨 · 法官后语
```

**统一 chunk 格式**（`chunk_common.py` 中 `make_chunk()` 保证字段完全一致）：
```json
{
  "source": "1 婚姻家庭继承.pdf",
  "doc_type": "case",
  "chunk_type": "case_section",
  "text": "……",
  "metadata": {
    "case_no": "1",
    "case_title": "彩礼适格返还主体……",
    "section": "基本案情",
    "path": ["第一编 总则", "第一章 ……"],
    "article_no": "第一条",
    "law_name": "民事诉讼法",
    "extra": {}
  }
}
```

### 7. 父子块检索设计

案例往往很长，整案做向量会让语义被稀释，子块做向量又会丢掉上下文。本项目采用**父子块**：

- **子块建向量**：每个 section（案件基本信息/基本案情/案件焦点/法院裁判要旨/法官后语）单独
  向量化，保证检索精度。
- **父块给 LLM**：`_init_data()` 中用 `GROUP_CONCAT` 把同一案例的
  **基本案情 + 案件焦点 + 法院裁判要旨** 拼成完整案情，存进 `parent_content` 字段
  （`法官后语` 与 `案件基本信息` 不参与拼接）。
- 命中子块后，重排与喂给 LLM 的都用**父块全文**，让它看到完整脉络。

---

## 🛠️ 技术栈

### 应用层
| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10 | 运行环境（开发使用 conda 环境 `law_rag`） |
| FastAPI | 0.115.12 | Web 框架（HTTP + WebSocket） |
| Uvicorn | 0.41.0 | ASGI 服务器 |
| Starlette | 0.46.2 | FastAPI 底层（WebSocket / StaticFiles） |
| Jinja2 | 3.1.6 | HTML 模板渲染 |
| WebSockets | 16.0 | 长连接 |

### 检索与生成层
| 组件 | 版本 | 用途 |
|------|------|------|
| PyMilvus | 2.5.4 | 向量数据库客户端（`hybrid_search` / `AnnSearchRequest` / `WeightedRanker`） |
| Milvus | — | 向量库（standalone，默认 `localhost:19530`） |
| **BGE-M3** | 本地 | 稠密 + 稀疏双向量编码 |
| **BGE-reranker-large** | 本地 | 交叉编码器重排序（v2 新增） |
| FlagEmbedding | 1.3.5 | `FlagReranker` 重排器 + BGE 工具链 |
| milvus-model | 0.2.5 | `BGEM3EmbeddingFunction` 封装 |
| rank-bm25 | 0.2.2 | BM25 关键词检索 |
| jieba | 0.42.1 | 中文分词 |
| transformers | 4.45.0 | 模型加载 |
| torch | 2.14.0+cu132 | 深度学习框架（CUDA 12.1） |
| OpenAI SDK | 2.24.0 | LLM 调用（OpenAI 兼容协议，支持 `stream=True`） |
| LangChain | 1.2.10 | `PromptTemplate` / `TextSplitter` |

### 存储层
| 组件 | 版本 | 用途 |
|------|------|------|
| MySQL | — | 结构化 chunk 与问答对 |
| PyMySQL | 1.1.1 | MySQL 驱动（`DictCursor`） |
| Redis | 5.3.1 | 问答缓存（TTL 1 小时） |

### 文档处理层
| 组件 | 版本 | 用途 |
|------|------|------|
| RapidOCR | onnxruntime 1.23.2 | OCR 文字识别 |
| PyMuPDF | 1.23.16 | PDF 解析 |
| python-docx | 1.1.2 | DOCX 解析 |
| python-pptx | 0.6.23 | PPTX 解析 |
| pandas / openpyxl | 2.3.1 / 3.1.5 | Excel 问答集解析 |
| Pillow / opencv-python | 9.5.0 / 4.10.0 | 图像处理 |

### 模型配置
| 项 | BGE-M3 | BGE-reranker-large |
|----|--------|--------------------|
| 路径 | `rag_qa/models/bge-m3/` | `rag_qa/models/bge-reranker-large/` |
| 体积 | ~4.3 GB | ~6.4 GB |
| 输出 | 稠密 1024 维 + 稀疏词权重 | 单个相关性分数 |
| 推理设备 | 自动检测：有 CUDA 用 GPU（fp16） | 同左 |
| batch_size | 32 | — |

> ⚠️ **两个模型合计约 10.7 GB**，请勿直接推送 GitHub，见「已知问题」。

---

## 🚀 快速开始

### 1. 前置依赖

需要先启动三个外部服务（推荐用 Docker）：

```bash
# Redis
docker run -d --name redis-law -p 6379:6379 redis:7 --requirepass 1234

# MySQL
docker run -d --name mysql-law -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=123456 \
  -e MYSQL_DATABASE=law_chain \
  mysql:8

# Milvus standalone（需要 etcd + MinIO，建议用官方 compose）
# 下载 milvus-standalone-docker-compose.yml 后：
docker compose -f milvus-standalone-docker-compose.yml up -d
```

> Milvus 启动较慢（依赖 etcd + MinIO，通常 30 秒 ~ 2 分钟）。
> **务必等 `localhost:19530` 可连通后**再启动本项目，否则 `LawChainClient` 初始化会失败。

### 2. 下载模型

两个模型权重都不随仓库提供，需自行下载后放到对应目录：

```bash
# 方式一：ModelScope（国内更快）
modelscope download --model BAAI/bge-m3              --local_dir rag_qa/models/bge-m3
modelscope download --model BAAI/bge-reranker-large  --local_dir rag_qa/models/bge-reranker-large

# 方式二：HuggingFace
huggingface-cli download BAAI/bge-m3              --local-dir rag_qa/models/bge-m3
huggingface-cli download BAAI/bge-reranker-large  --local-dir rag_qa/models/bge-reranker-large
```

### 3. 安装依赖

```bash
conda create -n law_rag python=3.10 -y
conda activate law_rag

# ⚠️ requirments.txt 是 UTF-16LE 编码，Linux/macOS 下直接安装会报错，
#    请先转成 UTF-8（见「已知问题」第 2 条）
pip install -r requirments.txt
```

### 4. 配置

编辑 `config.ini`：

```ini
[mysql]
host = localhost
user = root
password = 123456          # ← 改成你的密码
database = law_chain

[redis]
host = localhost
port = 6379
password = 1234            # ← 改成你的密码

[milvus]
host = localhost
port = 19530
data_name = law_chain
collection_cases = law_cases
collection_articles = law_clause

[llm]
model = deepseek-flash
dashscope_base_url = https://api.deepseek.com
dashscope_api_key = sk-xxxxxxxx    # ← 必须补上，否则调用 LLM 会 401

# v2：案例路与条款路参数独立配置
[retrieval]
vector_dim = 1024
case_retrieval_k = 5       # 案例：粗排召回条数
case_candidate_m = 3       # 案例：精排后交给 LLM 的条数
clause_retrieval_k = 6     # 条款：粗排召回条数
clause_candidate_m = 4     # 条款：精排后交给 LLM 的条数
```

> 也可以不写 `dashscope_api_key`，直接用环境变量 `DEEPSEEK_API_KEY`（优先级更高，适合不把密钥写进仓库）。

所有配置项都支持**环境变量覆盖**：
`MYSQL_HOST` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` /
`REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_DB` /
`MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_DATA_NAME` /
`DEEPSEEK_API_KEY` / `MODEL_NAME` / `DASHSCOPE_BASE_URL`

### 5. 启动

**方式一：Web 服务（推荐）**

```bash
uvicorn static.app:app --reload
# 浏览器打开 http://127.0.0.1:8000
```

启动时会通过 lifespan 初始化 `LawChainClient`（连接三个数据库 + 加载
bge-m3 与 bge-reranker-large 两个模型），首次启动约需 **10~30 秒**，
日志出现 `LawChainClient 初始化完成` 即就绪。

> 💡 改了 `chat.html` 后需要**重启 uvicorn**：Jinja2Templates 在非 debug 模式下会缓存模板，
> `--reload` 只监听 `.py` 文件。浏览器侧建议 `Ctrl+Shift+R` 强刷。

**方式二：命令行**

```bash
python main.py            # 完整检索链（流式打印）
python mysql_qa/main.py   # 仅缓存 + BM25
python rag_qa/main.py     # 检索 + LLM（流式）
```

### 6. 数据初始化

首次运行时 `MilvusClientSystem._init_data()` 会自动检查 Milvus 集合是否为空，
为空则从 MySQL 读取数据并向量化入库。因此**需要先保证 MySQL 有数据**：

```bash
# 触发一次 MySQL 数据灌入（幂等：表非空则跳过）
python mysql_qa/main.py    # 启动后直接 exit 即可
```

---

## 📡 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 渲染聊天页面 `chat.html` |
| GET | `/health` | 探活，返回 `{"status":"ok","heartbeat_interval":30}` |
| WS | `/ws/chat` | 问答长连接（流式 JSON 协议见上表） |
| GET | `/static/*` | 静态资源（头像等） |

---

## 🎨 前端说明

单文件自包含（`static/templates/chat.html`），**无任何外部 CDN 依赖**，离线可用。

- **风格**：须弥草系配色（草绿 `#3ba55d` / 米白 / 描金 `#d9b26a`），圆润气泡 + 柔和光晕
- **头像**：`static/data/nahida.jpg`（URL 为 `/static/data/nahida.jpg`），
  圆形裁切，顶栏 / 欢迎区 / 消息气泡三处复用；加载失败时降级为 🌿 不影响布局
- **流式渲染**：流式期间用**纯文本增量**（`white-space: pre-wrap`，只改 `textContent`），
  结束时才做一次完整 markdown 渲染 —— 避免增量解析 Markdown 出错导致气泡/头像消失
- **等待体验**：阶段提示「正在检索法规与案例…」→「正在生成回答…」，
  指示器保留到第一个字到达，**不留空白气泡**
- **Markdown 渲染**：自写轻量渲染器（标题 / 列表 / 引用 / 代码块 / 粗体 / 行内代码），
  **先 HTML 转义再渲染**，避免 XSS
- **交互**：引导问题卡片、Enter 发送 / Shift+Enter 换行、自动增高输入框、
  连接状态灯 + 实时延迟显示
- **容错**：所有 DOM 操作包 `try/catch`；断线时提问进队列并提示，重连后自动补发；
  页面重新可见时立即重连

---

## 🧩 关键设计决策

| 决策 | 原因 |
|------|------|
| **检索链有两种返回形态（str / 生成器）** | 缓存与关键词命中无需 LLM，直接返回字符串最省事；需要生成时返回生成器以支持流式 |
| **`asyncio.to_thread()` 包住检索链** | 检索链全是同步阻塞 IO（Redis/MySQL/Milvus/LLM），直接 `await` 会卡死事件循环，心跳随之失效 |
| **心跳协程不读 Socket** | Starlette 只允许单读取者；心跳若也 `receive`，会与业务循环抢消息 |
| **生成器用「线程推队列 → 事件循环消费」桥接** | 生成器必须在工作线程迭代，但要用 `loop.call_soon_threadsafe` 把每块送回事件循环发送 |
| **`_stream_and_cache` 的缓存写在 `finally`** | 无论正常跑完还是被中断，都要回填已生成的内容；完整迭代由 `app.py` 的泵线程保证 |
| **`answer_start` 不建气泡** | 首字前有 3~10 秒空窗，开空气泡会让用户以为卡死 |
| **流式期间不做增量 Markdown** | 每块重解析 Markdown 既浪费又脆弱，一次渲染失败就会丢掉整个气泡 |
| **混合检索粗排 + 交叉编码器精排** | 粗排保召回（稠密管语义、稀疏管关键词），精排保精度（交叉编码更准但贵，只对少量候选做） |
| **`WeightedRanker(1.0, 0.7)`** | 稠密权重高于稀疏：语义相关性为主，关键词命中为辅 |
| **重排按内容字段区分** | 案例按 `parent_content`（完整案情）、条款按 `text_content`（条文正文）评相关性 |
| **`_de_weight()` 按内容去重** | 同一案例的多个子块可能同时命中，内容去重避免重复占用配额 |
| **`_ensure_loaded()` 每次检索前检查** | Milvus 重启后集合会被自动卸载，不检查就会报 `collection not loaded` |
| **`mapping_table` 做下标→主键映射** | 避免在 BM25 层引入额外存储，用内存数组即可回查 MySQL |
| **子块向量 + 父块上下文** | 兼顾检索精度（子块语义集中）与生成质量（父块脉络完整） |
| **`create_schema(auto_id=False)` + MD5 主键** | 用 `md5(原id + case_no/article_no)` 保证重复入库时幂等 |
| **索引用 FLAT** | 当前数据量（1 万余条）下 FLAT 精度最高；超 10 万再换 HNSW（见 `优化项.md`） |
| **每个模块 `sys.path.insert(0, project_dir)`** | 支持"脚本直接运行"与"包内导入"两种方式共存 |

---

## ⚠️ 已知问题与当前局限

> 这是一份**诚实的清单**。v2 解决了 v1 的若干缺陷，但仍有下列问题已知未修。

### ✅ v2 已修复的 v1 问题

| 原 v1 问题 | v2 状态 |
|-----------|--------|
| 稀疏向量"白算"（检索只用稠密） | ✅ 已启用 `hybrid_search` + `WeightedRanker` |
| 检索结果无排序/去重/重排 | ✅ 已加 `_de_weight()` 去重 + BGE-reranker-large 重排 |
| 检索结果合并成嵌套列表、`str()` 进 prompt | ✅ 新增 `_format_context()` 结构化为 `[序号] 题目：来源：` |
| 无流式输出，用户全程白屏等待 | ✅ 已改为逐 token 流式 + 阶段提示 |
| 检索参数不可分路配置 | ✅ 已拆为 `case_*` / `clause_*` 两套 |
| Milvus 重启后 `collection not loaded` | ✅ 新增 `_ensure_loaded()` 前置检查 |

**v2 提交后修补**

| 问题 | 状态 |
|------|------|
| 重排模型 fp16 未生效（`FlagReranker(usp_fp16=...)` 参数名拼写错误） | ✅ 已修正为 `use_fp16`，GPU 显存占用减半 |
| `llm_client.py` 冗余导入 `from urllib import response` | ✅ 已移除 |

### 🔴 上传 GitHub 前必须处理

| # | 问题 | 说明 | 建议 |
|---|------|------|------|
| 1 | **模型权重合计约 10.7 GB** | `bge-m3` ~4.3 GB（`pytorch_model.bin` 2.17 GB + `onnx/model.onnx_data` 2.16 GB）；**v2 新增 `bge-reranker-large` ~6.4 GB**（`pytorch_model.bin` 2.14 GB + `model.safetensors` 2.14 GB 两份等价权重并存） | 已在 `.gitignore` 排除 `rag_qa/models/`；README 提供下载命令。若确需入库请用 **Git LFS**（单文件 >100 MB GitHub 直接拒绝）。可先删掉 `onnx/` 与重复的 `.safetensors` 省一半体积 |
| 2 | **`requirments.txt` 是 UTF-16LE** | Windows `pip freeze` 的产物，带 BOM；Linux/macOS 下 `pip install -r` 会解析失败；且文件名拼写错误（应为 `requirements`） | 转成 UTF-8 并重命名；建议精简为直接依赖 |
| 3 | **明文密码入库** | `config.ini` 含 MySQL `123456`、Redis `1234`；`base/config.py` 把它们写成了代码 fallback 默认值 | 移除代码里的明文默认值，改用 `.env` + `python-dotenv`，并提供 `.env.example` |
| 4 | **依赖清单是整环境快照** | 约 250 个包，包含大量与本项目无关的传递依赖（langchain 全家桶、ragas、kubernetes 等） | 按直接依赖重写，或用 `pipreqs` 生成 |
| 5 | **`common/data/` 有 79 MB 语料** | 24 个案例 PDF + 19 个法条 + 问答 Excel | 若版权/体积敏感，改放外部下载链接 |

### 🟡 代码层面的已知缺陷

**v2 新增代码相关**

| # | 位置 | 问题 |
|---|------|------|
| 6 | `milvus_client.py` 配额分配 | `top_m=max(case_top_m // num, 1)`：`subquery` 拆出 3 个子问题时，案例 `3//3=1`、条款 `4//3=1`，**最终条数反而少于单查询**（应为 3/4）。配额分配逻辑需要重新设计。 |
| 7 | `llm_client.py:97-99` | **异常退化为"没问题"**：`query_generate()` 出错时 `return query`（原问题），调用方无法区分"分类为 direct"与"分类失败"，检索质量静默下降。 |
| 8 | `llm_client.py:110-111` | **兜底有副作用**：`generate()` 在无上下文时 `yield` 一句提示后 `return`，该提示会被 `_stream_and_cache` 累积并写入缓存，导致"信息不足"被当成有效答案缓存起来。 |
| 9 | `milvus_client.py` 异常处理 | `search()` 里 `except MilvusException` 直接 `return [], []`，**静默吞掉检索失败**；调用方无法区分"没检索到"与"检索报错"。 |
| 10 | 意图分类的额外开销 | 每次未命中缓存的问题都要**多一次 LLM 调用**做分类（实测 2.3s）；`hyde`/`subquery`/`recall` 还要**再一次**做改写。可考虑分类结果缓存或与改写合并为一次调用。 |
| 11 | `llm_client.py:12` | **未使用导入 + 循环依赖隐患**：导入了 `MilvusClientSystem` 但全文未使用；而 `milvus_client.py` 会反向引用 LLM 模块，存在循环导入风险。 |

**v1 遗留未修**

| # | 位置 | 问题 |
|---|------|------|
| 12 | `mysql_qa/main.py:75-81` | **契约不匹配（功能性 bug）**：`BM25Search.search()` 返回 `str \| None`，但 `MySQLQASystem.search()` 按 `list[dict]` 处理（`for answer in answers: if "question" in answer`）。字符串会被逐**字符**迭代，`in` 判断恒为 False，导致"MySQL 命中 → 回填 Redis 缓存"**从未生效**；若返回 `None` 则抛 `TypeError` 被静默吞掉。**注：实际使用的 `LawChainClient`（根 `main.py`）调用方式正确，不受影响。** |
| 13 | `bm25_search.py:79-84` | **阈值几乎不可能命中**：`softmax` 归一化后榜首分数随语料规模迅速衰减，默认 `threshold=0.85` 在语料稍大时永远达不到，BM25 层形同废用；softmax 也抹掉了 BM25 原始分的绝对意义。**`优化项.md` 已记录改法**（改原始分 + 绝对阈值 + 与次高分的 gap 判断）。 |
| 14 | `llm_client.py:25-44` | **异常被吞**：`_init_llm` 捕获所有异常且**不重新抛出**，失败时 `self.llm` 保持 `None`，故障延迟到首次提问才暴露。 |
| 15 | `milvus_client.py:38` | **初始化失败即全站不可用**：`MilvusClientSystem` 构造失败会抛出，`LawChainClient` 随之失败，FastAPI 无法启动（页面也打不开）。建议把 Milvus 做成可降级组件。 |
| 16 | `bm25_search.py` / `redis_client.py` | **连接泄漏**：`BM25Search` 自建 `MySQLClient` 却**没有 `close()`**；`LawChainClient.close()` 也未关闭它。`redis.Redis()` 是惰性连接，`__init__` 里的 `except redis.RedisError` 基本捕获不到连接失败。 |
| 17 | `mysql_client.py` | **硬编码库名**：`insert_data` 里写死 `law_chain.law_qa` / `law_chain.law_chunk`；若 `MYSQL_DATABASE` 配置不同则直接失败。 |
| 18 | 唯一键设计 | `uk_question(question(255))` / `uk_chunk(source, text_content(255))` 配合 `insert ignore`，会**静默丢弃**前 255 字相同的不同问题/长法条，且无告警。 |
| 19 | `doc_loader.py:84` | **`break` 应为 `continue`**：目录遍历时遇到一个无法推断用途的文件就 `break`，会**直接放弃该目录下所有剩余文件**。对 `common/data` 这种混合目录是真实的数据丢失风险。 |
| 20 | `milvus_client.py:215,229` | **按字节截断中文**：`text_content.encode('utf-8')[:4000].decode('utf-8')` 若切在汉字中间会抛 `UnicodeDecodeError`；而 `VARCHAR(4000)` 本身按字符计长，这层截断既多余又错误。 |
| 21 | `embedding.py` / `reranker.py` | **模型无单例，重复加载**：两个工具类都在构造时立即加载数 GB 权重，`MilvusClientSystem` 与各测试脚本每次实例化都会重新加载。 |
| 22 | `llm_client.py:39-41` | **接口协议不一致**：自定义 LLM 用 `hasattr(llm, 'invoke')`（LangChain 风格）校验，实际却调用 `llm.chat.completions.create`（OpenAI 风格），传 LangChain 对象必然运行时报错。 |
| 23 | `llm_client.py` | **命名误导 + 参数缺失**：变量名 `DASHSCOPE_*` 实际指向 DeepSeek 后端；调用时**无 system prompt、未设 `temperature` / `max_tokens`**，输出稳定性不可控。 |
| 24 | `milvus_client.py:38` | **初始化有写库副作用**：`__init__` 里的 `_init_data()` 在集合为空时会自动从 MySQL 灌数据进 Milvus，让"构造对象"变成重操作。 |
| 25 | `law_text_spliter/main.py:18,30` | 测试入口路径 `r"/common/utils/test_data\..."` 写法错误（号称绝对路径却缺盘符）。 |
| 26 | `doc_loader.py:97` / `doc_spliter.py:40` | 测试代码硬编码开发机绝对路径 `D:\develop\...`，影响可移植性。 |

### 🔵 功能待完善

详见 [`优化项.md`](优化项.md)，摘要：

- **BM25 归一化问题**：去掉 softmax，改原始分 + 绝对阈值（建议 5.0）+ gap 判断
- **索引升级**：FLAT → HNSW（数据量 > 10 万后切换）
- **RRF 融合**：当前用 `WeightedRanker` 加权融合，可评估 RRF 是否更优
- **案例整案存储**：从子块存储改为整案聚合，向量基于"标题 + 关键facts"生成
- **增量更新**：支持增量 upsert，新文件入库不重复、不全量重建
- **多轮对话**：`generate()` 已预留 `history` 参数，但尚未接入
- **GPU 环境完善**：补全 cuDNN 8，OCR + embedding + reranker 全上 GPU
- **缺少自动化测试与 CI**

---

## 🗺️ 后续规划

| 版本 | 目标 | 状态 |
|------|------|------|
| v1 | 跑通完整链路：`search → prompt → LLM → 返回答案`，Web 端可用 | ✅ 已完成 |
| **v2** | 四模式检索 + 混合检索 + 重排序 + Prompt 优化 + 流式输出 | ✅ 已完成 |
| v3 | 修掉上述已知缺陷 + 多轮对话 + 增量更新 + 自动化测试与 CI | 规划中 |

---

## 📄 说明

- 本项目为学习/研究性质，检索结果**不构成法律意见**。
- `common/data/` 中的法律条文与案例来自公开渠道，版权归原作者/发布机构所有；
  若用于公开仓库请自行评估版权风险。
- 模型 `BGE-M3` 与 `BGE-reranker-large` 版权归 [BAAI](https://github.com/FlagOpen/FlagEmbedding) 所有。
- 页面头像素材来自网络，仅供学习交流，如涉版权请联系删除。
