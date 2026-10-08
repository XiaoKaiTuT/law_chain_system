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

## v3 更新要点

> 本次迭代主线是「**先判断问题类型，再决定走哪条链路**」与「**把缓存边界划清楚**」。
> 在 v2 已有的检索链路上，新增了一个**自训的 BERT 分类网关**，
> 并把 BM25 阈值从"失效"修正为"可用于生产"。

| # | 方向 | 具体内容 |
|---|------|---------|
| 1 | **BERT 分类网关（核心新增）** | 自训中文二分类模型判断「**法律问题**」/「**日常问题**」。日常问题**直接走通用问答，不触发任何检索** —— 从根本上解决了 v2 里「用户说『你好』被回『信息不足』」的体验问题 |
| 2 | **四级降级链路** | 缓存 → **BERT 网关分流** → BM25 关键词 → Milvus 混合检索 + 精排 + LLM 生成；非法律问题在第二级就分流出去 |
| 3 | **BM25 阈值修正** | 定位到 softmax 的**度量错误**（衡量的是"相对突出度"而非"绝对相关度"，且随语料规模漂移而失效）；改用原始分 `top1 ≥ 22 且 gap ≥ 5`。**误命中率 20% → 0%，召回保持 100%** |
| 4 | **缓存边界整改** | 兜底文案（"信息不足"）与报错信息**不再写入** Redis/MySQL；普通问题答案**只进 Redis 不进 MySQL**（`law_qa` 同时是 BM25 语料，混入非法律内容会摊薄 IDF） |
| 5 | **免责声明归属修正** | 免责声明**只加在法律回答上**，且**在缓存写入之后才推送** —— 既不污染检索语料，也不会出现「首次有、命中缓存没有」或重复叠加 |
| 6 | **训练流水线** | 新增 `rag_qa/classify/`：LLM 批量生成语料 → 严格 JSON 校验 → **四维分层切分**（label×长度×问候×人称）→ 微调 `chinese-roberta-wwm-ext` |
| 7 | **懒加载与后台预热** | 实测三模型均 ~2.3 GB / 8 GB 显存；bge-m3 与 reranker 是懒加载，启动仅 2.5 秒（Milvus 占 2.01s）。新增 `lifespan` 后台预热，**不阻塞启动**地把模型加载移到用户请求之前 |

---

## 📖 项目简介

针对中文法律领域问答场景，本项目实现了一条**完整可运行的 RAG 链路**：
把分散在法律条文（`.docx` / `.pdf`）与法院案例（`.pdf`）中的非结构化文本，经过 OCR 抽取、
结构化切块、向量化后存入 MySQL 与 Milvus；用户提问时按"**缓存 → 关键词 → 意图分类与改写
→ 混合检索与重排序 → 大模型流式生成**"召回答案，并通过 WebSocket 逐字推送到聊天页面。

**核心特点**

| 特点 | 说明 |
|------|------|
| 端到端可跑通 | 从原始 PDF/DOCX 到网页问答，全链路代码均在仓库内 |
|  **BERT 分类网关** | 自训中文二分类模型先判断「法律 / 日常」；**日常问题直接通用问答，省掉整条检索链** |
|  四级降级 | 缓存 → 分类网关 → 关键词命中 → 混合检索 + 重排序 + LLM 生成，逐级兜底 |
|  四模式检索 | 按问题类型自动选择 直接检索 / HyDE / 子查询 / 回溯检索，长短问题区别对待 |
|  混合检索 + 两段排序 | 稠密+稀疏融合粗排 → 内容去重 → 交叉编码器精排，兼顾召回率与精度 |
| ️ 精度优先的阈值策略 | 法律场景下"答错"的代价远高于"降级"，故所有阈值都朝**宁可漏、不可错**的方向设计 |
|  领域化切块 | 不用通用 `RecursiveCharacterTextSplitter` 硬切，而是**按法条结构 / 案例 section 语义切块** |
|  OCR 全格式支持 | PDF / DOCX / PPTX / 图片 统一走 RapidOCR 抽取文本 |
|  父子块设计 | 案例子块建向量检索，召回后回填**父块（完整案情）**给 LLM，兼顾精度与上下文 |
|  懒加载 + 后台预热 | 重模型按需加载，启动不被阻塞；预热把加载提前到用户请求之前 |
|  流式 + 阶段提示 | 逐 token 推送；等待期显示「正在检索法规与案例…」→「正在生成回答…」，首字前不留空白气泡 |
|  长连接前端 | WebSocket + 心跳保活 + 断线重连 + 请求队列，二次元风格 UI |
|  零外部模型下载 | `bge-m3` 与 `bge-reranker-large` 权重本地携带，离线可用 |

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
│  ② BERT 分类网关（rag_qa/utils/legal_classifier.py）                    │
│       │                                                                │
│       ├── 日常问题 ──► 通用问答（general_prompt）──► yield 逐块返回      │
│       │                 └─ 只回填 Redis（不进 MySQL，避免污染 BM25 语料）│
│       │                                                                │
│       └── 法律问题 ↓                                                    │
│  ③ BM25 关键词（原始分 top1≥22 且 gap≥5）──命中──► 返回 + 免责声明       │
│       │未命中                                                           │
│  ④ LLM 意图分类 ──► 问题改写（按类型选策略）                             │
│       │                                                                │
│  ⑤ Milvus 混合检索（粗排）─► 去重 ─► BGE-reranker-large（精排）          │
│       │                                                                │
│  ⑥ LLM 流式生成 ──► yield 逐块返回 ──► 末尾追加免责声明                  │
│       │                                                                │
│  ⑦ 生成完整成功 ──► 核心答案回填 Redis + MySQL（**免责声明不入缓存**）    │
│       │             兜底文案 / 报错信息【不缓存】                        │
└───────────────────────────────────────────────────────────────────────┘
         │                    │                      │
    ┌────▼────┐         ┌─────▼─────┐          ┌─────▼──────┐
    │  Redis  │         │   MySQL   │          │   Milvus   │
    │ 问答缓存 │         │ 结构化块  │          │ 向量 + 稀疏 │
    └─────────┘         └───────────┘          └────────────┘
```

**关键差异（对比 v2）**：

- **v2**：任何问题都走完 ②~⑥ 全链路，非法律问题（如"你好"）会因检索不到内容而返回"信息不足"
- **v3**：② 处由自训 BERT 判断问题类型，**日常问题在第 ② 级就分流**，不触发任何检索与改写

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
├── main.py                       # ★ 统一入口：LawChainClient（四级降级链 + 流式 + warmup）
├── config.ini                    # ★ 全局配置（数据库/模型/检索参数/BM25 阈值）
├── requirements.txt              # 依赖清单（UTF-8）
├── 优化项.md                      # 迭代待办清单（8 个章节，按版本归档）
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
│   ├── retrieval/bm25_search.py  #   ★ BM25 关键词检索（v3：原始分 + gap 双阈值）
│   └── utils/preprocess.py       #   jieba 分词
│
├── rag_qa/                       # 向量 + 分类 + 重排 + 生成层
│   ├── main.py                   #   RAGQAClient（CLI 版）
│   ├── db/milvus_client.py       #   建库/建集合/混合检索/去重/重排
│   ├── llm/llm_client.py         #   LLM 调用：意图分类 + 问题改写 + 法律/通用双路生成
│   ├── prompt/template.py        #   ★ 6 个提示词模板（v3 新增 general_prompt）
│   ├── utils/embedding.py        #   BGE-M3 向量化（稠密 + 稀疏）
│   ├── utils/reranker.py         #   BGE-reranker-large 重排序
│   ├── utils/legal_classifier.py #   ★ 法律/日常 BERT 分类网关（v3 新增，含单例）
│   ├── classify/                 #   ★ 分类器训练流水线（v3 新增，离线）
│   │   ├── generate_data.py      #     LLM 批量生成语料 + 严格 JSON 校验
│   │   ├── prepare_data.py       #     归一化 → 去重 → 四维分层切分 8:1:1
│   │   ├── train.py              #     微调 roberta-wwm-ext + 混淆矩阵报告
│   │   ├── data/                 #     raw/train/dev/test.jsonl
│   │   └── models/               #     训练产物（已在 .gitignore 中排除）
│   └── models/                   #   本地推理模型（需自行下载，勿推 GitHub）
│       ├── bge-m3/               #     ~4.3 GB
│       ├── bge-reranker-large/   #     ~6.4 GB
│       └── legal_classifier/     #     ~0.4 GB（v3 训练产物）
│
├── scripts/                      # ★ 诊断与标定脚本（v3 新增，只读不改数据）
│   ├── calib_bm25.py             #     BM25 阈值标定（正负样本分布 + 阈值扫描）
│   ├── cmp_bm25_metrics.py       #     softmax vs 原始分逐条对比
│   ├── diag_retrieval.py         #     检索失败静默问题复现
│   ├── probe_quota.py            #     子查询配额分配探针
│   ├── probe_quota2.py           #     跨子查询重复与重排轮次探针
│   ├── probe_gpu.py              #     GPU/cuDNN 可用性与显存预算探测
│   ├── verify_bm25.py            #     BM25 配置读取与判定效果验证
│   ├── verify_v3.py              #     检索链路多轮回归验证
│   └── verify_v4.py              #     空查询边界与配额回归验证
│
├── static/                       # Web 层
│   ├── app.py                    #   FastAPI 应用（流式协议 + 心跳 + 后台预热）
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

    # ② BERT 分类网关（v3 新增）：日常问题直接走通用问答，不触发检索
    if not self._is_legal_question(question):
        return self._stream_general(question)

    # ③ BM25 关键词检索：命中返回 str + 免责声明，并回填 Redis
    answer = self.bm25_search.search(question)
    if answer:
        answer += self.llm_client.rag_prompts.disclaimer(self.config.APP_PHONE)
        self.redis_client.set_question(question, answer)
        return answer

    # ④ 意图分类 + 问题改写
    optimizer_query = self.llm_client.query_generate(question)

    # ⑤ 混合检索 + 去重 + 重排序
    case_chunk, clause_chunk = self.milvus_client.search(optimizer_query)

    # ⑥ 返回生成器：边生成边 yield，边累积
    return self._stream_and_cache(question, case_chunk, clause_chunk)


def _is_legal_question(self, question) -> bool:
    """BERT 二分类：判断是否为法律专业问题"""
    return self.classifier.classify(question) == "法律"


def _stream_general(self, question):
    """通用问答（日常问题）：不检索知识库，只回填 Redis"""
    llm_gen = self.llm_client.generate_general(question)
    chunks = []
    try:
        for chunk in llm_gen:
            chunks.append(chunk)
            yield chunk
    except GenerationError:
        yield self.llm_client.rag_prompts.system_error_answer(self.config.APP_PHONE)
    else:
        if chunks:
            # ★ 只进 Redis，不进 MySQL（law_qa 是 BM25 语料，不能被日常内容污染）
            self.redis_client.set_question(question, "".join(chunks))


def _stream_and_cache(self, question, case_chunk, clause_chunk):
    """流式产出答案；仅在【完整生成成功】时回填缓存"""
    llm_gen = self.llm_client.generate(question, case_chunk, clause_chunk)
    chunks = []
    success = False
    try:
        for chunk in llm_gen:
            chunks.append(chunk)
            yield chunk                      # ← 逐块交给上层推给前端
        success = True
    except InsufficientContextError:         # ★ 兜底文案：给提示但不缓存
        yield self.llm_client.rag_prompts.insufficient_answer(self.config.APP_PHONE)
    except GenerationError:                  # ★ 报错：给提示但不缓存
        yield self.llm_client.rag_prompts.system_error_answer(self.config.APP_PHONE)
    else:                                    # ★ 只有完整成功才走这里
        if chunks:
            full_answer = "".join(chunks)    # 注意：此时 chunks 里【不含免责声明】
            self.redis_client.set_question(question, full_answer)
            self.mysql_client.insert_data([{"question": question, "answer": full_answer}])

    # ★ 免责声明在 try 之外、缓存写入之后才推送，保证不污染检索语料
    if success:
        yield self.llm_client.rag_prompts.disclaimer(self.config.APP_PHONE)
```

**降级策略的意义**：高频问题在第 ①/③ 级就被拦下；
**非法律问题在第 ② 级就被分流**，完全不触发检索；
只有真正"没见过"的**法律问题**才走完 ④⑤⑥。

### 1.1 为什么用 `try/except/else` 而不是 `try/finally`

这是 v3 修掉的一个真实 bug：

```
v2 用 try/finally  → finally 无论成功失败都执行
                  → "信息不足"这类兜底文案也被累积并写入缓存
                  → 用户下次问同样问题会直接拿到"信息不足"

v3 用 try/except/else → else 只在 try 块【无异常】时执行
                     → 只有完整成功才缓存
```

**更通用的原则**：**缓存的写入条件应该是"操作完整成功"，不是"操作结束了"**。
`finally` 表达的是后者，`else` 表达的是前者。

### 1.2 免责声明为什么不进缓存

`law_qa` 表**同时是 BM25 的检索语料**。免责声明（约 35 字）如果写进去：

| 影响 | 说明 |
|---|---|
| 文档变长 | 每篇答案多 35 字，改变 BM25 的文档长度归一化因子 |
| 引入共有高频词 | "客服""电话""仅供"在**所有**答案中都出现，稀释真正有区分度的法律术语 |
| 污染 IDF | 语料里凭空多出全文档共有词，其他词的 IDF 被相对拉平，**阈值 22 可能漂移** |

所以免责声明的定位是**展示层文案**：跟着回答给用户看，但**不落地到任何存储**。

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

### 8. BERT 分类网关（v3 核心）

**解决的问题**：v2 里所有问题都走完整检索链，导致用户说「你好」时，
BM25 与 Milvus 都检索不到相关内容，最终返回「信息不足，无法回答」—— 体验很差。

**v3 的做法**：用一个**自训的中文二分类模型**在链路第二级做分流。

```
用户问题
   │
   ▼
┌─────────────────────────────┐
│ BERT 二分类（法律 / 日常）    │   chinese-roberta-wwm-ext 微调
│ rag_qa/utils/legal_classifier.py │   单例，fp16，max_length=128
└──────────┬──────────────────┘
           │
   ┌───────┴───────┐
   ▼               ▼
 日常问题        法律问题
   │               │
   ▼               ▼
通用问答        完整 RAG 链路
（不检索）      （BM25 → 混合检索 → 精排 → 生成）
```

#### 训练数据与指标

| 项目 | 内容 |
|---|---|
| 语料规模 | 1208 条（法律 / 日常各约 500+，来源：LLM 批量生成 + `law_qa` 库内问题取用） |
| 切分方式 | **四维分层抽样**：`label × 长度档 × 是否含问候词 × 是否第一人称`，比例 8:1:1 |
| 基座模型 | `hfl/chinese-roberta-wwm-ext`（102M） |
| 超参 | `lr=2e-5`、`4 epochs`、`batch=32`、`weight_decay=0.01`、`warmup_ratio=0.1` |
| 选模型 | 按 dev **F1** 保存最佳 epoch（配 `EarlyStoppingCallback`） |
| **测试集结果** | **accuracy 98.35%**，「法律→日常」漏判 **0 条** |

#### ⚠️ 为什么必须"四维分层"而不是随机切分

数据里曾存在 **4 处「特征与标签虚假相关」（捷径）**：

| 特征 | `label=0`（日常） | `label=1`（法律） | 风险 |
|---|---|---|---|
| 含问候词 | 42 / 500（8.4%） | **0 / 500（0%）** | 模型会学成"看到『你好』→ 判日常" |
| 长度 > 40 字 | **0 条** | 103 条（20.6%） | 模型会学成"长句 → 法律" |
| 长度 ≤ 6 字 | 36 条 | **0 条** | 模型没见过"极短的法律问题" |
| 前 10 字含「我」 | 5 条（1.0%） | 85 条（17.0%） | 模型会学成"有『我』→ 法律" |

**这些捷径在训练集和测试集上都会表现良好**（两边都有同样规律），
**从分数上完全看不出来** —— 只有上线遇到真实用户才暴露。

**缓解措施**：补充针对性语料（带问候的法律问题、长句日常、极短法律问题、第一人称日常），
使两个类在这些特征上的比例接近。

#### 对抗测试（验证模型学的是语义而非捷径）

用 **25 条训练集之外**的针对性样本测试：

| 类别 | 样本数 | 通过 | 法律概率区间 |
|---|---|---|---|
| 问候 + 法律问题 | 8 | **8/8** | 0.948 ~ 0.985 |
| 极短法律问题（≤6 字） | 8 | **8/8** | 0.877 ~ 0.975 |
| 长句日常问题（>40 字） | 3 | **3/3** | 0.029 ~ 0.035 |
| 第一人称日常问题 | 4 | **4/4** | 0.027 ~ 0.036 |
| 控制组（明显样本） | 4 | **4/4** | — |

**两类概率分得很开，没有一条卡在 0.5 附近** —— 说明模型抓的是语义信号，不是表面特征。

#### 仍存在的局限

- **真实长度偏移**：训练/测试样本最长约 60 字，真实用户可能输入 150 字以上流水账，
  超过 `MAX_LENGTH=128` 会被静默截断
- **多轮追问**：`_is_legal_question()` 只看当前句，第二轮问「那他不还呢」可能判成日常
- **置信度阈值未落地**：目前取 `argmax` 硬标签，未利用概率。
  理论上应对模棱两可的输入采用「**低置信度默认走法律链路**」（与"宁可漏不可错"同源）

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
| **chinese-roberta-wwm-ext** | 本地 | **法律/日常二分类网关（v3 新增，自行微调）** |
| FlagEmbedding | 1.3.5 | `FlagReranker` 重排器 + BGE 工具链 |
| milvus-model | 0.2.5 | `BGEM3EmbeddingFunction` 封装 |
| rank-bm25 | 0.2.2 | BM25 关键词检索 |
| jieba | 0.42.1 | 中文分词 |
| transformers | 4.45.0 | 模型加载（推理 + 微调 `Trainer`） |
| datasets | 3.3.1 | 分类器训练数据容器（v3 新增） |
| scikit-learn | 1.7.2 | 分层切分 + 分类指标（v3 新增） |
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
| 项 | BGE-M3 | BGE-reranker-large | chinese-roberta-wwm-ext |
|----|--------|--------------------|-------------------------|
| 用途 | 稠密+稀疏向量化 | 交叉编码器精排 | 法律/日常二分类网关 |
| 路径 | `rag_qa/models/bge-m3/` | `rag_qa/models/bge-reranker-large/` | `rag_qa/models/legal_classifier/` |
| 体积 | ~4.3 GB | ~6.4 GB | ~0.4 GB |
| 输出 | 稠密 1024 维 + 稀疏词权重 | 单个相关性分数 | 2 类概率 |
| 加载方式 | **懒加载**（首次调用才加载） | **懒加载** | 饿汉式（构造即加载） |
| 单例 | ❌ 见「已知问题」 | ❌ 见「已知问题」 | ✅ `get_legal_classify()` |
| 推理设备 | 自动检测：有 CUDA 用 GPU（fp16） | 同左 | 同左 |
| batch_size | 32 | — | 单条 |

**实测耗时与显存**（RTX 4060 Laptop 8 GB）：

| 阶段 | 耗时 | 显存占用 |
|---|---|---|
| `VectorTools()` 构造 | 3.84 s | 0 GB |
| 首次 `encode_query()` | +2.14 s | 1.07 GB |
| `RerankerTools()` 构造 | 1.11 s | 0 GB |
| 首次 `rerank()` | +1.48 s | +1.05 GB |
| `ClassifyTools()` 构造 | 0.49 s | ~0.2 GB |
| **三模型合计** | — | **~2.3 GB / 8 GB** |

> ⚠️ **模型权重合计约 11.1 GB**，请勿直接推送 GitHub，见「已知问题」。
>
> 💡 **懒加载是库自身实现的**（`BGEM3EmbeddingFunction` / `FlagReranker`），
> 所以服务启动只需 2.5 秒（其中 Milvus 初始化 2.01 s）。
> 代价是"第一个用户"会等待模型加载 —— v3 用 `lifespan` 里的**后台预热**解决。

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

模型权重都不随仓库提供，需自行下载后放到对应目录：

```bash
# 方式一：ModelScope（国内更快）
modelscope download --model BAAI/bge-m3              --local_dir rag_qa/models/bge-m3
modelscope download --model BAAI/bge-reranker-large  --local_dir rag_qa/models/bge-reranker-large

# 方式二：HuggingFace
huggingface-cli download BAAI/bge-m3              --local-dir rag_qa/models/bge-m3
huggingface-cli download BAAI/bge-reranker-large  --local-dir rag_qa/models/bge-reranker-large
```

#### 3.1 分类器模型（v3）

分类器有**两种获取方式**：

**方式 A：直接使用仓库提供的训练产物**（推荐，开箱即用）

把 `rag_qa/classify/models/` 的内容复制到 `rag_qa/models/legal_classifier/`：

```bash
# Windows
xcopy /E /I rag_qa\classify\models rag_qa\models\legal_classifier

# Linux / macOS
cp -r rag_qa/classify/models rag_qa/models/legal_classifier
```

> ⚠️ 需要包含 `config.json` / `model.safetensors` / `tokenizer.json` / `vocab.txt` 等文件。
> 训练中间产物 `checkpoint-*/` 目录**不需要**（含 optimizer 状态，约 1.2 GB/个，可删）。

**方式 B：自己重新训练**

```bash
# 1) 生成语料（调用 LLM，需配置好 API Key）
python rag_qa/classify/generate_data.py

# 2) 清洗 + 分层切分为 train / dev / test
python rag_qa/classify/prepare_data.py

# 3) 微调（训练前请先停掉 uvicorn，避免显存冲突）
python rag_qa/classify/train.py
```

训练完成后 `rag_qa/classify/models/` 即为产物，按方式 A 复制到 `rag_qa/models/legal_classifier/`。

> 💡 **训练前必须停掉正在运行的服务**：bge-m3 + reranker 已占用约 2.1 GB 显存，
> 训练还需要 2~3 GB，同时在跑会 OOM。
>
> 💡 如果 `hfl/chinese-roberta-wwm-ext` 下载慢，可先用 ModelScope 拉到本地，
> 再把 `train.py` 里的 `MODEL_NAME` 改为本地路径。

### 3. 安装依赖

```bash
conda create -n law_rag python=3.10 -y
conda activate law_rag

pip install -r requirements.txt
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

# v3：BM25 原始分阈值（替代原 softmax + threshold=0.85）
[bm25]
min_top = 22               # 原始 BM25 最高分下限
min_gap = 5                # 最高分与次高分的最小差距（防"矮子里拔高个"）
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

启动时会通过 `lifespan` 初始化 `LawChainClient`（连接 Redis / MySQL / Milvus + 加载分类器），
**实测约 2.5 秒**即出现 `LawChainClient 初始化完成`：

```
Redis        0.00 s
MySQL        0.02 s
Milvus       2.01 s   ← 主要开销：连接 + 集合加载（load_collection）
LLM          0.01 s
分类器        0.49 s
BM25         0.02 s
```

**关于后台预热**：bge-m3 与 bge-reranker-large 是**懒加载**（构造时不加载，首次调用才加载），
所以启动很快。为避免"第一个用户"等待模型加载，`lifespan` 会启动一个**后台预热任务**：

```
已启动后台模型预热（不阻塞服务启动）      ← 启动立即完成，服务可访问
预热 bge-m3 完成，耗时 2.14 s            ← 后台静默进行
预热 reranker 完成, 耗时 1.48 s
模型预热全部完成
```

> 💡 预热失败只记 `warning`，不影响服务使用 —— 它是可选优化，不是必需项。
>
> 💡 如果想观察预热是否生效，可在启动后立刻用 `nvidia-smi` 看显存是否涨到约 2.3 GB。

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
| Milvus 重启后 `collection not loaded` | ⚠️ v2 曾加 `_ensure_loaded()` 前置检查；**v3 已撤销该方案**（见下） |

**v2 提交后修补**

| 问题 | 状态 |
|------|------|
| 重排模型 fp16 未生效（`FlagReranker(usp_fp16=...)` 参数名拼写错误） | ✅ 已修正为 `use_fp16`，GPU 显存占用减半 |
| `llm_client.py` 冗余导入 `from urllib import response` | ✅ 已移除 |

### ✅ v3 已修复的问题

| 原问题 | v3 状态 |
|--------|--------|
| **子查询配额被瓜分**：`top_m // num` 使 3 个子问题时案例配额降到 1 | ✅ 改为「每个子查询各取足额候选 → 汇总 → 跨子查询去重 → 用原问题统一精排一次 → 按配置截断」。实测子问题数 1/2/3/4/5 时上下文**恒定 3/4**，重复条数 2 → **0** |
| **检索失败静默**：`except MilvusException: return [], []` | ✅ 改为 `raise`；`main.py` 中对应的 `except → return "系统繁忙"` 一并移除，异常冒泡到 `app.py` 转成 `error` 帧 |
| **重排用子问题打分**：各轮分数不可比，排序标准偏离用户意图 | ✅ `query_generate()` 返回 `{"retrieval_queries", "rerank_query"}`，重排统一用**用户原问题** |
| **`_ensure_loaded()` 失效**：每次检索前 2 次远程 RPC，有竞态窗口且无重试 | ✅ **治本**：移除 `close()` 里的 `release_collection`，让集合常驻内存；`_ensure_loaded()` 整个删除。启动省掉重新加载开销 |
| **BM25 阈值几乎不可能命中**：softmax 归一化 + `threshold=0.85` | ✅ 改用原始分 `top1 ≥ 22 且 gap ≥ 5`（基于 30 正/20 负样本标定，两组完全分离）。**误命中 20% → 0%，召回保持 100%** |
| **兜底文案/报错被写入缓存**：`generate()` 无上下文时 `yield` 提示会被累积入库 | ✅ 改为抛 `InsufficientContextError` / `GenerationError`；`_stream_and_cache()` 用 `try/except/else` 替代 `try/finally`，**只有完整成功才缓存** |
| **免责声明归属错误**：会重复叠加、且未加在日常问答上 | ✅ **只加在法律回答上**，且在**缓存写入之后**才推送 —— 既不污染检索语料也不叠加 |
| **`_init_llm()` 异常被吞**：失败时 `self.llm` 保持 `None`，故障延迟到首次提问 | ✅ 改为 `raise`，故障在启动阶段暴露 |
| **`llm_client.py:12` 未使用导入** | ✅ 已移除 `MilvusClientSystem` 导入，消除循环依赖隐患 |
| **`Map`/JSON 结构错误**：生成语料时 LLM 输出无校验 | ✅ 新增 `generate_data.py` 的解析 + 校验双层检查（`OSError` 异常类、Pydantic 风格字段校验、带原始输出片段的错误信息） |
| **无启动预热**：懒加载导致"第一个用户"等待 3.6 秒 | ✅ `lifespan` 后台预热，进入 lifespan 仅 0.002s、关闭时正确取消（3 用例验证通过） |
| **`requirements.txt` 编码错误** | ✅ UTF-16LE → UTF-8，并修正文件名拼写 |

### 🔴 上传 GitHub 前必须处理

| # | 问题 | 说明 | 建议 |
|---|------|------|------|
| 1 | **模型权重合计约 10.9 GB** | `bge-m3` **4.27 GB**（`pytorch_model.bin` 2.17 GB + `onnx/model.onnx_data` 2.16 GB）；`bge-reranker-large` **6.28 GB**（`pytorch_model.bin` 2.14 GB + `model.safetensors` 2.14 GB 两份等价权重并存）；`legal_classifier` **0.38 GB**（v3 新增） | 已在 `.gitignore` 排除 `rag_qa/models/`；README 提供下载与获取方式。若确需入库请用 **Git LFS**（单文件 >100 MB GitHub 直接拒绝）。可先删掉 `onnx/` 与重复的 `.safetensors` 省一半体积 |
| 2 | **明文密码入库** | `config.ini` 含 MySQL `123456`、Redis `1234`；`base/config.py` 把它们写成了代码 fallback 默认值 | 移除代码里的明文默认值，改用 `.env` + `python-dotenv`，并提供 `.env.example` |
| 3 | **依赖清单是整环境快照** | 212 个包，包含大量与本项目无关的传递依赖（langchain 全家桶、ragas、kubernetes 等） | 按直接依赖重写，或用 `pipreqs` 生成 |
| 4 | **`common/data/` 有 69.7 MB 语料** | 24 个案例 PDF + 19 个法条 + 问答 Excel | 若版权/体积敏感，改放外部下载链接 |
| 5 | **`rag_qa/classify/models/checkpoint-*/`** | 训练中间产物含 optimizer 状态，每个约 1.2 GB（v3 训练产生） | 已在 `.gitignore` 排除；确认本地可删以释放空间 |

### 🟡 代码层面的已知缺陷

**v3 新增代码相关**

| # | 位置 | 问题 |
|---|------|------|
| 6 | `legal_classifier.py` `classify()` | **未利用置信度**：直接取 `argmax` 硬标签。对模棱两可的输入（如「押金不退」这类既可当法律咨询也可当吐槽的短句）可能只给出微弱优势。建议增加 `classify_with_score()` 返回 `(标签, 概率)`，并对低置信度样本记日志用于迭代 |
| 7 | `main.py` `_is_legal_question()` | **只用当前句判断**：若用户第二轮问「那他不还呢」「还有别的办法吗」，单看这句会像日常问题。需要先定产品决策：只看当前句（需补多轮追问样本）还是带上历史（需改输入格式） |
| 8 | 分类器长度偏移 | 训练/测试样本最长约 60 字，真实用户可能输入 150 字以上流水账；`MAX_LENGTH=128` 会**静默截断**超出部分，若关键信息在后半段可能导致误判。建议记录被截断的输入条数 |
| 9 | 意图分类的额外开销 | v3 已用 BERT 替代「是否法律问题」这一次判断，但**法律问题仍要多一次 LLM 调用**做四分类（实测 2.3s），`hyde`/`subquery`/`recall` 还要**再一次**做改写。可考虑四分类也换小模型、或与改写合并为一次调用 |
| 10 | `main.py` `_stream_general()` | **普通问题无"信息不足"兜底**：该方法只捕获 `GenerationError`，若 `generate_general()` 抛出其他异常会穿透到 `app.py`。因通用问答不检索，正常情况下不会缺上下文，故影响有限 |

**v1 / v2 遗留未修**

| # | 位置 | 问题 |
|---|------|------|
| 11 | `mysql_qa/main.py:75-81` | **契约不匹配（功能性 bug）**：`BM25Search.search()` 返回 `str \| None`，但 `MySQLQASystem.search()` 按 `list[dict]` 处理（`for answer in answers: if "question" in answer`）。字符串会被逐**字符**迭代，`in` 判断恒为 False，导致"MySQL 命中 → 回填 Redis 缓存"**从未生效**；若返回 `None` 则抛 `TypeError` 被静默吞掉。**注：实际使用的 `LawChainClient`（根 `main.py`）调用方式正确，不受影响。** |
| 12 | `milvus_client.py` `_recoder()` | **靠数据反推类型**：用 `de_weight_chunk[0]` 判断内容字段名（`text_content` / `parent_content`）。已在开头加 `if not chunk: return []` 挡住空列表，但更稳妥的做法是把字段名作为参数显式传入 |
| 13 | 去重维度 | `_recoder()` 按 `parent_content` / `text_content` **全文**去重，挡不住「同一案例的不同 section」同时命中（内容不同但案例相同）。应改为按 `case_no` / `law_name` + `article_no` 去重 |
| 14 | `milvus_client.py:38` | **初始化失败即全站不可用**：`MilvusClientSystem` 构造失败会抛出，`LawChainClient` 随之失败，FastAPI 无法启动（页面也打不开）。建议把 Milvus 做成可降级组件（分类器已做到"加载失败即启动报错"，但 Milvus 是更基础的硬依赖） |
| 15 | `bm25_search.py` / `redis_client.py` | **连接泄漏**：`BM25Search` 自建 `MySQLClient` 却**没有 `close()`**；`LawChainClient.close()` 也未关闭它。`redis.Redis()` 是惰性连接，`__init__` 里的 `except redis.RedisError` 基本捕获不到连接失败 |
| 16 | `embedding.py` / `reranker.py` | **模型无单例**：`VectorTools` / `RerankerTools` 仍是普通类，而新增的 `ClassifyTools` 已有模块级单例 `get_legal_classify()`。三处风格不一致，且误创建会导致显存翻倍（三模型约占 2.3 GB / 8 GB）。**注意：改单例不会加快启动**（新进程照样重新初始化），价值在防 OOM |
| 17 | `mysql_client.py` | **硬编码库名**：`insert_data` 里写死 `law_chain.law_qa` / `law_chain.law_chunk`；若 `MYSQL_DATABASE` 配置不同则直接失败 |
| 18 | 唯一键设计 | `uk_question(question(255))` / `uk_chunk(source, text_content(255))` 配合 `insert ignore`，会**静默丢弃**前 255 字相同的不同问题/长法条，且无告警 |
| 19 | `doc_loader.py:84` | **`break` 应为 `continue`**：目录遍历时遇到一个无法推断用途的文件就 `break`，会**直接放弃该目录下所有剩余文件**。对 `common/data` 这种混合目录是真实的数据丢失风险 |
| 20 | `milvus_client.py:215,229` | **按字节截断中文**：`text_content.encode('utf-8')[:4000].decode('utf-8')` 若切在汉字中间会抛 `UnicodeDecodeError`；而 `VARCHAR(4000)` 本身按字符计长，这层截断既多余又错误 |
| 21 | `llm_client.py:39` | **接口协议不一致**：自定义 LLM 用 `hasattr(llm, 'invoke')`（LangChain 风格）校验，实际却调用 `llm.chat.completions.create`（OpenAI 风格），传 LangChain 对象必然运行时报错 |
| 22 | `llm_client.py` | **命名误导 + 参数缺失**：变量名 `DASHSCOPE_*` 实际指向 DeepSeek 后端；调用时**无 system prompt、未设 `temperature` / `max_tokens`**，输出稳定性不可控 |
| 23 | `milvus_client.py:38` | **初始化有写库副作用**：`__init__` 里的 `_init_data()` 在集合为空时会自动从 MySQL 灌数据进 Milvus，让"构造对象"变成重操作（有 `row_count == 0` 检查，非空时跳过） |
| 24 | `law_text_spliter/main.py:18,30` | 测试入口路径 `r"/common/utils/test_data\..."` 写法错误（号称绝对路径却缺盘符） |
| 25 | `doc_loader.py:97` / `doc_spliter.py:40` | 测试代码硬编码开发机绝对路径 `D:\develop\...`，影响可移植性 |
| 26 | `main.py` `_is_legal_question()` docstring | 注释仍写着「临时桩」，但实现已换成 BERT 模型推理，需同步更新 |

### 🔵 功能待完善

详见 [`优化项.md`](优化项.md)（8 个章节，39 条待办），摘要：

- **置信度阈值策略**：低置信度时默认走法律链路（与"宁可漏不可错"同源）
- **去重维度**：内容去重 → 按 `case_no` / `article_no` 去重
- **向量化质量门控**：为粗排结果加最低相似度阈值
- **索引升级**：FLAT → HNSW（数据量 > 10 万后切换）
- **RRF 融合**：当前用 `WeightedRanker` 加权融合，可评估 RRF 是否更优
- **模型单例化**：`VectorTools` / `RerankerTools` 补单例（防 OOM，非加速）
- **BM25 阈值重标定**：语料翻倍后需重跑 `scripts/calib_bm25.py`
- **增量更新**：支持增量 upsert，新文件入库不重复、不全量重建
- **多轮对话**：`generate()` 已预留 `history` 参数，但尚未接入真实历史
- **缺少自动化测试与 CI**：所有模块的 `__main__` 都是手工验证脚本；
  可把 `scripts/` 下已有的回归脚本改造成可重复运行的集成测试

---

## 🗺️ 后续规划

| 版本 | 目标 | 状态 |
|------|------|------|
| v1 | 跑通完整链路：`search → prompt → LLM → 返回答案`，Web 端可用 | ✅ 已完成 |
| v2 | 四模式检索 + 混合检索 + 重排序 + Prompt 优化 + 流式输出 | ✅ 已完成 |
| **v3** | **BERT 分类网关（法律/日常分流）+ 通用问答链路 + BM25 阈值修正 + 缓存边界整改 + 懒加载与后台预热** | ✅ 已完成 |
| **v4** | **RAG 项目评估**：建立评测集与指标（检索命中率、答案忠实度、上下文相关性等），用数据驱动后续优化 | 🎯 下一步 |
| v5 | 四分类也换成小模型、多轮对话、增量更新、模型单例化、自动化测试与 CI | 规划中 |

### v4 为什么先做评估

v3 的所有改动（BM25 阈值、分类网关、缓存边界）都是**靠少量手工样本验证**的：

- BM25 阈值：30 正 + 20 负样本
- 分类器：25 条对抗样本 + 121 条测试集

**样本量小、覆盖有限，且无法回答"整体效果到底提升多少"**。
v4 要建立一套可重复运行的评测流程，把「凭样本感觉」变成「有指标可依」，
之后的每一次优化才能量化收益、避免退化。

---

## 📄 说明

- 本项目为学习/研究性质，检索结果**不构成法律意见**。
- `common/data/` 中的法律条文与案例来自公开渠道，版权归原作者/发布机构所有；
  若用于公开仓库请自行评估版权风险。
- 模型 `BGE-M3` 与 `BGE-reranker-large` 版权归 [BAAI](https://github.com/FlagOpen/FlagEmbedding) 所有。
- 页面头像素材来自网络，仅供学习交流，如涉版权请联系删除。
