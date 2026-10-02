# LawChain · 法律问答系统

> 基于 **RAG（检索增强生成）** 的中文法律智能问答系统。
> 多级检索链：Redis 缓存 → MySQL 关键词检索（BM25）→ Milvus 向量检索 → LLM 生成答案。
> 附带一个须弥草系二次元风格的 WebSocket 聊天前端。


---

## 📖 项目简介

针对中文法律领域问答场景，本项目实现了一条**完整可运行的 RAG 链路**：
把分散在法律条文（`.docx` / `.pdf`）与法院案例（`.pdf`）中的非结构化文本，经过 OCR 抽取、
结构化切块、向量化后存入 MySQL 与 Milvus；用户提问时按"**缓存 → 关键词 → 向量 → 大模型**"
四级降级策略召回答案，并通过 WebSocket 实时推送到聊天页面。

**核心特点**

| 特点 | 说明 |
|------|------|
| 端到端可跑通 | 从原始 PDF/DOCX 到网页问答，全链路代码均在仓库内 |
| 四级检索降级 | 缓存命中 → 关键词命中 → 向量召回 + LLM 生成，逐级兜底 |
| 领域化切块 | 不用通用 `RecursiveCharacterTextSplitter` 硬切，而是**按法条结构 / 案例 section 语义切块** |
| OCR 全格式支持 | PDF / DOCX / PPTX / 图片 统一走 RapidOCR 抽取文本 |
| 父子块设计 | 案例子块建向量检索，召回后回填**父块（完整案情）**给 LLM，兼顾精度与上下文 |
| 长连接前端 | WebSocket + 心跳保活 + 断线重连 + 请求队列，二次元风格 UI |
| 零外部模型下载 | `bge-m3` 权重随仓库携带，离线可用 |

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                          浏览器 (chat.html)                           │
│         WebSocket /ws/chat  · 心跳 ping/pong  ·  断线指数退避重连        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ JSON 协议
┌───────────────────────────────▼──────────────────────────────────────┐
│                      FastAPI (static/app.py)                         │
│    GET /  →  渲染 chat.html        GET /health  →  探活                │
│    WS /ws/chat  →  单读取者循环 + 并发心跳协程                            │
│    asyncio.to_thread(...)  →  同步检索链不阻塞事件循环                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                  LawChainClient (main.py) —— 统一入口                  │
│                                                                      │
│   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌─────────┐ │
│   │ ①Redis 缓存  │ → │ ②BM25 关键词  │ → │ ③Milvus 向量 │ → │ ④LLM 生成│ │
│   │  命中即返回   │   │  命中即返回    │   │  召回上下文   │   │  出答案  │ │
│   └─────────────┘   └──────────────┘   └─────────────┘   └────┬────┘ │
│         ▲                                                      │      │
│         └──────────── 答案回填缓存 (TTL 1h) ◄───────────────────┘      │
└───────────────────────────────────────────────────────────────────────┘
         │                    │                      │
    ┌────▼────┐         ┌─────▼─────┐          ┌─────▼──────┐
    │  Redis  │         │   MySQL   │          │   Milvus   │
    │ 问答缓存 │         │ 结构化块  │          │  向量索引   │
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
                                              MySQL → 父块拼接 → bge-m3 向量化
                                                       │
                                                       ▼
                                            Milvus: law_cases / law_clause
```

---

## 📁 目录结构

```
law_chain_system/
├── main.py                       # ★ 统一入口：LawChainClient（四级检索链）
├── config.ini                    # ★ 全局配置（数据库/模型/检索参数）
├── requirments.txt               # 依赖清单（注意：文件名拼写 & 编码问题，见「已知问题」）
├── 优化项.md                      # 迭代待办清单
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
├── rag_qa/                       # 向量 + 生成层
│   ├── main.py                   #   RAGQAClient（CLI 版）
│   ├── db/milvus_client.py       #   建库/建集合/建索引/向量检索
│   ├── llm/llm_client.py         #   LLM 调用（OpenAI 兼容 SDK）
│   ├── prompt/template.py        #   RAG 提示词模板
│   ├── utils/embedding.py        #   bge-m3 向量化
│   └── models/bge-m3/            #   本地模型(需要在魔搭或者HuggingFace官网下载)
│
├── static/                       # Web 层
│   ├── app.py                    #   FastAPI 应用（路由 + WebSocket + 心跳）
│   │                             #   app.mount("/static", StaticFiles(directory=static/))
│   ├── data/nahida.jpg           #   页面头像（子目录随挂载点自动可访问）
│   └── templates/chat.html       #   二次元风格聊天页
│
└── logs/app.log                  # 运行日志
```

---

## 🔄 核心流程详解

### 1. 提问 → 答案（在线链路）

`main.py` 中 `LawChainClient.search()` 是整条链路的核心：

```python
def search(self, question) -> str:
    # ① 空问题兜底
    if not question:
        return "消息为空，请重新输入有效问题 QAQ"

    # ② Redis 缓存：命中直接返回（省掉后面所有开销）
    answer = self.redis_client.get_answer(question)
    if answer:
        return answer

    # ③ MySQL BM25 关键词检索：命中直接返回
    answer = self.bm25_search.search(question)
    if answer:
        return answer

    # ④ Milvus 向量检索 → 拼装 prompt → LLM 生成
    context = self.milvus_client.search(question)
    answer = self.llm_client.generate(question, context)

    # ⑤ 生成成功则回填缓存（下次同样问题走 ②）
    if answer:
        self.redis_client.set_question(question, answer)
    return answer
```

**降级策略的意义**：绝大多数高频问题在第 ②/③ 级就被拦下，只有真正"没见过"的问题才会
触发向量检索 + 大模型生成，显著降低 LLM 调用成本与响应延迟。

### 2. WebSocket 通信协议

前后端约定 JSON 消息（为兼容旧客户端，**纯文本消息仍按提问处理**）：

| 方向 | 消息体 | 说明 |
|------|--------|------|
| C → S | `{"type":"question","text":"..."}` | 用户提问 |
| C → S | `{"type":"ping"}` / `{"type":"pong"}` | 心跳 |
| S → C | `{"type":"answer","text":"..."}` | 答案 |
| S → C | `{"type":"ping"}` / `{"type":"pong"}` | 心跳 |
| S → C | `{"type":"error","text":"..."}` | 检索异常兜底 |

### 3. 心跳保活机制

长连接最容易出问题的地方是"连接看起来还在，实际已经死了"。本项目做法：

- **服务端**：`_send_heartbeat()` 每 `HEARTBEAT_INTERVAL`(30s) 主动发 `ping`，
  并检查共享的 `ConnectionState.last_seen` 时间戳；连续 `HEARTBEAT_MAX_MISS`(2) 个窗口
  没收到**任何**帧，即判定链路已死并 `close(1001)`。
- **关键设计：单一读取者**。Starlette 的 WebSocket 只允许一个协程在读，
  因此**心跳协程从不调用 `receive_text()`**，只负责发送与计时；由 `_receive_loop()`
  统一读取并在收到任何帧时刷新时间戳。这样避免了心跳与业务逻辑争抢消息、
  甚至把用户提问"吃掉"的经典 bug。
- **前端**：收到 `ping` 立即回 `pong`；同时有 90s 看门狗（任何服务端帧都会重置），
  超时主动断开并**指数退避重连**（1s→2s→4s→8s，上限 15s）；离线期间的提问进队列，
  重连成功后自动补发。

### 4. 领域化切块（核心亮点）

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

### 5. 父子块检索设计

案例往往很长，整案做向量会让语义被稀释，子块做向量又会丢掉上下文。本项目采用**父子块**：

- **子块建向量**：每个 section（案件基本信息/基本案情/案件焦点/法院裁判要旨/法官后语）单独
  向量化，保证检索精度。
- **父块给 LLM**：`_init_data()` 中用 `GROUP_CONCAT` 把同一案例的
  **基本案情 + 案件焦点 + 法院裁判要旨** 拼成完整案情，存进 `parent_content` 字段
  （`法官后语` 与 `案件基本信息` 不参与拼接）。
- 命中子块后，把**父块全文**交给 LLM，让它看到完整脉络。

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
| PyMilvus | 2.5.4 | 向量数据库客户端 |
| Milvus | — | 向量库（standalone，默认 `localhost:19530`） |
| rank-bm25 | 0.2.2 | BM25 关键词检索 |
| jieba | 0.42.1 | 中文分词 |
| transformers | 4.45.0 | 模型加载 |
| torch | 2.14.0+cu132 | 深度学习框架（CUDA 12.1） |
| milvus-model | 0.2.5 | `BGEM3EmbeddingFunction` 封装 |
| sentence-transformers | 3.0.1 | 句向量工具链 |
| FlagEmbedding | 1.3.5 | BGE 系列模型工具链 |
| OpenAI SDK | 2.24.0 | LLM 调用（OpenAI 兼容协议） |
| LangChain | 1.2.10 | PromptTemplate / TextSplitter |

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

### 向量模型
| 项 | 值 |
|----|-----|
| 模型 | **BGE-M3**（`rag_qa/models/bge-m3/`，本地离线） |
| 稠密向量维度 | 1024 |
| 稀疏向量 | 支持（`SPARSE_INVERTED_INDEX`） |
| 推理设备 | 自动检测：有 CUDA 用 GPU（fp16），否则 CPU（fp32） |
| batch_size | 32 |

> ⚠️ 权重目录约 **4.3 GB**（`pytorch_model.bin` 单文件 2.1 GB），**不要直接推送 GitHub**，见文末。

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

### 2. 安装依赖

```bash
conda create -n law_rag python=3.10 -y
conda activate law_rag

# ⚠️ requirments.txt 是 UTF-16LE 编码，Linux/macOS 下直接安装会报错，
#    请先转成 UTF-8（见「已知问题」第 2 条）
pip install -r requirments.txt
```

### 3. 配置

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

[retrieval]
vector_dim = 1024
retrieval_k = 3
candidate_m = 2
```

所有配置项都支持**环境变量覆盖**（优先级更高，推荐用于 CI / 部署）：
`MYSQL_HOST` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` /
`REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_DB` /
`MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_DATA_NAME` /
`DEEPSEEK_API_KEY` / `MODEL_NAME` / `DASHSCOPE_BASE_URL`

### 4. 启动

**方式一：Web 服务（推荐）**

```bash
uvicorn static.app:app --reload
# 浏览器打开 http://127.0.0.1:8000
```

启动时会通过 lifespan 初始化 `LawChainClient`（连接三个数据库 + 加载 bge-m3 模型），
首次启动约需 **3~15 秒**，日志出现 `LawChainClient 初始化完成` 即就绪。

**方式二：命令行**

```bash
python main.py            # 完整四级检索链
python mysql_qa/main.py   # 仅缓存 + BM25
python rag_qa/main.py     # 仅向量 + LLM
```

### 5. 数据初始化

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
| WS | `/ws/chat` | 问答长连接（JSON 协议见表） |
| GET | `/static/*` | 静态资源（头像等） |

---

## 🎨 前端说明

单文件自包含（`static/templates/chat.html`），**无任何外部 CDN 依赖**，离线可用。

- **风格**：须弥草系配色（草绿 `#3ba55d` / 米白 / 描金 `#d9b26a`），圆润气泡 + 柔和光晕
- **头像**：`static/data/nahida.jpg`（URL 为 `/static/data/nahida.jpg`），
  圆形裁切，顶栏 / 欢迎区 / 消息气泡三处复用
- **Markdown 渲染**：自写轻量渲染器（标题 / 列表 / 引用 / 代码块 / 粗体），
  **先 HTML 转义再渲染**，避免 XSS
- **交互**：引导问题卡片、打字中指示、Enter 发送 / Shift+Enter 换行、
  自动增高输入框、连接状态灯 + 实时延迟显示
- **容错**：断线时提问进队列并提示，重连后自动补发；页面重新可见时立即重连

---

## 🧩 关键设计决策

| 决策 | 原因 |
|------|------|
| **BM25 用 softmax 归一化后比阈值** | 想让阈值与语料规模解耦；实际效果欠佳，见「已知问题」第 3 条 |
| **`mapping_table` 做下标→主键映射** | 避免在 BM25 层引入额外存储，用内存数组即可回查 MySQL |
| **`asyncio.to_thread()` 包住检索链** | 检索链全是同步阻塞 IO（Redis/MySQL/Milvus/LLM），直接 `await` 会卡死事件循环，心跳随之失效 |
| **心跳协程不读 Socket** | Starlette 只允许单读取者；心跳若也 `receive`，会与业务循环抢消息 |
| **子块向量 + 父块上下文** | 兼顾检索精度（子块语义集中）与生成质量（父块脉络完整） |
| **`create_schema(auto_id=False)` + MD5 主键** | 用 `md5(原id + case_no/article_no)` 保证重复入库时幂等 |
| **`insert ignore` + 唯一键** | 让数据灌入可重复执行而不报错 |
| **每个模块 `sys.path.insert(0, project_dir)`** | 支持"脚本直接运行"与"包内导入"两种方式共存 |
| **索引用 FLAT** | 当前数据量（数千条）下 FLAT 精度最高；超 10 万再换 HNSW（见 `优化项.md`） |

---

## ⚠️ 已知问题与 v1 局限

> 这是一份**诚实的清单**。v1 的目标是跑通链路，下列问题已知但未修。

### 🔴 上传 GitHub 前必须处理

| # | 问题 | 说明 | 建议 |
|---|------|------|------|
| 1 | **仓库体积 4.4 GB** | `rag_qa/models/bge-m3/` 占 4.3 GB，其中 `pytorch_model.bin`(**2.17 GB**)、`onnx/model.onnx_data`(**2.16 GB**) | 加入 `.gitignore`；模型改为启动时从 HuggingFace/ModelScope 下载，或用 **Git LFS**（单文件 >100 MB GitHub 会直接拒绝） |
| 2 | **`requirments.txt` 是 UTF-16LE** | Windows `pip freeze` 的产物，带 BOM；Linux/macOS 下 `pip install -r` 会解析失败；且文件名拼写错误（应为 `requirements`） | 转成 UTF-8 并重命名；建议精简为直接依赖 |
| 3 | **明文密码入库** | `config.ini` 含 MySQL `123456`、Redis `1234`；`base/config.py` 把它们写成了代码 fallback 默认值 | 移除代码里的明文默认值，改用 `.env` + `python-dotenv`，并提供 `.env.example` |
| 4 | **依赖清单是整环境快照** | 约 250 个包，包含大量与本项目无关的传递依赖 | 按直接依赖重写，或用 `pipreqs` 生成 |
| 5 | **缺少 `.gitignore` / `.git`** | `logs/*.log`（0.7 MB）、`__pycache__/`、`.idea/`（含本地数据库配置）都会被打包 | 补 `.gitignore`，初始化 git 仓库 |
| 6 | **`common/data/` 有 79 MB 语料** | 24 个案例 PDF + 21 个法条 + 问答 Excel | 若版权/体积敏感，改放外部下载链接 |

### 🟡 代码层面的已知缺陷

| # | 位置 | 问题 |
|---|------|------|
| 7 | `mysql_qa/main.py:75-81` | **契约不匹配（功能性 bug）**：`BM25Search.search()` 返回 `str \| None`，但 `MySQLQASystem.search()` 按 `list[dict]` 处理（`for answer in answers: if "question" in answer`）。字符串会被逐**字符**迭代，`in` 判断恒为 False，导致"MySQL 命中 → 回填 Redis 缓存"**从未生效**；若返回 `None` 则 `for answer in None` 抛 `TypeError`，被 `except Exception` 静默吞掉并返回 `[]`。**注：项目实际使用的 `LawChainClient`（根 `main.py`）调用方式正确，不受此影响。** |
| 8 | `bm25_search.py:79-84` | **阈值几乎不可能命中**：`softmax` 之后榜首分数随语料规模迅速衰减，默认 `threshold=0.85` 在语料稍大时永远达不到，BM25 层形同废用。且 softmax 抹掉了 BM25 原始分的绝对意义。仅 1 条语料时 softmax 恒为 1.0，阈值完全失效。 |
| 9 | `llm_client.py:43-44` | **异常被吞**：`_init_llm` 捕获所有异常且**不重新抛出**，失败时 `self.llm` 保持 `None`，故障延迟到首次提问才以 `AttributeError` 形式暴露。 |
| 10 | `milvus_client.py:281-283` | **异常类型收窄**：`except MilvusException` 只捕获一种异常，若 `vector_tools.encode_query()` 内部报错（如显存不足），异常会直接穿透。 |
| 11 | `milvus_client.py:33` | **初始化失败即全站不可用**：`MilvusClientSystem` 构造失败会抛出，`LawChainClient` 随之失败，FastAPI 无法启动（页面也打不开）。建议把 Milvus 做成可降级组件。 |
| 12 | `bm25_search.py` / `redis_client.py` | **连接泄漏**：`BM25Search` 自建 `MySQLClient` 却**没有 `close()`**；`LawChainClient.close()` 也未关闭它。`redis.Redis()` 是惰性连接，`__init__` 里的 `except redis.RedisError` 基本捕获不到连接失败。 |
| 13 | `mysql_client.py` | **硬编码库名**：`insert_data` 里写死 `law_chain.law_qa` / `law_chain.law_chunk`；若 `MYSQL_DATABASE` 配置不同则直接失败。 |
| 14 | 唯一键设计 | `uk_question(question(255))` / `uk_chunk(source, text_content(255))` 配合 `insert ignore`，会**静默丢弃**前 255 字相同的不同问题/长法条，且无告警。 |
| 15 | `doc_loader.py:84` | **`break` 应为 `continue`**（已核实）：目录遍历时遇到一个无法推断用途的文件就 `break`，会**直接放弃该目录下所有剩余文件**。对 `common/data` 这种混合目录是真实的数据丢失风险。 |
| 16 | `milvus_client.py:215,229` | **按字节截断中文**（已核实）：`text_content.encode('utf-8')[:4000].decode('utf-8')` 若切在汉字中间会抛 `UnicodeDecodeError`（无 `errors` 参数）；而 `VARCHAR(4000)` 本身按字符计长，这层截断既多余又错误。 |
| 17 | `milvus_client.py:278-280` | **结果合并粗糙**（已核实）：`result.extend(result_case)` 使 context 成为含 `distance` 等噪声的**嵌套 Python 列表**，直接 `str()` 进 prompt，token 浪费且不利于来源溯源；两集合结果无排序、无去重、无阈值、无重排。 |
| 18 | `embedding.py:19-23` | **模型无单例，重复加载**：`VectorTools()` 构造时立即加载 bge-m3（数 GB 权重），每次实例化都要重新加载一遍（`MilvusClientSystem` 与各测试脚本都会新建）。 |
| 19 | `llm_client.py:39-41` | **接口协议不一致**：自定义 LLM 用 `hasattr(llm, 'invoke')`（LangChain 风格）校验，实际却调用 `llm.chat.completions.create`（OpenAI 风格），传 LangChain 对象必然运行时报错。 |
| 20 | `llm_client.py:34-37,57-63` | **命名误导 + 参数缺失**：变量名 `DASHSCOPE_*` 实际指向 DeepSeek 后端；调用时**无 system prompt、未设 `temperature` / `max_tokens`**，输出稳定性不可控。 |
| 21 | `rag_qa/main.py:31-33` | **异常当答案返回**：`RAGQAClient.query()` 捕获异常后 `return str(e)`，用户会收到一串英文报错当作"答案"。 |
| 22 | `milvus_client.py:131` | **初始化有写库副作用**：`__init__` 里的 `_init_data()` 在集合为空时会自动从 MySQL 灌数据进 Milvus。虽方便，但让"构造对象"变成了重操作，首次启动耗时/内存高，且难以预料。 |
| 23 | 硬编码不可配 | 模型路径 `rag_qa/models/bge-m3`、`batch_size=32`（向量化）/ `200`（入库）、`use_fp16`、`VECTOR_DIM=1024` 均写死；`VECTOR_DIM` 与模型实际维度需人工保持一致，不一致时建集合即失败。Milvus 无用户名/密码/token 支持。 |
| 24 | `law_text_spliter/main.py:18,30` | 测试入口路径 `r"/common/utils/test_data\..."` 写法错误（号称绝对路径却缺盘符），仅在作为脚本直跑时失败。 |
| 25 | `doc_loader.py:97` / `doc_spliter.py:40` | 测试代码硬编码开发机绝对路径 `D:\develop\...`，影响可移植性与观感。 |
| 26 | 稀疏向量"白算" | 文档侧生成了稀疏向量并建好 `SPARSE_INVERTED_INDEX` 索引，但 `search()` 里 `query_embedding, _ = encode_query(query)` **丢弃了查询稀疏向量**，全程只用稠密向量 —— 混合检索未生效，稀疏算力被浪费。 |

### 🔵 功能待完善

详见 [`优化项.md`](优化项.md)，摘要：

- 混合检索（稠密 + 稀疏 + RRF 融合排序）
- 索引升级 FLAT → HNSW（数据量 > 10 万后）
- 检索结果按 `case_no` / `law_name` 去重
- 案例整案聚合存储 + 增量 upsert（避免全量重建）
- 多轮对话上下文、流式输出（SSE / WebSocket 分片）
- GPU 环境补全 cuDNN 8，OCR + embedding 全上 GPU
- 缺少自动化测试与 CI

---

## 🗺️ 后续规划

| 版本 | 目标 |
|------|------|
| v1（当前） | 跑通完整链路：`search → prompt → LLM → 返回答案`，Web 端可用 |
| v2 | 混合检索 + 结果去重 + 索引升级 + 修掉上述已知缺陷 |
| v3 | 多轮对话、流式输出、增量更新、自动化测试与 CI |

---

## 📄 说明

- 本项目为学习/研究性质，检索结果**不构成法律意见**。
- `common/data/` 中的法律条文与案例来自公开渠道，版权归原作者/发布机构所有；
  若用于公开仓库请自行评估版权风险。
- 模型 `BGE-M3` 版权归 [BAAI](https://github.com/FlagOpen/FlagEmbedding) 所有。
- 页面头像素材来自网络，仅供学习交流，如涉版权请联系删除。
