"""
函数功能：RAG 评估模块的集中配置

设计原则：
    1. 所有可调参数集中在此，避免散落在各个脚本里
    2. 路径用【本文件位置】推导，无论从哪里运行都能找到
    3. 评估配置与生产配置（config.ini）分离 —— 评估不该影响线上行为

⚠️ 重要约束：本目录下的代码【只读】调用项目其他模块的接口，
   不修改任何现有文件，也不调用任何会写 Redis / MySQL 的方法。
"""
import os
import sys
from pathlib import Path

# ============================================================================
# 路径推导
# ============================================================================
# 本文件为 rag_qa/evaluation/config.py
#   → EVAL_DIR    = rag_qa/evaluation
#   → RAG_QA_DIR  = rag_qa
#   → PROJECT_DIR = 项目根目录
EVAL_DIR: Path = Path(__file__).resolve().parent
RAG_QA_DIR: Path = EVAL_DIR.parent
PROJECT_DIR: Path = RAG_QA_DIR.parent

# 把项目根目录加入 sys.path，保证能 import main / base / rag_qa 等模块
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 评估产物目录（数据集、中间结果、报告）
# 可用环境变量 EVAL_OUTPUT_ROOT 重定向到项目外（例如写入受限的环境、或想把产物放别处）
_output_root_env = os.getenv("EVAL_OUTPUT_ROOT")
_output_root: Path = Path(_output_root_env).resolve() if _output_root_env else EVAL_DIR

OUTPUT_DIR: Path = _output_root / "outputs"

# 评估专用日志（与线上 logs/app.log 隔离，避免互相干扰）
EVAL_LOG_DIR: Path = _output_root / "logs"


def ensure_dirs() -> None:
    """
    函数功能：创建评估所需的输出目录

    为什么不在模块顶层直接 mkdir：
        模块顶层执行 mkdir 会让「import config」产生文件系统副作用 ——
        仅仅想读取几个常量就会改动磁盘，且权限受限时连 import 都会失败。
        因此把建目录改为显式调用，由各入口脚本在真正要写文件前调用一次。
    :return: None
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 一、数据集构建参数
# ============================================================================
# 抽样条数
# ⚠️ 为什么取 20 而不是 10：
#     实测约 1/3 的评估问题仍会被 BM25 字面命中（答案直接来自 law_qa 表），
#     这些样本没有检索上下文，无法参与 RAGAS 指标计算而必须剔除。
#     抽 20 条、剔除后约剩 13~14 条有效样本，加上 6 条手工样本，
#     可参与计算的样本约 19~20 条，足够支撑一份有意义的报告。
SAMPLE_SIZE: int = 20

# 固定随机种子，保证「同一次抽样结果可复现」
RANDOM_SEED: int = 42

# law_qa 表中答案的最短长度门槛
# 太短的答案（如「是的」「可以」）没有足够法律要点，无法作为 ground_truth
MIN_ANSWER_CHARS: int = 80

# law_qa 表中问题的最短长度门槛，过滤掉无意义短问题
MIN_QUESTION_CHARS: int = 6

# 问题的最长长度门槛
# 超过这个长度的问题多是「执行法院以物抵债裁定…」这类专业表述，
# 真实用户不会这么长，且改写成口语后评测意义有限
MAX_QUESTION_CHARS: int = 80

# 【PII 过滤】抽样时跳过含以下模式的问答，避免把个人隐私送进外部 LLM API
# 注意：这是粗筛，不是完整脱敏方案
PII_PATTERNS: tuple = (
    r"1[3-9]\d{9}",                      # 中国手机号
    r"\d{17}[\dXx]",                     # 身份证号
    r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|同志)",   # 张先生 / 李女士
)

# 【关键】改写评估问题的要求：
#    库里的原问题必然能被 Redis 缓存 / BM25 命中，那样就测不出真实检索能力。
#    因此必须把问题改写成「语义相同、措辞不同」的口语化提问。
REWRITE_TEMPERATURE: float = 0.3      # 偏低，保证改写稳定

REWRITE_SYSTEM_PROMPT: str = (
    "你是一个法律问答系统的测试数据构造助手。\n"
    "你的任务：把一条「书面化的法律问题」改写成一个普通用户会问的口语化问题。\n"
    "\n"
    "硬性要求：\n"
    "1. 必须保持原问题的【核心法律争议点】完全不变\n"
    "2. 必须换用不同的措辞，禁止与原问题有连续 8 个字以上的重合\n"
    "3. 可以补充一个具体的生活场景（如租房、上班、网购、借钱），"
    "但不得引入原问题没有的法律条件\n"
    "4. 只输出改写后的问题本身，不要任何解释、前缀、引号或编号\n"
    "5. 输出必须是一个完整的中文问句"
)

# 改写结果的合法性校验
REWRITE_MIN_CHARS: int = 8
# 改写后问题的【基础】长度上限
REWRITE_MAX_CHARS: int = 60
# 允许的「相对原问题的最大膨胀系数」
#   ⚠️ 为什么需要：法答网有相当一部分问题本身就很长（如 108 字的破产财产顺位问题），
#   其口语化改写必然也长。若只用绝对上限 60 字，这类问题会被连续判失败而丢弃。
#   实测 20 条抽样中有 5 条因此被误杀，占 25% —— 样本损失太大。
#   规则：改写后的长度不得超过「原问题长度 × 本系数 + 常数余量」。
#   这样短问题的改写仍受严格控制，长问题则按比例放宽。
REWRITE_MAX_EXPAND_RATIO: float = 1.2
REWRITE_MAX_EXPAND_SLACK: int = 15
# 与原问题允许的最长连续重合字符数
#   ⚠️ 不能设太小：law_qa 的问题里含【不可改写的法律术语】，
#   例如「组织、领导传销活动罪」本身就有 10 个字，改写时必须原样保留，
#   否则法律含义会变。设成 8 会把所有含长术语的合法改写都判成失败。
REWRITE_MAX_OVERLAP: int = 14
# 相对重合率上限：最长公共子串 / 改写后问题长度
#   ⚠️ 为什么还需要这个：绝对阈值对【短问题】太宽松。
#   例如原问题「租房押金不退怎么办？」共 10 字，若 LLM 原样返回，
#   最长公共子串是 10，仍小于绝对上限 14，会被误判为「改写成功」。
#   加了相对阈值（0.7）后，10/10 = 1.0 > 0.7，会被正确拦下。
REWRITE_MAX_OVERLAP_RATIO: float = 0.7
# 单条问题最多尝试几次改写
REWRITE_MAX_ATTEMPTS: int = 3

# ============================================================================
# 一之二、场景过滤（关键：决定评估结论是否可信）
# ============================================================================
# 【为什么需要这个】
#     law_qa 来自法答网，其中相当一部分是「法官视角」的问题，例如：
#         「驳回申诉通知书补正使用何种文书样式？」
#         「司法救助…人民法院能否以50%为标准认定救助不到位？」
#         「商标权纠纷案件中，法院在审查合法来源抗辩时应当把握何种标准？」
#     这些问题的用户是法官/律师，不是普通老百姓。
#     而本项目的定位是【面向普通用户的法律咨询】。
#     用前者评估后者，会得到系统性偏低的分数 ——
#     但那不代表系统差，而是评估集选错了场景，结论不可信。
#
# 【策略】
#     1. 问题必须命中至少一个「真实场景关键词」
#     2. 问题不能命中任何「司法程序/文书类关键词」
#     两者都通过，才认为这条样本代表普通用户的法律咨询场景。
SCENE_FILTER_ENABLED: bool = True

# 真实用户常见法律场景关键词（命中任一即可）
SCENE_KEYWORDS: tuple = (
    "合同", "劳动", "工资", "加班", "辞退", "解雇", "离职", "社保", "工伤",
    "租房", "租赁", "押金", "房东", "买房", "商品房", "物业", "装修",
    "离婚", "抚养", "赡养", "继承", "遗嘱", "结婚", "彩礼",
    "借款", "欠款", "欠条", "借条", "利息", "担保", "抵押", "网贷",
    "网购", "退款", "退货", "假货", "消费者", "快递", "外卖",
    "赔偿", "医疗", "事故", "侵权", "隐私", "名誉", "肖像",
    "公司", "股东", "合伙", "股权", "拖欠", "预付", "会员卡",
    "未成年人", "校园", "动物", "宠物", "邻居", "漏水", "噪音",
)

# 司法程序 / 司法机关内部事务关键词（命中任一即排除）
PROCEDURE_KEYWORDS: tuple = (
    "文书样式", "案号", "立案", "申诉", "再审", "抗诉", "检察",
    "管辖", "送达", "公告", "执行法院", "强制措施", "羁押",
    "法官", "合议庭", "审判委员会", "司法解释适用", "裁判文书",
    "公安机关", "侦查", "起诉阶段", "量刑", "缓刑", "假释",
    "司法救助", "法律援助指派", "鉴定意见", "证据规则",
)

# ============================================================================
# 二、评估流水线参数
# ============================================================================
# 单个检索片段送入 RAGAS 前的截断长度（字符）
# 法律条文/案例很长，不截断会让裁判模型的上下文过长而漏判
MAX_CONTEXT_CHARS: int = 1200

# 上下文片段总数上限（案例 + 法条合计），防止个别样本塞入过多片段
MAX_CONTEXT_CHUNKS: int = 12

# 生成答案时的最大等待（秒）
# ⚠️ 这个值【必须真正生效】。项目里 OpenAI 客户端没设 timeout，
#   用的是 openai 库默认的 600 秒；实测某条"海上货物运输保险"样本
#   卡了整整 10 分钟才抛 ReadTimeout，整个评估被拖住。
#   评估场景下宁可放弃这一条，也不能卡 10 分钟。
#   实现见 pipeline_runner._iter_with_timeout()
GENERATE_TIMEOUT: int = 180

# 单条样本生成失败（超时/网络错误）时的最多尝试次数
#   注意：重试会重新消耗 token，因此次数不宜多
MAX_GENERATE_ATTEMPTS: int = 2

# 每条样本之间的间隔（秒），避免对 LLM API 造成突发压力
SLEEP_BETWEEN_SAMPLES: float = 0.5

# ============================================================================
# 三、裁判模型配置（RAGAS 用）
# ============================================================================
# 裁判 LLM：复用项目已有的 DeepSeek（环境变量 DEEPSEEK_API_KEY）
# 为什么不用小模型当裁判：RAGAS 官方提示，弱裁判模型会给出不可信分数，
# 那比不评估更糟 —— 会误导后续优化方向。
JUDGE_MODEL: str = "deepseek-flash"
JUDGE_BASE_URL: str = "https://api.deepseek.com"
JUDGE_TEMPERATURE: float = 0.0        # 裁判必须确定性输出
JUDGE_MAX_TOKENS: int = 2048
JUDGE_TIMEOUT: int = 120
JUDGE_MAX_RETRIES: int = 3

# 裁判 Embedding：复用项目本地的 BGE-M3（不额外依赖 Ollama）
# 仅用于 answer_relevancy 的余弦相似度计算，BGE-M3 完全够用
EMBEDDING_BATCH_SIZE: int = 32

# ============================================================================
# 四、评估指标开关
# ============================================================================
# 「检索层」指标：衡量上下文质量，不看生成的答案
#     context_precision —— 捞上来的材料有没有噪声（排序质量）
#     context_recall    —— 标准答案的要点有没有漏掉（召回能力）
METRICS_RETRIEVAL: tuple = ("context_precision", "context_recall")

# 「生成层」指标：衡量答案质量
#     faithfulness      —— 答案有没有编造（幻觉检测）
#     answer_relevancy  —— 答案有没有跑题
METRICS_GENERATION: tuple = ("faithfulness", "answer_relevancy")

# 「参考指标」：对「答案措辞差异」高度敏感
#     answer_correctness —— 生成答案 vs 标准答案
# ⚠️ 法律答案允许多种正确表达路径（引用不同法条/案例），该指标会系统性偏低，
#    因此【仅作观察】，不作为优化依据。要跑就一起跑，但报告里单独标注。
ENABLE_ANSWER_CORRECTNESS: bool = True

# 报告里分数保留几位小数
SCORE_DECIMALS: int = 4

# ============================================================================
# 五、输出文件名
# ============================================================================
DATASET_FILE: Path = OUTPUT_DIR / "eval_dataset.json"        # ① 评估数据集
RUN_FILE: Path = OUTPUT_DIR / "eval_run.json"                # ② 流水线中间结果
RUN_RAW_FILE: Path = OUTPUT_DIR / "eval_run_raw.json"        #    含检索原文，便于人工核查
RAGAS_FILE: Path = OUTPUT_DIR / "eval_ragas.json"            # ③ RAGAS 原始分数
REPORT_FILE: Path = OUTPUT_DIR / "eval_report.md"            # ④ 最终报告


def describe() -> str:
    """返回配置摘要，便于脚本启动时打印（一眼看清本次评估用什么参数）"""
    lines = [
        "=" * 68,
        "RAG 评估配置",
        "=" * 68,
        f"  项目根目录      : {PROJECT_DIR}",
        f"  产物目录        : {OUTPUT_DIR}",
        f"  评估日志目录    : {EVAL_LOG_DIR}",
        "-" * 68,
        f"  抽样条数        : {SAMPLE_SIZE}",
        f"  随机种子        : {RANDOM_SEED}",
        f"  law_qa 答案门槛 : >= {MIN_ANSWER_CHARS} 字",
        f"  片段截断长度    : {MAX_CONTEXT_CHARS} 字 / 最多 {MAX_CONTEXT_CHUNKS} 片段",
        "-" * 68,
        f"  裁判 LLM        : {JUDGE_MODEL}",
        f"  裁判 Base URL   : {JUDGE_BASE_URL}",
        f"  裁判 Embedding  : BGE-M3（本地）",
        "-" * 68,
        f"  检索层指标      : {', '.join(METRICS_RETRIEVAL)}",
        f"  生成层指标      : {', '.join(METRICS_GENERATION)}",
        f"  观察指标        : answer_correctness（{'启用' if ENABLE_ANSWER_CORRECTNESS else '关闭'}）",
        "=" * 68,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
