"""
函数功能：构建 RAG 评估数据集

流程：
    ① 从 MySQL 的 law_qa 表【只读】抽样（按种子固定，可复现）
    ② 过滤：答案太短 / 问题太短或太长 / 含 PII 的样本跳过
    ③ 用裁判 LLM 把「书面化法律问题」改写成「用户口语化提问」
       —— 目的是避开 Redis 缓存与 BM25 的字面命中，测出真实检索能力
    ④ 校验改写结果（长度、与原问题的连续重合度）
    ⑤ 落盘成 eval_dataset.json

产出字段：
    qa_id             law_qa 主键，便于回溯
    original_question 库里的原始问题（仅记录，不用于评估提问）
    eval_question     改写后的问题（★ 评估时用这个提问）
    ground_truth      参考答案（用库里的答案，法答网权威答案）
    answer_chars      参考答案长度，便于分析
    rewrite_attempts  改写用了几次才成功

⚠️ 本模块只执行 SELECT，绝不写 MySQL / Redis。
"""
import argparse
import json
import os
import random
import re
import sys
import time

# ---- 路径注入 --------------------------------------------------------------
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_QA_DIR = os.path.dirname(_EVAL_DIR)
_PROJECT_DIR = os.path.dirname(_RAG_QA_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ★ 必须最先导入 bootstrap：把 DEEPSEEK_API_KEY 注入 os.environ。
#   否则若 key 只存在于 Windows 用户变量（而未进入进程环境），
#   本模块用 build_judge_llm() 改写问题时会拿到占位符并报 401。
from rag_qa.evaluation import bootstrap                                # noqa: E402,F401
from rag_qa.evaluation import config                                  # noqa: E402
from rag_qa.evaluation.wrappers import (                               # noqa: E402
    MissingApiKeyError,
    _get_logger,
    build_judge_llm,
    resolve_api_key,
)

logger = _get_logger()


# ============================================================================
# 一、从 law_qa 抽样（只读）
# ============================================================================
def scene_filter(question: str) -> tuple[bool, str]:
    """
    函数功能：判断问题是否属于「普通用户的法律咨询场景」

    为什么需要：
        law_qa 来自法答网，其中大量问题是法官视角的司法政策与文书问题
        （如「驳回申诉通知书补正使用何种文书样式？」）。
        本项目的定位是面向普通用户的法律咨询，
        用前者评估后者会得到系统性偏低且不可信的分数。

    规则：
        1. 必须命中至少一个 SCENE_KEYWORDS（真实场景关键词）
        2. 不能命中任何 PROCEDURE_KEYWORDS（司法程序/文书类关键词）

    :param question: 问题文本
    :return: (是否保留, 原因说明)
    """
    if not config.SCENE_FILTER_ENABLED:
        return True, "场景过滤已关闭"

    hit_proc = [k for k in config.PROCEDURE_KEYWORDS if k in question]
    if hit_proc:
        return False, f"命中司法程序词 {hit_proc[:2]}"

    hit_scene = [k for k in config.SCENE_KEYWORDS if k in question]
    if not hit_scene:
        return False, "未命中任何真实场景关键词"

    return True, f"命中场景词 {hit_scene[:3]}"


# 各场景的判定关键词：命中越多越可能属于该场景
# 用途仅是把法答网样本【大致】分成两组，便于报告分组对比，不追求精确分类
_EVERYDAY_HINTS: tuple = (
    "租房", "押金", "房东", "工资", "加班", "辞退", "离职", "社保", "工伤",
    "离婚", "抚养", "赡养", "继承", "遗嘱", "彩礼",
    "借款", "欠款", "欠条", "借条", "利息", "欠薪", "拖欠",
    "网购", "退款", "退货", "假货", "消费者", "快递",
    "装修", "漏水", "噪音", "邻居", "物业",
)
_PROFESSIONAL_HINTS: tuple = (
    "效力如何认定", "如何审查", "应否", "可否", "是否应当认定",
    "制度", "抗辩", "竞合", "溯及", "司法解释", "审判", "裁判",
    "破产", "刑事", "数罪并罚", "量刑", "顺位", "保全", "管辖",
)


def infer_scene(question: str) -> str:
    """
    函数功能：把法答网样本大致分类为「生活化」或「专业」

    为什么要分组：
        law_qa 来自法答网（法官/律师提问），而本项目面向普通用户。
        报告里把「生活化样本」与「专业样本」分开统计，
        就能看出分数偏低主要是「题目难」还是「系统弱」——
        这个对比是评估结论可信度的关键。

    ⚠️ 这只是启发式规则，不是精确分类，报告里会注明。
    :param question: 问题文本
    :return: "everyday" 或 "professional"
    """
    n_everyday = sum(1 for k in _EVERYDAY_HINTS if k in question)
    n_prof = sum(1 for k in _PROFESSIONAL_HINTS if k in question)
    return "everyday" if n_everyday > n_prof else "professional"


def fetch_candidates(limit: int = 0) -> list[dict]:
    """
    函数功能：从 law_qa 表读取候选问答（只读 SELECT）

    过滤条件：
        - 答案长度 >= MIN_ANSWER_CHARS（太短的答案没有法律要点，无法当参考答案）
        - 问题长度在 [MIN_QUESTION_CHARS, MAX_QUESTION_CHARS] 之间
        - 问题与答案都不命中 PII 正则
        - 通过场景过滤（面向普通用户，而非法院内部事务）

    :param limit: 最多读取多少条，0 表示不限制
    :return: [{"qa_id": int, "question": str, "answer": str}, ...]
    """
    import pymysql

    from base import config as base_config

    conn = pymysql.connect(
        host=base_config.MYSQL_HOST,
        user=base_config.MYSQL_USER,
        password=base_config.MYSQL_PASSWORD,
        database=base_config.MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        cur = conn.cursor()

        # 长度过滤交给 SQL，减少传输量
        sql = """
            select id, question, answer
            from law_qa
            where char_length(answer) >= %s
              and char_length(question) >= %s
              and char_length(question) <= %s
        """
        params = [config.MIN_ANSWER_CHARS, config.MIN_QUESTION_CHARS,
                  config.MAX_QUESTION_CHARS]
        if limit:
            sql += " limit %s"
            params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()          # ★ 只读连接用完立刻关闭

    logger.info(f"law_qa 读取到 {len(rows)} 条候选（已按长度过滤）")

    # PII 过滤在 Python 侧做：MySQL 的 regexp 不支持 \d 之类的写法，行为不一致
    pii_res = [re.compile(p) for p in config.PII_PATTERNS]
    kept, skipped_pii, skipped_scene = [], 0, []
    for r in rows:
        text = f"{r['question']}\n{r['answer']}"
        if any(p.search(text) for p in pii_res):
            skipped_pii += 1
            continue

        question = r["question"].strip()
        ok, reason = scene_filter(question)
        if not ok:
            skipped_scene.append((question, reason))
            continue

        kept.append({
            "qa_id": r["id"],
            "question": question,
            "answer": r["answer"].strip(),
        })

    if skipped_pii:
        logger.warning(f"PII 过滤跳过 {skipped_pii} 条")
    if skipped_scene:
        logger.info(f"场景过滤跳过 {len(skipped_scene)} 条，示例：")
        for q, reason in skipped_scene[:3]:
            logger.info(f"    {q[:40]}... → {reason}")
    logger.info(f"最终候选 {len(kept)} 条")
    return kept


def sample(candidates: list[dict], n: int, seed: int) -> list[dict]:
    """
    函数功能：按固定种子随机抽样，保证可复现

    :param candidates: 候选问答列表
    :param n: 抽样条数
    :param seed: 随机种子
    :return: 抽样结果
    """
    if n >= len(candidates):
        logger.warning(f"候选只有 {len(candidates)} 条，少于请求的 {n} 条，全部采用")
        return list(candidates)
    rng = random.Random(seed)
    picked = rng.sample(candidates, n)
    logger.info(f"按种子 {seed} 抽取 {len(picked)} 条")
    return picked


# ============================================================================
# 二、问题改写（避开缓存与 BM25 字面命中）
# ============================================================================
def longest_common_substring_len(a: str, b: str) -> int:
    """
    函数功能：求两个字符串的最长连续公共子串长度

    用途：判断改写后的问题是否与原文过于相似。
        如果连续重合太多，说明改写幅度不够，
        该问题仍可能被 BM25 字面命中，评估就失去意义。

    实现：滚动数组的动态规划，O(len(a)*len(b)) 时间、O(len(b)) 空间。
        对 100 字以内的中文问题完全够用。
    :param a: 字符串一
    :param b: 字符串二
    :return: 最长连续公共子串长度
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def max_allowed_chars(original: str) -> int:
    """
    函数功能：算出这条问题允许的改写长度上限

    规则：取「绝对上限」与「相对上限」中的较大者
        绝对上限 = REWRITE_MAX_CHARS
        相对上限 = len(original) * REWRITE_MAX_EXPAND_RATIO + REWRITE_MAX_EXPAND_SLACK

    为什么不能只用绝对上限：
        law_qa 里不少问题本身就有 90~110 字（如破产财产清偿顺位类问题），
        它们的口语化改写必然也长。实测 20 条抽样里有 5 条因为「太长」被连续判失败
        而整条丢弃（占 25%），样本损失过大。

    为什么仍要保留相对上限：
        如果完全不限长度，模型可能把问题扩写成一大段场景描述，
        偏离「一个完整问句」的形态，也会让 BM25 的字面重合度失控。
    :param original: 原问题
    :return: 允许的最大字符数
    """
    relative = int(len(original) * config.REWRITE_MAX_EXPAND_RATIO) \
        + config.REWRITE_MAX_EXPAND_SLACK
    return max(config.REWRITE_MAX_CHARS, relative)


def validate_rewrite(original: str, rewritten: str) -> tuple[bool, str]:
    """
    函数功能：校验改写结果是否可用

    校验项：
        1. 非空
        2. 长度在 [REWRITE_MIN_CHARS, max_allowed_chars(original)] 内
        3. 与原问题的【最长连续公共子串】不超过 REWRITE_MAX_OVERLAP（绝对上限）
        4. 且该重合占改写后长度的比例不超过 REWRITE_MAX_OVERLAP_RATIO（相对上限）

    为什么要同时用绝对与相对两个重合上限：
        只用绝对值时，短问题会被漏放 ——
        原问题「租房押金不退怎么办？」10 字，若原样返回，
        最长公共子串 10 < 绝对上限 14，会被误判为「改写成功」。
        相对上限 0.7 能拦住这种情况（10/10 = 1.0 > 0.7）。

    为什么长度上限要随原问题变化：
        见 max_allowed_chars() 的说明。

    :param original: 原问题
    :param rewritten: 改写后的问题
    :return: (是否通过, 原因说明)
    """
    if not rewritten:
        return False, "改写结果为空"

    rw = rewritten.strip().strip('"“”\'‘’')
    n = len(rw)
    if n < config.REWRITE_MIN_CHARS:
        return False, f"太短（{n} 字）"

    limit = max_allowed_chars(original)
    if n > limit:
        return False, f"太长（{n} 字，上限 {limit} 字）"

    overlap = longest_common_substring_len(original, rw)
    if overlap > config.REWRITE_MAX_OVERLAP:
        return False, f"与原问题连续重合 {overlap} 字（绝对上限 {config.REWRITE_MAX_OVERLAP}）"

    ratio = overlap / n
    if ratio > config.REWRITE_MAX_OVERLAP_RATIO:
        return False, (
            f"重合占比过高：{overlap}/{n} = {ratio:.0%}"
            f"（上限 {config.REWRITE_MAX_OVERLAP_RATIO:.0%}）"
        )

    return True, f"通过（{n} 字，最长重合 {overlap} 字，占 {ratio:.0%}）"


def rewrite_question(llm, original: str, attempt: int = 1) -> str:
    """
    函数功能：调用裁判 LLM 把书面化问题改写成口语化提问

    :param llm: ChatOpenAI 实例
    :param original: 原始问题
    :param attempt: 当前是第几次尝试（用于调整提示强度）
    :return: 改写后的问题（失败时返回空字符串）
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = config.REWRITE_SYSTEM_PROMPT
    limit = max_allowed_chars(original)
    if attempt > 1:
        # 上次改得不够，加强要求（并明确给出长度上限，别让模型猜）
        system += (
            f"\n\n【重要】上一次的改写不合格。这次请务必："
            f"① 改变句式结构和用词，与原文的连续重合不得超过 "
            f"{config.REWRITE_MAX_OVERLAP} 个字；"
            f"② **总长度必须控制在 {limit} 个字以内**（越简洁越好）；"
            f"③ 仍然只输出一个问题，不要解释、不要分点。"
        )

    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"原问题：{original}\n\n改写后："),
        ])
        text = (resp.content or "").strip()
        # 模型偶尔会带前缀，如「改写后：xxx」
        text = re.sub(r"^(改写后|改写结果|答案)[:：]\s*", "", text)
        return text.strip().strip('"“”')
    except Exception as e:  # noqa: BLE001
        logger.error(f"改写调用失败：{e}")
        return ""


def build_one(llm, item: dict) -> dict | None:
    """
    函数功能：为单条候选构建评估样本（含改写与校验）

    :param llm: ChatOpenAI 实例
    :param item: {"qa_id", "question", "answer"}
    :return: 评估样本 dict；改写失败返回 None
    """
    original = item["question"]
    for attempt in range(1, config.REWRITE_MAX_ATTEMPTS + 1):
        rewritten = rewrite_question(llm, original, attempt)
        ok, reason = validate_rewrite(original, rewritten)
        if ok:
            logger.info(f"  [qa_id={item['qa_id']}] 改写成功（第 {attempt} 次）：{reason}")
            return {
                "qa_id": item["qa_id"],
                "original_question": original,
                "eval_question": rewritten,
                "ground_truth": item["answer"],
                "answer_chars": len(item["answer"]),
                "rewrite_attempts": attempt,
                "scene": infer_scene(original),
                "source": "法答网",
            }
        logger.warning(f"  [qa_id={item['qa_id']}] 第 {attempt} 次改写不合格：{reason}")

    logger.error(f"  [qa_id={item['qa_id']}] {config.REWRITE_MAX_ATTEMPTS} 次改写均失败，舍弃该条")
    return None


# ============================================================================
# 三、落盘
# ============================================================================
def save_dataset(samples: list[dict]) -> str:
    """把评估数据集写成 JSON，返回文件路径"""
    config.ensure_dirs()
    path = config.DATASET_FILE
    payload = {
        "meta": {
            "sample_size": len(samples),
            "seed": config.RANDOM_SEED,
            "min_answer_chars": config.MIN_ANSWER_CHARS,
            "question_chars_range": [config.MIN_QUESTION_CHARS, config.MAX_QUESTION_CHARS],
            "rewrite_max_overlap": config.REWRITE_MAX_OVERLAP,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "samples": samples,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"数据集已写入：{path}")
    return str(path)


def load_dataset() -> list[dict]:
    """读取已生成的评估数据集（供 pipeline_runner 使用）"""
    path = config.DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"找不到评估数据集：{path}\n请先运行：python -m rag_qa.evaluation.dataset_builder"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["samples"]


# ============================================================================
# 四、命令行入口
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="构建 RAG 评估数据集（从 law_qa 抽样 + LLM 改写成口语化问题）",
    )
    parser.add_argument("--size", type=int, default=config.SAMPLE_SIZE,
                        help=f"抽样条数（默认 {config.SAMPLE_SIZE}）")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help=f"随机种子（默认 {config.RANDOM_SEED}）")
    parser.add_argument("--preview-only", action="store_true",
                        help="只抽样并预览，不调用 LLM 改写、不落盘（零成本）")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="不调用 LLM 改写，直接用原问题（⚠️ 会导致评估被缓存/BM25 短路）")
    parser.add_argument("--with-manual", action="store_true", default=True,
                        help="合并手工构造的生活化样本（默认开启）")
    parser.add_argument("--no-manual", action="store_true",
                        help="不合并手工样本，只用 law_qa 抽样")
    args = parser.parse_args()

    print(config.describe())

    # ① 抽样
    print("\n[1/4] 从 law_qa 只读抽样 ...")
    candidates = fetch_candidates()
    if not candidates:
        print("  ❌ 没有可用候选，检查 law_qa 是否有数据")
        return 1
    picked = sample(candidates, args.size, args.seed)
    print(f"  ✅ 抽到 {len(picked)} 条")

    if args.preview_only:
        print("\n[预览模式] 抽样结果：")
        for i, it in enumerate(picked, 1):
            print(f"\n  {i}. [qa_id={it['qa_id']}] {it['question']}")
            print(f"     答案 {len(it['answer'])} 字：{it['answer'][:80]}...")
        print("\n未调用 LLM、未落盘。")
        return 0

    # ② 构造样本
    if args.no_rewrite:
        print("\n[2/4] 跳过改写（--no-rewrite），直接用原问题")
        print("  ⚠️ 警告：原问题必然被 Redis/BM25 命中，评估结果会失真！")
        samples = [
            {
                "qa_id": it["qa_id"],
                "original_question": it["question"],
                "eval_question": it["question"],
                "ground_truth": it["answer"],
                "answer_chars": len(it["answer"]),
                "rewrite_attempts": 0,
            }
            for it in picked
        ]
    else:
        print("\n[2/4] 调用 DeepSeek 改写问题 ...")
        try:
            llm = build_judge_llm(resolve_api_key())
        except MissingApiKeyError as e:
            print(f"  ❌ {e}")
            return 1
        samples = []
        for i, it in enumerate(picked, 1):
            print(f"\n  ({i}/{len(picked)}) qa_id={it['qa_id']}")
            print(f"    原问题：{it['question']}")
            sample_obj = build_one(llm, it)
            if sample_obj:
                print(f"    改写后：{sample_obj['eval_question']}")
                samples.append(sample_obj)
            time.sleep(config.SLEEP_BETWEEN_SAMPLES)

    if not samples:
        print("\n❌ 没有任何样本构建成功")
        return 1

    # 合并手工构造的生活化样本（方案 C：两组对比）
    if args.with_manual and not args.no_manual:
        print("\n[合并] 加入手工构造的生活化样本 ...")
        from rag_qa.evaluation.manual_samples import load_manual_samples

        manual = load_manual_samples()
        samples.extend(manual)
        print(f"  ✅ 加入 {len(manual)} 条生活化样本")

    # ③ 落盘
    print(f"\n[3/4] 写入数据集（{len(samples)} 条）...")
    path = save_dataset(samples)
    print(f"  ✅ {path}")

    # ④ 摘要
    print("\n[4/4] 摘要")
    print(f"  成功样本      : {len(samples)} 条")
    avg_gt = sum(s["answer_chars"] for s in samples) / len(samples)
    print(f"  参考答案均长  : {avg_gt:.0f} 字")
    retried = sum(1 for s in samples if s["rewrite_attempts"] > 1)
    print(f"  改写重试过的  : {retried} 条")

    # 按场景分组统计
    groups: dict[str, int] = {}
    for s in samples:
        groups[s.get("scene", "unknown")] = groups.get(s.get("scene", "unknown"), 0) + 1
    scene_label = {"everyday": "生活化", "professional": "专业", "manual": "手工构造"}
    print("  场景分组      :")
    for k, v in sorted(groups.items()):
        print(f"     {scene_label.get(k, k):<8} {v} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
