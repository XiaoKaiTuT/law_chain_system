from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from pathlib import Path
import os, sys
import json
import re

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger, config
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# 用于生成数据训练的语料
class GenerateDataClient:
    def __init__(self):
        self.logger = logger
        self.config = config
        try:
            self.client = OpenAI(
                api_key=self.config.DASHSCOPE_API_KEY,
                base_url=self.config.DASHSCOPE_BASE_URL,
            )
        except Exception as e:
            self.logger.error(f"LLM 初始化失败：{e}")
            raise

    @staticmethod
    def _generate_legal_prompt() -> PromptTemplate:
        """
        函数功能：批量生成法律专业问题的提示词(单词生成一批，循环调用累积)
        :return: PromptTemplate对象
        """
        return PromptTemplate(
            template="""
            你是法律问答数据集构建专家。请生成 {num} 条【法律专业问题】，用于训练法律问题分类器。

            ## 什么算「法律专业问题」
            回答该问题需要引用法律条文、司法解释或法律概念。包含以下两种：

            A. 显式法律问题：直接使用法律术语
               例：离婚冷静期是多久？/ 盗窃罪的构成要件是什么？

            B. 口语化法律问题：不使用法律术语，但回答它需要法律知识
               例：老板拖欠我工资三个月了咋办？/ 租房押金不退怎么办？

            ## 硬性要求
            1. 必须全部是【疑问句】或明确的问题表述，不能是陈述句。
            2. **至少 40% 必须是 B 类口语化问题** —— 这是本次生成的重点。
               B 类问题中**不允许出现**「法律」「法条」「诉讼」「违法」「权益」等词，
               要用老百姓的日常说法（如"押金不退""老板不给钱""房东赶我走"）。
            3. 覆盖以下法律领域，尽量均匀分布：
               婚姻家庭 / 劳动争议 / 房屋租赁买卖 / 民间借贷 / 合同纠纷 /
               交通事故 / 消费者权益 / 侵权赔偿 / 继承 / 刑事犯罪 /
               公司股权 / 知识产权 / 行政处罚 / 执行与保全
            4. **禁止重复或高度相似**：不得出现仅个别字词不同的问题。
            5. 问题长度要有差异：60% 为 30 字以上的完整问题，
               40% 为 20 字以内的简短提问。
            6. 不能与历史记录中的语义或者内容重复。

            ## 输出格式（严格遵守）
            只输出一个 JSON 数组，不要任何解释、前言、Markdown 代码块标记。
            数元组的每个素是只含一个键 "text" 的对象：

            [
              {{"text": "离婚时两周岁以下的孩子抚养权怎么判"}},
              {{"text": "老板拖欠工资三个月了一直不给怎么办"}},
              {{"text": "租的房子还没到期房东就要赶我走"}},
            ]

            注意：JSON 的键和字符串值必须使用双引号；不要在数组外输出任何字符。
            
            历史记录：{history}
            现在请生成 {num} 条法律专业问题：
            """,
            input_variables=["num", "history"],
        )

    @staticmethod
    def _generate_daily_prompt() -> PromptTemplate:
        """
        函数功能：批量生成日常非法律问题的提示词(单词生成一批，循环调用累积)
        :return: PromptTemplate对象
        """
        return PromptTemplate(
            template="""
            你是对话数据集构建专家。请生成 {num} 条【日常非法律问题】，用于训练法律问题分类器。

            ## 什么算「日常非法律问题」
            回答该问题**不需要**引用任何法律条文、司法解释或法律概念。
            注意：其他专业领域（医学、编程、金融投资等）也算日常问题。

            ## 必须覆盖的 10 个方面（每条问题只属于一个方面，尽量均匀分布）
            1. 问候寒暄：如"你好""在吗""你是谁""谢谢"（**至少 30% 是这类短句，长度 2~15 字**）
            2. 科技与编程：如"Python 怎么读取大文件""怎么把手机照片传到电脑"
            3. 健康医疗：如"感冒了吃什么药好得快""血压高平时要注意什么"
            4. 学习教育：如"考研数学怎么复习""怎么快速背单词"
            5. 职场非法律：如"怎么跟领导提加薪""简历怎么写吸引人"
            6. 生活常识：如"水烧开是多少度""洗衣机怎么清洗"
            7. 情感心理：如"和朋友吵架了怎么和好""总是焦虑怎么办"
            8. 娱乐休闲：如"推荐几部好看的悬疑电影""周末去哪玩比较好"
            9. 旅行出行：如"去杭州三天怎么安排""坐飞机能带充电宝吗"
            10. 美食烹饪：如"红烧肉怎么做不腻""空气炸锅能做什么"
            11. 不能与历史记录中的语义或者内容重复。

            ## 硬性要求
            1. 必须全部是【疑问句】或明确的问题表述。
            2. **严禁涉及法律**：不得出现法律术语，也不得是"需要法律判断"的场景。
               反例（不要生成）："租房押金不退怎么办"（这是法律问题）
               正例（要生成）："租房要注意什么"（这是生活常识）
            3. **不允许出现**「法律」「法条」「诉讼」「合同」「赔偿」「权益」等词。
            4. **禁止重复或高度相似**。
            5. 长度分布：30% 为 2~15 字的短句，70% 为 15~40 字。

            ## 输出格式（严格遵守）
            只输出一个 JSON 数组，不要任何解释、前言、Markdown 代码块标记。
            数组的每个元素是只含一个键 "text" 的对象：

            [
              {{"text": "你好"}},
              {{"text": "Python 怎么读取大文件"}},
              {{"text": "感冒了吃什么药好得快"}},
            ]

            注意：JSON 的键和字符串值必须使用双引号；不要在数组外输出任何字符。
            
            历史记录：{history}
            现在请生成 {num} 条日常非法律问题：
            """,
            input_variables=["num", "history"],
        )

    @staticmethod
    def _generate_daily_prompt_missing() -> PromptTemplate:
        """
        函数功能：针对于生成出的日常问题语料分析，进行补充缺口：长句、第一人称、金融/数学/医疗领域
        :return: PromptTemplate对象
        """
        return PromptTemplate(
            template="""
            你是对话数据集构建专家。请生成 {num} 条【日常非法律问题】。

            ## 什么是「日常非法律问题」
            回答该问题【不需要】引用法律条文、司法解释或法律概念。
            注意：其他专业领域（金融、医学、数学等）也算日常问题。

            ## 本批次的强制配比（必须严格遵守，这是本次生成的核心要求）

            ### 要求 1：长度
            - **至少 50% 必须是 40 字以上的长句**，且要分成 2~3 个短句，用逗号连接。
              像是用户在认真描述自己的情况，例如：
              "我家的洗衣机用了五六年，最近甩干的时候声音特别大，
               而且衣服还是湿的，我想自己看看是不是哪里松了，
               但不知道从哪拆起，该注意什么"
            - 其余不超过 40 字。

            ### 要求 2：人称
            - **至少 40% 必须以「我」为主语**描述自己的情况或困惑。
            - 其余可以用"怎么才能…""有什么办法"等无人称问法。

            ### 要求 3：领域覆盖（必须同时包含以下几类，每类至少 3 条）
            1. **金融理财**（本批重点）：基金定投怎么选、信用卡分期划不划算、
               存款和货币基金哪个更适合短期闲钱、房贷提前还款值不值、
               保险怎么挑、记账方法、如何控制冲动消费
            2. **数学物理**（本批重点）：公式理解、解题思路、
               物理概念怎么想通、单位换算、统计概念
            3. **医学健康**（本批重点）：常见症状的日常护理、
               体检指标怎么看、作息饮食调整、康复锻炼
            4. 编程与数码
            5. 学习教育
            6. 生活技能（家电清洁保养、收纳整理、小维修）

            ## 硬性禁止
            1. **严禁涉及法律**：不得出现"违法""侵权""赔偿""合同""维权"
               "起诉""责任"等词，也不得是"需要法律判断"的场景。
               反例（不要生成）："房东不退押金怎么办" ← 这是法律问题
               正例（要生成）："租房时看房应该重点检查哪些地方"
            2. **禁止重复或高度相似**：不能只改个别字词。
            3. 必须全部是疑问句或明确的问题表述。
            4. 不能与历史记录中的语义或者内容重复。

            ## 输出格式（严格遵守）
            只输出一个 JSON 数组，不要任何解释、前言、Markdown 代码块标记。
            每个元素是只含一个键 "text" 的对象：

            [
              {{"text": "我家的冰箱用了八年，冷藏室总是积水，擦完过两天又有了，想自己看看是不是排水孔堵了，该从哪里下手"}},
              {{"text": "基金定投是每个月固定投一笔好，还是跌的时候多投一点好"}},
              {{"text": "我总忍不住月底就花超，有没有适合上班族的记账方法"}}
            ]

            注意：JSON 的键和值必须用双引号；数组外不要输出任何字符。

            ## 输出前自检（请在心里过一遍再输出）
            - 长句（40字以上）数量是否达到 {num} 的 50%？
            - 以「我」开头或含「我」的比例是否达到 40%？
            - 金融、数学物理、医学健康三类是否各至少 3 条？
            - 有没有不小心写出需要法律判断的问题？
            
            历史记录：{history}
            现在请生成 {num} 条日常非法律问题：
            """,
            input_variables=["num", "history"],
        )

    @staticmethod
    def _generate_legal_prompt_missing() -> PromptTemplate:
        """
        函数功能：针对于生成出的法律问题语料分析，进行补充缺口：带问候客套、极短句、第一人称
        :return:PromptTemplate对象
        """
        return PromptTemplate(
            template="""
            你是法律问答数据集构建专家。请生成 {num} 条【法律专业问题】。

            ## 什么是「法律专业问题」
            回答该问题需要引用法律条文、司法解释或法律概念。

            ## 本批次的强制配比（必须严格遵守，这是本次生成的核心要求）

            ### 要求 1：必须带问候 / 客套词 —— 至少 50%
            在问题前面加上自然的问候或客套语，让人读起来像是用户随口打的。
            可用的问候/客套词（轮换使用，不要只用"你好"）：
              你好 / 您好 / 请问 / 麻烦问下 / 想咨询一下 / 请教一下 /
              不好意思打扰了 / 谢谢 / 嗨 / 在吗 / 想问个事

            示例：
              "你好，我想问一下租房押金不退怎么办"
              "麻烦问下，公司拖欠工资三个月了可以告他吗"
              "不好意思打扰了，我买的手机有质量问题该怎么维权"
              "嗨，想咨询一下朋友借钱不还只有转账记录能不能要回来"
              "谢谢，那如果对方一直不还钱，我能直接去起诉吗"

            ### 要求 2：极短提问 —— 至少 20%（6 字以内）
            像用户打字偷懒一样，只给出核心词，不带任何修饰。
            示例：
              "押金不退" / "欠钱不还" / "被辞退了" / "工伤怎么办" /
              "离婚财产" / "合同违约" / "交通事故" / "工资拖欠" /
              "房东赶人" / "借钱不还" / "试用期辞退" / "抚养费"

            ### 要求 3：第一人称真实遭遇 —— 至少 30%
            以「我」为主语，描述自己遇到的具体情况，用口语说。
            示例：
              "我在工厂上班手指被机器压伤了，老板没给我交工伤保险，医药费该谁出"
              "我朋友开公司让我挂名当股东，说不用出钱，现在公司欠债法院找上我，我要不要替公司还钱"
              "我租的房子还没到期，房东突然说要收回自己住，让我一个月内搬走，这合法吗"

            ### 要求 4：覆盖多个法律领域
            婚姻家庭 / 劳动争议 / 房屋租赁买卖 / 民间借贷 / 合同纠纷 /
            交通事故 / 消费者权益 / 侵权赔偿 / 继承 / 刑事 /
            公司股权 / 知识产权 / 行政处罚 / 执行保全

            ## 硬性要求
            1. **禁止重复或高度相似**：不能只改个别字词。
            2. 口语化的那部分**不要堆砌法律术语**，用老百姓的说法
               （如"押金不退"而不是"出租人拒不返还租赁保证金"）。
            3. 每个问题的疑问点要单一明确，不要一次问五六个问题。
            4. 不能与历史记录中的语义或者内容重复。

            ## 输出格式（严格遵守）
            只输出一个 JSON 数组，不要任何解释、前言、Markdown 代码块标记。
            每个元素是只含一个键 "text" 的对象：

            [
              {{"text": "你好，我想问一下租房押金不退怎么办"}},
              {{"text": "欠钱不还"}},
              {{"text": "我在网上买的手机收到就是翻新机，商家不承认，我该怎么维权"}}
            ]

            注意：JSON 的键和值必须用双引号；数组外不要输出任何字符。

            ## 输出前自检（请在心里过一遍再输出）
            - 带问候/客套词的是否达到 {num} 的 50%？
            - 6 字以内的极短提问是否达到 20%？
            - 以「我」为主语的是否达到 30%？
            - 法律领域是否覆盖了 5 个以上？
            
            历史记录：{history}
            现在请生成 {num} 条法律专业问题：
            """,
            input_variables=["num", "history"],
        )

    def generate_data(self, num, prompt_type, history=None):
        """
        函数功能：生成数据训练的语料
        :param num: 生成语料的条数
        :param history: 历史记录
        :param prompt_type: 提示词类型，可选值有："daily"、"legal"
        :return:
        """
        # 判断提示词类型
        PROMPT_TYPE = {
            # 第一次生成语料记录
            # "daily": self._generate_daily_prompt,
            # "legal": self._generate_legal_prompt,
            "daily": self._generate_daily_prompt_missing,
            "legal": self._generate_legal_prompt_missing,
        }
        if prompt_type in PROMPT_TYPE:
            generate_prompt = PROMPT_TYPE[prompt_type]
        else:
            self.logger.error(f"提示词类型错误: {prompt_type}")
            raise ValueError(f"提示词类型错误: {prompt_type}")
        prompt = generate_prompt().format(num=num, history=history)

        # 调用LLM生成语料
        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            answer = response.choices[0].message.content
            return self._parse_llm_json(answer)
        except Exception as e:
            self.logger.error(f"LLM 调用失败：{e}")
            raise

    def _parse_llm_json(self, answer: str|None) -> list:
        """
        函数功能：把 LLM 返回的文本解析成 Python 列表
        :param answer: LLM 原始输出
        :return: 解析后的列表
        """
        # 判断LLM返回内容是否为空
        if not answer or not answer.strip():
            raise ValueError("LLM返回内容为空")

        text = answer.strip()

        # 尝试去掉 Markdown 代码块包裹
        # ```json ...``` 或 ```...```
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        # 尝试解析 JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 尝试进行修复，找到最外层的 JSON 数组
            start = text.find("[")
            end = text.rfind("]")
            # 判断是否找到 JSON 数组边界
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"未找到 JSON 数组边界，原始输出前200字：{answer[:200]}")
            # 尝试继续解析 JSON
            try:
                data = json.loads(text[start:end+1])
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {e};原始输出前 200 字: {answer[:200]}") from e

        return data

    def append_jsonl(self, path: Path, records: list[dict]) -> int:
        """
        函数功能：将记录追加到 JSONL 文件中 (UTF-8，不带 BOM)
        :param path: 目标文件路径
        :param records: 记录列表
        :return: 实际写入的条数
        """
        if not records:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(records)

    def load_jsonl(self, path: Path) -> list[dict]:
        """
        函数功能：读取 JSONL 文件，跳过空行与解析失败的行
        :param path:文件路径
        :return:记录列表
        """
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"第 {lineno} 行解析失败，已跳过: {e}")
                    continue
                if isinstance(obj, dict) and obj.get("text"):
                    records.append(obj)
        return records

def main():
    client = GenerateDataClient()
    num = 20
    target_per_type = 600
    out_path = Path(__file__).parent / "data" / "raw.jsonl"

    for prompt_type in ("daily", "legal"):
        existing = client.load_jsonl(out_path)
        seen = {r["text"] for r in existing if r.get("label") == (0 if prompt_type == "daily" else 1)}
        history = list(seen)

        rounds_needed = (target_per_type - len(seen) + num - 1) // num
        for i in range(rounds_needed):
            try:
                data = client.generate_data(num, prompt_type, history)
            except ValueError as e:
                client.logger.error(f"生成数据解析/效验失败: {e}")
                continue
            except Exception as e:
                client.logger.error(f"LLM 调用失败: {e}")
                continue

            # 去重后再写
            fresh = [d for d in data if d["text"] not in seen]
            label = 0 if prompt_type == "daily" else 1
            for d in fresh:
                seen.add(d["text"])
                history.append(d["text"])
                d["label"] = label
            n = client.append_jsonl(out_path, fresh)
            client.logger.info(
                f"[{prompt_type}] 第 {i+1}/{rounds_needed} 轮，"
                f"新增 {n} 条 (重复 {len(data) - n} 条)，累积 {len(seen)}/{target_per_type}"
            )

            if len(seen) >= target_per_type:
                break

# TODO 测试函数
if __name__ == '__main__':
    main()

