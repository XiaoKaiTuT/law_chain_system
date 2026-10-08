from openai import OpenAI
from typing import Iterator
import os
import sys

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import config, setup_logger
from rag_qa.prompt.template import RAGPrompts
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# 设置参数
MAX_HISTORY_MSG_CHARS = 400         # 历史消息单条最大长度
RESOLVE_LENGTH_FACTOR = 5           # 消解结果异常长 → 判定为模型在编造

class InsufficientContextError(Exception):
    """
    函数目的：检索没有拿到可用上下文，无法生成答案
    """

class GenerationError(Exception):
    """
    函数目的：LLM调用或流式生成过程中失败
    """

# LLM客户端类
class LLMClient:
    def __init__(self, llm=None):
        self.rag_prompts = RAGPrompts()
        self.logger = logger
        self.config = config
        self.llm = None
        self._init_llm(llm)

    def _init_llm(self, llm) -> None:
        """
        函数功能：初始化LLM
        :param llm: 对接的LLM模型
        :return: None
        """
        # 如果llm参数为空，则使用配置中的默认参数
        try:
            if llm is None:
                self.llm = OpenAI(
                    api_key=self.config.DASHSCOPE_API_KEY,
                    base_url=self.config.DASHSCOPE_BASE_URL,
                )
            else:
                if not hasattr(llm, 'invoke'):
                    raise TypeError(f'存入的llm对象不存在invoke方法，不支持该类型: {type(llm)}')
                self.llm = llm
            self.logger.info("LLM初始化成功")
        except Exception as e:
            self.logger.error(f"LLM初始化失败: {e}")
            raise

    def _search_classification(self, query: str) -> str:
        """
        函数功能：根据用户问题，进行检索分类
        :param query: 用户问题
        :return: 检索方式
        """
        try:
            prompt = self.rag_prompts.classification_prompt().format(query=query)
            response = self.llm.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
            if answer.strip().lower() not in ["direct", "hyde", "subquery", "recall"]:
                self.logger.warning(f"检索分类异常：{answer}，将使用直接检索方式。")
                return "direct"
            self.logger.info(f"检索分类成功：{answer}")
            return answer.strip().lower()
        except Exception as e:
            self.logger.error(f"检索分类异常: {e}")
            return "direct"

    def query_generate(self, query: str) -> dict:
        """
        函数功能：根据用户问题，生成优化后的问题，有利于提高检索效率
        :param query: 用户问题
        :return: 字典形式：{优化后的问题，原问题}
        """
        SEARCH_PATTERN = {
            "hyde": self.rag_prompts.hyde_search(),
            "subquery": self.rag_prompts.subquery_search(),
            "recall": self.rag_prompts.recall_search(),
        }
        question = {"retrieval_queries": None, "rerank_query": query}
        try:
            classification = self._search_classification(query)
            if classification not in SEARCH_PATTERN:
                question["retrieval_queries"] = query
                return question
            prompt = SEARCH_PATTERN[classification].format(query=query)
            response = self.llm.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
            if classification == "subquery":
                question["retrieval_queries"] = result
                return question
            question["retrieval_queries"] = result
            return question
        except Exception as e:
            self.logger.error(f"生成优化问题异常: {e}")
            question["retrieval_queries"] = query
            return question

    def _format_history(self, history: list | None) -> str:
        """
        函数功能：把历史列表转成给 LLM 读的对话文本
        :param history: 历史列表，形如 [{"role":"user","content":"..."}, ...]
        :return: 形如 "用户：xxx\\n助手：xxx" 的文本；无历史时返回 "无"
        """
        # 没有历史 → 返回"无"，绝不返回字符串 "None"
        if not history:
            return "无"

        # 是列表就直接当没有历史（避免逐字符迭代崩溃）
        if not isinstance(history, list):
            if history:  # 传了非空但类型不对，记个日志
                self.logger.warning(f"history 类型异常({type(history).__name__})，按无历史处理")
            return "无"

        lines = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            # 单条过长就截断（法律回答可能上千字）
            if len(content) > MAX_HISTORY_MSG_CHARS:
                content = content[:MAX_HISTORY_MSG_CHARS] + "…"

            # role 翻译成中文，模型读起来更自然
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"助手：{content}")

        return "\n".join(lines) if lines else "无"

    def resolve_query(self, question: str, history: list | None = None,
                      on_stage=None) -> str:
        """
        函数功能：结合对话历史，把省略式追问改写为自包含的完整问题
        :param question: 用户当前问题
        :param history: 对话历史列表
        :param on_stage: 阶段回调（可为 None），仅在真正调用 LLM 时上报一次
        :return: 消解后的问题；无历史 / 失败 / 结果异常时，均回退原问题
        """
        # 没有历史 → 第 1 轮，不需要消解（省一次 LLM 调用）
        if not history:
            return question

        # 没有可用历史文本 → 同上
        history_text = self._format_history(history)
        if history_text == "无":
            return question

        # 到这一步说明确实要调用 LLM 做消解了，此时才上报阶段
        if on_stage:
            on_stage("正在理解您的问题…")

        try:
            prompt = self.rag_prompts.resolve_prompt().format(
                history=history_text, question=question,
            )
            response = self.llm.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 改写任务要忠实，别发挥
            )
            resolved = (response.choices[0].message.content or "").strip()

            # 空结果 → 回退
            if not resolved:
                self.logger.warning(f"指代消解返回空，回退原问题: {question!r}")
                return question

            # 结果异常长 → 判定为模型在编造
            limit = len(question) * RESOLVE_LENGTH_FACTOR + 30
            if len(resolved) > limit:
                self.logger.warning(
                    f"指代消解结果异常长({len(resolved)}字)疑似编造，回退原问题：{resolved!r}"
                )
                return question

            # 结果被拆成多行 → 可能拆成了多个子问题，回退
            if "\n" in resolved:
                self.logger.warning(f"指代消解返回多行，回退原问题: {resolved!r}")
                return question

            # 正常：记录前后对照，方便排查
            self.logger.info(f"指代消解: {question!r} → {resolved!r}")
            return resolved

        except Exception as e:
            # 任何异常都回退原问题 —— 消解是增强功能，失败不应中断提问
            self.logger.error(f"指代消解失败，回退原问题: {e}")
            return question

    def generate(self, query: str, case: list[dict]|None = None, legal_provision: list[dict]|None = None, history: list|None = None) -> Iterator[str]:
        """
        函数功能：根据用户问题，检索到的上下文
        :param query: 用户问题
        :param case: 案例
        :param legal_provision: 法律条文
        :param history: 历史对话
        :return: 回答
        """
        if not query or not (case or legal_provision):
            raise InsufficientContextError(
                f"检索未获得可用上下文 (case={len(case) if case else 0})，"
                f"clause={len(legal_provision) if legal_provision else 0}"
            )

        try:
            # 处理优化case与legal_provision
            case_text = self._format_context(case, "parent_content") if case else None
            provision_text = self._format_context(legal_provision, "text_content") if legal_provision else None
            # 传入到提示词模板中
            prompt = self.rag_prompts.rag_prompt().format(
                question=query,
                case=case_text,
                legal_provision=provision_text,
                history=self._format_history(history)
            )
            # 调用LLM模型
            response = self.llm.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )
            # 流式处理LLM的返回结果
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            self.logger.error(f"LLM生成异常: {e}", exc_info=True)
            raise GenerationError(f'LLM 生成失败: {e}') from e

    def generate_general(self, query: str, history: list|None = None) -> Iterator[str]:
        """
        函数功能：通用问答，不检索知识库，直接用LLM回答非法律问题
        :param query: 用户问题
        :return: 流式回答
        """
        prompt = self.rag_prompts.general_prompt().format(
            query=query,
            history=self._format_history(history)
        )
        try:
            response = self.llm.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            self.logger.error(f"通用问答生成异常: {e}", exc_info=True)
            raise GenerationError(f"通用问答生成失败: {e}") from e


    @staticmethod
    def _format_context(documents: list[dict], content_field: str) -> str:
        """
        函数功能：将列表形式的文档转换为字符串形式
        :param content_field: 文档内容字段
        :return: 格式化后的文档内容
        """
        # 每个子文档整理后的列表集合
        parts = []
        for i, doc in enumerate(documents):
            part = f'[{i}] 题目：{doc.get("title", "")} 来源：{doc.get("source", "")}\n'
            part += doc.get(content_field, "")
            parts.append(part)
        return "\n\n".join(parts)




# TODO 测试函数
if __name__ == '__main__':
    llm_client = LLMClient()
    # query = "离婚案件中，孩子选择跟随生活的一方条件比另一方差很多，应如何处理？"
    # case = "《中华人民共和国民法典》第一千零八十四条：父母与子女间的关系，不因父母离婚而消除。离婚后，子女无论由父或者母直接抚养，仍是父母双方的子女。离婚后，父母对于子女仍有抚养、教育、保护的权利和义务。"
    # print(llm_client.generate(query, case))

    # query = "离婚案件中，孩子选择跟随生活的一方条件比另一方差很多，应如何处理？"
    # print(llm_client.query_generate(query))

    # query_list = [
    #     "离婚冷静期是多久？",
    #     "甲借给乙10万没打欠条，只有微信转账记录，乙能要回来吗？",
    #     "婚前一方贷款买房，婚后共同还贷，离婚时房屋权属如何认定、共同还贷部分如何补偿、增值部分如何分割？",
    #     "我在工厂上了五年班了，厂里一直让我签劳务派遣合同但实际在正式岗位上班，上个月突然把我辞了也没给补偿，我该怎么维权？"
    # ]
    # for query in query_list:
    #     result = llm_client.query_generate(query)
    #     print(result)

    # # 用例 1：正常历史
    # print(llm_client._format_history([
    #     {"role": "user", "content": "押金不退怎么办"},
    #     {"role": "assistant", "content": "可以要求房东返还…"},
    # ]))
    #
    # # 用例 2：没有历史（第 1 轮）
    # print(llm_client._format_history(None))
    # print(llm_client._format_history([]))
    #
    # # 用例 3：某条内容为空（脏数据）
    # print(llm_client._format_history([
    #     {"role": "user", "content": "押金不退怎么办"},
    #     {"role": "assistant", "content": "   "},  # ← 空白
    #     {"role": "user", "content": "那他不给呢"},
    # ]))
    #
    # # 用例 4：某条超长
    # print(llm_client._format_history([{"role": "assistant", "content": "甲" * 500}]))
    #
    # # 无历史 → 应该原样返回且不报错
    # print(llm_client.resolve_query("那他不给呢", None))
    # print(llm_client.resolve_query("那他不给呢", []))
    #
    # # 空内容历史 → 也原样返回
    # print(llm_client.resolve_query("那他不给呢", [{"role": "user", "content": "  "}]))
    #
    # h = [{"role": "user", "content": "租房押金不退怎么办"},
    #      {"role": "assistant", "content": "可以要求房东返还押金，协商不成可以起诉。"}]
    # print(llm_client.resolve_query("那他不给呢", h))

    # 1. 不传 history（老调用方式）→ 不该报错
    for m in llm_client.generate_general("你好"):  # 应该正常流式返回
        print(m, end="")
    print()

    # 2. 传 history → 不该报错
    for m in llm_client.generate_general("那还有什么办法", history=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是 LawChain 智能小助手。"},
    ]):
        print(m, end="")
    print()

