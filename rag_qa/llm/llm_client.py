from openai import OpenAI
import os
import sys

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import config, setup_logger
from rag_qa.prompt.template import RAGPrompts
from rag_qa.db.milvus_client import MilvusClientSystem
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])


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
            if response.choices[0].message.content.strip().lower() not in ["direct", "hyde", "subquery", "recall"]:
                self.logger.warning(f"检索分类异常：{response.choices[0].message.content}，将使用直接检索方式。")
                return "direct"
            self.logger.info(f"检索分类成功：{response.choices[0].message.content}")
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            self.logger.error(f"检索分类异常: {e}")
            return "direct"

    def query_generate(self, query: str) -> str|list:
        """
        函数功能：根据用户问题，生成优化后的问题，有利于提高检索效率
        :param query: 用户问题
        :return: 优化后的问题
        """
        SEARCH_PATTERN = {
            "hyde": self.rag_prompts.hyde_search(),
            "subquery": self.rag_prompts.subquery_search(),
            "recall": self.rag_prompts.recall_search(),
        }
        try:
            classification = self._search_classification(query)
            if classification not in SEARCH_PATTERN:
                return query
            prompt = SEARCH_PATTERN[classification].format(query=query)
            response = self.llm.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
            if classification == "subquery":
                return [q.strip() for q in result.strip().split("\n") if q.strip()]
            return result
        except Exception as e:
            self.logger.error(f"生成优化问题异常: {e}")
            return query

    def generate(self, query: str, case: list[dict]|None = None, legal_provision: list[dict]|None = None, history: list|None = None):
        """
        函数功能：根据用户问题，检索到的上下文
        :param query: 用户问题
        :param case: 案例
        :param legal_provision: 法律条文
        :param history: 历史对话
        :return: 回答
        """
        if not query or not (case or legal_provision):
            yield f"信息不足，无法回答。\n如有问题请联系人工客服，电话：{self.config.APP_PHONE}"
            return

        try:
            # 处理优化case与legal_provision
            case_text = self._format_context(case, "parent_content") if case else None
            provision_text = self._format_context(legal_provision, "text_content") if legal_provision else None
            # 传入到提示词模板中
            prompt = self.rag_prompts.rag_prompt().format(question=query, case=case_text, legal_provision=provision_text, history=history)
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
            yield f"\n\n上述回答仅供参考。如有疑问，请联系人工客服，电话：{self.config.APP_PHONE}"
        except Exception as e:
            self.logger.error(f"LLM生成异常: {e}")
            yield f"抱歉，系统出现问题。如有疑问，请拨打人工客服，电话：{self.config.APP_PHONE}"

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

    query_list = [
        "离婚冷静期是多久？",
        "甲借给乙10万没打欠条，只有微信转账记录，乙能要回来吗？",
        "婚前一方贷款买房，婚后共同还贷，离婚时房屋权属如何认定、共同还贷部分如何补偿、增值部分如何分割？",
        "我在工厂上了五年班了，厂里一直让我签劳务派遣合同但实际在正式岗位上班，上个月突然把我辞了也没给补偿，我该怎么维权？"
    ]
    for query in query_list:
        result = llm_client.query_generate(query)
        print(result)

