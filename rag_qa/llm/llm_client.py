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

    def generate(self, query: str, contexts: list) -> str:
        """
        函数功能：根据用户问题，检索到的上下文
        :param query: 用户问题
        :param contexts: 上下文
        :return: 回答
        """
        if not query or not contexts:
            return f"信息不足，请联系人工客服，电话：{self.config.APP_PHONE}"

        try:
            prompt = self.rag_prompts.rag_prompt().format(context=contexts, question=query, phone=self.config.APP_PHONE)
            response = self.llm.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM生成异常: {e}")
            raise e

# TODO 测试函数
if __name__ == '__main__':
    milvus_client = MilvusClientSystem()
    llm_client = LLMClient()
    query = "离婚案件中，孩子选择跟随生活的一方条件比另一方差很多，应如何处理？"
    contexts = milvus_client.search(query)
    print(llm_client.generate(query, contexts))
