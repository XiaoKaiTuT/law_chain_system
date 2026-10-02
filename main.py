import os, sys

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
from rag_qa.db.milvus_client import MilvusClientSystem
from rag_qa.llm.llm_client import LLMClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# LawChain项目统一入口类
class LawChainClient:
    def __init__(self, llm=None):
        self.redis_client = RedisClient()
        self.mysql_client = MySQLClient()
        self.milvus_client = MilvusClientSystem()
        self.llm_client = LLMClient(llm)
        self.bm25_search = BM25Search()

    def search(self, question) -> str:
        """
        函数功能：根据问题进行查询，步骤：缓存查询 -> bm25关键字查询Mysql -> milvus相似度查询 -> llm生成答案
        :param question: 用户问题
        :return: llm生成答案
        """
        if not question:
            return "消息为空，请重新输入有效问题 QAQ"
        logger.info(f"开始进行问题查询，问题为：{question}")
        answer = self.redis_client.get_answer(question)
        if answer:
            logger.info(f"从缓存中获取问题答案，答案为：{answer}")
            return answer
        answer = self.bm25_search.search(question)
        if answer:
            logger.info(f"从Mysql中获取问题答案，答案为：{answer}")
            return answer
        context = self.milvus_client.search(question)
        answer = self.llm_client.generate(question, context)
        logger.info(f"从Milvus中获取问题答案，答案为：{answer}")

        # 为缓存增加QA问答对
        if answer:
            self.redis_client.set_question(question, answer)

        return answer

    def close(self) -> None:
        self.redis_client.close()
        self.mysql_client.close()
        self.milvus_client.close()

def main():
    system = LawChainClient()
    logger.info("LawChain Q&A系统启动")
    try:
        print("\n欢迎使用LawChain Q&A系统")
        print("请输入问题，输入'exit'退出系统")
        while True:
            query = input("请输入问题：\n")
            if query == "exit":
                logger.info("退出LawChain Q&A系统")
                print("感谢使用LawChain Q&A系统")
                break
            answer = system.search(query)
            print(answer)
    except Exception as e:
        logger.error(f"系统错误: {e}")
    finally:
        system.close()

if __name__ == '__main__':
    main()