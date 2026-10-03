import os, sys

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger, config
from rag_qa.db.milvus_client import MilvusClientSystem
from rag_qa.llm.llm_client import LLMClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# LawChain项目统一入口类
class LawChainClient:
    def __init__(self, llm=None):
        self.config = config
        self.redis_client = RedisClient()
        self.mysql_client = MySQLClient()
        self.milvus_client = MilvusClientSystem()
        self.llm_client = LLMClient(llm)
        self.bm25_search = BM25Search()

    def search(self, question: str):
        """
        函数功能：根据问题进行查询，步骤：缓存查询 -> bm25关键字查询Mysql -> milvus相似度查询 -> llm生成答案
        :param question: 用户问题
        :return: llm流式答案
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
        # 使用LLM生成优化后的问题
        optimizer_query = self.llm_client.query_generate(question)
        # 使用Milvus进行相似度查询
        case_chunk, clause_chunk = self.milvus_client.search(optimizer_query)

        return self._stream_and_cache(question, case_chunk, clause_chunk)

    def _stream_and_cache(self, question: str, case_chunk: list, clause_chunk: list):
        """
        函数功能：流式返回答案并缓存，步骤：LLM生成答案 -> 流式返回答案 -> 缓存答案
        :param question: 问题
        :param case_chunk: 参考案例
        :param clause_chunk: 法律依据
        :yield: 逐块产出答案文本
        """
        chunks = []
        try:
            for chunk in self.llm_client.generate(question, case_chunk, clause_chunk):
                chunks.append(chunk)
                yield chunk
        except Exception as e:
            logger.error(f"LLM生成答案过程中发生错误: {e}")
            error_msg = f"抱歉，系统出现问题。如有疑问，请拨打人工客服，电话：{self.config.APP_PHONE}"
            chunks.append(error_msg)
            yield error_msg
        finally:
            # 生成器被消费完毕 (或异常中断) 后，缓存完整答案
            full_answer = "".join(chunks)
            if full_answer:
                logger.info("LLM流式生成完毕，缓存完整答案")
                self.redis_client.set_question(question, full_answer)
                self.mysql_client.insert_data([{"question": question, "answer": full_answer}])

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
            result = system.search(query)
            if isinstance(result, str):
                print(result)
            else:
                for chunk in result:
                    print(chunk, end="", flush=True)
                print()
    except Exception as e:
        logger.error(f"系统错误: {e}")
    finally:
        system.close()

if __name__ == '__main__':
    main()