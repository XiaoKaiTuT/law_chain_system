import os, sys
from typing import Iterator

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
from rag_qa.db.milvus_client import MilvusClientSystem
from rag_qa.llm.llm_client import LLMClient
from rag_qa.utils.legal_classifier import get_legal_classify
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

class RAGQAClient:
    def __init__(self, llm=None):
        self.milvus_client = MilvusClientSystem()
        self.llm_client = LLMClient(llm)
        self.classify_tools = get_legal_classify()
        self.logger = logger
        self.llm = llm

    def query(self, query) -> Iterator[str] | str:
        """
        函数功能：根据用户问题和上下文，生成答案
        :param query: 用户问题
        :param context: 上下文
        :return: llm模型生成的答案
        """
        try:
            classify = self.classify_tools.classify(query)
            if classify == "日常":
                response = self.llm_client.generate_general(query)
                return response
            optimizer_query = self.llm_client.query_generate(query)
            case_chunk, clause_chunk = self.milvus_client.search(optimizer_query)
            response = self.llm_client.generate(query, case_chunk, clause_chunk)
            return response
        except Exception as e:
            self.logger.error(f"RAG Q&A查询异常: {e}")
            error_msg = "系统繁忙，请稍后重试"
            return error_msg

def main():
    # 创建RAGQAClient对象
    rag_qa_client = RAGQAClient()
    logger.info("RAG Q&A系统启动")
    # 进入系统
    try:
        print("\n欢迎使用MySQL Q&A系统")
        print("请输入问题，输入'exit'退出系统")
        while True:
            query = input("请输入问题：\n")
            if query == "exit":
                break
            result = rag_qa_client.query(query)
            if isinstance(result, str):
                print(result)
            else:
                for chunk in result:
                    print(chunk, end="", flush=True)
                print()
        logger.info("RAG Q&A系统退出")
    except Exception as e:
        logger.error(f"RAG Q&A系统异常: {e}")
    finally:
        rag_qa_client.milvus_client.close()
    print("感谢使用RAG Q&A系统")
    logger.info("RAG Q&A系统退出")

if __name__ == '__main__':
    main()