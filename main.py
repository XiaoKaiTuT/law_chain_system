import os, sys
import time
from typing import Iterator

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger, config
from rag_qa.db.milvus_client import MilvusClientSystem
from rag_qa.utils.legal_classifier import get_legal_classify
from rag_qa.llm.llm_client import LLMClient, InsufficientContextError, GenerationError
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# LawChain项目统一入口类
class LawChainClient:
    def __init__(self, llm=None):
        self.logger = logger
        self.config = config
        self.redis_client = RedisClient()
        self.mysql_client = MySQLClient()
        self.milvus_client = MilvusClientSystem()
        self.llm_client = LLMClient(llm)
        self.classifier = get_legal_classify()
        self.bm25_search = BM25Search()

    def search(self, question: str) -> Iterator[str] | str:
        """
        函数功能：根据问题进行查询，步骤：缓存查询 -> bm25关键字查询Mysql -> milvus相似度查询 -> llm生成答案
        :param question: 用户问题
        :return: llm流式答案
        """
        if not question:
            raise InsufficientContextError("问题为空")
        logger.info(f"开始进行问题查询，问题为：{question}")

        # Redis检索
        answer = self.redis_client.get_answer(question)
        if answer:
            logger.info(f"从缓存中获取问题答案，答案为：{answer}")
            return answer

        # 判断问题是否专业问题
        if not self._is_legal_question(question):
            logger.info(f'判断为普通问题，走通用问答: {question}')
            return self._stream_general(question)

        # MySQL + BM25检索
        answer = self.bm25_search.search(question)
        if answer:
            logger.info(f"从Mysql中获取问题答案，答案为：{answer}")
            answer += self.llm_client.rag_prompts.disclaimer(self.config.APP_PHONE)
            self.redis_client.set_question(question, answer)
            return answer

        # Milvus混合检索 + 去重 + bge-reranker重排序
        # 使用LLM生成优化后的问题
        optimizer_query = self.llm_client.query_generate(question)
        # 使用Milvus进行相似度查询
        case_chunk, clause_chunk = self.milvus_client.search(optimizer_query)
        return self._stream_and_cache(question, case_chunk, clause_chunk)

    def _is_legal_question(self, question: str) -> bool:
        """
        函数功能：BERT模型，判断问题是否是法律专业问题。
        :param question: 用户问题
        :return: 布尔值，表示问题是否是法律专业问题，True则专业问题
        """
        classify = self.classifier.classify(question)
        prob = True if classify == "法律" else False
        return prob

    def _stream_general(self, question: str):
        """
        函数目的：流式返回通用问题答案
        :param question: 用户问题
        :return:
        """
        llm_gen = self.llm_client.generate_general(question)
        chunks = []
        try:
            for chunk in llm_gen:
                chunks.append(chunk)
                yield chunk
        except GenerationError as e:
            logger.error(f"答案生成失败，不缓存: {e}")
            yield self.llm_client.rag_prompts.system_error_answer(self.config.APP_PHONE)
        else:
            full_answer = "".join(chunks)
            if full_answer:
                logger.info(f"生成完成，写入缓存 (长度 {len(full_answer)})")
                self.redis_client.set_question(question, full_answer)
        finally:
            # 客户端可能中途断开
            # 显示关闭触发 generate() 的清理，并终端前文未完成的 LLM 请求
            if hasattr(llm_gen, "close"):
                llm_gen.close()

    def _stream_and_cache(self, question: str, case_chunk: list, clause_chunk: list):
        """
        函数功能：流式返回答案并缓存，步骤：LLM生成答案 -> 流式返回答案 -> 缓存答案
        :param question: 问题
        :param case_chunk: 参考案例
        :param clause_chunk: 法律依据
        :yield: 逐块产出答案文本（法律回答的末尾会附免责声明；该免责声明不入缓存）
        """
        llm_gen = self.llm_client.generate(question, case_chunk, clause_chunk)
        chunks = []
        success = False
        try:
            for chunk in llm_gen:
                chunks.append(chunk)
                yield chunk
            success = True
        except InsufficientContextError:
            # 检索没有拿到上下文：给提示，但不缓存、不加免责声明（文案已含客服电话）
            logger.warning(f"检索未获得上下文，不缓存: {question}")
            yield self.llm_client.rag_prompts.insufficient_answer(self.config.APP_PHONE)
        except GenerationError as e:
            # LLM 生成失败：给提示，但不缓存
            logger.error(f"答案生成失败，不缓存: {e}")
            yield self.llm_client.rag_prompts.system_error_answer(self.config.APP_PHONE)
        else:
            # 正常跑完才回执行 else - 完整成功
            full_answer = "".join(chunks)
            if full_answer:
                logger.info(f"生成完成，写入缓存 (长度 {len(full_answer)})")
                self.redis_client.set_question(question, full_answer + self.llm_client.rag_prompts.disclaimer(self.config.APP_PHONE))
                self.mysql_client.insert_data([{"question": question, "answer": full_answer}])
        finally:
            # 客户端可能中途断开
            # 显示关闭触发 generate() 的清理，并终端前文未完成的 LLM 请求
            if hasattr(llm_gen, "close"):
                llm_gen.close()

        if success:
            yield self.llm_client.rag_prompts.disclaimer(self.config.APP_PHONE)

    def close(self) -> None:
        self.redis_client.close()
        self.mysql_client.close()
        self.milvus_client.close()

    def warmup(self) -> None:
        """
        函数功能：预热模型，把懒加载的模型提前加载好，避免用户的等待
        :return: None
        """
        # 预热 bge-m3 模型
        t0 = time.time()
        self.milvus_client.vector_tools.encode_query("预热")
        self.logger.info(f"预热 bge-m3 完成，耗时 {time.time() - t0:.2f}s")

        # 预热 bge-reranker 模型
        t1 = time.time()
        dummy = [{"text_content": "预热文档一"}, {"text_content": "预热文档二"}]
        self.milvus_client.reranker_tool.rerank("预热", dummy, "text_content", 1)
        logger.info(f"预热 reranker 完成, 耗时 {time.time() - t1:.2f}s")

        logger.info("模型预热全部完成")


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
            start = time.time()
            result = system.search(query)
            if isinstance(result, str):
                print(result)
            else:
                for chunk in result:
                    print(chunk, end="", flush=True)
                print()
            system.logger.info(f"耗时：{time.time() - start:.2f}")
    except Exception as e:
        logger.error(f"系统错误: {e}")
    finally:
        system.close()

if __name__ == '__main__':
    main()