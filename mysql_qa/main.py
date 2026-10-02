import os
import sys
import pandas as pd
import time

# 将项目根目录添加到系统路径，目的是为了让Python能找到项目中的模块
current_dir: str = os.path.dirname(os.path.abspath(__file__))
mysql_qa_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(mysql_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import config, setup_logger
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search
from common import spliter
from common import loader

logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# MySQL Q&A系统类
class MySQLQASystem:
    def __init__(self):
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = None
        self.logger = logger
        self._init_mysql()
        self._init_bm25()

    def _init_mysql(self) -> None:
        """
        函数功能：初始化MySQL数据库表数据
        :return: None
        """
        qa_search = "select count(*) from law_qa"
        self.mysql_client.cursor.execute(qa_search)
        qa_result = self.mysql_client.cursor.fetchone()["count(*)"]
        if qa_result == 0:
            try:
                qa_data = pd.read_excel(os.path.join(config.DATA_DIR, "qa", "法答网精选答问答题集.xlsx"))
                self.mysql_client.insert_data(qa_data.to_dict(orient="records"))
            except FileNotFoundError as e:
                self.logger.error(f"文件未找到: {e}")

        chunk_search = "select count(*) from law_chunk"
        self.mysql_client.cursor.execute(chunk_search)
        chunk_result = self.mysql_client.cursor.fetchone()["count(*)"]
        if chunk_result == 0:
            try:
                dir_path = config.DATA_DIR
                documents = loader(dir_path)
                split_result = spliter(documents)
                self.mysql_client.insert_data(split_result)
            except Exception as e:
                self.logger.error(f"数据加载或分割失败: {e}")
                raise

        self.logger.info("MySQL数据初始化完成")

    def _init_bm25(self) -> None:
        """
        函数功能：初始化BM25搜索
        :return: None
        """
        self.bm25_search = BM25Search()

    def search(self, query: str) -> list:
        start_time = time.time()
        answer = self.redis_client.get_answer(query)
        try:
            if answer:
                self.logger.info(f"从Redis缓存中获取到问题答案，耗时{time.time() - start_time:.2f}s")
                return [answer]
            answers = self.bm25_search.search(query)
            for answer in answers:
                if "question" in answer:
                    self.logger.info(f"MySQL问答对已存入Redis缓存中")
                    self.redis_client.set_question(query, answer["answer"])
            self.logger.info(f"从Mysql中获取到相关答案，耗时{time.time() - start_time:.2f}s")
            return answers
        except Exception as e:
            self.logger.error(f"搜索错误: {e}")
            return []


    def close(self) -> None:
        self.redis_client.close()
        self.mysql_client.close()

# TODO 测试函数
def main():
    system = MySQLQASystem()
    logger.info("MySQL Q&A系统启动")
    try:
        print("\n欢迎使用MySQL Q&A系统")
        print("请输入问题，输入'exit'退出系统")
        while True:
            query = input("请输入问题：\n")
            if query == "exit":
                logger.info("退出MySQL系统")
                print("感谢使用MySQL Q&A系统")
                break
            answer = system.search(query)
            print(answer)
    except Exception as e:
        logger.error(f"系统错误: {e}")
    finally:
        system.close()


if __name__ == '__main__':
    main()