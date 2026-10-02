import os
import sys
import numpy as np
import pymysql
from rank_bm25 import BM25Okapi
from mysql_qa.db.mysql_client import MySQLClient

# 将项目根目录添加到系统路径，目的是为了让Python能找到项目中的模块
current_dir: str = os.path.dirname(os.path.abspath(__file__))
mysql_qa_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(mysql_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from mysql_qa.utils.preprocess import preprocess_text
from base import config, setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# BM25搜索类
class BM25Search:
    def __init__(self):
        self.logger = logger                    # 日志
        self.threshold = config.THRESHOLD       # 阈值
        self.mysql_client = MySQLClient()       # MySQL客户端
        self.bm25 = None                        # 初始化BM25
        self.mapping_table = None               # 初始化Mysql映射表
        self._init_data()                       # 内置函数，初始化属性

    def _init_data(self) -> None:
        """
        函数功能：初始化映射表、原始数据表、BM25
        :return: None
        """
        self.mapping_table = []
        original_data = []

        query_qa = "select id, question from law_qa"
        try:
            self.mysql_client.cursor.execute(query_qa)
            for row in self.mysql_client.cursor.fetchall():
                idx, question = row["id"], row["question"]
                self.mapping_table.append(("law_qa", idx))
                original_data.append(question)
            self.logger.info(f"MySQL检索成功，映射表与数据表初始化完成")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL检索异常，映射表与数据表初始化异常: {e}")
            raise

        try:
            self.bm25 = BM25Okapi([preprocess_text(text) for text in original_data])
        except Exception as e:
            self.logger.error(f"BM25初始化异常: {e}")
            raise

    @staticmethod
    def _softmax(scores: list) -> list:
        """
        函数功能：对分数进行softmax处理
        :param scores: 原始分数列表
        :return: 概率列表
        """
        scores = np.array(scores)
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    def search(self, query: str) -> str | None:
        """
        函数功能：根据用户问题进行bm25检索
        :param query: 用户问题
        :return: 检索结果
        """
        self.logger.info("开始进行bm25检索")
        # 检查提问内容的有效性
        if not query or not isinstance(query, str):
            self.logger.error(f"无效查询：查询结果为空或者非字符串类型 {query}")
            return None

        query_word = preprocess_text(query)
        scores = self.bm25.get_scores(query_word)
        scores = self._softmax(scores)

        # 检测最大分数是否超过阈值
        max_index = np.argmax(scores)
        max_score = scores[max_index]
        if max_score >= self.threshold:
            mapping = self.mapping_table[max_index]
            answer = self.mysql_client.get_chunks_by_ids([mapping])
            self.logger.info(f"查询到满足阈值的结果: scores: {max_score: .4f}, answer: {answer}")
            return answer[0]['answer']
        else:
            self.logger.info(f"未查询到满足阈值的结果: {query}")
            return None

# TODO 测试函数
if __name__ == '__main__':
    test_query1 = "网络主播为公司带货，双方是否存在劳动关系？"
    test_query2 = "离婚案件中，孩子选择跟随生活的一方条件比另一方差很多，应如何处理？"
    test_query3 = ""
    test_query4 = ["非字符串样式"]
    query = [test_query1, test_query2, test_query3, test_query4]

    bm25_search = BM25Search()
    for q in query:
        result = bm25_search.search(q)
        print(result)



