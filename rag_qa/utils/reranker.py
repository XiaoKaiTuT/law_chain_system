import os
import sys
import torch
from FlagEmbedding import FlagReranker

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from base import setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

class RerankerTools:
    def __init__(self):
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.reranker = FlagReranker(
                model_name_or_path=os.path.join(rag_qa_dir, "models", "bge-reranker-large"),
                use_fp16=(self.device == 'cuda')
            )
        except Exception as e:
            self.logger.error(f"重排序初始化错误: {e}")
            raise

    def rerank(self, query: str, documents:list[dict], content_field: str, top_m: int = 3) -> list[dict]:
        """
        函数功能：对文档进行重排序
        :param query: 需要查询的文本
        :param documents: 需要重排序的文档列表
        :param content_field: 需要重排序的文档列表中的字段名
        :param top_m: 需要返回的文档数量
        :return: 重排序后的文档列表
        """
        # 检查文档列表是否为空
        if not documents:
            return []

        # 检查文档列表长度是否小于等于需要返回的文档数量
        if len(documents) <= top_m:
            return documents

        try:
            pairs = [[query, doc.get(content_field, "")] for doc in documents]
            scores = self.reranker.compute_score(pairs)

            if isinstance(scores, (int, float)):
                scores = [scores]

            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)

            ranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

            return ranked[:top_m]

        except Exception as e:
            self.logger.error(f"重排序异常：{e}")
            return documents[:top_m]


