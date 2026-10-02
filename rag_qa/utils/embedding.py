import os
import sys
import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
model_dir = os.path.join(rag_qa_dir, 'models')
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# 向量工具类
class VectorTools:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=os.path.join(model_dir, 'bge-m3'),
            use_fp16=(self.device == 'cuda'),
            batch_size=32
        )
        self.logger = logger

    @staticmethod
    def _encode(vector: dict):
        """
        函数功能：内置函数，对向量化进行统一的处理
        :param vector: 向量字典，包含稠密向量与稀疏向量
        :return: 稠密向量 和 稀疏向量
        """
        dense_vector = [vec.astype('float32') for vec in vector['dense']]
        sparse_vector = vector['sparse']
        sparse_list = []
        for i, row in enumerate(sparse_vector):
            sparse_indices = row.indices.tolist()
            sparse_values = row.data.tolist()
            sparse_dict = {}
            for idx, values in zip(sparse_indices, sparse_values):
                sparse_dict[idx] = values
            sparse_list.append(sparse_dict)
        return dense_vector, sparse_list

    def encode_document(self, document: list):
        """
        函数功能：对文档进行批量向量化
        :param document: 需要向量化的文档列表
        :return: 稠密向量 和 稀疏向量
        """
        if not document:
            self.logger.warning("文档列表为空，跳过向量化")
            return [], []
        try:
            document_vector = self.embedding_function.encode_documents(document)
            dense_vector, sparse_vector = self._encode(document_vector)
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"文档向量化失败: {e}")
            raise

    def encode_query(self, query: str):
        """
        函数功能：对用户问题进行向量化
        :param query: 用户问题
        :return: 稠密向量 和 稀疏向量
        """
        if not query:
            self.logger.warning("文本为空，跳过向量化")
            return [], []
        try:
            query_vector = self.embedding_function.encode_queries([query])
            dense_vector, sparse_vector = self._encode(query_vector)
            return dense_vector[0], sparse_vector[0]
        except Exception as e:
            self.logger.error(f"文本向量化失败: {e}")
            raise



# TODO 测试函数
if __name__ == '__main__':
    # 1. 测试文本向量化
    document = ["this is a test", "this is another test", "this is yet another test"]
    # {'dense': [
    #       array([-0.01884 ,  0.02014 , -0.0468  , ..., -0.01297 ,  0.001735, 0.02133 ], shape=(1024,), dtype=float16),
    #       array([-0.01807  ,  0.02457  , -0.05276  , ..., -0.01888  , -0.0003164, 0.03238  ], shape=(1024,), dtype=float16),
    #       array([-0.02129 ,  0.03088 , -0.05038 , ..., -0.01892 , -0.003077, 0.04443 ], shape=(1024,), dtype=float16)],
    #  'sparse': <Compressed Sparse Row sparse array of dtype 'float64' with 13 stored elements and shape (3, 250002)>}
    encode = VectorTools()
    dense, sparse = encode.encode_document(document)
    print(f'稠密向量：\n{dense}')
    print(f'稀疏向量：\n{sparse}')
    print('-' * 30)

    # 2. 测试问题向量化
    query = "this is a test"
    encode = VectorTools()
    dense, sparse = encode.encode_query(query)
    print(f'稠密向量：\n{dense}')
    print(f'稀疏向量：\n{sparse}')


