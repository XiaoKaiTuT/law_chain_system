import os
import sys
import hashlib
from pymilvus import MilvusClient, MilvusException, DataType, AnnSearchRequest, WeightedRanker
# LoadState 在 pymilvus 2.5.4 未从顶层导出，需从 client.types 导入
from pymilvus.client.types import LoadState

current_dir: str = os.path.dirname(os.path.abspath(__file__))
rag_qa_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(rag_qa_dir)
model_dir = os.path.join(rag_qa_dir, 'models')
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import config, setup_logger
from rag_qa.utils.embedding import VectorTools
from mysql_qa.db.mysql_client import MySQLClient
from rag_qa.utils.reranker import RerankerTools
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# Milvus客户端类
class MilvusClientSystem:
    def __init__(self):
        self.logger = logger
        self.mysql_client = MySQLClient()
        self.vector_tools = VectorTools()
        self.reranker_tool = RerankerTools()
        self.dense_dim = config.VECTOR_DIM
        self.database = config.MILVUS_DATA_NAME
        self.collection_case = config.MILVUS_COLLECTION_CASES
        self.collection_clause = config.MILVUS_COLLECTION_ARTICLES
        try:
            self.client = MilvusClient(
                uri=f'http://{config.MILVUS_HOST}:{config.MILVUS_PORT}',
            )
            self.logger.info("Milvus 连接成功")
        except MilvusException as e:
            self.logger.error(f"Milvus 连接异常: {e}")
            raise
        self._init_milvus()
        self._init_data()

    def _init_milvus(self) -> None:
        """
        函数作用：内置函数，初始化Milvus
        :return: None
        """
        try:
            # 创建数据库
            databases = self.client.list_databases()
            if self.database not in databases:
                self.client.create_database(self.database)
                self.logger.info(f"Milvus数据库 {self.database} 创建成功")
            self.client.use_database(self.database)
            self.logger.info(f"使用Milvus数据库 {self.database}")
        except MilvusException as e:
            self.logger.error(f"Milvus 数据库创建异常: {e}")
            raise

        try:
            # 创建集合
            if not self.client.has_collection(self.collection_case):
                schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, max_length=32)
                schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=300)
                schema.add_field(field_name='text_content', datatype=DataType.VARCHAR, max_length=4000)
                schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
                schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
                schema.add_field(field_name='parent_content', datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=300)
                schema.add_field(field_name='timestamp', datatype=DataType.VARCHAR, max_length=50)

                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name='dense_vector',
                    index_name='dense_index',
                    index_type='FLAT',
                    metric_type='COSINE',
                )
                index_params.add_index(
                    field_name='sparse_vector',
                    index_name='sparse_index',
                    index_type='SPARSE_INVERTED_INDEX',
                    metric_type='IP',
                )

                self.client.create_collection(
                    collection_name=self.collection_case,
                    schema=schema,
                    index_params=index_params,
                )
                self.logger.info(f"Milvus集合 {self.collection_case} 创建成功")

            if not self.client.has_collection(self.collection_clause):
                schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, max_length=32)
                schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=400)
                schema.add_field(field_name='text_content', datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
                schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
                schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=150)
                schema.add_field(field_name='timestamp', datatype=DataType.VARCHAR, max_length=50)

                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name='dense_vector',
                    index_name='dense_index',
                    index_type='FLAT',
                    metric_type='COSINE',
                )
                index_params.add_index(
                    field_name='sparse_vector',
                    index_name='sparse_index',
                    index_type='SPARSE_INVERTED_INDEX',
                    metric_type='IP',
                )

                self.client.create_collection(
                    collection_name=self.collection_clause,
                    schema=schema,
                    index_params=index_params,
                )
                self.logger.info(f"Milvus集合 {self.collection_clause} 创建成功")
        except MilvusException as e:
            self.logger.error(f"Milvus 集合创建异常: {e}")
            raise

        try:
            self.client.load_collection(self.collection_clause)
            self.client.load_collection(self.collection_case)
            self.logger.info(f'Milvus 集合加载成功')
        except MilvusException as e:
            self.logger.error(f"Milvus 集合加载异常: {e}")
            raise

    def _init_data(self) -> None:
        """
        函数作用：内置函数，初始化数据库数据
        :return: None
        """
        # 子文本拼接成父文本(案情)
        select_parent_cases = """
            select
                source,
                case_no,
                group_concat(text_content order by id separator '\n') as parent_content
            from
                law_chunk
            where
                doc_type = 'case' and section in ('基本案情', '案件焦点', '法院裁判要旨')
            group by
                source, case_no;
        """

        # 案情子文本
        select_child_cases = """
            select 
                id, source, doc_type, text_content, case_no, case_title, created_at
            from 
                law_chunk
            where
                doc_type = 'case'
        """

        # 条款
        select_clauses = """
            select 
                id, source, doc_type, text_content, article_no, law_name, created_at
            from 
                law_chunk
            where
                doc_type = 'law'
        """
        try:
            if int(self.client.get_collection_stats(self.collection_case)['row_count']) == 0:
                self.mysql_client.cursor.execute("set session group_concat_max_len = 65535")
                self.mysql_client.cursor.execute(select_parent_cases)
                parent_cases = self.mysql_client.cursor.fetchall()

                parent_map = {(parent["source"], parent["case_no"]): parent["parent_content"] for parent in parent_cases}
                self.mysql_client.cursor.execute(select_child_cases)
                child_cases = self.mysql_client.cursor.fetchall()
                # 案情的父文本以一对多的方式进行拼接子文本
                for child in child_cases:
                    child['parent_content'] = parent_map.get((child['source'], child['case_no']), '')
                self.insert_data(child_cases)
            if int(self.client.get_collection_stats(self.collection_clause)['row_count']) == 0:
                self.mysql_client.cursor.execute(select_clauses)
                clauses = self.mysql_client.cursor.fetchall()
                self.insert_data(clauses)
        except Exception as e:
            self.logger.error(f"MySQL 查询异常: {e}")
            raise
        self.logger.info(f"数据初始化完成")

    def insert_data(self, datas: list[dict]) -> None:
        """
        函数作用：通过MySQL获取指定数据并插入到Milvus中
        :return: None
        """
        batch_size = 200
        total = len(datas)
        self.logger.info(f'开始插入数据，总数量: {total}, 每批 {batch_size} 条')
        # 分批插入数据，原因：防止一次性插入大量数据导致内存不足
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_data = datas[start:end]
            try:
                collection = None
                text_content = [data['text_content'] for data in batch_data]
                dense_vector, sparse_vector = self.vector_tools.encode_document(text_content)
                # 判断batch_data属于案情还是条款
                if 'case_no' in batch_data[0]:
                    id = [hashlib.md5((str(data['id']) + "_" + data['case_no']).encode('utf-8')).hexdigest() for data in batch_data]
                    collection = self.collection_case
                    milvus_data = [
                        {
                            'id': id[i],
                            'source': data['source'],
                            'text_content': data['text_content'].encode('utf-8')[:4000].decode('utf-8'),
                            'dense_vector': dense_vector[i],
                            'sparse_vector': sparse_vector[i],
                            'parent_content': data['parent_content'],
                            'title': data['case_title'],
                            'timestamp': str(data['created_at'])
                        }
                        for i, data in enumerate(batch_data)
                    ]
                elif 'article_no' in batch_data[0]:
                    id = [hashlib.md5((str(data['id']) + "_" + data['article_no']).encode('utf-8')).hexdigest() for data in batch_data]
                    collection = self.collection_clause
                    milvus_data = [
                        {
                            'id': id[i],
                            'source': data['source'],
                            'text_content': data['text_content'],
                            'dense_vector': dense_vector[i],
                            'sparse_vector': sparse_vector[i],
                            'title': data['law_name'],
                            'timestamp': str(data['created_at'])
                        }
                        for i, data in enumerate(batch_data)
                    ]
                else:
                    self.logger.error("数据格式错误，请检查数据")
                    raise ValueError("数据格式错误，请检查数据")
                self.client.upsert(collection_name=collection, data=milvus_data)
            except Exception as e:
                self.logger.error(f"Milvus 插入数据异常: {e}")
                raise
        self.logger.info(f"数据插入成功")

    def search(
            self, query: str|list,
            case_top_k: int = config.CASE_RETRIEVAL_K,
            case_top_m: int = config.CASE_CANDIDATE_M,
            clause_top_k: int = config.CLAUSE_RETRIEVAL_K,
            clause_top_m: int = config.CLAUSE_CANDIDATE_M,
    ) -> tuple[list, list]:
        """
        函数作用：通过查询语句查询相似数据
        :param query:查询语句
        :param top_k:稠密向量相似度数量
        :return:查询到最相似的top_k数据
        """
        # 查询语句为空时返回空列表
        if not query:
            return [], []

        self._ensure_loaded()

        # 对于topK和topM的数量进行管控
        if case_top_k < config.CASE_RETRIEVAL_K:
            self.logger.warning(f"top_k {case_top_k} 小于配置值 {config.CASE_RETRIEVAL_K}，已设置为默认值")
            case_top_k = config.CASE_RETRIEVAL_K
        if case_top_m < config.CASE_CANDIDATE_M:
            self.logger.warning(f"top_m {case_top_m} 小于配置值 {config.CASE_CANDIDATE_M}，已设置为默认值")
            case_top_m = config.CASE_CANDIDATE_M
        if clause_top_k < config.CLAUSE_RETRIEVAL_K:
            self.logger.warning(f"top_k {clause_top_k} 小于配置值 {config.CLAUSE_RETRIEVAL_K}，已设置为默认值")
            clause_top_k = config.CLAUSE_RETRIEVAL_K
        if clause_top_m < config.CLAUSE_CANDIDATE_M:
            self.logger.warning(f"top_m {clause_top_m} 小于配置值 {config.CLAUSE_CANDIDATE_M}，已设置为默认值")
            clause_top_m = config.CLAUSE_CANDIDATE_M

        # 统一用户问题的类型为列表
        query = [query] if isinstance(query, str) else query
        num = len(query)

        # 汇总查询结果
        case_chunk = []
        clause_chunk = []

        try:
            # 对每个子问题进行查询
            for question in query:
                # 混合检索(粗排)
                query_dense, query_sparse = self.vector_tools.encode_query(question)
                # 案情
                dense_case = AnnSearchRequest(
                    data=[query_dense],
                    anns_field="dense_vector",
                    limit=case_top_k,
                    param={"metric_type": "COSINE"},
                )
                sparse_case = AnnSearchRequest(
                    data=[query_sparse],
                    anns_field="sparse_vector",
                    limit=case_top_k,
                    param={"metric_type": "IP"},
                )
                ranker = WeightedRanker(1.0, 0.7)
                results_case = self.client.hybrid_search(
                    collection_name=self.collection_case,
                    reqs=[dense_case, sparse_case],
                    ranker=ranker,
                    limit=case_top_k,
                    output_fields=['parent_content', 'title', 'source']
                )
                case_result = [hit["entity"] for hit in results_case[0]]

                # 法律条例
                dense_clause = AnnSearchRequest(
                    data=[query_dense],
                    anns_field="dense_vector",
                    limit=clause_top_k,
                    param={"metric_type": "COSINE"},
                )
                sparse_clause = AnnSearchRequest(
                    data=[query_sparse],
                    anns_field="sparse_vector",
                    limit=clause_top_k,
                    param={"metric_type": "IP"}
                )
                results_clause = self.client.hybrid_search(
                    collection_name=self.collection_clause,
                    reqs=[dense_clause, sparse_clause],
                    ranker=ranker,
                    limit=clause_top_k,
                    output_fields=['text_content', 'title', 'source']
                )
                clause_result = [hit["entity"] for hit in results_clause[0]]

                # 对查询结果去重
                case_parent = self._de_weight(case_result)
                clause_parent = self._de_weight(clause_result)

                # 重排序(细排)
                case_chunk.extend(self.reranker_tool.rerank(query=question, documents=case_parent, content_field='parent_content', top_m=max(case_top_m // num, 1)))
                clause_chunk.extend(self.reranker_tool.rerank(query=question, documents=clause_parent, content_field='text_content', top_m=max(clause_top_m // num, 1)))
            return case_chunk, clause_chunk
        except MilvusException as e:
            self.logger.error(f"Milvus 查询异常: {e}")
            return [], []

    def _de_weight(self, results: list[dict]) -> list[dict]:
        """
        函数作用：对查询结果进行去重
        :param results: 检索到的字典列表
        :return: 去重后的字典列表
        """
        seen = set()
        unique_results = []
        for item in results:
            content = None
            if "parent_content" in item:
                content = item.get("parent_content")
            elif "text_content" in item:
                content = item.get("text_content")
            if content and content not in seen:
                seen.add(content)
                unique_results.append(item)
        return unique_results

    def _ensure_loaded(self) -> None:
        """
        函数作用：确保集合已加载到内存（Milvus 重启后集合会被自动卸载）

        注意：判断加载状态必须用 get_load_state()，不能查 get_collection_stats() ——
        后者只返回 row_count 之类的统计信息，没有加载状态字段。
        """
        for collection in (self.collection_case, self.collection_clause):
            try:
                state = self.client.get_load_state(collection)["state"]
                if state != LoadState.Loaded:
                    self.logger.warning(f"集合 {collection} 当前状态 {state!r}，重新加载")
                    self.client.load_collection(collection)
                    self.logger.info(f"集合 {collection} 重新加载完成")
            except MilvusException as e:
                self.logger.error(f"集合 {collection} 加载检查异常: {e}")
            except Exception as e:
                # 兜底：检查阶段失败不应拖垮整个检索（此前 KeyError 就是这么穿透的）
                self.logger.error(f"集合 {collection} 状态检查异常: {type(e).__name__}: {e}")

    def close(self) -> None:
        """
        函数作用：关闭数据库连接
        :return: None
        """
        try:
            self.client.release_collection(self.collection_case)
            self.client.release_collection(self.collection_clause)
            self.logger.info(f"Milvus 集合释放成功")
        except MilvusException as e:
            self.logger.error(f"Milvus 集合释放异常: {e}")

        try:
            self.client.close()
            self.logger.info(f"Milvus 连接关闭成功")
        except MilvusException as e:
            self.logger.error(f"Milvus 连接关闭异常: {e}")

# TODO 测试函数
if __name__ == '__main__':
    # 1. 测试数据库连接
    milvus_client = MilvusClientSystem()

    # 2. 查询用户问题
    # query = "婚前借名购房合意不明时房屋权属认定规则"
    query = """
    婚前一方贷款买房婚后共同还贷离婚时如何分割
    婚前借名购房合意不明时房屋权属如何认定
    离婚时房产增值部分如何补偿
    """
    sub_queries = [q.strip() for q in query.split("\n") if q.strip()]
    print(sub_queries)
    result = milvus_client.search(sub_queries)
    print(result)

    # 3. 测试数据库关闭
    milvus_client.close()