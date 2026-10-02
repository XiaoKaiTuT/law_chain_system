# 该脚本用于读取配置文件config.ini

import configparser
import os

class Config:
    def __init__(self, config_file=None):
        # 配置文件路径
        self.PROJECT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.LOG_DIR: str = os.path.join(self.PROJECT_DIR, 'logs')
        self.LAW_DOCUMENT_LOADERS_DIR: str = os.path.join(self.PROJECT_DIR, 'common', 'utils', 'law_document_loaders')
        self.LAW_TEXT_SPLITER_DIR: str = os.path.join(self.PROJECT_DIR, 'common', 'utils', 'law_text_spliter')
        self.DATA_DIR = os.path.join(self.PROJECT_DIR, 'common', 'data')
        if config_file is None:
            config_file = os.path.join(self.PROJECT_DIR, 'config.ini')

        # 读取配置文件
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(config_file, encoding='utf-8')

        # 解析mysql数据库配置
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='law_chain'))

        # 解析redis数据库配置
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        self.REDIS_DB = os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0))

        # 解析milvus数据库配置
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        self.MILVUS_PORT = int(os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback=19530)))
        self.MILVUS_DATA_NAME = os.getenv('MILVUS_DATA_NAME', self.config.get('milvus', 'data_name', fallback='law_chain'))
        self.MILVUS_COLLECTION_CASES = os.getenv('MILVUS_COLLECTION_CASES', self.config.get('milvus', 'collection_cases', fallback='law_cases'))
        self.MILVUS_COLLECTION_ARTICLES = os.getenv('MILVUS_COLLECTION_ARTICLES', self.config.get('milvus', 'collection_articles', fallback='law_clause'))

        # 解析llm配置
        self.MODEL_NAME = os.getenv('MODEL_NAME', self.config.get('llm', 'model', fallback='deepseek-flash'))
        self.DASHSCOPE_API_KEY = os.getenv('DEEPSEEK_API_KEY', self.config.get('llm', 'dashscope_api_key', fallback='YOUR_DEEPSEEK_API_KEY'))
        self.DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL', self.config.get('llm', 'dashscope_base_url', fallback='https://api.deepseek.com'))

        # 解析检索参数配置
        self.VECTOR_DIM = int(self.config.get('retrieval', 'vector_dim', fallback=1024))
        self.RETRIEVAL_K = int(self.config.get('retrieval', 'retrieval_k', fallback=5))
        self.CANDIDATE_M = int(self.config.get('retrieval', 'candidate_m', fallback=2))

        # 解析BM25搜索配置
        self.THRESHOLD = float(self.config.get('bm25', 'threshold', fallback=0.85))

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, self.config.get('logger', 'log_file', fallback='app.log'))

        # 应用配置
        self.APP_PHONE = self.config.get('app', 'phone', fallback='123-456-7890')


config = Config()

# 测试函数
if __name__ == '__main__':
    print('MySQL配置', config.MYSQL_HOST, config.MYSQL_USER, config.MYSQL_PASSWORD, config.MYSQL_DATABASE)
    print('Redis配置', config.REDIS_HOST, config.REDIS_PORT, config.REDIS_PASSWORD, config.REDIS_DB)
    print('日志文件路径', config.LOG_FILE)
    print('BM25搜索配置', config.THRESHOLD)
    print('Milvus配置', config.MILVUS_HOST, config.MILVUS_PORT, config.MILVUS_DATA_NAME, config.MILVUS_COLLECTION_CASES, config.MILVUS_COLLECTION_ARTICLES)
    print('LLM配置', config.MODEL_NAME, config.DASHSCOPE_API_KEY, config.DASHSCOPE_BASE_URL)
    print('检索参数配置', config.VECTOR_DIM, config.RETRIEVAL_K, config.CANDIDATE_M)
    print('应用配置', config.APP_PHONE)
