import os
import sys
import json
import redis

# 将项目根目录添加到系统路径，目的是为了让Python能找到项目中的模块
current_dir: str = os.path.dirname(os.path.abspath(__file__))
mysql_qa_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(mysql_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import config, setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# 会话 TTL 配置
QA_TTL = 3600
CONVERSATION_TTL = 1800

# Redis客户端类
class RedisClient:
    def __init__(self):
        self.logger = logger
        try:
            self.client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                decode_responses=True,
            )
            self.logger.info("Redis 连接成功")
        except redis.RedisError as e:
            self.logger.error(f"Redis 连接异常: {e}")
            raise

    def _set_data(self, key: str, value: str) -> None:
        """
        函数功能：向Redis中存储数据
        :param key: key
        :param value: value
        :return: None
        """
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False), ex=QA_TTL)
            self.logger.info(f"Redis存储数据成功: {key}")
        except redis.RedisError as e:
            self.logger.error(f'Redis存储数据失败: {e}')
            raise

    def _get_data(self, key: str) -> str | None:
        """
        函数功能：根据key从Redis中获取value值
        :param key: key
        :return: value
        """
        try:
            data: str = self.client.get(key)
            self.logger.info(f"Redis获取数据成功: {key}")
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis获取数据失败: {e}')
            return None

    def set_question(self, key: str, value: str) -> None:
        """
        函数功能：处理问题和答案格式问题，并存入Redis中
        :param key: 问题
        :param value: 答案
        :return: None
        """
        key = "".join(key.strip().split())
        value = value.strip()
        self._set_data(key, value)

    def get_answer(self, key: str) -> str | None:
        """
        函数功能：处理问题格式问题，并从Redis中获取答案
        :param key: 问题
        :return: 答案
        """
        key = "".join(key.strip().split())
        return self._get_data(key)

    @staticmethod
    def _conv_key(session_id: str) -> str:
        """
        函数功能：拼会话的 Redis key
        :param session_id: 会话ID
        :return: 形如 conversation:abc123 的 key
        """
        return f"conversation:{session_id}"

    def set_conversation(self, session_id: str, history: list| None) -> None:
        """
        函数功能：把整个会话历史写入 Redis（覆盖写）
        :param session_id: 会话ID
        :param history: 历史列表，形如 [{"role":"user","content":"..."}, ...]
        :return: None
        """
        if not session_id or not isinstance(history, list) or not history:
            self.logger.warning("session_id 为空，跳过会话写入")
            return
        try:
            key = self._conv_key(session_id)
            self.client.set(
                key,
                json.dumps(history, ensure_ascii=False),
                ex=CONVERSATION_TTL,
            )
            # 每次写入都刷新保鲜期，用户一直聊就一直保留
            self.logger.info(f"会话历史写入成功: {key}，共 {len(history)} 条")
        except redis.RedisError as e:
            self.logger.error(f"会话历史写入失败: {e}")

    def get_conversation(self, session_id: str) -> list:
        """
        函数功能：读取会话历史
        :param session_id: 会话ID
        :return: 历史列表；不存在或出错时返回【空列表】，绝不返回 None
        """
        if not session_id:
            return []
        try:
            data = self.client.get(self._conv_key(session_id))
            if not data:
                return []
            history = json.loads(data)
            return history if isinstance(history, list) else []
        except (redis.RedisError, json.JSONDecodeError) as e:
            self.logger.error(f"会话历史读取失败，返回空历史: {e}")
            return []

    def delete_conversation(self, session_id: str) -> None:
        """
        函数功能：删除会话历史（用户点"新对话"或连接断开时用）
        :param session_id: 会话ID
        :return: None
        """
        if not session_id:
            return
        try:
            self.client.delete(self._conv_key(session_id))
            self.logger.info(f"会话历史已删除: {self._conv_key(session_id)}")
        except redis.RedisError as e:
            self.logger.error(f"会话历史删除失败: {e}")

    def close(self) -> None:
        """
        函数功能：关闭Redis连接
        :return: None
        """
        try:
            self.client.close()
            self.logger.info("Redis 连接关闭")
        except redis.RedisError as e:
            self.logger.error(f"Redis 连接关闭异常: {e}")

# TODO 测试函数
if __name__ == '__main__':
    # 1. 初始化Redis客户端
    redis_client = RedisClient()

    # 3. 存储格式异常的问题和答案
    # question2 = "  客户信  息是否属 于公司 的商业秘 密？  "
    # answer2 = """ 客户信息主要包括两部分，一部分是客户的名称、地址、联系方式等信息，即基础信息；另外一部分是交易习惯、意向、价格承受能力等信息，即深度信息。但该分类并不必然影响客户信息是否构成商业秘密的认定。判断客户信息是否构成商业秘密的标准，在于其是否满足法律规定的“不为公众所知悉、具有商业价值并经权利人采取相应保密措施”，即秘密性、价值性、保密性。值得注意的是，秘密性要求不为公众普遍知悉和容易获得，既不要求绝无他人知晓，也不要求他人付出足够代价仍然不能得到。客户信息的商业秘密相较于技术秘密的商业秘密存在一定特殊性：客户信息实质系可经收集获得的信息，故侵害客户信息商业秘密行为的实质通常是侵权人通过该侵权行为节省了搜集信息所需要的时间和金钱成本。因此，客户信息的商业秘密保护通常有时间限制。故尽管基础信息较之深度信息容易获取，但这仅导致基础信息秘密性的认定更困难及相应保护期限更短。如果基础信息确有商业价值、数量足够庞大，收集足够困难，其亦可能满足价值性、保密性要求，进而可被认定构成商业秘密，对此需要根据案件具体情况予以认定。 """
    # redis_client.set_question(question2, answer2)
    # print(redis_client.get_answer(question2))

    redis_client.set_conversation("123456", [{"role": "user", "content": "测试会话历史写入"}, {"role": "assistant", "content": "这是一条测试消息"}])
    history = redis_client.get_conversation("123456")
    print(history)
    redis_client.delete_conversation("123456")

    # 4. 关闭Redis连接
    redis_client.close()


