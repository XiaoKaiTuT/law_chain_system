import os, sys
import time
from typing import Callable, Iterator

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

# 设置参数
MAX_HISTORY_TURNS = 4           # 最多保留最近几轮
MAX_HISTORY_CHARS = 1500        # 历史部分的总字符预算
MAX_ASSISTANT_CHARS = 400       # 单条助手回复存入历史时的上限
MAX_SESSION_MESSAGES = 40       # 单个 session 最多保留的历史条数（防止异常情况下无限增长）

# 阶段提示文案：由 LawChainClient.search() 在关键节点通过 on_stage 回调上报，
# Web 层把它转成 {"type": "stage"} 帧，前端用打字机气泡动态显示"当前在做什么"。
STAGE_RESOLVE = "正在理解您的问题…"
STAGE_CLASSIFY = "正在判断问题类型…"
STAGE_RETRIEVE = "正在检索法规与案例…"
STAGE_GENERATE = "正在生成回答…"


def _noop_stage(stage: str) -> None:
    """默认的空阶段回调（不关心阶段时使用）"""
    return None


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

    def search(
            self,
            question: str,
            history: list | None = None,
            session_id: str = "",
            on_stage: Callable[[str], None] | None = None,
    ) -> Iterator[str] | str:
        """
        函数功能：根据问题进行查询
        流程：首轮查询缓存 -> 指代消解 -> BERT 分类 -> BM25+MYSQL -> milvus混合检索 + 去重 + bge-reranker -> 生成
        :param question: 用户问题 (当前这一轮)
        :param history: 历史对话列表
        :param session_id: 会话ID
        :param on_stage: 阶段回调，用于向 Web 层上报"当前在做什么"（可为 None）
        :return: llm流式答案
        """
        if not question:
            raise InsufficientContextError("问题为空")

        on_stage = on_stage or _noop_stage

        # ① 历史只保留最近几轮（动态截断，保最新）
        history = self._trim_history(history or [])
        is_first_turn = not history
        logger.info(
            f"开始进行问题查询，问题为：{question}"
            f"（{'首轮' if is_first_turn else f'带 {len(history)} 条历史'}）"
        )

        # ② Redis检索：只对首轮生效
        if is_first_turn:
            answer = self.redis_client.get_answer(question)
            if answer:
                logger.info(f"从缓存中获取问题答案，答案为：{answer}")
                self._record_turn(history, session_id, question, answer)
                return answer

        # ③ 指代消解：将联系上下文的问题补成自包含问题
        #    （resolve_query 内部会在真正调用 LLM 时先上报阶段，首轮无历史时不上报也不调用）
        resolved = self.llm_client.resolve_query(question, history, on_stage) or question

        # ④ BERT 分类：用 [消解后] 的问题判断是否专业问题
        on_stage(STAGE_CLASSIFY)
        if not self._is_legal_question(resolved):
            logger.info(f'判断为普通问题，走通用问答: {resolved}')
            return self._stream_general(question, history, session_id, resolved)

        # ⑤ MySQL + BM25检索
        on_stage(STAGE_RETRIEVE)
        answer = self.bm25_search.search(resolved)
        if answer:
            logger.info(f"从Mysql中获取问题答案，答案为：{answer}")
            self._record_turn(history, session_id, question, answer)
            answer += self.llm_client.rag_prompts.disclaimer(self.config.APP_PHONE)
            self.redis_client.set_question(resolved, answer)
            return answer

        # ⑥ Milvus混合检索 + 去重 + bge-reranker重排序
        # 使用LLM生成优化后的问题
        optimizer_query = self.llm_client.query_generate(resolved)
        # 使用Milvus进行相似度查询
        case_chunk, clause_chunk = self.milvus_client.search(optimizer_query)

        # ⑦ 交给 LLM 流式生成
        on_stage(STAGE_GENERATE)
        return self._stream_and_cache(question, case_chunk, clause_chunk, history, session_id, resolved)

    @staticmethod
    def _history_turn_chars(turn: list) -> int:
        """
        函数功能： 一轮（user + assistant）的字符数
        :param turn: 用户和助手的一轮对话
        :return: 一轮对话的字符数
        """
        return sum(len(m.get("content") or "") for m in turn)

    def _save_history(self, session_id: str, history: list | None) -> None:
        """
        函数功能：把历史写回 Redis（含长度保护，防异常情况下无限增长）
        :param session_id: 会话ID
        :param history: 历史列表
        :return: None
        """
        if not session_id or not isinstance(history, list) or not history:
            return
        if len(history) > MAX_SESSION_MESSAGES:
            history = history[-MAX_SESSION_MESSAGES:]
        self.redis_client.set_conversation(session_id, history)

    def _record_turn(self, history: list | None, session_id: str,
                     question: str, answer: str) -> None:
        """
        函数功能：记录一轮问答到历史，并写回 Redis
                  （供 search() 里"提前 return"的两条路径调用）
        :param history: 历史列表
        :param session_id: 会话ID
        :param question: 用户问题
        :param answer: 助手回答（会按 MAX_ASSISTANT_CHARS 截断）
        :return: None
        """
        self._append_history(history, "user", question)
        self._append_history(history, "assistant", answer)
        self._save_history(session_id, history)

    def _trim_history(self, history: list) -> list:
        """
        函数功能：对历史做动态截断，保证【最新一轮一定保留】
        策略：先按轮数上限截断，再按字符预算从新到旧累加；无论如何最新一轮无条件保留。
        :param history: 完整历史列表
        :return: 截断后的历史列表（顺序不变）
        """
        if not history:
            return []

        # 按"轮"分组：一轮 = 一个 user + 一个 assistant
        turns, cur = [], []
        for msg in history:
            cur.append(msg)
            if msg.get("role") == "assistant":
                turns.append(cur)
                cur = []
        if cur:  # 末尾可能只有 user（当前问题尚未作答）
            turns.append(cur)

        # 先按轮数上限：只留最近 MAX_HISTORY_TURNS 轮
        turns = turns[-MAX_HISTORY_TURNS:]

        # 再按字符预算：从最新往回累加
        kept, used = [], 0
        for turn in reversed(turns):
            turn_len = self._history_turn_chars(turn)
            if used + turn_len > MAX_HISTORY_CHARS and kept:
                break
            used += turn_len
            kept.insert(0, turn)  # 从头部插入，保持时间顺序

        result = [m for turn in kept for m in turn]
        self.logger.info(
            f"历史截断：{len(history)} 条 → {len(result)} 条"
            f"（保留 {len(kept)} 轮 / {used} 字）"
        )
        return result

    def _append_history(self, history: list | None, role: str, content: str) -> None:
        """
        函数功能：向历史追加一条消息（助手回复会截断，防止历史膨胀）
        :param history: 历史列表（原地修改）
        :param role: user / assistant
        :param content: 内容
        :return: None
        """
        if not isinstance(history, list):
            return
        content = (content or "").strip()
        if not content:
            return
        if role == "assistant" and len(content) > MAX_ASSISTANT_CHARS:
            content = content[:MAX_ASSISTANT_CHARS] + "…"
        history.append({"role": role, "content": content})

    def _is_legal_question(self, question: str) -> bool:
        """
        函数功能：BERT模型，判断问题是否是法律专业问题。
        :param question: 用户问题
        :return: 布尔值，表示问题是否是法律专业问题，True则专业问题
        """
        classify = self.classifier.classify(question)
        prob = True if classify == "法律" else False
        return prob

    def _stream_general(
            self,
            question: str,
            history: list | None = None,
            session_id: str = "",
            resolved: str = ""
        ):
        """
        函数目的：流式返回通用问题答案
        :param question: 用户问题
        :return:
        """
        llm_gen = self.llm_client.generate_general(question, history)
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
                self.redis_client.set_question(resolved, full_answer)
                # 成功分支里记录这一轮，无需 success 标志
                self._append_history(history, "user", question)
                self._append_history(history, "assistant", full_answer)
                self._save_history(session_id, history)
        finally:
            # 客户端可能中途断开
            # 显示关闭触发 generate() 的清理，并终端前文未完成的 LLM 请求
            if hasattr(llm_gen, "close"):
                llm_gen.close()

    def _stream_and_cache(
            self,
            question: str,
            case_chunk: list,
            clause_chunk: list,
            history: list | None = None,
            session_id: str ="",
            resolved: str = "",
        ):
        """
        函数功能：流式返回答案并缓存，步骤：LLM生成答案 -> 流式返回答案 -> 缓存答案
        :param question: 问题
        :param case_chunk: 参考案例
        :param clause_chunk: 法律依据
        :yield: 逐块产出答案文本（法律回答的末尾会附免责声明；该免责声明不入缓存）
        """
        llm_gen = self.llm_client.generate(question, case_chunk, clause_chunk, history)
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
                self.redis_client.set_question(
                    resolved,
                    full_answer + self.llm_client.rag_prompts.disclaimer(self.config.APP_PHONE)
                )
                self.mysql_client.insert_data([{"question": resolved, "answer": full_answer}])
                # 记录历史
                self._append_history(history, "user", question)
                self._append_history(history, "assistant", full_answer)
                self._save_history(session_id, history)
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