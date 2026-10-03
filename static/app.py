import os
import sys

current_dir: str = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import setup_logger
from main import LawChainClient

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

import asyncio
import inspect
import json

logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# ============================================================================
# 配置
# ============================================================================
HEARTBEAT_INTERVAL = 30      # 心跳间隔（秒）
HEARTBEAT_MAX_MISS = 2       # 连续多少个心跳窗口没收到任何帧就判定断线

# 单例：由 lifespan 创建 / 关闭
law_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化检索链，关闭时释放资源"""
    global law_chain
    logger.info("正在初始化 LawChainClient ...")
    law_chain = LawChainClient()
    logger.info("LawChainClient 初始化完成")
    try:
        yield
    finally:
        try:
            law_chain.close()
            logger.info("LawChainClient 已关闭")
        except Exception as e:  # noqa: BLE001
            logger.error(f"关闭 LawChainClient 出错：{e}")


app = FastAPI(title="LawChain 法律问答系统", lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
# 挂载静态目录：templates 与 data 都在 static/ 下
app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")


# ============================================================================
# 连接状态
# ============================================================================
class ConnState:
    """
    连接存活状态。

    Starlette 的 WebSocket 只允许一个协程读取，所以心跳【绝不】自己去 receive，
    只负责发送与计时；由接收循环在收到任何一帧时调用 touch() 刷新时间戳。
    收不到任何帧（含 pong）即说明链路已死。
    """

    def __init__(self) -> None:
        self.last_seen = asyncio.get_event_loop().time()
        self.awaiting_pong = False

    def touch(self) -> None:
        self.last_seen = asyncio.get_event_loop().time()
        self.awaiting_pong = False


# ============================================================================
# 发送
# ============================================================================
async def send_json(ws: WebSocket, payload: dict) -> bool:
    """统一发送出口；连接已断开时返回 False，不抛异常"""
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


# ============================================================================
# 心跳
# ============================================================================
async def heartbeat_loop(ws: WebSocket, state: ConnState) -> None:
    """周期性发 ping，并在窗口结束时检查这段时间里是否收到过任何帧"""
    missed = 0
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        if not await send_json(ws, {"type": "ping"}):
            logger.warning("心跳发送失败，连接疑似已断开")
            return
        state.awaiting_pong = True

        # 给出一个完整的应答窗口
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        if not state.awaiting_pong:
            missed = 0
            continue

        missed += 1
        logger.warning(f"心跳无响应 {missed}/{HEARTBEAT_MAX_MISS}")
        if missed >= HEARTBEAT_MAX_MISS:
            logger.warning("心跳连续超时，关闭无效连接")
            try:
                await ws.close(code=1001)
            except (RuntimeError, WebSocketDisconnect):
                pass
            return


# ============================================================================
# 流式回答
# ============================================================================
async def stream_answer(ws: WebSocket, result) -> None:
    """
    把 search() 的返回值以流式方式推给前端。

    search() 有两种形态：
      1. str      —— 缓存 / BM25 命中 / 空问题兜底，一次性发完
      2. 生成器    —— LLM 流式生成，逐块推送

    生成器内部是同步阻塞 IO（LLM 网络、检索链），必须在工作线程里迭代，
    否则会把事件循环连同心跳一起拖死。这里用「线程推队列 + 事件循环消费」桥接：
    事件循环只在 await queue.get() 上挂起，心跳协程仍可正常运行。
    完整迭代生成器同时会触发 main.py 里 _stream_and_cache 的 finally，完成答案回填。
    """
    # 1) 开始
    if not await send_json(ws, {"type": "answer_start"}):
        return

    # 2) 非流式：一次性发完
    if not inspect.isgenerator(result):
        text = result if result else "（没有返回内容）"
        await send_json(ws, {"type": "answer_delta", "text": text})
        await send_json(ws, {"type": "answer_end"})
        return

    # 3) 流式：线程迭代 → 队列 → 事件循环
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def pump() -> None:
        try:
            for piece in result:
                if piece:
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
        except Exception as exc:  # noqa: BLE001 —— 线程内异常需带回事件循环
            logger.error(f"流式生成异常：{exc}")
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    pump_task = asyncio.create_task(asyncio.to_thread(pump))
    try:
        while True:
            item = await queue.get()
            if item is DONE:
                break
            if isinstance(item, Exception):
                await send_json(ws, {"type": "answer_delta", "text": f"\n\n[生成中断] {item}"})
                break
            if not await send_json(ws, {"type": "answer_delta", "text": item}):
                break       # 客户端已断开
    finally:
        # 无论客户端是否还在，都等泵线程收尾，确保 main.py 的缓存回写完成
        await pump_task

    await send_json(ws, {"type": "answer_end"})


async def handle_question(ws: WebSocket, question: str) -> None:
    """处理一次提问"""
    question = (question or "").strip()
    if not question:
        await send_json(ws, {"type": "answer", "text": "消息为空，请重新输入有效问题 QAQ"})
        return

    logger.info(f"开始处理用户问题：{question}")
    try:
        # 检索链是同步阻塞的，必须放到线程池，否则心跳会被拖死
        result = await asyncio.to_thread(law_chain.search, question)
    except Exception as e:  # noqa: BLE001
        logger.error(f"检索链处理异常：{e}", exc_info=True)
        await send_json(ws, {"type": "error", "text": f"检索出错了 QAQ：{e}"})
        return

    if result is None:
        await send_json(ws, {"type": "answer", "text": "抱歉，没有查询到相关内容呢 QAQ"})
        return

    try:
        await stream_answer(ws, result)
    except (WebSocketDisconnect, RuntimeError):
        logger.info("客户端断开，停止推送")
    except Exception as e:  # noqa: BLE001
        logger.error(f"流式推送异常：{e}", exc_info=True)
        await send_json(ws, {"type": "error", "text": f"推送出错 QAQ：{e}"})


# ============================================================================
# 接收循环（唯一的读取者）
# ============================================================================
async def receive_loop(ws: WebSocket, state: ConnState) -> None:
    """
    统一处理三类消息：question / ping / pong。
    收到非 JSON 纯文本时按提问处理（兼容旧客户端）。
    """
    while True:
        raw = await ws.receive_text()
        state.touch()

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = {"type": "question", "text": raw}
        if not isinstance(payload, dict):
            payload = {"type": "question", "text": raw}

        msg_type = payload.get("type", "question")

        if msg_type == "ping":
            await send_json(ws, {"type": "pong"})
        elif msg_type == "pong":
            continue        # 已在 state.touch() 记账
        else:
            text = payload.get("text") or payload.get("question") or ""
            logger.info(f"收到用户信息: {text}")
            await handle_question(ws, text)


# ============================================================================
# 路由
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "heartbeat_interval": HEARTBEAT_INTERVAL}


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    logger.info("用户建立连接")
    state = ConnState()
    hb = asyncio.create_task(heartbeat_loop(ws, state))
    try:
        await receive_loop(ws, state)
    except WebSocketDisconnect:
        logger.info("用户断开连接")
    except Exception as e:  # noqa: BLE001
        logger.error(f"WebSocket 异常：{e}", exc_info=True)
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
