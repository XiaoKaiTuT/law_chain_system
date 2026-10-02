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
from fastapi.templating import Jinja2Templates      # 模板引擎，用来渲染HTML页面
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

import asyncio
import json

logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

HEARTBEAT_INTERVAL = 30  # 心跳间隔时间，单位为秒
HEARTBEAT_MAX_MISS = 2   # 连续丢失多少个 pong 就判定连接已死

law_chain = None

@asynccontextmanager
async def websocket_manager(app: FastAPI):
    global law_chain
    law_chain = LawChainClient()
    logger.info("LawChainClient 初始化完成")
    yield
    law_chain.close()
    logger.info("LawChainClient 关闭完成")

app = FastAPI(title="LawChain 法律问答系统", lifespan=websocket_manager)
# 创建模板引擎，模板在templates目录下
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
# 挂载静态资源目录：否则 chat.html 里的 /static/nahida.jpg 会 404
app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")



async def _send_json(websocket: WebSocket, payload: dict) -> bool:
    """统一的消息发送出口，发送失败（连接已断开）返回 False"""
    try:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


class ConnectionState:
    """
    连接存活状态。

    关键约束：Starlette 的 WebSocket 只能有一个协程在读。
    因此心跳不自己去 receive（否则会和 receive_loop 抢消息，甚至把用户提问吃掉），
    而是由 receive_loop 在收到任何消息时刷新 last_seen —— 收到 pong 就等于拿到了
    带内（in-band）心跳应答，收不到任何帧就说明链路已死。
    """

    def __init__(self) -> None:
        self.last_seen = asyncio.get_event_loop().time()
        self.awaiting_pong = False

    def touch(self) -> None:
        self.last_seen = asyncio.get_event_loop().time()
        self.awaiting_pong = False

    def elapsed(self) -> float:
        return asyncio.get_event_loop().time() - self.last_seen


async def _send_heartbeat(websocket: WebSocket, state: ConnectionState) -> None:
    """
    服务端心跳循环：每隔 HEARTBEAT_INTERVAL 秒主动 ping 一次，并检查是否收到过任何帧。
    连续 HEARTBEAT_MAX_MISS 个窗口都没有收到任何消息（含 pong）即判定为死连接并主动关闭。
    本协程只负责发送与计时，从不读取 WebSocket，因此不会与 receive_loop 争抢消息。
    """
    missed = 0
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        if not await _send_json(websocket, {"type": "ping", "ts": int(asyncio.get_running_loop().time())}):
            logger.warning("心跳发送失败，连接疑似已断开")
            return
        state.awaiting_pong = True

        # 给出 HEARTBEAT_INTERVAL 的应答窗口，再回看这段时间里有没有任何帧到达
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        if not state.awaiting_pong:
            missed = 0
            continue

        missed += 1
        logger.warning(f"心跳无响应，第 {missed}/{HEARTBEAT_MAX_MISS} 次")
        if missed >= HEARTBEAT_MAX_MISS:
            logger.warning("心跳连续超时，主动关闭无效连接")
            try:
                await websocket.close(code=1001)
            except (RuntimeError, WebSocketDisconnect):
                # 连接可能已被对端先行关闭，此处无需处理
                pass
            return


async def _handle_question(websocket: WebSocket, question: str) -> None:
    """业务处理：交给线程池执行同步检索链，避免阻塞事件循环（否则心跳会被拖死）"""
    question = (question or "").strip()
    if not question:
        await _send_json(websocket, {"type": "answer", "text": "消息为空，请重新输入有效问题 QAQ"})
        return
    logger.info(f"开始处理用户问题：{question}")
    try:
        # LawChainClient.search 内部包含 Redis / MySQL / Milvus / LLM 同步阻塞调用，
        # 必须放到线程池，保证心跳协程在此期间仍能正常工作
        answer = await asyncio.to_thread(law_chain.search, question)
    except Exception as e:
        logger.error(f"检索链处理异常：{e}")
        await _send_json(websocket, {"type": "error", "text": f"检索出错了 QAQ：{e}"})
        return
    await _send_json(websocket, {"type": "answer", "text": answer if answer else "抱歉，没有查询到相关内容呢 QAQ"})


async def _receive_loop(websocket: WebSocket, state: ConnectionState) -> None:
    """
    客户端消息循环：作为唯一的 WebSocket 读取者，统一处理 question / ping / pong 三类消息。
    每收到一帧都刷新 state，心跳据此判定链路是否存活。
    为兼容旧客户端，收到非 JSON 纯文本时按普通问题处理。
    """
    while True:
        raw = await websocket.receive_text()
        state.touch()   # 收到任何帧（含 pong）都算链路存活
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = {"type": "question", "text": raw}
        if not isinstance(payload, dict):
            payload = {"type": "question", "text": raw}

        msg_type = payload.get("type", "question")
        if msg_type == "ping":
            # 客户端主动探活，立即回 pong
            await _send_json(websocket, {"type": "pong"})
        elif msg_type == "pong":
            # 心跳应答，已在上面通过 state.touch() 记账，无需额外处理
            continue
        else:
            text = payload.get("text") or payload.get("question") or ""
            logger.info(f'收到用户信息: {text}')
            await _handle_question(websocket, text)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # 使用模板引擎渲染 chat.html，并传入参数
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health")
async def health():
    """探活接口，便于部署时检查服务是否存活"""
    return {"status": "ok", "heartbeat_interval": HEARTBEAT_INTERVAL}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("用户建立连接")
    state = ConnectionState()
    heartbeat_task = asyncio.create_task(_send_heartbeat(websocket, state))
    try:
        await _receive_loop(websocket, state)
    except WebSocketDisconnect:
        logger.info("用户断开连接")
    except Exception as e:
        logger.error(f"WebSocket 异常：{e}")
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
