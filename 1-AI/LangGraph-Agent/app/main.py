"""FastAPI：用户端对话 + 坐席端工单审批。"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy import text

import asyncio
from collections import defaultdict

_thread_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

from app.config import DATABASE_URL, engine
from app.db_init import init_db
from app.graph import build_graph
from app.logger import get_logger, setup_logging
from app.ticket_store import can_approve

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERT_NODES = {"order_expert", "after_sale_expert", "knowledge_expert", "human_handoff"}
AGENT_TOKENS = {"dev-agent-token": "agent", "dev-supervisor-token": "supervisor",
                "dev-finance-token": "finance"}

state = {}
log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("服务启动：初始化数据库与检查点")
    await init_db()
    pg_uri = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(pg_uri) as cp:
        await cp.setup()
        state["graph"] = build_graph(cp)
        log.info("服务就绪，监听 :8000")
        yield
    log.info("服务关闭")


api = FastAPI(lifespan=lifespan)


class ChatReq(BaseModel):
    user_id: str
    session_id: str
    message: str


class ApproveReq(BaseModel):
    decision: str


def _cfg(uid: str, sid: str) -> dict:
    return {"configurable": {"thread_id": f"{uid}-{sid}"}, "recursion_limit": 25}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _role(token: str | None) -> str:
    r = AGENT_TOKENS.get(token or "")
    if not r:
        raise HTTPException(403, "无效令牌")
    return r


async def _stream(inputs, cfg):
    graph = state["graph"]
    yield _sse({"type": "start"})
    try:
        async for chunk, meta in graph.astream(inputs, config=cfg, stream_mode="messages"):
            if meta.get("langgraph_node") in EXPERT_NODES and chunk.content:
                yield _sse({"type": "token", "content": chunk.content})

        snap = await graph.aget_state(cfg)
        if snap.next:
            async with engine.connect() as conn:
                t = (await conn.execute(
                    text("SELECT ticket_id, required_level FROM tickets "
                         "WHERE thread_id=:th AND status='pending' "
                         "ORDER BY created_at DESC LIMIT 1"),
                    {"th": cfg["configurable"]["thread_id"]},
                )).mappings().first()
            yield _sse({"type": "ticket_created",
                        "ticket_id": t["ticket_id"] if t else "",
                        "required_level": t["required_level"] if t else "unknown",
                        "message": f"退款申请已提交，工单 {t['ticket_id'] if t else ''}，客服将在 4 小时内处理。"})
        else:
            yield _sse({"type": "done",
                        "answer": snap.values["messages"][-1].content,
                        "trace": snap.values.get("trace", [])})
    except Exception as e:
        log.exception(f"图执行异常: {e}")
        yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})


@api.post("/api/chat/stream")
async def chat_stream(req: ChatReq):
    cfg = _cfg(req.user_id, req.session_id)
    thread_id = cfg["configurable"]["thread_id"]
    log.info(f"对话请求 user={req.user_id} session={req.session_id} msg={req.message[:50]!r}")
    async def _guarded():
        async with _thread_locks[thread_id]:
            async for chunk in _stream(
                {"messages": [HumanMessage(content=req.message)],
                 "user_id": req.user_id}, cfg
            ):
                yield chunk

    return StreamingResponse(_guarded(), media_type="text/event-stream")


@api.get("/api/agent/tickets")
async def list_tickets(status: str = "pending", x_agent_token: str = Header(None)):
    role = _role(x_agent_token)
    async with engine.connect() as conn:
        rows = (await conn.execute(
            text("SELECT * FROM tickets WHERE status=:s ORDER BY created_at DESC"),
            {"s": status},
        )).mappings().all()
    tickets = []
    for r in rows:
        d = dict(r)
        d["can_approve"] = can_approve(role, d["amount"])
        d["tool_args"] = d.get("tool_args") or {}
        # datetime 序列化
        for k in ("created_at", "decided_at"):
            if d.get(k):
                d[k] = d[k].isoformat()
        tickets.append(d)
    return {"role": role, "tickets": tickets}


@api.post("/api/agent/tickets/{ticket_id}/decide")
async def decide(ticket_id: str, req: ApproveReq, x_agent_token: str = Header(None)):
    role = _role(x_agent_token)
    log.info(f"审批请求 ticket={ticket_id} decision={req.decision} role={role}")

    async with engine.begin() as conn:
        t = (await conn.execute(
            text("SELECT * FROM tickets WHERE ticket_id=:tid"), {"tid": ticket_id}
        )).mappings().first()
        if not t:
            log.warning(f"审批失败：工单不存在 ticket={ticket_id}")
            raise HTTPException(404, "工单不存在")
        if not can_approve(role, t["amount"]):
            log.warning(f"审批被拒：角色权限不足 ticket={ticket_id} role={role}")
            raise HTTPException(403, f"角色 {role} 无权审批 ¥{t['amount']}")

        new_status = "approved" if req.decision == "approve" else "rejected"
        result = await conn.execute(text("""
            UPDATE tickets SET status=:st, decided_at=now(), decided_by=:by
            WHERE ticket_id=:tid AND status='pending'
        """), {"st": new_status, "by": role, "tid": ticket_id})
        if result.rowcount == 0:
            log.warning(f"审批冲突：工单已被处理 ticket={ticket_id}")
            raise HTTPException(409, "工单已被处理")
        thread_id = t["thread_id"]

    log.info(f"工单状态已更新 ticket={ticket_id} status={new_status} by={role}")

    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    await state["graph"].ainvoke(
        Command(resume={"decision": req.decision, "approver": role, "ticket_id": ticket_id}),
        config=cfg,
    )
    snap = await state["graph"].aget_state(cfg)
    log.info(f"图已恢复执行 ticket={ticket_id} thread={thread_id}")
    return {"ticket_id": ticket_id, "status": new_status,
            "final_answer": snap.values["messages"][-1].content}


_static = BASE_DIR / "static"
if _static.exists():
    api.mount("/agent", StaticFiles(directory=str(_static), html=True), name="agent")
    api.mount("/", StaticFiles(directory=str(_static), html=True), name="static")