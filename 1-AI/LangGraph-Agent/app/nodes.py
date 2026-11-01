"""节点：路由、三个专家、建工单、审批闸门、转人工。"""
import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_config
from langgraph.types import interrupt
from sqlalchemy import text

from app.config import engine
from app.llm import (AFTER_SALE_PROMPT, KNOWLEDGE_PROMPT, ORDER_PROMPT,
                     ROUTER_PROMPT, get_llm)
from app.logger import get_logger
from app.state import SupportState
from app.ticket_store import required_level
from app.tools import (create_refund_request, query_after_sale, query_order,
                       query_refund_ticket, search_knowledge)

log = get_logger("graph")

order_llm = get_llm([query_order])
after_sale_llm = get_llm([query_after_sale, query_refund_ticket, create_refund_request])
knowledge_llm = get_llm([search_knowledge])
router_llm = get_llm()

SHORT_CONFIRMS = ("确认", "好的", "提交", "同意", "可以", "是", "对", "没问题")


def _needs_retry(text: str, has_tool_call: bool) -> bool:
    """是否强制重试写工具。

    只在以下两个条件同时成立时返回 True：
    1. 模型完全没调用任何工具（has_tool_call=False）
    2. 用户消息是明确的"发起新退款"或"确认提交"，且不是查询类语句
    """
    if has_tool_call:
        # 模型已经调了工具（查询或写），不管调什么，都不重试
        return False
    # 查询类语句排除
    if any(k in text for k in ("如何", "进度", "到哪", "状态", "审批结果",
                                "通过了吗", "怎么样", "查询")):
        return False
    # 明确发起新退款
    return any(k in text for k in ("申请", "提交", "办理", "我要退", "帮我退",
                                    "退掉", "不想要", "确认", "同意"))


# ---------- 路由 ----------
async def router_node(state: SupportState) -> dict:
    last = state["messages"][-1].content.strip()

    if last in SHORT_CONFIRMS and len(state["messages"]) >= 2:
        prev = state.get("intent") or "after_sale"
        log.info(f"路由：确认词继承意图 intent={prev}")
        return {"intent": prev, "trace": [f"router:inherit:{prev}"]}

    resp = await router_llm.ainvoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=last),
    ])
    intent = resp.content.strip().lower()
    if intent not in ("order", "after_sale", "knowledge", "human"):
        intent = "human"
    log.info(f"路由：intent={intent}")
    return {"intent": intent, "trace": [f"router:{intent}"]}


def after_route(state: SupportState) -> str:
    return state["intent"]


# ---------- 三个专家 ----------
async def _expert_call(state: SupportState, llm, system_prompt: str) -> dict:
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    return {"messages": [await llm.ainvoke(messages)]}


async def order_agent(state: SupportState) -> dict:
    result = await _expert_call(state, order_llm, ORDER_PROMPT)
    msg = result["messages"][0]
    log.info(f"订单专家：tool_calls={[tc['name'] for tc in (getattr(msg, 'tool_calls', None) or [])]}")
    return {**result, "trace": ["order_expert"]}


async def after_sale_agent(state: SupportState) -> dict:
    result = await _expert_call(state, after_sale_llm, AFTER_SALE_PROMPT)
    msg = result["messages"][0]

    tool_names = [tc["name"] for tc in (getattr(msg, "tool_calls", None) or [])]
    log.info(f"售后专家：tool_calls={tool_names}")

    # 工具刚执行完 → 生成总结，不重试
    last_msg = state["messages"][-1]
    if getattr(last_msg, "type", "") == "tool":
        log.info("售后专家：工具执行完毕，生成总结回复（不触发重试）")
        return {**result, "trace": ["after_sale_expert"]}

    last_user = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"), ""
    )

    if not _needs_retry(last_user, has_tool_call=bool(tool_names)):
        return {**result, "trace": ["after_sale_expert"]}

    log.warning(f"售后专家：模型未调工具且用户明确要退款，第 1 次重试 user={last_user[:40]!r}")
    correction = HumanMessage(content=(
        "你刚才没有调用 create_refund_request 工具。"
        "用户已明确要求退款，请立即调用 create_refund_request。"
        "如果用户消息或历史中已有订单号，直接用那个订单号，不要重复询问。"
    ))
    retry1 = await after_sale_llm.ainvoke(
        [SystemMessage(content=AFTER_SALE_PROMPT)] + state["messages"] + [correction]
    )
    if any(tc["name"] == "create_refund_request"
           for tc in (getattr(retry1, "tool_calls", None) or [])):
        log.info("售后专家：第 1 次重试成功调用写工具")
        return {"messages": [retry1],
                "trace": ["after_sale_expert", "after_sale:retry1"]}

    log.warning("售后专家：第 1 次重试仍失败，第 2 次重试（仅绑定写工具）")
    write_only_llm = get_llm([create_refund_request])
    retry2 = await write_only_llm.ainvoke(
        [SystemMessage(content=AFTER_SALE_PROMPT)] + state["messages"] + [correction]
    )
    return {"messages": [retry2],
            "trace": ["after_sale_expert", "after_sale:retry2_write_only"]}


async def knowledge_agent(state: SupportState) -> dict:
    result = await _expert_call(state, knowledge_llm, KNOWLEDGE_PROMPT)
    msg = result["messages"][0]
    log.info(f"知识专家：tool_calls={[tc['name'] for tc in (getattr(msg, 'tool_calls', None) or [])]}")
    return {**result, "trace": ["knowledge_expert"]}


# ---------- 独立建工单节点（不在 interrupt 重放范围） ----------
async def create_refund_ticket_node(state: SupportState) -> dict:
    """创建退款工单。执行一次后写入 checkpoint，interrupt 恢复时不会重放。

    同一 thread 已有 pending 工单则复用，避免用户重复请求产生多张待审批工单。
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    refund_tc = next(
        (tc for tc in tool_calls if tc["name"] == "create_refund_request"), None
    )
    if refund_tc is None:
        return {"trace": ["create_ticket:skip"]}

    order_id = refund_tc["args"].get("order_id", "")
    cfg = get_config()
    thread_id = cfg.get("configurable", {}).get("thread_id", "unknown")

    async with engine.connect() as conn:
        # 同 thread 已有 pending 工单则复用
        existing = (await conn.execute(
            text("SELECT ticket_id FROM tickets "
                 "WHERE thread_id = :th AND status = 'pending' LIMIT 1"),
            {"th": thread_id},
        )).mappings().first()

        if existing:
            ticket_id = existing["ticket_id"]
            log.info(f"建工单：复用已有 pending 工单 ticket={ticket_id} thread={thread_id}")
            return {"ticket_id": ticket_id, "trace": [f"create_ticket:reuse:{ticket_id}"]}

        row = (await conn.execute(
            text("SELECT amount FROM orders WHERE order_id = :oid"), {"oid": order_id}
        )).mappings().first()
        amount = float(row["amount"]) if row else 0.0
        ticket_id = "RF-" + uuid.uuid4().hex[:6].upper()
        level = required_level(amount)

        log.info(f"建工单：thread={thread_id} order={order_id} "
                 f"amount={amount} level={level} ticket={ticket_id}")
        await conn.execute(text("""
            INSERT INTO tickets
                (ticket_id, thread_id, user_id, order_id, amount, reason,
                 tool_name, tool_args, required_level, status)
            VALUES (:tid, :th, :uid, :oid, :amt, :reason, :tn,
                    CAST(:ta AS jsonb), :lvl, 'pending')
        """), {
            "tid": ticket_id, "th": thread_id,
            "uid": state.get("user_id", ""), "oid": order_id,
            "amt": amount, "reason": refund_tc["args"].get("reason", ""),
            "tn": refund_tc["name"],
            "ta": json.dumps(refund_tc["args"], ensure_ascii=False),
            "lvl": level,
        })
        await conn.commit()

        # 落库验证
        verify = (await conn.execute(
            text("SELECT ticket_id FROM tickets WHERE ticket_id = :tid"),
            {"tid": ticket_id},
        )).mappings().first()
        if not verify:
            log.error(f"建工单失败：落库验证未通过 ticket={ticket_id}")
            raise RuntimeError(f"工单 {ticket_id} 落库失败")

    log.info(f"建工单完成 ticket={ticket_id}，进入审批闸门")
    return {"ticket_id": ticket_id, "trace": [f"create_ticket:{ticket_id}"]}


# ---------- 审批闸门（只做 interrupt，不建工单） ----------
async def approval_gate(state: SupportState) -> dict:
    """暂停等审批。工单已在上一节点创建，本节点恢复时不会重复建。"""
    ticket_id = state.get("ticket_id")
    if not ticket_id:
        log.error("审批闸门：state 里没有 ticket_id，拒绝放行")
        return {"approval_decision": "rejected",
                "trace": ["approval:missing_ticket"]}

    async with engine.connect() as conn:
        row = (await conn.execute(
            text("SELECT order_id, amount, required_level FROM tickets "
                 "WHERE ticket_id = :tid"), {"tid": ticket_id}
        )).mappings().first()

    if not row:
        log.error(f"审批闸门：工单不存在 ticket={ticket_id}")
        return {"approval_decision": "rejected", "trace": ["approval:missing"]}

    order_id = row["order_id"]
    amount = float(row["amount"])
    level = row["required_level"]

    log.info(f"审批闸门：暂停 ticket={ticket_id} order={order_id} "
             f"amount={amount} level={level}")
    decision = interrupt({
        "type": "approval_required", "ticket_id": ticket_id,
        "order_id": order_id, "amount": amount, "required_level": level,
    })
    log.info(f"审批闸门：恢复 ticket={ticket_id} decision={decision}")

    if isinstance(decision, dict) and decision.get("decision") == "approve":
        return {"approval_decision": "approved",
                "trace": [f"approval:approved:{ticket_id}"]}

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    tms = [ToolMessage(content="人工审批拒绝：未执行该写操作", tool_call_id=tc["id"])
           for tc in tool_calls if tc["name"] == "create_refund_request"]
    return {"messages": tms, "approval_decision": "rejected",
            "trace": [f"approval:rejected:{ticket_id}"]}


# ---------- 转人工 ----------
async def human_handoff_node(state: SupportState) -> dict:
    ticket = "CS-" + uuid.uuid4().hex[:6].upper()
    log.info(f"转人工：工单号 {ticket}")
    return {"messages": [AIMessage(content=f"已为您转接人工客服。工单号：{ticket}")],
            "trace": ["human_handoff"]}