"""图组装。checkpointer 由外部注入。"""
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from app.nodes import (after_route, after_sale_agent, approval_gate,
                       create_refund_ticket_node, human_handoff_node,
                       knowledge_agent, order_agent, router_node)
from app.state import SupportState
from app.tools import (create_refund_request, query_after_sale, query_order,
                       query_refund_ticket, search_knowledge)

order_tools = ToolNode([query_order], handle_tool_errors=True)
after_sale_read_tools = ToolNode([query_after_sale, query_refund_ticket],
                                 handle_tool_errors=True)
after_sale_write_tools = ToolNode([create_refund_request], handle_tool_errors=True)
knowledge_tools = ToolNode([search_knowledge], handle_tool_errors=True)


def route_order(state):
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


def route_after_sale(state):
    tcs = getattr(state["messages"][-1], "tool_calls", None) or []
    if not tcs:
        return END
    if any(tc["name"] == "create_refund_request" for tc in tcs):
        return "create_refund_ticket"
    return "after_sale_read_tools"


def route_approval(state):
    return ("after_sale_write_tools" if state.get("approval_decision") == "approved"
            else "after_sale_expert")


def route_knowledge(state):
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


def build_graph(checkpointer):
    builder = StateGraph(SupportState)
    retry = RetryPolicy(max_attempts=3, initial_interval=0.5,
                        backoff_factor=2.0, jitter=True)

    builder.add_node("router", router_node, retry_policy=retry)
    builder.add_node("order_expert", order_agent, retry_policy=retry)
    builder.add_node("after_sale_expert", after_sale_agent, retry_policy=retry)
    builder.add_node("knowledge_expert", knowledge_agent, retry_policy=retry)
    builder.add_node("human_handoff", human_handoff_node)
    builder.add_node("order_tools", order_tools)
    builder.add_node("after_sale_read_tools", after_sale_read_tools)
    builder.add_node("after_sale_write_tools", after_sale_write_tools)
    builder.add_node("knowledge_tools", knowledge_tools)
    builder.add_node("create_refund_ticket", create_refund_ticket_node)
    builder.add_node("approval_gate", approval_gate)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", after_route, {
        "order": "order_expert", "after_sale": "after_sale_expert",
        "knowledge": "knowledge_expert", "human": "human_handoff",
    })
    builder.add_conditional_edges("order_expert", route_order,
                                  {"tools": "order_tools", END: END})
    builder.add_edge("order_tools", "order_expert")

    builder.add_conditional_edges("after_sale_expert", route_after_sale, {
        "create_refund_ticket": "create_refund_ticket",
        "after_sale_read_tools": "after_sale_read_tools",
        END: END,
    })
    builder.add_edge("create_refund_ticket", "approval_gate")
    builder.add_conditional_edges("approval_gate", route_approval, {
        "after_sale_write_tools": "after_sale_write_tools",
        "after_sale_expert": "after_sale_expert",
    })
    builder.add_edge("after_sale_read_tools", "after_sale_expert")
    builder.add_edge("after_sale_write_tools", "after_sale_expert")

    builder.add_conditional_edges("knowledge_expert", route_knowledge,
                                  {"tools": "knowledge_tools", END: END})
    builder.add_edge("knowledge_tools", "knowledge_expert")
    builder.add_edge("human_handoff", END)

    return builder.compile(checkpointer=checkpointer)