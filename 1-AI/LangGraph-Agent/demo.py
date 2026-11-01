"""python demo.py  —— 跑通完整对话与人工审批流程。"""
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.graph import graph

CONFIG = {"configurable": {"thread_id": f"demo-{uuid.uuid4().hex[:6]}"},
          "recursion_limit": 25}


def chat(text: str) -> None:
    print(f"\n用户 > {text}")
    for _ in graph.stream(
        {"messages": [HumanMessage(content=text)], "user_id": "u-001"},
        config=CONFIG, stream_mode="values",
    ):
        pass

    state = graph.get_state(CONFIG)
    if state.next:                                   # 停在中断点
        pending = state.values["messages"][-1].tool_calls
        print(f"[待审批] {[(tc['name'], tc['args']) for tc in pending]}")
        answer = input("批准？(y/n): ").strip().lower()
        resume = "approve" if answer == "y" else "reject"
        for _ in graph.stream(Command(resume=resume), config=CONFIG,
                              stream_mode="values"):
            pass
        state = graph.get_state(CONFIG)

    print(f"客服 > {state.values['messages'][-1].content}")
    print(f"[trace] {' → '.join(state.values.get('trace', []))}")


if __name__ == "__main__":
    chat("帮我查一下订单 1001 的状态")
    chat("我那个 1299 块的东西签收了，想退掉")
    chat("七天无理由退货怎么算")
    chat("我要申请退款，订单 1003")