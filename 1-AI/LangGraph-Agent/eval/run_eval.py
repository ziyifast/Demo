"""python -m eval.run_eval"""
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.graph import graph

CASES = [
    {"input": "帮我查订单 1001", "intent": "order", "contains": "已发货"},
    {"input": "订单 1002 怎么还没发货", "intent": "order", "contains": "待支付"},
    {"input": "我要申请退款，订单 1003", "intent": "after_sale",
     "interrupt": True, "contains": "退款"},
    {"input": "我那个 1299 块的东西签收了想退掉", "intent": "after_sale"},
    {"input": "七天无理由怎么算", "intent": "knowledge", "contains": "七天"},
    {"input": "退款多久到账", "intent": "knowledge", "contains": "到账"},
    {"input": "我要转人工投诉", "intent": "human", "contains": "工单号"},
]


def run_case(case: dict) -> tuple[bool, str]:
    cfg = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:6]}"},
           "recursion_limit": 25}
    graph.invoke({"messages": [HumanMessage(content=case["input"])],
                  "user_id": "eval"}, config=cfg)

    state = graph.get_state(cfg)
    if state.next:
        if not case.get("interrupt"):
            return False, "不应中断却中断了"
        graph.invoke(Command(resume="approve"), config=cfg)
        state = graph.get_state(cfg)

    values = state.values
    if values.get("intent") != case["intent"]:
        return False, f"intent={values.get('intent')}，期望 {case['intent']}"
    answer = values["messages"][-1].content
    if "contains" in case and case["contains"] not in answer:
        return False, f"回答缺少「{case['contains']}」：{answer[:60]}"
    return True, "ok"


if __name__ == "__main__":
    passed = 0
    for i, c in enumerate(CASES, 1):
        ok, msg = run_case(c)
        print(f"[{i}] {'PASS' if ok else 'FAIL'} {c['input']}  ({msg})")
        passed += ok
    print(f"\n通过 {passed}/{len(CASES)}")