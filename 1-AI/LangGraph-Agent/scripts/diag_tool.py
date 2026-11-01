# scripts/diag_tool.py
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from app.llm import get_llm, AFTER_SALE_PROMPT
from app.tools import create_refund_request, query_after_sale

async def main():
    llm = get_llm([query_after_sale, create_refund_request])
    history = []

    for text in ["帮我查1003订单", "我想要退款", "确认"]:
        history.append(HumanMessage(content=text))
        resp = await llm.ainvoke([SystemMessage(content=AFTER_SALE_PROMPT)] + history)
        print(f"\n用户: {text}")
        print(f"模型: {resp.content[:60]}")
        print(f"tool_calls: {[(tc['name'], tc['args']) for tc in (resp.tool_calls or [])]}")
        history.append(resp)

        # 模拟工具返回，让下一轮有完整上下文
        for tc in (resp.tool_calls or []):
            if tc["name"] == "query_after_sale":
                fake = f"售后单 AS-001：类型=退货退款，状态=审核中，金额=¥1299.0"
            else:
                fake = f"已为订单 {tc['args'].get('order_id')} 创建退款申请"
            history.append(ToolMessage(content=fake, tool_call_id=tc["id"]))

asyncio.run(main())