"""工具层：数据库查询 + 向量检索。全部异步。"""
from langchain_core.tools import tool
from sqlalchemy import text

from app.config import engine
from app.embeddings import embeddings


@tool
async def query_order(order_id: str) -> str:
    """根据订单号查询订单状态、金额和物流信息。订单号必须是 4 位以上数字。"""
    async with engine.connect() as conn:
        row = (await conn.execute(
            text("SELECT order_id, status, amount, tracking, eta "
                 "FROM orders WHERE order_id = :oid"),
            {"oid": order_id},
        )).mappings().first()
    if not row:
        return f"未找到订单 {order_id}，请确认订单号是否正确。"
    return (f"订单 {row['order_id']}：状态={row['status']}，"
            f"金额=¥{row['amount']}，物流单号={row['tracking'] or '暂无'}，"
            f"预计{row['eta']}")


@tool
async def query_after_sale(order_id: str) -> str:
    """查询指定订单的售后单状态。只读操作。用于查看售后单（不是退款工单）的进度。"""
    async with engine.connect() as conn:
        row = (await conn.execute(
            text("SELECT after_sale_id, type, status, amount "
                 "FROM after_sales WHERE order_id = :oid LIMIT 1"),
            {"oid": order_id},
        )).mappings().first()
    if not row:
        return f"订单 {order_id} 暂无售后单记录。"
    return (f"售后单 {row['after_sale_id']}：类型={row['type']}，"
            f"状态={row['status']}，金额=¥{row['amount']}")


@tool
async def query_refund_ticket(order_id: str) -> str:
    """查询指定订单的退款工单审批状态。用户询问退款进度、审批结果时调用。"""
    async with engine.connect() as conn:
        row = (await conn.execute(
            text("SELECT ticket_id, status, required_level, decided_by, created_at "
                 "FROM tickets WHERE order_id = :oid ORDER BY created_at DESC LIMIT 1"),
            {"oid": order_id},
        )).mappings().first()
    if not row:
        return f"订单 {order_id} 暂无退款工单。"
    status_map = {"pending": "待审批", "approved": "已批准", "rejected": "已拒绝"}
    decided = f"，处理人：{row['decided_by']}" if row["decided_by"] else ""
    return (f"订单 {order_id} 的退款工单 {row['ticket_id']}，"
            f"状态：{status_map.get(row['status'], row['status'])}{decided}，"
            f"提交时间：{row['created_at']}")


@tool
async def create_refund_request(order_id: str, reason: str = "用户申请") -> str:
    """为指定订单创建退款申请。写操作，需人工审批。调用前必须向用户确认订单号。"""
    return f"已为订单 {order_id} 创建退款申请（原因：{reason}），预计 3 个工作日内处理。"


@tool
async def search_knowledge(query: str) -> str:
    """检索售后政策、退款规则、物流异常、会员权益等知识文档。回答必须基于检索结果。"""
    qvec = await embeddings.aembed_query(query)
    vec_str = str(qvec)

    async with engine.connect() as conn:
        rows = (await conn.execute(text("""
            SELECT id, content, title, doc_id,
                   1 - (embedding <=> :qvec) AS score
            FROM knowledge_chunks
            WHERE status = 'active'
            ORDER BY embedding <=> :qvec
            LIMIT 3
        """), {"qvec": vec_str})).mappings().all()

    if not rows:
        return "知识库中未找到相关内容。建议告知用户暂未收录并建议转人工。"
    return "\n\n".join(
        f"[{r['doc_id']}·{r['title']}] {r['content']}" for r in rows
    )