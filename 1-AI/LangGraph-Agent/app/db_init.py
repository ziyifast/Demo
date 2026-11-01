"""建扩展、建表、灌种子数据。"""
from sqlalchemy import text

from app.config import engine
from app.models import AfterSale, Base, Order


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    async with engine.begin() as conn:
        orders_n = (await conn.execute(text("SELECT count(*) FROM orders"))).scalar()
        as_n = (await conn.execute(text("SELECT count(*) FROM after_sales"))).scalar()
        if orders_n and as_n:
            return
        await conn.run_sync(_seed)


def _seed(sync_conn):
    from sqlalchemy.orm import Session
    with Session(sync_conn) as s:
        s.add_all([
            Order(order_id="1001", status="已发货", amount=299.0,
                  tracking="SF123456789", eta="明天"),
            Order(order_id="1002", status="待支付", amount=89.0,
                  tracking=None, eta="支付后 48 小时内"),
            Order(order_id="1003", status="已签收", amount=1299.0,
                  tracking="JD987654321", eta="已送达"),
            AfterSale(after_sale_id="AS-001", order_id="1003",
                      type="退货退款", status="审核中", amount=1299.0),
        ])
        s.commit()