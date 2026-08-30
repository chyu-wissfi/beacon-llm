"""traces 表结构与写入约束单测（M09 任务 1/2）。

列面 = spec 任务 2 的 19 字段一个不漏（+ 存储层自身的 id 坐标）；
request_id 唯一约束是"恰好一次落库"的底层防线。
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from llm_gateway.storage.engine import get_engine
from llm_gateway.storage.models import TraceRow

pytestmark = pytest.mark.asyncio

# spec 任务 2 的字段面（存储行另带 id 主键坐标）。
SPEC_FIELDS = {
    "request_id",
    "timestamp",
    "caller",
    "requested_model",
    "actual_model",
    "final_endpoint",
    "route_reason",
    "prompt_name",
    "prompt_version",
    "validation_profile",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "price_version",
    "latency_ms",
    "ttft_ms",
    "attempts",
    "status",
    "error_code",
}


def _full_row_values(**overrides) -> dict:
    # 全字段行值：未覆盖的字段吃 None/零值，便于逐字段回读断言。
    values = {
        "request_id": "req-1",
        "timestamp": datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc),
        "caller": "演示调用方",
        "requested_model": "general-primary",
        "actual_model": "general-backup",
        "final_endpoint": "https://api.deepseek.com",
        "route_reason": "general-primary: exhausted",
        "prompt_name": "knowledge_decision",
        "prompt_version": "v1",
        "validation_profile": "order_decision/v1",
        "input_tokens": 13,
        "output_tokens": 5,
        "cost_usd": 0.000033,
        "price_version": "2026-08-30",
        "latency_ms": 120,
        "ttft_ms": 40,
        "attempts": 4,
        "status": "success",
        "error_code": None,
    }
    values.update(overrides)
    return values


async def test_traces_columns_match_spec_fields():
    # 列面与 spec 任务 2 的 19 字段一一对应（多一个存储层自身的 id）。
    assert set(TraceRow.__table__.columns.keys()) == SPEC_FIELDS | {"id"}


async def test_request_id_unique_constraint():
    # 底层防线：同 request_id 二次落库触发唯一约束（trace_service 按幂等吞掉）。
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.execute(insert(TraceRow), _full_row_values())
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(insert(TraceRow), _full_row_values(status="failed"))


async def test_full_row_roundtrip():
    # 全字段写入后逐字段回读一致（时间戳允许丢时区：方言行为，读路径统一
    # coerce 为 UTC——这里按无时区口径对账）。
    engine = await get_engine()
    async with AsyncSession(engine) as session:
        await session.execute(insert(TraceRow), _full_row_values())
        await session.commit()
    async with AsyncSession(engine) as session:
        row = (await session.execute(select(TraceRow))).scalars().one()
    for field in SPEC_FIELDS:
        expected = _full_row_values()[field]
        actual = getattr(row, field)
        if field == "timestamp":
            actual = actual.replace(tzinfo=timezone.utc) if actual.tzinfo is None else actual
        assert actual == expected, field
