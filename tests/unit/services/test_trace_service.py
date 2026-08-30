"""trace_service 单测（M09 任务 3/4）：落库恰好一次、价格快照、崩溃安全。

服务层直调（不经 HTTP）：record_trace -> 进程内缓存 + 异步入库；
TraceDraft 的恰好一次终态写与"终态前无半条 trace"的崩溃安全边界。
引擎夹具在 services/conftest.py（每用例一份内存库）。
"""

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_gateway.core.schemas import PromptSelection, Usage
from llm_gateway.services.catalog import PRICE_PER_MILLION, PRICE_VERSION
from llm_gateway.services.run_context import Budget, TraceDraft
from llm_gateway.services.trace_service import (
    CALL_TRACES,
    calculate_cost,
    flush_pending,
    persist_trace,
    record_trace,
)
from llm_gateway.storage.engine import get_engine
from llm_gateway.storage.models import TraceRow

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clean_traces():
    # CALL_TRACES 是进程内缓存面，用例间清零。
    CALL_TRACES.clear()
    yield
    CALL_TRACES.clear()


async def _db_rows() -> list[TraceRow]:
    engine = await get_engine()
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(TraceRow).order_by(TraceRow.id))).scalars().all())


def _record(**overrides) -> None:
    # 一条标准成功 trace：覆盖全部记账面，用例按需覆盖。
    kwargs = {
        "request_id": "req-1",
        "requested_model": "general-primary",
        "actual_model": "general-primary",
        "prompt": PromptSelection(name="knowledge_decision", version="v1"),
        "usage": Usage(input_tokens=13, output_tokens=5),
        "latency_ms": 120,
        "attempts": 1,
        "status": "success",
        "caller": "演示调用方",
        "ttft_ms": 40,
        "final_endpoint": "https://api.deepseek.com",
        "validation_profile": "order_decision/v1",
        "price_version": PRICE_VERSION,
    }
    kwargs.update(overrides)
    record_trace(**kwargs)  # type: ignore[arg-type]


async def test_record_trace_persists_single_row_with_full_fields():
    # 记账恰好一次的基础形态：一次终态写 -> 缓存一条 + 库里一行，字段逐一对上。
    _record()
    await flush_pending()
    assert len(CALL_TRACES) == 1
    rows = await _db_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.request_id == "req-1"
    assert row.caller == "演示调用方"
    assert row.requested_model == "general-primary"
    assert row.actual_model == "general-primary"
    assert row.final_endpoint == "https://api.deepseek.com"
    assert row.prompt_name == "knowledge_decision"
    assert row.prompt_version == "v1"
    assert row.validation_profile == "order_decision/v1"
    assert row.input_tokens == 13
    assert row.output_tokens == 5
    assert row.price_version == PRICE_VERSION
    assert row.latency_ms == 120
    assert row.ttft_ms == 40
    assert row.attempts == 1
    assert row.status == "success"
    assert row.error_code is None
    # 成本按 primary 牌价：(13*1.0 + 5*4.0) / 1e6。
    assert row.cost_usd == pytest.approx((13 * 1.0 + 5 * 4.0) / 1_000_000)


async def test_duplicate_persist_is_idempotent_exactly_once():
    # 底层防线：同一 trace 重复落库（编程错误路径）不炸、不重复——
    # request_id 唯一约束 + 幂等吞掉，库里仍恰好一行。
    _record()
    await flush_pending()
    await persist_trace(CALL_TRACES[0])  # 模拟重复写（不经过调度，直接重放）
    await flush_pending()
    rows = await _db_rows()
    assert len(rows) == 1


def _draft(clock_value: float = 0.0) -> TraceDraft:
    clock = lambda: clock_value  # noqa: E731  # 注入时钟：冻结时间便于断言
    draft = TraceDraft(
        request_id="req-draft",
        requested_model="general-primary",
        prompt=None,
        caller="演示调用方",
        started_at=0.0,
        clock=clock,
        validation_label="order_decision/v1",
        price_version=PRICE_VERSION,
    )
    draft.bind_budget(Budget(timeout_seconds=30, clock=clock))
    return draft


async def test_finalize_writes_trace_exactly_once():
    # 主防线：终态写入口幂等——状态机任何出口重复触发也只落一条。
    draft = _draft()
    draft.observe_usage(Usage(input_tokens=3, output_tokens=2))
    draft.finalize("success", "general-primary")
    draft.finalize("failed", "general-primary", error_code="model_unavailable")  # no-op
    await flush_pending()
    assert len(CALL_TRACES) == 1
    rows = await _db_rows()
    assert len(rows) == 1
    assert rows[0].status == "success"  # 以首次终态为准


async def test_no_trace_in_db_before_finalize():
    # 崩溃安全（spec 任务 6）：终态迁移前进程取消 -> 库里零行，无半条 trace。
    draft = _draft()
    draft.observe_usage(Usage(input_tokens=3, output_tokens=2))
    # 刻意不 finalize：模拟终态迁移前进程被取消。
    await flush_pending()
    assert CALL_TRACES == []
    assert await _db_rows() == []


async def test_usage_accumulates_across_attempts_exactly_once():
    # 每次上游响应的 usage 恰好记账一次：各尝试观测值之和进 run 总量，
    # 不重复、不遗漏（失败尝试的缺口观测不到、不伪造，同口径）。
    draft = _draft()
    draft.observe_usage(Usage(input_tokens=10, output_tokens=4))  # 第 1 次尝试
    draft.observe_usage(Usage(input_tokens=7, output_tokens=3))  # 第 2 次尝试
    draft.observe_usage(None)  # 上游未回传：跳过不伪造
    draft.observe_usage(Usage(input_tokens=5, output_tokens=2))  # 修复调用
    draft.finalize("success", "general-primary")
    await flush_pending()
    rows = await _db_rows()
    assert len(rows) == 1
    assert rows[0].input_tokens == 22
    assert rows[0].output_tokens == 9


async def test_finalize_carries_price_version_and_validation_label():
    # 价格版本快照与校验档案坐标随终态写一并落库（spec 任务 2/4）。
    draft = _draft()
    draft.finalize("failed", None, error_code="model_unavailable")
    await flush_pending()
    rows = await _db_rows()
    assert len(rows) == 1
    assert rows[0].price_version == PRICE_VERSION
    assert rows[0].validation_profile == "order_decision/v1"
    assert rows[0].final_endpoint is None  # 未服务到任何模型：不伪造


async def test_calculate_cost_follows_price_table_snapshot():
    # 成本口径回归：牌价表快照取值，与 demo 语义逐字一致。
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000)
    primary = PRICE_PER_MILLION["general-primary"]
    expected = primary["input"] * 1.0 + primary["output"] * 0.5
    assert calculate_cost("general-primary", usage) == pytest.approx(expected)


async def test_record_trace_without_event_loop_keeps_cache_only(caplog):
    # 同步直调边界（无事件循环）：只留进程内缓存并告警——不静默。
    # 在工作线程里调同步入口：线程内无运行中事件循环（主循环不穿透）。
    await asyncio.to_thread(
        record_trace,
        request_id="req-sync",
        requested_model="general-primary",
        actual_model=None,
        prompt=None,
        usage=Usage(input_tokens=0, output_tokens=0),
        latency_ms=0,
        attempts=0,
        status="cancelled",
    )
    assert any("无事件循环" in record.message for record in caplog.records)
    assert CALL_TRACES[-1].request_id == "req-sync"
