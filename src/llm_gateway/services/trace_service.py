"""调用记账：成本核算与调用 trace 落库（M09：SQLite 持久化）。

trace 的合法数据源 = SQLite（storage/）：终态迁移时恰好一次落库（spec 任务 3，
含 failed/cancelled）。CALL_TRACES 降级为进程内缓存面——存量测试断言与结构化
日志继续消费它，/v1/traces 改读库（api/governance.py）。

恰好一次落库：record_trace 仍是同步入口（流式取消路径在 GeneratorExit 处理中
不得 await，M06 铁律），内部以 loop.create_task 调度异步持久化任务（登记进
_PENDING）；/v1/traces 读库前先 await flush_pending()——同进程 read-your-
writes。"恰好一次"双防线：TraceDraft.finalized 旗标（主防线）+ request_id
唯一约束（底层防线，IntegrityError 按幂等吞掉）。持久化失败记日志不波及
请求路径：trace 是观测面，落库故障不得反转业务终态（Controller 裁决）。

崩溃安全：trace 只在终态迁移产生单条 INSERT——终态前进程取消 => 库里零行，
不存在"半条 trace"（spec 任务 6）。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from llm_gateway.core.schemas import CallTrace, PromptSelection, Usage
from llm_gateway.observability.metrics import (
    LATENCY_SECONDS,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
)
from llm_gateway.services.catalog import PRICE_PER_MILLION
from llm_gateway.storage.engine import get_engine
from llm_gateway.storage.models import TraceRow

# 与 demo 同名的具名 logger：logging.getLogger 按名单例，日志行为等价。
logger = logging.getLogger("llm_gateway")

# 进程内缓存面（M09 降级）：合法数据源是 SQLite；本 list 保留存量断言面与
# 日志语义，测试仍 import 后清空。
CALL_TRACES: list[CallTrace] = []

# 已调度未完成的持久化任务：flush_pending 是读路径唯一的对账面。
_PENDING: set[asyncio.Task[None]] = set()


def calculate_cost(model: str, usage: Usage) -> float:
    # 按实际模型和输入输出 Token 计算本次调用成本。价格表是启动期一次性
    # 加载的快照（CONFIG）；trace 的 price_version 列标识本条按哪个版本计价。
    price = PRICE_PER_MILLION[model]
    return (usage.input_tokens * price["input"] + usage.output_tokens * price["output"]) / 1_000_000


def record_trace(
    request_id: str,
    requested_model: str,
    actual_model: str | None,
    prompt: PromptSelection | None,
    usage: Usage,
    latency_ms: int,
    attempts: int,
    status: Literal["success", "failed", "cancelled"],
    error_code: str | None = None,
    caller: str | None = None,
    route_reason: str | None = None,
    ttft_ms: int | None = None,
    final_endpoint: str | None = None,
    validation_profile: str | None = None,
    price_version: str | None = None,
) -> None:
    # 留存调用元数据，支持成本、延迟、模型和模板版本治理。
    # M06 起可选记账面：caller / route_reason / ttft_ms；M09 补全：
    # final_endpoint / validation_profile / price_version（语义见
    # core/schemas.py CallTrace 字段注）。
    trace = CallTrace(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc),
        requested_model=requested_model,
        actual_model=actual_model,
        prompt_name=prompt.name if prompt else None,
        prompt_version=prompt.version if prompt else None,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=calculate_cost(actual_model, usage) if actual_model else 0,
        latency_ms=latency_ms,
        attempts=attempts,
        status=status,
        error_code=error_code,
        caller=caller,
        route_reason=route_reason,
        ttft_ms=ttft_ms,
        final_endpoint=final_endpoint,
        validation_profile=validation_profile,
        price_version=price_version,
    )
    CALL_TRACES.append(trace)
    logger.info("llm_call_trace=%s", trace.model_dump_json())
    _observe_metrics(trace)
    _schedule_persist(trace)


def _observe_metrics(trace: CallTrace) -> None:
    # 指标与 trace 是同一终态事件的两本账（M10 任务 4）：metrics 是速率、
    # trace 是审计。record_trace 是唯一终态出口（恰好一次由 TraceDraft 幂等
    # 防线保证），终态类指标在此集中记账，无遗漏无重复：
    # - requests_total 的 model 维用请求模型（三终态恒存在；实际服务模型在
    #   tokens_total 维度承载）；
    # - tokens 只在实际服务到模型时入账，取消/早退的缺口与 trace 同款语义。
    REQUESTS_TOTAL.labels(model=trace.requested_model, status=trace.status).inc()
    LATENCY_SECONDS.observe(trace.latency_ms / 1000)
    if trace.actual_model is not None:
        TOKENS_TOTAL.labels(model=trace.actual_model, direction="input").inc(trace.input_tokens)
        TOKENS_TOTAL.labels(model=trace.actual_model, direction="output").inc(trace.output_tokens)


def _schedule_persist(trace: CallTrace) -> None:
    # 调度异步落库：record_trace 是同步入口（GeneratorExit 处理中不得 await），
    # create_task 是同步上下文里唯一的持久化路径。无事件循环（同步直调边界）
    # 只留进程内缓存并告警——不静默。
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("trace %s：无事件循环，仅留存进程内缓存", trace.request_id)
        return
    task = loop.create_task(persist_trace(trace))
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


async def persist_trace(trace: CallTrace) -> None:
    # 落库单条：单行 INSERT 原子 + request_id 唯一约束。IntegrityError 是
    # "恰好一次"底层防线的预期命中（幂等吞掉）；其余异常只记日志。
    try:
        engine = await get_engine()
        async with engine.begin() as conn:
            await conn.execute(insert(TraceRow), _row_values(trace))
    except IntegrityError:
        logger.warning("trace %s：重复落库，幂等吞掉", trace.request_id)
    except Exception:
        logger.exception("trace %s：落库失败（不波及请求路径）", trace.request_id)


def _row_values(trace: CallTrace) -> dict[str, Any]:
    # CallTrace -> traces 表行值：字段面一一对应（id 是存储层自身坐标除外）。
    return {
        "request_id": trace.request_id,
        "timestamp": trace.timestamp,
        "caller": trace.caller,
        "requested_model": trace.requested_model,
        "actual_model": trace.actual_model,
        "final_endpoint": trace.final_endpoint,
        "route_reason": trace.route_reason,
        "prompt_name": trace.prompt_name,
        "prompt_version": trace.prompt_version,
        "validation_profile": trace.validation_profile,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "cost_usd": trace.cost_usd,
        "price_version": trace.price_version,
        "latency_ms": trace.latency_ms,
        "ttft_ms": trace.ttft_ms,
        "attempts": trace.attempts,
        "status": trace.status,
        "error_code": trace.error_code,
    }


async def flush_pending() -> None:
    # 对账：等掉全部已调度的持久化任务。/v1/traces 读库前调用，保证同进程
    # read-your-writes；测试夹具的 teardown 也用它收尾（避免悬空任务噪声）。
    if not _PENDING:
        return
    pending = list(_PENDING)
    _PENDING.clear()
    await asyncio.gather(*pending, return_exceptions=True)
