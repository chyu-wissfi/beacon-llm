"""调用记账：成本核算与调用 trace 留存（M01 仍为内存 list，M09 才落库）。"""

import logging
from datetime import datetime, timezone
from typing import Literal

from llm_gateway.core.schemas import CallTrace, PromptSelection, Usage
from llm_gateway.services.catalog import PRICE_PER_MILLION

# 与 demo 同名的具名 logger：logging.getLogger 按名单例，日志行为等价。
logger = logging.getLogger("llm_gateway")

# 模块级可变全局：/v1/traces 直接读它，测试也 import 后清空；保持 demo 语义。
CALL_TRACES: list[CallTrace] = []


def calculate_cost(model: str, usage: Usage) -> float:
    # 按实际模型和输入输出 Token 计算本次调用成本。
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
    status: Literal["success", "failed"],
    error_code: str | None = None,
) -> None:
    # 留存调用元数据，支持成本、延迟、模型和模板版本治理。
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
    )
    CALL_TRACES.append(trace)
    logger.info("llm_call_trace=%s", trace.model_dump_json())
