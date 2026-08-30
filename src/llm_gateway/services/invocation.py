"""调用编排：模型白名单校验、有限重试 + fallback 链、流式事件编码。

本层面向 Provider 协议编程（providers/base.py），不 import 任何供应商 SDK；
可重试与否由 provider 层映射后的错误码判定（PROVIDER_RETRYABLE_CODES），
重试节奏（次数、退避、换模型）留在本模块。

熔断反馈（M05）：逐尝试、按实际模型向 core/breaker 报告成败——计入失败
的只有 MODEL_UNAVAILABLE（连接/超时/上游 5xx 的映射码）；上游 429（限流非
损坏）不计，口径在 core/breaker.py 模块注中钉死。
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate

from llm_gateway.core.breaker import get_breaker
from llm_gateway.core.errors import (
    INVALID_JSON,
    MODEL_UNAVAILABLE,
    SCHEMA_VALIDATION_FAILED,
    STRUCTURED_OUTPUT_UNSUPPORTED,
    UNKNOWN_MODEL,
    UPSTREAM_STREAM_FAILED,
    GatewayError,
)
from llm_gateway.core.schemas import LLMRequest, LLMResponse, ModelConfig, Usage
from llm_gateway.providers import PROVIDER_REGISTRY
from llm_gateway.providers.base import (
    PROVIDER_RETRYABLE_CODES,
    ContentDelta,
    Provider,
    StreamCompleted,
)
from llm_gateway.services.catalog import MODEL_CONFIGS
from llm_gateway.services.prompt_service import build_messages
from llm_gateway.services.trace_service import record_trace

# 与 demo 同名的具名 logger：logging.getLogger 按名单例，日志行为等价。
logger = logging.getLogger("llm_gateway")

# 编排层面向协议编程，具体 Adapter 按配置驱动的注册表查表装配（M04 任务 5）。


def _get_provider(config: ModelConfig) -> Provider:
    # provider 取值在配置层已被 Literal 收窄（core/config.py），越界值在启动期
    # fail-fast；走到这里的 KeyError 只能是绕过配置中心的编程错误。
    return PROVIDER_REGISTRY[config.provider]


def validate_model(model: str, response_schema: dict[str, Any] | None) -> ModelConfig:
    # 校验模型白名单和结构化能力，阻止不等价的 fallback。
    config = MODEL_CONFIGS.get(model)
    if config is None:
        # 错误码与默认三元组取自注册表（core/errors.py），调用点不写字面量。
        raise GatewayError(UNKNOWN_MODEL)
    if response_schema is not None and not config.supports_structured_output:
        raise GatewayError(STRUCTURED_OUTPUT_UNSUPPORTED)
    return config


async def call_with_fallback(request: LLMRequest) -> LLMResponse:
    # 对临时故障有限重试，并在主模型不可用时切换能力等价的备用模型。
    requested_model = request.model
    request_id = str(uuid4())
    started = time.perf_counter()
    attempts = 0
    last_error: Exception | None = None
    messages = build_messages(request)
    for model_name in dict.fromkeys([requested_model, "general-backup"]):
        try:
            config = validate_model(model_name, request.response_schema)
        except GatewayError as exc:
            if model_name == requested_model:
                raise exc
            last_error = exc
            continue
        for retry_number in range(2):
            attempts += 1
            try:
                content, usage, finish_reason = await _get_provider(config).complete(
                    config,
                    messages,
                    request.timeout_seconds,
                    request.response_schema,
                    # 白名单放行字段的透传（M04 接线）：None 时 provider 不传参。
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    json_mode=request.json_mode,
                )
                parsed: dict[str, Any] | list[Any] | None = None
                if request.response_schema is not None:
                    try:
                        parsed = json.loads(content)
                        validate(instance=parsed, schema=request.response_schema)
                    except json.JSONDecodeError as exc:
                        raise GatewayError(INVALID_JSON) from exc
                    except JsonSchemaError as exc:
                        raise GatewayError(SCHEMA_VALIDATION_FAILED) from exc
                response = LLMResponse(
                    request_id=request_id,
                    model=model_name,
                    content=content,
                    parsed=parsed,
                    usage=usage,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempts,
                    finish_reason=finish_reason,
                )
                # 熔断反馈（M05）：成功的尝试闭合/复位对应模型的状态机。
                get_breaker(model_name).record_success()
                record_trace(request_id, requested_model, model_name, request.prompt, usage, response.latency_ms, attempts, "success")
                return response
            except GatewayError as exc:
                # M04 起 provider 层把 SDK 异常映射为 GatewayError（错误不穿透）
                # ，可重试性改由错误码承载（PROVIDER_RETRYABLE_CODES）。请求/配
                # 置类错误（unknown_model / gateway_misconfigured 等）不在集合
                # 内，首次失败即抛；可重试错误沿用旧节奏：重试一次后 break 去
                # fallback，最终对外仍归一为 model_unavailable（对外码不漂移）。
                last_error = exc
                if exc.code == MODEL_UNAVAILABLE:
                    # 熔断反馈（M05）：传输故障逐尝试计入（重试两次 = 两次失败）；
                    # 上游 429（provider_overloaded）不计——限流非损坏（见模块注）。
                    get_breaker(model_name).record_failure()
                if exc.code not in PROVIDER_RETRYABLE_CODES:
                    raise
                if retry_number == 0:
                    await asyncio.sleep(0.1)
                    continue
                break
            except Exception:
                # 走到这里的是网关自身的编程错误（provider 层已收编一切上游
                # 异常形态），不重试直接失败——与旧 "except GatewayError: raise"
                # 对非映射异常的处置一致。
                raise
    latency_ms = int((time.perf_counter() - started) * 1000)
    error_code = MODEL_UNAVAILABLE
    record_trace(request_id, requested_model, None, request.prompt, Usage(input_tokens=0, output_tokens=0), latency_ms, attempts, "failed", error_code)
    raise GatewayError(error_code) from last_error


async def stream_with_fallback(request: LLMRequest) -> AsyncIterator[dict[str, Any]]:
    # 上游首块前可切备用模型；首块后仅发送流内错误，避免文本重复。
    # 产出内部事件流（非线格式），三类事件：
    #   {"type": "content.delta", "delta": <文本增量>, "model": <实际服务模型>}
    #   {"type": "response.completed", "model": <实际服务模型>,
    #    "finish_reason": <词表值>, "usage": <Usage | None>}
    #   {"type": "response.failed", "error": <注册表错误码>}
    # finish_reason/usage 由 provider 的 StreamCompleted 终态事件承载（M04）
    # ，消费点在 api 层（终态 chunk / usage chunk）；output_truncated 关卡
    # （§3.7 第 2 关）属 M06/M08 编排面，本层只透传不消费。
    # SSE / OpenAI chunk / [DONE] 等线格式由 api 层一次性翻译（design.md §3.3：
    # 方言不进 services 层）；demo 期的 encode_sse 已随 /v1/llm/stream 删除。
    # delta 事件携带 model：OpenAI chunk 逐块标注实际服务方，fallback 后调用方
    # 在每个块上（而非仅终态）都能看到真实模型。
    messages = build_messages(request)
    started = time.perf_counter()
    attempts = 0
    emitted = False
    last_error: Exception | None = None
    for model_name in dict.fromkeys([request.model, "general-backup"]):
        try:
            config = validate_model(model_name, None)
            attempts += 1
            completed: StreamCompleted | None = None
            async for event in _get_provider(config).stream(
                config, messages, request.timeout_seconds, include_usage=request.include_usage
            ):
                if isinstance(event, ContentDelta):
                    emitted = True
                    yield {"type": "content.delta", "delta": event.text, "model": model_name}
                else:
                    completed = event
            record_trace(str(uuid4()), request.model, model_name, request.prompt, Usage(input_tokens=0, output_tokens=0), int((time.perf_counter() - started) * 1000), attempts, "success")
            # 熔断反馈（M05）：流正常走完同样算该模型一次成功。
            get_breaker(model_name).record_success()
            yield {
                "type": "response.completed",
                "model": model_name,
                "finish_reason": completed.finish_reason if completed else "stop",
                "usage": completed.usage if completed else None,
            }
            return
        except Exception as exc:
            last_error = exc
            if isinstance(exc, GatewayError) and exc.code == MODEL_UNAVAILABLE:
                # 熔断反馈（M05）：流式路径的传输故障同样按实际模型计入。
                get_breaker(model_name).record_failure()
            if emitted:
                break
            # 与非流式同款码表判定：provider 层映射后的可重试错误才切
            # fallback；请求/配置类错误与编程错误首次失败即终止。
            if isinstance(exc, GatewayError) and exc.code in PROVIDER_RETRYABLE_CODES:
                continue
            break
    logger.exception("upstream stream failed", exc_info=last_error)
    record_trace(str(uuid4()), request.model, None, request.prompt, Usage(input_tokens=0, output_tokens=0), int((time.perf_counter() - started) * 1000), attempts, "failed", UPSTREAM_STREAM_FAILED)
    yield {"type": "response.failed", "error": UPSTREAM_STREAM_FAILED}
