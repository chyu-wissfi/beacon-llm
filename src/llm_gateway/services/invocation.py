"""调用编排：模型白名单校验、有限重试 + fallback 链、流式事件编码。

本层面向 Provider 协议编程（providers/base.py），不 import 任何供应商 SDK；
可重试与否由 Adapter 的 is_retryable 谓词判定，重试节奏（次数、退避、换模型）
留在本模块。
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
from llm_gateway.providers.base import Provider
from llm_gateway.providers.openai_compatible import (
    OpenAICompatibleProvider,
    is_retryable,
)
from llm_gateway.services.catalog import MODEL_CONFIGS
from llm_gateway.services.prompt_service import build_messages
from llm_gateway.services.trace_service import record_trace

# 与 demo 同名的具名 logger：logging.getLogger 按名单例，日志行为等价。
logger = logging.getLogger("llm_gateway")

# 编排层面向协议编程，具体 Adapter 在此装配；M01 保持 demo 的模块级单例语义。
provider: Provider = OpenAICompatibleProvider()


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
                content, usage = await provider.complete(config, messages, request.timeout_seconds, request.response_schema)
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
                )
                record_trace(request_id, requested_model, model_name, request.prompt, usage, response.latency_ms, attempts, "success")
                return response
            except GatewayError:
                raise
            except Exception as exc:
                last_error = exc
                if is_retryable(exc) and retry_number == 0:
                    await asyncio.sleep(0.1)
                    continue
                break
    latency_ms = int((time.perf_counter() - started) * 1000)
    error_code = MODEL_UNAVAILABLE
    record_trace(request_id, requested_model, None, request.prompt, Usage(input_tokens=0, output_tokens=0), latency_ms, attempts, "failed", error_code)
    raise GatewayError(error_code) from last_error


async def stream_with_fallback(request: LLMRequest) -> AsyncIterator[dict[str, Any]]:
    # 上游首块前可切备用模型；首块后仅发送流内错误，避免文本重复。
    # 产出内部事件流（非线格式），三类事件：
    #   {"type": "content.delta", "delta": <文本增量>, "model": <实际服务模型>}
    #   {"type": "response.completed", "model": <实际服务模型>}
    #   {"type": "response.failed", "error": <注册表错误码>}
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
            async for delta in provider.stream(config, messages, request.timeout_seconds):
                emitted = True
                yield {"type": "content.delta", "delta": delta, "model": model_name}
            record_trace(str(uuid4()), request.model, model_name, request.prompt, Usage(input_tokens=0, output_tokens=0), int((time.perf_counter() - started) * 1000), attempts, "success")
            yield {"type": "response.completed", "model": model_name}
            return
        except Exception as exc:
            last_error = exc
            if emitted or not is_retryable(exc):
                break
    logger.exception("upstream stream failed", exc_info=last_error)
    record_trace(str(uuid4()), request.model, None, request.prompt, Usage(input_tokens=0, output_tokens=0), int((time.perf_counter() - started) * 1000), attempts, "failed", UPSTREAM_STREAM_FAILED)
    yield {"type": "response.failed", "error": UPSTREAM_STREAM_FAILED}
