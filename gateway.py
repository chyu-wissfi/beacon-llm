import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Template
from typing import Any, Literal, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, model_validator


logger = logging.getLogger("llm_gateway")


class Message(BaseModel):
    # 定义跨模型通用的单条对话消息，隔离供应商消息格式差异。
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class PromptSelection(BaseModel):
    # 只允许调用方选择受控模板及变量，不能提交或覆盖模板正文。
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    variables: dict[str, str] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    # 统一 Gateway 请求协议，并在 HTTP 入口拦截不合法组合和字段。
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=100)
    messages: list[Message] = Field(min_length=1, max_length=100)
    stream: bool = False
    response_schema: dict[str, Any] | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    prompt: PromptSelection | None = None

    @model_validator(mode="after")
    def check_supported_combination(self) -> "LLMRequest":
        if self.stream and self.response_schema is not None:
            raise ValueError("stream 与 response_schema 不能同时使用")
        return self


class Usage(BaseModel):
    # 统一输入与输出 Token 统计口径，用于成本和用量治理。
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class LLMResponse(BaseModel):
    # 统一模型调用结果，并作为 FastAPI 响应出口的 Pydantic 校验契约。
    model_config = ConfigDict(extra="forbid")

    request_id: str
    model: str
    content: str
    parsed: dict[str, Any] | list[Any] | None = None
    usage: Usage
    latency_ms: int = Field(ge=0)
    attempts: int = Field(ge=1)


class PromptTemplate(BaseModel):
    # 表示由 Gateway 发布和版本化管理的系统 Prompt 模板资产。
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    system_template: str


class CallTrace(BaseModel):
    # 保存单次调用的模型、Token、成本、延迟与状态，默认不记录文本内容。
    model_config = ConfigDict(extra="forbid")

    request_id: str
    timestamp: datetime
    requested_model: str
    actual_model: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    attempts: int = Field(ge=0)
    status: Literal["success", "failed"]
    error_code: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    # 将平台模型名映射为供应商模型、地址、密钥与能力配置。
    provider_model: str
    base_url: str
    api_key_env: str
    supports_structured_output: bool
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"


MODEL_CONFIGS = {
    "general-primary": ModelConfig(
        provider_model=os.getenv("PRIMARY_PROVIDER_MODEL", "deepseek-v4-flash"),
        base_url=os.getenv("PRIMARY_BASE_URL", "https://api.deepseek.com"),
        api_key_env="DEEPSEEK_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
    ),
    "general-backup": ModelConfig(
        provider_model=os.getenv("BACKUP_PROVIDER_MODEL", "deepseek-chat"),
        base_url=os.getenv("BACKUP_BASE_URL", "https://api.deepseek.com"),
        api_key_env="DEEPSEEK_BACKUP_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
    ),
}

PROMPT_TEMPLATES = {
    ("knowledge_decision", "v1"): PromptTemplate(
        name="knowledge_decision",
        version="v1",
        system_template="你是${product_name}的知识库决策器。资料不足时搜索，资料充分时结束回答。不得编造制度内容。",
    )
}

PRICE_PER_MILLION = {
    "general-primary": {"input": 1.0, "output": 4.0},
    "general-backup": {"input": 0.8, "output": 3.2},
}

CALL_TRACES: list[CallTrace] = []


class GatewayError(Exception):
    # 将内部错误标准化为可安全暴露给调用方的稳定错误码和 HTTP 状态。
    def __init__(self, code: str, message: str, status_code: int = 502) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class Provider(Protocol):
    # 规定供应商 Adapter 的统一接口，业务流程不依赖具体 SDK。
    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]: ...

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]: ...


class OpenAICompatibleProvider:
    # 实现 OpenAI Compatible Adapter，集中处理供应商协议与认证细节。
    # 将 API Key 保留在 Gateway 内，业务 Agent 无需接触供应商密钥。
    def create_client(self, config: ModelConfig) -> AsyncOpenAI:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise GatewayError("gateway_misconfigured", "Gateway 模型凭据未配置", 503)
        return AsyncOpenAI(api_key=api_key, base_url=config.base_url, max_retries=0)

    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]:
        # 将统一请求转换为 OpenAI Compatible 调用，隔离厂商协议差异。
        request_data: dict[str, Any] = {
            "model": config.provider_model,
            "messages": [message.model_dump() for message in messages],
            "timeout": timeout_seconds,
        }
        if response_schema is not None:
            if config.structured_output_mode == "json_schema":
                request_data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
            else:
                request_data["response_format"] = {"type": "json_object"}
                request_data["messages"] = [
                    {
                        "role": "system",
                        "content": (
                            "只返回一个合法 JSON 对象，必须严格符合下列 JSON Schema，"
                            "不要返回 Markdown 或额外文字："
                            f"{json.dumps(response_schema, ensure_ascii=False)}"
                        ),
                    },
                    *request_data["messages"],
                ]
        completion = await self.create_client(config).chat.completions.create(**request_data)
        content = completion.choices[0].message.content or ""
        usage = completion.usage
        return content, Usage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        # 逐块读取上游响应，为 Gateway 的实时流式代理提供标准增量文本。
        response = await self.create_client(config).chat.completions.create(
            model=config.provider_model,
            messages=[message.model_dump() for message in messages],
            stream=True,
            timeout=timeout_seconds,
        )
        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta.content:
                yield choice.delta.content


provider: Provider = OpenAICompatibleProvider()


def render_prompt(selection: PromptSelection) -> Message:
    # 从受控模板库渲染系统提示词，调用方只能传版本和变量。
    template = PROMPT_TEMPLATES.get((selection.name, selection.version))
    if template is None:
        raise GatewayError("unknown_prompt_template", "Prompt 模板不存在", 400)
    try:
        content = Template(template.system_template).substitute(selection.variables)
    except KeyError as exc:
        raise GatewayError("missing_prompt_variable", f"缺少 Prompt 变量: {exc.args[0]}", 400) from exc
    return Message(role="system", content=content)


def build_messages(request: LLMRequest) -> list[Message]:
    # 将模板系统消息统一注入调用上下文，避免 Prompt 分散在各个 Agent 中。
    if request.prompt is None:
        return request.messages
    return [render_prompt(request.prompt), *request.messages]


def validate_model(model: str, response_schema: dict[str, Any] | None) -> ModelConfig:
    # 校验模型白名单和结构化能力，阻止不等价的 fallback。
    config = MODEL_CONFIGS.get(model)
    if config is None:
        raise GatewayError("unknown_model", "模型不在 Gateway 允许列表中", 400)
    if response_schema is not None and not config.supports_structured_output:
        raise GatewayError("structured_output_unsupported", "模型不支持 Structured Output", 400)
    return config


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


def is_retryable(exc: Exception) -> bool:
    return isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError, ConnectionError))


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
                        raise GatewayError("invalid_json", "模型没有返回合法 JSON") from exc
                    except JsonSchemaError as exc:
                        raise GatewayError("schema_validation_failed", "模型结果不符合 response_schema") from exc
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
    error_code = "model_unavailable"
    record_trace(request_id, requested_model, None, request.prompt, Usage(input_tokens=0, output_tokens=0), latency_ms, attempts, "failed", error_code)
    raise GatewayError(error_code, "主模型和备用模型均不可用") from last_error


def encode_sse(event: dict[str, Any]) -> str:
    # 将统一事件编码为浏览器和 Agent 都可消费的 SSE 格式。
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def stream_with_fallback(request: LLMRequest) -> AsyncIterator[str]:
    # 上游首块前可切备用模型；首块后仅发送流内错误，避免文本重复。
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
                yield encode_sse({"type": "content.delta", "delta": delta})
            record_trace(str(uuid4()), request.model, model_name, request.prompt, Usage(input_tokens=0, output_tokens=0), int((time.perf_counter() - started) * 1000), attempts, "success")
            yield encode_sse({"type": "response.completed", "model": model_name})
            return
        except Exception as exc:
            last_error = exc
            if emitted or not is_retryable(exc):
                break
    logger.exception("upstream stream failed", exc_info=last_error)
    record_trace(str(uuid4()), request.model, None, request.prompt, Usage(input_tokens=0, output_tokens=0), int((time.perf_counter() - started) * 1000), attempts, "failed", "upstream_stream_failed")
    yield encode_sse({"type": "response.failed", "error": "upstream_stream_failed"})


app = FastAPI(title="Agent LLM Gateway", version="0.0.1")


@app.post("/v1/llm", response_model=LLMResponse)
async def create_llm_response(request: LLMRequest) -> LLMResponse:
    # FastAPI 在入口校验请求、在 response_model 校验统一响应出口。
    if request.stream:
        raise HTTPException(status_code=400, detail={"code": "use_stream_endpoint", "message": "流式请求请使用 /v1/llm/stream"})
    try:
        return await call_with_fallback(request)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc


@app.post("/v1/llm/stream")
async def create_stream(request: LLMRequest) -> StreamingResponse:
    # 提供独立流式入口，明确禁止与 Structured Output 混用。
    if request.response_schema is not None:
        raise HTTPException(status_code=400, detail={"code": "unsupported_combination", "message": "流式输出不支持 response_schema"})
    try:
        validate_model(request.model, None)
        build_messages(request)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc
    return StreamingResponse(stream_with_fallback(request), media_type="text/event-stream")


@app.get("/v1/traces", response_model=list[CallTrace])
async def list_traces() -> list[CallTrace]:
    # 暴露调用审计记录，供成本分析与故障排查使用。
    return CALL_TRACES
