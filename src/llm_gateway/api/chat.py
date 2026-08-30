"""POST /v1/chat/completions：OpenAI 兼容主端点（M03 任务 2）。

design.md §3.1/§3.9：本端点取代 demo 的 /v1/llm 与 /v1/llm/stream——单一端点内
以 stream 字段分流非流式 / SSE 两分支，openai SDK 零改动可调（不变量 #3）。
OpenAI 方言（chat.completion 结构、SSE chunk、[DONE] 终态）只在本层一次性翻译：
进入 services 层的仍是内部 LLMRequest / LLMResponse / 内部事件流（§3.3），
demo 的 content.delta / response.completed 线格式不在这里之外出现。

字段语义边界（与 api/schemas.py 的白名单注释互为表里）：
- temperature / max_tokens / response_format / include_usage 在 M04 接线落地：
  前两者透传到 Provider（None 不传参）；response_format 的 json_object /
  json_schema 两形态在这里翻译成内部协议（json_mode / response_schema），
  未知 type 或缺 schema 报 unsupported_field；深层形态校验仍后置（M07/M08）。
- 本层不透传白名单外字段，也不伪造行为：任何"放行但未消费"的字段都必须
  在注释里标出后置里程碑，而不是静默吞掉。

准入边界（M05）：请求路径先认证（401），再模型白名单（400），然后按固化序列
进准入（全局并发 -> 熔断 -> RPM -> TPM -> 供应商并发）；非流式随上下文退出释放，
流式持有到生成器结束（spec 任务 2）。治理端点（/v1/models、/v1/traces）本里程碑
不鉴权（演进项，台账已录）。TPM 事后记账也在本层（调用完成后按实际 usage）。
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Any, Final, Literal
from uuid import uuid4

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from llm_gateway.api.errors import openai_error_body
from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.core import ratelimit
from llm_gateway.core.auth import authenticate
from llm_gateway.core.config import CONFIG
from llm_gateway.core.errors import (
    ERROR_REGISTRY,
    UNSUPPORTED_COMBINATION,
    UNSUPPORTED_FIELD,
    GatewayError,
)
from llm_gateway.core.schemas import LLMRequest, LLMResponse
from llm_gateway.services.invocation import (
    call_with_fallback,
    stream_with_fallback,
    validate_model,
)
from llm_gateway.services.prompt_service import build_messages

router = APIRouter()

# SSE 终态标记：OpenAI 协议里 [DONE] 是"成功终态"标记，只在正常完成后发送；
# 流内失败以 OpenAI 风格错误事件收场、不再发 [DONE]，避免客户端把失败流
# 当作完整回答（design.md §3.9：流内失败发错误事件后终止）。
_SSE_DONE: Final[str] = "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 响应侧 wire 模型：OpenAI chat.completion / chunk 的对外契约
# ---------------------------------------------------------------------------
# 与请求白名单（api/schemas.py）分离：那边只管"接受哪些字段"，这里描述
# "网关承诺返回什么"。字段集合取 OpenAI 标准形态的最小集，本里程碑不夹带
# 网关扩展字段——attempts / latency 等治理信息走 /v1/traces（审计不属于
# OpenAI 协议，design.md §3.1）。


class CompletionUsage(BaseModel):
    # OpenAI usage 三键口径；内部 Usage 的 input/output_tokens 在翻译时换名，
    # total_tokens 由网关补齐（OpenAI 客户端按此口径消费）。
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"]
    content: str


class ChatChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    message: ChatMessage
    # 内部词表收窄（M04）：只承认 stop / length（截断识别依据）；值取
    # LLMResponse.finish_reason，None 仅出自旧调用路径，对外兜底 stop。
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["chat.completion"]
    created: int
    # 实际服务方的平台模型名（fallback 后可见），provider_model 不出网关
    # （CONTEXT.md：Provider Model 只存在于网关内部）。
    model: str
    choices: list[ChatChoice] = Field(min_length=1)
    usage: CompletionUsage


class ChunkDelta(BaseModel):
    # 流式只产出文本增量，delta 不含 role 首块（openai SDK 对缺 role 容忍）；
    # 允许空串：终态块的 delta=""（M04）。
    model_config = ConfigDict(extra="forbid")

    content: str


class ChunkChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    delta: ChunkDelta
    # 与 ChatChoice 同款收窄（M04）：内容块恒 None，终态块取词表值。
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChunkChoice]
    # usage chunk 的载体（M04）：仅 include_usage 时在 [DONE] 前附一块，
    # choices 为空（OpenAI 惯例）；其余块此键为 null。
    usage: CompletionUsage | None = None


# ---------------------------------------------------------------------------
# 方言翻译：OpenAI 请求/响应 <-> 内部协议（design.md §3.3 的一次性翻译点）
# ---------------------------------------------------------------------------


def _finish_reason_or_stop(value: str | None) -> Literal["stop", "length"]:
    # provider 层保证内部词表（base.py normalize）；对旧调用路径的 None 与
    # 理论上的词表外值兜底 stop——非流式出口的终态原因不得为 None。
    return "length" if value == "length" else "stop"


def _translate_response_format(fmt: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    # response_format -> (response_schema, json_mode)。OpenAI 的包装结构在这里
    # 一次性翻译为内部协议：json_schema 提取内层 schema 走 response_schema 既有
    # 链路（supports_structured_output 检查 + 本地校验）；json_object 无 schema，
    # 只开 JSON 模式（本地校验无从谈起）。未知 type / 缺 schema 用
    # UNSUPPORTED_FIELD 显式点名 response_format（动态 message 是注册表的合法覆盖）。
    fmt_type = fmt.get("type")
    if fmt_type == "json_object":
        return None, True
    if fmt_type == "json_schema":
        inner = fmt.get("json_schema")
        schema = inner.get("schema") if isinstance(inner, dict) else None
        if isinstance(schema, dict):
            return schema, False
        raise GatewayError(UNSUPPORTED_FIELD, message="response_format.json_schema 缺少合法的 schema")
    raise GatewayError(
        UNSUPPORTED_FIELD,
        message=f"response_format.type 只支持 json_object / json_schema，收到：{fmt_type!r}",
    )


def _to_internal_request(request: ChatCompletionRequest) -> LLMRequest:
    # stream 不进内部请求：它只在本端点内选分支（内部协议的 stream 死字段已随
    # M04 清理）。response_format 不直接当 schema 用：它是包装结构，必须经翻译。
    response_schema, json_mode = (
        _translate_response_format(request.response_format)
        if request.response_format is not None
        else (None, False)
    )
    include_usage = (
        request.stream_options.include_usage is True if request.stream_options is not None else False
    )
    return LLMRequest(
        model=request.model,
        messages=request.messages,
        prompt=request.prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        include_usage=include_usage,
        json_mode=json_mode,
        response_schema=response_schema,
    )


def _to_chat_completion(response: LLMResponse, created: int) -> ChatCompletionResponse:
    # id 复用 request_id：调用方可用响应 id 与 /v1/traces 记录对账（demo 的
    # trace 关联语义在 OpenAI 面的唯一落点）。
    return ChatCompletionResponse(
        id=response.request_id,
        object="chat.completion",
        created=created,
        model=response.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response.content),
                # provider 回传的上游终态原因；None 仅出自旧调用路径，对外兜 stop。
                finish_reason=_finish_reason_or_stop(response.finish_reason),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        ),
    )


def _encode_chunk(chunk: ChatCompletionChunk) -> str:
    # ensure_ascii 语义与 demo 一致：中文增量原样输出（pydantic json 不转义
    # 非 ASCII），浏览器与 SDK 均按 UTF-8 解。
    return f"data: {chunk.model_dump_json()}\n\n"


async def _chunk_stream(
    internal_request: LLMRequest,
    completion_id: str,
    created: int,
) -> AsyncIterator[str]:
    # 编排层产出内部事件流（dict，含 type/delta/model 或 type/error），这里逐
    # 事件翻译为 OpenAI chunk 线格式。流式语义铁律（首块前可 fallback、首块后
    # 不重生成）留在编排层（design.md review 锚点 ④：invocation.py 是唯一状态机）。
    async for event in stream_with_fallback(internal_request):
        if event["type"] == "content.delta":
            yield _encode_chunk(
                ChatCompletionChunk(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=event["model"],
                    choices=[
                        ChunkChoice(index=0, delta=ChunkDelta(content=event["delta"]), finish_reason=None)
                    ],
                )
            )
        elif event["type"] == "response.completed":
            # 终态 chunk（M04）：finish_reason 非 None、delta 空串——依赖终态原因
            # 的调用方（截断识别等，消费在 M06/M08）从这里拿到信号。
            yield _encode_chunk(
                ChatCompletionChunk(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=event["model"],
                    choices=[
                        ChunkChoice(index=0, delta=ChunkDelta(content=""), finish_reason=event["finish_reason"])
                    ],
                )
            )
            usage = event["usage"]
            if usage is not None:
                # TPM 事后记账（M05 任务 3）：按准入模型入账（实际服务方可能
                # 是 fallback 备用——事后记账的已知近似，归入准入模型的预算）；
                # 上游未回传 usage（如 chat 流式未开 include_usage）则跳过，
                # 不伪造用量。
                ratelimit.ADMISSION.record_usage(internal_request.model, usage)
            if internal_request.include_usage and usage is not None:
                # usage chunk（OpenAI 惯例）：[DONE] 前附一块，choices 为空；
                # 仅当调用方显式请求（include_usage）才发，不请求不谎报。
                yield _encode_chunk(
                    ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=event["model"],
                        choices=[],
                        usage=CompletionUsage(
                            prompt_tokens=usage.input_tokens,
                            completion_tokens=usage.output_tokens,
                            total_tokens=usage.input_tokens + usage.output_tokens,
                        ),
                    )
                )
            yield _SSE_DONE
        else:
            # response.failed：UPSTREAM_STREAM_FAILED 按注册表口径渲染成 OpenAI
            # 风格错误事件（该码不作 GatewayError 抛出，只作 trace error_code 与
            # 流内终态 payload——core/errors.py 注释的既有约定）。事件里的 error
            # 是注册表码字符串（事件 dict 的 Any 标注不携带字面量类型，M03 唯一
            # 生产者是 UPSTREAM_STREAM_FAILED），三元组仍只取自注册表。
            code = event["error"]
            spec = ERROR_REGISTRY[code]
            yield (
                "data: "
                + json.dumps(
                    openai_error_body(code, spec.message, spec.status_code),
                    ensure_ascii=False,
                )
                + "\n\n"
            )


async def _admitted_chunk_stream(
    permit: ratelimit.AdmissionPermit,
    internal_request: LLMRequest,
    completion_id: str,
    created: int,
) -> AsyncIterator[str]:
    # 流式的准入持有形态（spec 任务 2：并发计数从准入持有到流式结束）：
    # 端点在返回 StreamingResponse 前完成准入，释放推迟到生成器终结——
    # 无论正常收梢、流内失败还是客户端提前断开（GeneratorExit）。
    try:
        async for chunk in _chunk_stream(internal_request, completion_id, created):
            yield chunk
    finally:
        permit.release()


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
) -> ChatCompletionResponse | StreamingResponse:
    # 认证（M05 任务 1）：准入序列的第一层，失败 401 unauthorized；
    # caller 标识注入请求上下文（request.state，M09 trace 消费）。
    caller = authenticate(authorization, CONFIG.callers)
    http_request.state.caller = caller.display_name

    # stream + response_format 互斥沿用 demo 的 unsupported_combination（400，
    # controller 裁决）。在端点层以 GatewayError 抛稳定码，而不是放进 Pydantic
    # validator——后者只能给 code=None 的通用 400，会丢失稳定错误码（任务 A
    # 报告给任务 B 的衔接点）。message 覆盖为 OpenAI 面的字段名：注册表默认
    # message 引用的是旧协议字段 response_schema，双冻结不可改，动态覆盖是
    # 注册表允许的唯一合法覆盖面（同 missing_prompt_variable 先例）。
    if request.stream and request.response_format is not None:
        raise GatewayError(UNSUPPORTED_COMBINATION, message="流式输出不支持 response_format")

    internal_request = _to_internal_request(request)

    # 准入前先定模型（白名单 400 在准入之前）：准入需要 provider 与限流参数。
    # 结构化能力检查仍由编排层随 response_schema 一并做，此处只取配置。
    model_config = validate_model(internal_request.model, None)
    rate = model_config.rate_limit

    if request.stream:
        # 统一端点下 stream=true 直接走流式分支（controller 裁决）：M03 起
        # use_stream_endpoint 运行时不再抛出（注册表双冻结保留，仅作历史码）。
        # 请求级校验（模型白名单、Prompt 渲染）保持在返回 StreamingResponse
        # 之前——与 demo 一致：请求问题走 HTTP 错误，只有上游/流中途失败才
        # 走流内错误事件。准入拒绝同为请求期 HTTP 错误（429/503）。
        build_messages(internal_request)
        permit = await ratelimit.ADMISSION.acquire(
            internal_request.model,
            model_config.provider,
            rpm=rate.rpm if rate is not None else None,
            tpm=rate.tpm if rate is not None else None,
        )
        return StreamingResponse(
            _admitted_chunk_stream(
                permit,
                internal_request,
                completion_id=str(uuid4()),
                created=int(time.time()),
            ),
            media_type="text/event-stream",
        )

    # 非流式：准入资源随上下文退出释放；拒绝（429/503）时编排层零波及，
    # 上游不会收到任何请求。
    async with ratelimit.ADMISSION.admit(
        internal_request.model,
        model_config.provider,
        rpm=rate.rpm if rate is not None else None,
        tpm=rate.tpm if rate is not None else None,
    ):
        response = await call_with_fallback(internal_request)
    # TPM 事后记账（spec 任务 3）：只在成功完成后按实际 usage 入账。
    ratelimit.ADMISSION.record_usage(internal_request.model, response.usage)
    return _to_chat_completion(response, created=int(time.time()))
