"""POST /v1/chat/completions：OpenAI 兼容主端点（M03 任务 2）。

design.md §3.1/§3.9：本端点取代 demo 的 /v1/llm 与 /v1/llm/stream——单一端点内
以 stream 字段分流非流式 / SSE 两分支，openai SDK 零改动可调（不变量 #3）。
OpenAI 方言（chat.completion 结构、SSE chunk、[DONE] 终态）只在本层一次性翻译：
进入 services 层的仍是内部 LLMRequest / LLMResponse / 内部事件流（§3.3），
demo 的 content.delta / response.completed 线格式不在这里之外出现。

M03 字段语义边界（与 api/schemas.py 的白名单注释互为表里）：
- temperature / max_tokens / response_format 已过白名单，但下游协议尚未承载——
  Provider Protocol 定稿在 M04（finish_reason / 流式 usage 回传），结构化输出
  （response_format 内部结构）消费在 M06-M08。本层不透传它们，也不伪造行为。
- stream_options.include_usage 同理：M03 流式链路拿不到上游 usage（provider.stream
  只回传文本增量，M04 定稿后才有真值），附一个 usage=0 的块等于向调用方撒谎，
  故 include_usage 的语义消费与 M04 一并落地，本里程碑只放行不消费。
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Final, Literal
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from llm_gateway.api.errors import openai_error_body
from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.core.errors import (
    ERROR_REGISTRY,
    UNSUPPORTED_COMBINATION,
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
    # M03 恒为 stop：Provider Protocol 尚未回传上游终态原因，真实映射
    # （length 等）随 M04 落地；Literal 锁死当前唯一取值，M04 扩展即显式变更。
    finish_reason: Literal["stop"]


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
    # M03 流式只产出文本增量，delta 不含 role 首块（openai SDK 对缺 role 容忍）。
    model_config = ConfigDict(extra="forbid")

    content: str


class ChunkChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    delta: ChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChunkChoice]
    # usage 字段随 M04 流式 usage 回传一起加入；本里程碑 chunk 无 usage 键。


# ---------------------------------------------------------------------------
# 方言翻译：OpenAI 请求/响应 <-> 内部协议（design.md §3.3 的一次性翻译点）
# ---------------------------------------------------------------------------


def _to_internal_request(request: ChatCompletionRequest) -> LLMRequest:
    # stream 不进内部请求：它只在本端点内选分支，内部协议无此字段。
    # response_format 不译成 response_schema——OpenAI 的 response_format 是
    # 包装结构（type: json_object / json_schema），直接当 schema 用会对内容
    # 做错误校验；其内部结构的消费在 M06-M08 按设计接线。
    return LLMRequest(model=request.model, messages=request.messages, prompt=request.prompt)


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
                finish_reason="stop",
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


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
    # stream + response_format 互斥沿用 demo 的 unsupported_combination（400，
    # controller 裁决）。在端点层以 GatewayError 抛稳定码，而不是放进 Pydantic
    # validator——后者只能给 code=None 的通用 400，会丢失稳定错误码（任务 A
    # 报告给任务 B 的衔接点）。message 覆盖为 OpenAI 面的字段名：注册表默认
    # message 引用的是旧协议字段 response_schema，双冻结不可改，动态覆盖是
    # 注册表允许的唯一合法覆盖面（同 missing_prompt_variable 先例）。
    if request.stream and request.response_format is not None:
        raise GatewayError(UNSUPPORTED_COMBINATION, message="流式输出不支持 response_format")

    internal_request = _to_internal_request(request)

    if request.stream:
        # 统一端点下 stream=true 直接走流式分支（controller 裁决）：M03 起
        # use_stream_endpoint 运行时不再抛出（注册表双冻结保留，仅作历史码）。
        # 请求级校验（模型白名单、Prompt 渲染）保持在返回 StreamingResponse
        # 之前——与 demo 一致：请求问题走 HTTP 错误，只有上游/流中途失败才
        # 走流内错误事件。
        validate_model(internal_request.model, None)
        build_messages(internal_request)
        return StreamingResponse(
            _chunk_stream(
                internal_request,
                completion_id=str(uuid4()),
                created=int(time.time()),
            ),
            media_type="text/event-stream",
        )

    response = await call_with_fallback(internal_request)
    return _to_chat_completion(response, created=int(time.time()))
