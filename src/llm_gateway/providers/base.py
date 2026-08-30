"""供应商 Adapter 的统一接口协议（M04 任务 1 定稿）。

编排层只面向本 Protocol 编程，不依赖具体供应商 SDK（design.md §2.2）；
新增供应商时实现本协议即可，对编排零波及。

M04 相对 M01 的三处定稿（design.md §3.6）：
- complete 返回 (content, usage, finish_reason) 三元组——demo 丢弃了
  finish_reason，导致截断无法识别（真实盲点）；output_truncated 关卡的
  消费在 M06/M08 编排层，本层只保证不丢；
- stream 产出 tagged 事件流（ContentDelta / StreamCompleted），替代裸
  str 增量——OpenAI 终态 chunk 与流式 usage 回传需要结构化载体（M03
  账本 deferred：流翻译器依赖注册表 KeyError 的 tagged 类型收编）；
- SDK 异常只允许出现在 providers/ 的具体 Adapter 模块内，向上只抛
  GatewayError（map_provider_error 是唯一映射出口，review 锚点：
  错误不穿透到响应体）。
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Final, Protocol

from llm_gateway.core.errors import MODEL_UNAVAILABLE, PROVIDER_OVERLOADED, GatewayError
from llm_gateway.core.schemas import Message, ModelConfig, Usage

# finish_reason 统一词表：内部只承认这两个值。design.md §3.7 的截断识别只
# 关心 length；其余上游终态原因（tool_calls 等）M04 消费不到，归一为 stop
# 并告警——宁可标注未知，不静默编造语义。
FINISH_REASON_STOP: Final[str] = "stop"
FINISH_REASON_LENGTH: Final[str] = "length"

# 编排层可重试的 provider 错误码：传输故障与上游过载。GATEWAY_MISCONFIGURED、
# UNKNOWN_MODEL 等配置/请求类错误首次失败即终止（重试节奏仍由编排层决定，
# 本集合只回答"该不该重试"，与旧 is_retryable 谓词同位替换）。
PROVIDER_RETRYABLE_CODES: Final[frozenset[str]] = frozenset({MODEL_UNAVAILABLE, PROVIDER_OVERLOADED})


@dataclass(frozen=True)
class ContentDelta:
    # 流式文本增量：上游 delta 文本片段的内部形态。
    text: str


@dataclass(frozen=True)
class StreamCompleted:
    # 流式终态事件：finish_reason 必带（截断识别依据），usage 仅在上游回传时
    # 有值（openai chat 需 include_usage 开关；anthropic / responses 原生回传）。
    finish_reason: str
    usage: Usage | None = None


def normalize_finish_reason(value: str | None) -> str:
    # 上游终态原因归一到内部词表。None（上游未给终态）按自然结束处理，不告警。
    if value == FINISH_REASON_STOP or value == FINISH_REASON_LENGTH:
        return value
    if value is not None:
        logging.getLogger("llm_gateway").warning("未知 finish_reason %r，按 stop 处理", value)
    return FINISH_REASON_STOP


def map_provider_error(exc: Exception) -> GatewayError:
    # SDK 异常 -> GatewayError 的唯一出口。本函数不 import 任何 SDK：上游 429
    # 通过 status_code 鸭子类型识别（openai / anthropic 的 APIStatusError 系
    # 异常都携带该属性），其余一律按传输故障处理——provider 模块负责先捕获
    # 自家 SDK 异常类型再转投这里，保证 SDK 类型不出 providers/。
    if getattr(exc, "status_code", None) == 429:
        return GatewayError(PROVIDER_OVERLOADED)
    return GatewayError(MODEL_UNAVAILABLE)


class Provider(Protocol):
    # 规定供应商 Adapter 的统一接口，业务流程不依赖具体 SDK。

    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
        # temperature / max_tokens：M03 白名单放行字段的协议承载（M04 接线），
        # None 表示调用方未指定——Adapter 不传参，不伪造上游默认值。
        # json_mode：response_format={"type":"json_object"}（无 schema）的
        # JSON 模式开关；json_schema 形态走 response_schema 既有链路。
        # 三个关键字参数带默认值：编排层 M04 尚未全部接线（Task C），先保
        # 旧调用点零改动。
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> tuple[str, Usage, str]: ...

    # stream 用普通 def 声明而非 async def：Adapter 的实现是 async generator，
    # 调用即返回 AsyncIterator（不产生 awaitable）。若协议声明成 async def，
    # 类型上承诺的是 Coroutine[..., AsyncIterator[str]]，任何 async generator
    # 实现都无法结构匹配，调用方的 `async for ... in provider.stream(...)`
    # 也会被判错。纯类型声明调整：Protocol 成员从不执行，无运行期影响。
    def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        # include_usage：openai chat 协议需显式向上游请求 usage 回传
        # （stream_options）；anthropic / responses 原生回传，忽略此开关。
        include_usage: bool = False,
    ) -> AsyncIterator[ContentDelta | StreamCompleted]: ...
