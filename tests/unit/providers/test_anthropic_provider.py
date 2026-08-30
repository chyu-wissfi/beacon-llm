"""AnthropicProvider 单测（M04 任务 6）：SDK 方法层 mock。

与 openai_compatible 单测的分工差异（实测裁定，非偏离规格）：spec 写"respx
注入 anthropic 的 httpx 层"，但 anthropic 1.2.0 实测走 httpx2 传输且无
legacy httpx1 分支（httpx2.URL 参数与 httpx1 build_request 不兼容，替换客户端
工厂方案已实测失败），respx 0.23.1 视野外——因此退而求其次：在
AsyncMessages.create 方法层 mock，断言面不变（错误映射后的稳定错误码、
语义参数是否到达请求参数面、finish_reason 映射、流式事件序列）。

断言的是行为边界，不是 SDK 内部细节；上游响应体用与真实 Messages API 一致的
最小 JSON 形态，经 SDK 自家类型解析（TypeAdapter），不手造 SDK 模型。
"""

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import TypeAdapter

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import Message
from llm_gateway.providers.anthropic_provider import AnthropicProvider
from llm_gateway.providers.base import ContentDelta, StreamCompleted

pytestmark = pytest.mark.asyncio

_MESSAGES = [Message(role="user", content="你好")]
_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

_EVENT_ADAPTER = TypeAdapter(anthropic.types.MessageStreamEvent)


# -- 上游响应体构造器（与真实 Messages API 一致的最小形态）--


def _message_body(*, stop_reason: str | None = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "你好"}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _stream_event_dicts(*, stop_reason: str = "end_turn") -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "你好"}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
        {"type": "message_stop"},
    ]


# -- AsyncMessages.create 方法层 mock --


class CreateMock:
    # 每个用例一个新实例：captured 记录最后一次调用参数面，行为由用例配置
    # （非流式响应体 / 流式事件序列 / 直接抛异常三选一）。
    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}
        self._body: dict[str, Any] = _message_body()
        self._events: list[dict[str, Any]] | None = None
        self._exception: Exception | None = None

    def respond(self, body: dict[str, Any]) -> None:
        self._body = body

    def stream_events(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def raise_exc(self, exc: Exception) -> None:
        self._exception = exc

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        # self 是 AsyncMessages 实例（monkeypatch 装在类上）；这里只收参数。
        del args
        self.captured = dict(kwargs)
        if self._exception is not None:
            raise self._exception
        if kwargs.get("stream"):
            events = [
                _EVENT_ADAPTER.validate_python(event) for event in (self._events or _stream_event_dicts())
            ]

            async def _gen() -> AsyncIterator[Any]:
                for event in events:
                    yield event

            return _gen()
        return anthropic.types.Message.model_validate(self._body)


@pytest.fixture
def create_mock(monkeypatch) -> CreateMock:
    mock = CreateMock()
    monkeypatch.setattr("anthropic.resources.messages.AsyncMessages.create", mock.create)
    return mock


def _httpx2_request() -> httpx2.Request:
    return httpx2.Request("POST", "http://upstream.test/v1/messages")


# -- 非流式 --


async def test_complete_success(anthropic_config, create_mock):
    content, usage, finish_reason = await AnthropicProvider().complete(
        anthropic_config, _MESSAGES, 30.0, None
    )
    assert content == "你好"
    assert (usage.input_tokens, usage.output_tokens) == (5, 3)
    assert finish_reason == "stop"
    assert create_mock.captured["model"] == "claude-test"
    assert create_mock.captured["messages"] == [{"role": "user", "content": "你好"}]


async def test_system_messages_extracted_to_system_param(anthropic_config, create_mock):
    messages = [
        Message(role="system", content="甲"),
        Message(role="system", content="乙"),
        Message(role="user", content="你好"),
    ]
    await AnthropicProvider().complete(anthropic_config, messages, 30.0, None)
    # system 角色提取合并为顶层 system 参数，不留在 messages 里。
    assert create_mock.captured["system"] == "甲\n\n乙"
    assert create_mock.captured["messages"] == [{"role": "user", "content": "你好"}]


async def test_max_tokens_defaults_4096_when_absent(anthropic_config, create_mock):
    # Anthropic 必填参数：调用方未指定时网关裁决默认 4096（代码注释注明出处）。
    await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None)
    assert create_mock.captured["max_tokens"] == 4096

    await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None, max_tokens=128)
    assert create_mock.captured["max_tokens"] == 128


async def test_temperature_passthrough(anthropic_config, create_mock):
    await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None, temperature=0.3)
    assert create_mock.captured["temperature"] == 0.3

    await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None)
    assert "temperature" not in create_mock.captured


async def test_finish_reason_max_tokens_maps_length(anthropic_config, create_mock):
    create_mock.respond(_message_body(stop_reason="max_tokens"))
    _, _, finish_reason = await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None)
    assert finish_reason == "length"


async def test_unknown_stop_reason_falls_back_to_stop(anthropic_config, create_mock):
    # tool_use 等词表外值归一为 stop（normalize 告警，行为断言只看结果）。
    create_mock.respond(_message_body(stop_reason="tool_use"))
    _, _, finish_reason = await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None)
    assert finish_reason == "stop"


async def test_json_object_schema_injection(anthropic_config, create_mock):
    await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, _SCHEMA)
    assert "JSON Schema" in create_mock.captured["system"]


async def test_json_schema_mode_rejected(anthropic_config, create_mock):
    # 裁定：配置 json_schema 的 anthropic 模型显式报不支持，不静默降级。
    json_schema_config = replace(anthropic_config, structured_output_mode="json_schema")
    with pytest.raises(GatewayError) as exc_info:
        await AnthropicProvider().complete(json_schema_config, _MESSAGES, 30.0, _SCHEMA)
    assert exc_info.value.code == "structured_output_unsupported"
    assert create_mock.captured == {}  # 拒绝发生在任何上游调用之前


# -- 异常映射：SDK 类型不出 provider --


async def test_missing_api_key_maps_gateway_misconfigured(anthropic_config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(GatewayError) as exc_info:
        await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None)
    assert exc_info.value.code == "gateway_misconfigured"


async def test_upstream_429_maps_provider_overloaded(anthropic_config, create_mock):
    create_mock.raise_exc(
        anthropic.RateLimitError("rate limited", response=httpx2.Response(429, request=_httpx2_request()), body=None)
    )
    with pytest.raises(GatewayError) as exc_info:
        await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None)
    assert exc_info.value.code == "provider_overloaded"
    assert exc_info.value.status_code == 429


async def test_timeout_maps_model_unavailable(anthropic_config, create_mock):
    create_mock.raise_exc(anthropic.APITimeoutError(request=_httpx2_request()))
    with pytest.raises(GatewayError) as exc_info:
        await AnthropicProvider().complete(anthropic_config, _MESSAGES, 30.0, None)
    assert exc_info.value.code == "model_unavailable"


# -- 流式 --


async def test_stream_events(anthropic_config, create_mock):
    create_mock.stream_events(_stream_event_dicts())
    events = [
        event async for event in AnthropicProvider().stream(anthropic_config, _MESSAGES, 30.0)
    ]
    # Anthropic 原生回传 usage（input 在 message_start、output 在 message_delta
    # 累积），无需 include_usage 开关。
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.usage is not None
    assert events == [ContentDelta("你好"), StreamCompleted("stop", completed.usage)]
    assert (completed.usage.input_tokens, completed.usage.output_tokens) == (5, 3)


async def test_stream_max_tokens_maps_length(anthropic_config, create_mock):
    create_mock.stream_events(_stream_event_dicts(stop_reason="max_tokens"))
    events = [
        event async for event in AnthropicProvider().stream(anthropic_config, _MESSAGES, 30.0)
    ]
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.finish_reason == "length"


async def test_stream_error_mid_iteration_maps_gateway_error(anthropic_config, monkeypatch):
    # 流中途断开：SDK 异常同样不穿透，映射为 MODEL_UNAVAILABLE。
    async def _broken_create(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs

        async def _gen() -> AsyncIterator[Any]:
            yield _EVENT_ADAPTER.validate_python(_stream_event_dicts()[1])
            raise anthropic.APITimeoutError(request=_httpx2_request())

        return _gen()

    monkeypatch.setattr("anthropic.resources.messages.AsyncMessages.create", _broken_create)
    with pytest.raises(GatewayError) as exc_info:
        _ = [event async for event in AnthropicProvider().stream(anthropic_config, _MESSAGES, 30.0)]
    assert exc_info.value.code == "model_unavailable"
