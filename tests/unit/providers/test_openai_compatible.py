"""OpenAICompatibleProvider 单测（M04 任务 6）：respx 注入 openai 的 httpx 层。

断言的是行为边界：错误映射后的稳定错误码、语义参数是否到达上游请求体、
finish_reason 词表归一、流式事件序列——不断言 SDK 内部细节。上游响应体用
与真实 API 一致的最小 JSON 形态，避免钉死 SDK 模型的非契约字段。
"""

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest
import respx

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import Message
from llm_gateway.providers.base import ContentDelta, StreamCompleted
from llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

pytestmark = pytest.mark.asyncio

_MESSAGES = [Message(role="user", content="你好")]
_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

# -- 上游响应体构造器（与真实 API 一致的最小形态）--


def _chat_completion_body(finish_reason: str | None = "stop") -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "你好"},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _responses_body(
    *,
    status: str = "completed",
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "deepseek-v4-flash",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "你好", "annotations": []}],
            }
        ],
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
    }


def _sse(lines: list[str]) -> httpx.Response:
    payload = "".join(f"{line}\n\n" for line in lines)
    return httpx.Response(200, content=payload.encode(), headers={"content-type": "text/event-stream"})


def _chat_sse_chunks(*, finish_reason: str | None = "stop", with_usage: bool = False) -> httpx.Response:
    lines = [
        json.dumps(
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "deepseek-v4-flash",
                "choices": [{"index": 0, "delta": {"content": "你好"}, "finish_reason": None}],
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "deepseek-v4-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            },
            ensure_ascii=False,
        ),
    ]
    if with_usage:
        lines.append(
            json.dumps(
                {
                    "id": "cmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "deepseek-v4-flash",
                    "choices": [],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }
            )
        )
    lines.append("[DONE]")
    return _sse([f"data: {line}" for line in lines])


def _responses_sse_events(*, status: str = "completed") -> httpx.Response:
    lines = [
        json.dumps(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "你好",
                "sequence_number": 1,
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "response.completed" if status == "completed" else "response.incomplete",
                "sequence_number": 2,
                "response": _responses_body(
                    status=status,
                    incomplete_reason="max_output_tokens" if status == "incomplete" else None,
                ),
            },
            ensure_ascii=False,
        ),
    ]
    return _sse([f"data: {line}" for line in lines])


# -- 非流式：chat 模式 --


async def test_chat_complete_success(chat_config):
    with respx.mock(assert_all_mocked=True) as mock:
        route = mock.post("https://upstream.test/v1/chat/completions").respond(json=_chat_completion_body())
        content, usage, finish_reason = await OpenAICompatibleProvider().complete(
            chat_config, _MESSAGES, 30.0, None
        )
    assert content == "你好"
    assert (usage.input_tokens, usage.output_tokens) == (5, 3)
    assert finish_reason == "stop"
    assert route.call_count == 1


async def test_chat_semantic_params_passthrough(chat_config):
    # temperature/max_tokens 指定 -> 进请求体；未指定 -> 键缺席（不伪造上游默认）。
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_completion_body())

    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").mock(side_effect=_capture)
        await OpenAICompatibleProvider().complete(
            chat_config, _MESSAGES, 30.0, None, temperature=0.3, max_tokens=128
        )
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 128

    captured.clear()
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").mock(side_effect=_capture)
        await OpenAICompatibleProvider().complete(chat_config, _MESSAGES, 30.0, None)
    assert "temperature" not in captured
    assert "max_tokens" not in captured


async def test_chat_json_schema_mode(chat_config):
    chat_config = replace(chat_config, structured_output_mode="json_schema")
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_completion_body())

    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").mock(side_effect=_capture)
        await OpenAICompatibleProvider().complete(chat_config, _MESSAGES, 30.0, _SCHEMA)
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["schema"] == _SCHEMA
    assert captured["response_format"]["json_schema"]["strict"] is True


async def test_chat_json_object_injection(chat_config):
    # json_object + schema：供应商约束 + system 注入 schema 双通道。
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_completion_body())

    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").mock(side_effect=_capture)
        await OpenAICompatibleProvider().complete(chat_config, _MESSAGES, 30.0, _SCHEMA)
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["messages"][0]["role"] == "system"
    assert "JSON Schema" in captured["messages"][0]["content"]


async def test_chat_json_mode_without_schema(chat_config):
    # json_mode（response_format=json_object，无 schema）：只开 JSON 模式，
    # 无 system 注入——本地 schema 校验无从谈起，不谎报约束。
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_completion_body())

    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").mock(side_effect=_capture)
        await OpenAICompatibleProvider().complete(chat_config, _MESSAGES, 30.0, None, json_mode=True)
    assert captured["response_format"] == {"type": "json_object"}
    assert all(m["role"] != "system" for m in captured["messages"])


# -- 非流式：Responses 模式 --


async def test_responses_complete_success(responses_config):
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_responses_body())

    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/responses").mock(side_effect=_capture)
        content, usage, finish_reason = await OpenAICompatibleProvider().complete(
            responses_config, _MESSAGES, 30.0, None, max_tokens=64
        )
    assert content == "你好"
    assert (usage.input_tokens, usage.output_tokens) == (5, 3)
    assert finish_reason == "stop"
    # max_tokens 在 Responses 协议的对应物是 max_output_tokens。
    assert captured["max_output_tokens"] == 64
    assert "max_tokens" not in captured


async def test_responses_incomplete_maps_length(responses_config):
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/responses").respond(
            json=_responses_body(status="incomplete", incomplete_reason="max_output_tokens")
        )
        _, _, finish_reason = await OpenAICompatibleProvider().complete(
            responses_config, _MESSAGES, 30.0, None
        )
    assert finish_reason == "length"


async def test_responses_json_schema_text_format(responses_config):
    # json_schema 走 text.format；json_object 走 instructions 注入。
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_responses_body())

    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/responses").mock(side_effect=_capture)
        await OpenAICompatibleProvider().complete(responses_config, _MESSAGES, 30.0, _SCHEMA)
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["schema"] == _SCHEMA

    captured.clear()
    responses_json_object = replace(responses_config, structured_output_mode="json_object")
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/responses").mock(side_effect=_capture)
        await OpenAICompatibleProvider().complete(responses_json_object, _MESSAGES, 30.0, _SCHEMA)
    assert captured["text"]["format"] == {"type": "json_object"}
    assert "JSON Schema" in captured["instructions"]


# -- 异常映射：SDK 类型不出 provider --


async def test_upstream_429_maps_provider_overloaded(chat_config):
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").respond(429)
        with pytest.raises(GatewayError) as exc_info:
            await OpenAICompatibleProvider().complete(chat_config, _MESSAGES, 30.0, None)
    assert exc_info.value.code == "provider_overloaded"
    assert exc_info.value.status_code == 429


async def test_transport_error_maps_model_unavailable(chat_config):
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        with pytest.raises(GatewayError) as exc_info:
            await OpenAICompatibleProvider().complete(chat_config, _MESSAGES, 30.0, None)
    assert exc_info.value.code == "model_unavailable"


async def test_missing_api_key_maps_gateway_misconfigured(chat_config, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(GatewayError) as exc_info:
        await OpenAICompatibleProvider().complete(chat_config, _MESSAGES, 30.0, None)
    assert exc_info.value.code == "gateway_misconfigured"


# -- 流式：chat 模式 --


async def test_chat_stream_events(chat_config):
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").return_value = _chat_sse_chunks()
        events = [
            event
            async for event in OpenAICompatibleProvider().stream(chat_config, _MESSAGES, 30.0)
        ]
    assert events == [ContentDelta("你好"), StreamCompleted("stop", None)]


async def test_chat_stream_include_usage_switch(chat_config):
    # include_usage=True 才向上游传 stream_options，且回传的 usage 进终态事件。
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _chat_sse_chunks(with_usage=True)

    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").mock(side_effect=_capture)
        events = [
            event
            async for event in OpenAICompatibleProvider().stream(
                chat_config, _MESSAGES, 30.0, include_usage=True
            )
        ]
    assert captured["stream_options"] == {"include_usage": True}
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.finish_reason == "stop"
    assert completed.usage is not None
    assert (completed.usage.input_tokens, completed.usage.output_tokens) == (5, 3)

    captured.clear()
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").mock(side_effect=_capture)
        _ = [
            event
            async for event in OpenAICompatibleProvider().stream(chat_config, _MESSAGES, 30.0)
        ]
    assert "stream_options" not in captured


async def test_chat_stream_unknown_finish_reason_falls_back_to_stop(chat_config):
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/chat/completions").return_value = _chat_sse_chunks(
            finish_reason="tool_calls"
        )
        events = [
            event
            async for event in OpenAICompatibleProvider().stream(chat_config, _MESSAGES, 30.0)
        ]
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.finish_reason == "stop"


# -- 流式：Responses 模式 --


async def test_responses_stream_events(responses_config):
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/responses").return_value = _responses_sse_events()
        events = [
            event
            async for event in OpenAICompatibleProvider().stream(responses_config, _MESSAGES, 30.0)
        ]
    # Responses 原生回传 usage，无需 include_usage 开关；不等价于 chat 的
    # usage=None 终态。
    assert events[0] == ContentDelta("你好")
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.finish_reason == "stop"
    assert completed.usage is not None
    assert (completed.usage.input_tokens, completed.usage.output_tokens) == (5, 3)


async def test_responses_stream_incomplete_maps_length(responses_config):
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post("https://upstream.test/v1/responses").return_value = _responses_sse_events(
            status="incomplete"
        )
        events = [
            event
            async for event in OpenAICompatibleProvider().stream(responses_config, _MESSAGES, 30.0)
        ]
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.finish_reason == "length"
