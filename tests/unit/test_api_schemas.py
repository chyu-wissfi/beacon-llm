"""api/schemas.py 的单元测试：OpenAI 请求字段白名单（M03 任务 1）。

白名单本身就是契约：字段集合必须恰好是 spec 点名的 7 个 OpenAI 标准字段 +
2 个扩展字段，多放一个字段就等于绕过不变量 #4（不支持的字段必须在调用模型
之前失败）。纯 Pydantic 层面的断言不经过 HTTP——HTTP 侧的 400 unsupported_field
翻译在 api/errors.py 的处理器里，由 test_api_errors.py 覆盖。
"""

from typing import Any

import pytest
from pydantic import ValidationError

from llm_gateway.api.schemas import ChatCompletionRequest, ValidationSelection
from llm_gateway.core.schemas import PromptSelection

# ---------------------------------------------------------------------------
# 公共构造：最小合法请求（只含必填字段），其余字段按用例需要叠加
# ---------------------------------------------------------------------------


def _minimal_request(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "你好"}],
    }
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# 白名单的封闭性
# ---------------------------------------------------------------------------


def test_whitelist_is_exactly_the_spec_field_set() -> None:
    # 字段集合逐字对齐 M03 spec 任务 1：7 个 OpenAI 标准字段 + prompt/validation
    # 扩展字段。用相等断言（而非子集）锁死封闭性——有人加字段必须先改这里，
    # 避免"顺手放行"绕过白名单评审。
    assert set(ChatCompletionRequest.model_fields) == {
        "model",
        "messages",
        "stream",
        "temperature",
        "max_tokens",
        "response_format",
        "stream_options",
        "prompt",
        "validation",
    }


def test_unknown_field_rejected_as_extra_forbidden() -> None:
    # 不变量 #4 的解析期半边：白名单外字段在 Pydantic 层报 extra_forbidden，
    # HTTP 侧由 api/errors.py 翻译成 400 unsupported_field。
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest.model_validate(_minimal_request(top_k=5))
    errors = exc_info.value.errors()
    assert [error["type"] for error in errors] == ["extra_forbidden"]
    assert errors[0]["loc"] == ("top_k",)


def test_unknown_nested_field_in_stream_options_rejected() -> None:
    # stream_options 已被放行，但其内部同样遵守 extra="forbid" 约定——
    # 放行一个字段不等于放行它内部的任意键。
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest.model_validate(
            _minimal_request(stream=True, stream_options={"include_usage": True, "foo": 1})
        )
    errors = exc_info.value.errors()
    assert [error["type"] for error in errors] == ["extra_forbidden"]
    assert errors[0]["loc"] == ("stream_options", "foo")


# ---------------------------------------------------------------------------
# 字段解析与默认值
# ---------------------------------------------------------------------------


def test_minimal_request_uses_defaults() -> None:
    request = ChatCompletionRequest.model_validate(_minimal_request())
    assert request.stream is False
    assert request.temperature is None
    assert request.max_tokens is None
    assert request.response_format is None
    assert request.stream_options is None
    assert request.prompt is None
    assert request.validation is None


def test_openai_standard_fields_parse() -> None:
    request = ChatCompletionRequest.model_validate(
        _minimal_request(
            stream=True,
            temperature=0.5,
            max_tokens=128,
            response_format={"type": "json_object"},
            stream_options={"include_usage": True},
        )
    )
    assert request.stream is True
    assert request.temperature == 0.5
    assert request.max_tokens == 128
    # response_format 本里程碑只透传（M07/M08 消费内部结构），原值保留。
    assert request.response_format == {"type": "json_object"}
    assert request.stream_options is not None
    assert request.stream_options.include_usage is True


def test_extension_fields_parse_into_core_selections() -> None:
    # 扩展字段经 openai SDK 的 extra_body 通道提交：prompt 复用 core 的
    # PromptSelection（受控模板 + 变量），validation 是 {name, version} 点名。
    request = ChatCompletionRequest.model_validate(
        _minimal_request(
            prompt={"name": "support", "version": "v2", "variables": {"product": "X"}},
            validation={"name": "answer-shape", "version": "v1"},
        )
    )
    assert request.prompt == PromptSelection(name="support", version="v2", variables={"product": "X"})
    assert request.validation == ValidationSelection(name="answer-shape", version="v1")


def test_validation_selection_rejects_unknown_subfield() -> None:
    # validation 载体同样封闭：混入未知键按白名单外处理，而不是静默丢弃
    # （静默丢弃会让 M08 的 profile 选择悄悄失效）。
    with pytest.raises(ValidationError) as exc_info:
        ValidationSelection.model_validate({"name": "p", "version": "v1", "extra": 1})
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


# ---------------------------------------------------------------------------
# 取值约束（沿用 OpenAI 文档面，网关作为契约第一道防线）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_temperature_outside_openai_range_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_minimal_request(temperature=value))


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_minimal_request(max_tokens=0))


def test_messages_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_minimal_request(messages=[]))


def test_message_shape_inherits_core_message_constraints() -> None:
    # messages 复用 core.Message：未知 role / 空 content 在白名单层即拒绝，
    # 供应商消息格式差异仍只存在于 Provider 层（CONTEXT.md：Provider 条目）。
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_minimal_request(messages=[{"role": "tool", "content": "x"}]))
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_minimal_request(messages=[{"role": "user", "content": ""}]))
