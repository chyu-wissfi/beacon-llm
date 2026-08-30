"""字段白名单的 HTTP 面契约测试（M03 任务 1 的收尾）。

不变量 #4 的 HTTP 半边：白名单外字段 / 不合法取值一律 400（OpenAI 体），且
**在调用模型之前失败**——每例都断言上游请求计数为 0。Pydantic 解析层的封闭
性单测在 tests/unit/test_api_schemas.py，错误体翻译单测在
tests/unit/test_api_errors.py；本文件只钉真实 app 上的端到端行为。
"""

from typing import Any

import pytest

from tests.contract.helpers import (
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    chat_request,
    completion,
)

pytestmark = pytest.mark.asyncio

CHAT_PATH = "/v1/chat/completions"


def _error_body(response_json: dict[str, Any]) -> dict[str, Any]:
    return response_json["error"]


# ---------------------------------------------------------------------------
# 白名单外字段：400 unsupported_field + 上游零调用
# ---------------------------------------------------------------------------


async def test_unknown_top_level_field_rejected_400_no_upstream(client, mock_upstream):
    # 不变量 #4：白名单外字段在调用模型之前明确失败，错误 message 点名字段名。
    upstream = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("不应被调用")
    )
    response = await client.post(CHAT_PATH, json=chat_request(top_k=5))
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unsupported_field"
    assert error["type"] == "invalid_request_error"
    assert "top_k" in error["message"]
    assert upstream.call_count == 0


async def test_unknown_nested_field_rejected_400_no_upstream(client, mock_upstream):
    # 放行 stream_options 不等于放行其内部任意键：嵌套白名单外字段报完整路径。
    upstream = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("不应被调用")
    )
    response = await client.post(
        CHAT_PATH,
        json=chat_request(stream=True, stream_options={"include_usage": True, "foo": 1}),
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unsupported_field"
    assert "stream_options.foo" in error["message"]
    assert upstream.call_count == 0


async def test_multiple_unknown_fields_all_named_in_message(client, mock_upstream):
    # 并存多个白名单外字段时逐一点名（去重），调用方一次改全、不用试错。
    upstream = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("不应被调用")
    )
    response = await client.post(
        CHAT_PATH, json=chat_request(frequency_penalty=0.5, top_k=5)
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unsupported_field"
    assert "frequency_penalty" in error["message"]
    assert "top_k" in error["message"]
    assert upstream.call_count == 0


async def test_whitelist_field_on_openai_surface_takes_precedence_over_other_errors(client, mock_upstream):
    # 白名单外字段与缺字段并存：先报 unsupported_field（不变量 #4 的语义是
    # "不接受白名单外的请求"，字段层面的问题优先告知）。
    upstream_any = mock_upstream.route()
    response = await client.post(CHAT_PATH, json={"top_k": 5})
    assert response.status_code == 400
    assert _error_body(response.json())["code"] == "unsupported_field"
    assert upstream_any.call_count == 0


# ---------------------------------------------------------------------------
# 白名单内字段的不合法取值：400 code=None + 上游零调用
# ---------------------------------------------------------------------------


async def test_missing_required_field_rejected_400_without_code(client, mock_upstream):
    # 缺字段没有稳定注册码（注册表封闭、双冻结禁止扩员），code=None 但形态
    # 仍为 OpenAI 三键，message 携带字段路径。
    upstream_any = mock_upstream.route()
    response = await client.post(CHAT_PATH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 400
    error = _error_body(response.json())
    assert set(error) == {"message", "type", "code"}
    assert error["code"] is None
    assert error["type"] == "invalid_request_error"
    assert "model" in error["message"]
    assert upstream_any.call_count == 0


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"temperature": 2.1}, id="temperature 越上界"),
        pytest.param({"temperature": -0.1}, id="temperature 越下界"),
        pytest.param({"max_tokens": 0}, id="max_tokens 非正"),
        pytest.param({"messages": []}, id="messages 空列表"),
        pytest.param({"messages": [{"role": "tool", "content": "x"}]}, id="未知消息角色"),
    ],
)
async def test_out_of_range_values_rejected_400_no_upstream(client, mock_upstream, overrides):
    # 取值约束沿用 OpenAI 文档面（网关是契约第一道防线），越界值在白名单层
    # 400，不透传给上游。
    upstream_any = mock_upstream.route()
    response = await client.post(CHAT_PATH, json=chat_request(**overrides))
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] is None
    assert error["type"] == "invalid_request_error"
    assert upstream_any.call_count == 0


# ---------------------------------------------------------------------------
# 白名单内请求不受惩罚
# ---------------------------------------------------------------------------


async def test_whitelisted_optional_fields_accepted(client, mock_upstream):
    # 白名单内的可选字段（含本里程碑不透传的 temperature / max_tokens /
    # response_format——见 api/chat.py 的 M03 边界注释）不触发 400：拒绝它们
    # 会破坏"openai SDK 零改动可调"（不变量 #3）。
    upstream = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )
    response = await client.post(
        CHAT_PATH,
        json=chat_request(
            temperature=0.5,
            max_tokens=128,
            response_format={"type": "json_object"},
            stream=False,
        ),
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert upstream.call_count == 1
