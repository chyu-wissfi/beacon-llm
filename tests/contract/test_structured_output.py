"""双重校验与 Validation Profile 的 HTTP 面契约测试（M08 任务 5）。

不变量 #9（供应商约束 + 本地双重校验）与 #10（JSON 合法但业务非法不进
Agent Loop）的端到端断言：
- 结构合法但业务非法的输出绝不进入响应：走恰好 1 次修复后终报
  business_validation_failed（502）；
- 本地 jsonschema 捕获绕过供应商约束的输出（Fake Adapter 忠实返回坏内容，
  结构关由网关本地兜底）；
- 未注册 Profile：400 unknown_validation_profile 且上游请求数 == 0；
- 修复链：每种失败类型恰好 1 次修复调用（计数断言）；
- stream + validation 互斥（业务关卡无法在流式增量上执行）。

Fake Adapter 用例把剧本实例挂到 openai_compatible 键（主备模型共享实例，
adapter.attempts 即上游请求总数）；mock_upstream 仍全程生效——任何真 HTTP
请求会因未注册路由直接失败（离线保证的双重保险）。
"""

import json
from typing import Any

import pytest

from llm_gateway.providers import PROVIDER_REGISTRY
from llm_gateway.providers.fake import (
    FakeAdapter,
    InvalidOutput,
    Scenario,
    ScenarioSequence,
    Success,
)
from llm_gateway.services.trace_service import CALL_TRACES
from tests.contract.helpers import (
    ANSWER_SCHEMA,
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    chat_request,
    completion,
)

pytestmark = pytest.mark.asyncio

CHAT_PATH = "/v1/chat/completions"

# 与 order_decision/v1 Profile 字段结构对齐的 JSON Schema（结构关）：
# 业务非法样本（approve+reject 同真）在此结构下完全合法——供应商约束与
# 本地结构关都放行，业务关才是最后防线（不变量 #10 的靶场）。
ORDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["order_id", "approve", "reject"],
    "properties": {
        "order_id": {"type": "string"},
        "approve": {"type": "boolean"},
        "reject": {"type": "boolean"},
        "escalate": {"type": "boolean"},
        "escalation_reason": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

BUSINESS_INVALID = json.dumps({"order_id": "A-1", "approve": True, "reject": True})
BUSINESS_VALID = json.dumps({"order_id": "A-1", "approve": True, "reject": False})
ORDER_VALIDATION = {"name": "order_decision", "version": "v1"}


def _order_response_format() -> dict[str, Any]:
    # OpenAI json_schema 形态：内层 schema 由 api 层翻译进 response_schema 链路。
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "order_decision_shape",
            "strict": True,
            "schema": ORDER_SCHEMA,
        },
    }


@pytest.fixture
def fake_provider(monkeypatch):
    """把剧本化 FakeAdapter 挂到 openai_compatible 键并返回实例。

    主备模型共享同一实例：adapter.attempts 是全链路上游请求总数，
    "每种失败恰好 1 次修复"由此可断言。
    """

    def _install(scenario: Scenario) -> FakeAdapter:
        adapter = FakeAdapter(scenario)
        monkeypatch.setitem(PROVIDER_REGISTRY, "openai_compatible", adapter)
        return adapter

    return _install


def _error_body(response_json: dict[str, Any]) -> dict[str, Any]:
    return response_json["error"]


# ---------------------------------------------------------------------------
# 双重校验：结构合法但业务非法 -> 不进响应、走修复、终报（不变量 #10）
# ---------------------------------------------------------------------------


async def test_business_rule_blocked_after_single_repair_never_in_response(
    client, fake_provider, mock_upstream
):
    # Fake Adapter 返回"结构合法但业务非法"的 JSON：两次都违规 -> 恰好 1 次
    # 修复后终报 business_validation_failed（502）；非法内容绝不进入任何响应体。
    adapter = fake_provider(
        ScenarioSequence(
            [
                InvalidOutput(content=BUSINESS_INVALID),
                InvalidOutput(content=BUSINESS_INVALID),
            ]
        )
    )
    response = await client.post(
        CHAT_PATH,
        json=chat_request(
            response_format=_order_response_format(), validation=ORDER_VALIDATION
        ),
    )
    assert response.status_code == 502
    error = _error_body(response.json())
    assert error["code"] == "business_validation_failed"
    assert error["type"] == "api_error"
    assert BUSINESS_INVALID not in response.text  # 非法输出不进响应（不进 Agent Loop）
    assert adapter.attempts == 2  # 原始 + 恰好 1 次修复
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.error_code == "business_validation_failed"


async def test_business_repair_success_returns_valid_output(
    client, fake_provider, mock_upstream
):
    # 修复链正向半边：首次业务非法、修复后合法 -> 200 且响应是修复后的合法输出。
    adapter = fake_provider(
        ScenarioSequence(
            [InvalidOutput(content=BUSINESS_INVALID), Success(content=BUSINESS_VALID)]
        )
    )
    response = await client.post(
        CHAT_PATH,
        json=chat_request(
            response_format=_order_response_format(), validation=ORDER_VALIDATION
        ),
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == BUSINESS_VALID
    assert adapter.attempts == 2
    assert CALL_TRACES[-1].status == "success"


# ---------------------------------------------------------------------------
# 结构双重：本地 jsonschema 捕获绕过供应商约束的输出（不变量 #9）
# ---------------------------------------------------------------------------


async def test_jsonschema_local_catches_output_bypassing_vendor_constraints(
    client, fake_provider, mock_upstream
):
    # Fake Adapter 忠实返回违反 response_schema 的内容（供应商约束被绕过）：
    # 本地 jsonschema 兜底捕获 -> 修复仍违规 -> 终报 schema_validation_failed。
    adapter = fake_provider(
        ScenarioSequence(
            [
                InvalidOutput(content=json.dumps({"wrong": 1})),
                InvalidOutput(content=json.dumps({"wrong": 1})),
            ]
        )
    )
    response = await client.post(
        CHAT_PATH,
        json=chat_request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_shape",
                    "strict": True,
                    "schema": ANSWER_SCHEMA,
                },
            }
        ),
    )
    assert response.status_code == 502
    assert _error_body(response.json())["code"] == "schema_validation_failed"
    assert adapter.attempts == 2  # 本地关卡触发恰好 1 次修复


# ---------------------------------------------------------------------------
# 未注册 Profile：400 + 上游请求数 == 0（spec 任务 3）
# ---------------------------------------------------------------------------


async def test_unknown_validation_profile_rejected_400_zero_upstream(
    client, mock_upstream
):
    upstream = mock_upstream.post(
        PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL
    ).respond(200, json=completion("不应被调用"))
    response = await client.post(
        CHAT_PATH,
        json=chat_request(validation={"name": "no_such_profile", "version": "v9"}),
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unknown_validation_profile"
    assert error["type"] == "invalid_request_error"
    assert "no_such_profile" in error["message"]
    assert upstream.call_count == 0  # 准入之前拒绝，上游零请求


async def test_stream_with_validation_rejected_400_zero_upstream(client, mock_upstream):
    # stream + validation 互斥（controller 裁决）：业务关卡需要完整输出，
    # 流式增量上无法执行——放行即绕过不变量 #9/#10。
    upstream = mock_upstream.post(
        PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL
    ).respond(200, json=completion("不应被调用"))
    response = await client.post(
        CHAT_PATH, json=chat_request(stream=True, validation=ORDER_VALIDATION)
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unsupported_combination"
    assert upstream.call_count == 0


# ---------------------------------------------------------------------------
# 修复链：每种失败类型恰好 1 次修复调用（计数断言）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scenario", "overrides", "expected_code"),
    [
        pytest.param(
            ScenarioSequence(
                [InvalidOutput(content="{bad"), InvalidOutput(content="{bad")]
            ),
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_shape",
                        "strict": True,
                        "schema": ANSWER_SCHEMA,
                    },
                }
            },
            "invalid_json",
            id="invalid_json",
        ),
        pytest.param(
            ScenarioSequence(
                [
                    Success(content="half", finish_reason="length"),
                    Success(content="half", finish_reason="length"),
                ]
            ),
            {},
            "output_truncated",
            id="output_truncated",
        ),
        pytest.param(
            ScenarioSequence(
                [
                    InvalidOutput(content=json.dumps({"answer": 123})),
                    InvalidOutput(content=json.dumps({"answer": 123})),
                ]
            ),
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_shape",
                        "strict": True,
                        "schema": ANSWER_SCHEMA,
                    },
                }
            },
            "schema_validation_failed",
            id="schema_validation_failed",
        ),
        pytest.param(
            ScenarioSequence(
                [
                    InvalidOutput(content=BUSINESS_INVALID),
                    InvalidOutput(content=BUSINESS_INVALID),
                ]
            ),
            {
                "response_format": _order_response_format(),
                "validation": ORDER_VALIDATION,
            },
            "business_validation_failed",
            id="business_validation_failed",
        ),
    ],
)
async def test_repair_chain_single_repair_per_failure_type(
    client, fake_provider, mock_upstream, scenario, overrides, expected_code
):
    # spec 任务 5：每层关卡失败各走恰好 1 次修复调用——上游请求总数 == 2
    # （原始 + 1 次修复），修复仍失败才终报对应稳定码。
    adapter = fake_provider(scenario)
    response = await client.post(CHAT_PATH, json=chat_request(**overrides))
    assert response.status_code == 502
    assert _error_body(response.json())["code"] == expected_code
    assert adapter.attempts == 2
    assert CALL_TRACES[-1].error_code == expected_code
