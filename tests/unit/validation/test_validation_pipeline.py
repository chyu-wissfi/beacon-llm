"""业务校验流水线服务层回归（M08 任务 5）。

直调 call_with_fallback + Fake Adapter 剧本，断言行为边界（上游请求计数、
终态码、trace、修复反馈内容）：
- 结构合法但业务非法的输出绝不进入响应，走恰好 1 次修复后终报
  business_validation_failed（不变量 #10）；
- 无 response_schema 仅有 Profile 时 json.loads + 业务关同样生效；
- 修复反馈携带对应错误细节（业务违规项原文）；
- 未注册 Profile 在 Run 状态机启动前拒绝：零上游请求、零 trace。
"""

import json
from typing import Any

import pytest
import pytest_asyncio

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import LLMRequest, Message, ModelConfig, Usage
from llm_gateway.providers.fake import (
    FakeAdapter,
    InvalidOutput,
    ScenarioSequence,
    Success,
)
from llm_gateway.services import invocation as inv
from llm_gateway.services.trace_service import CALL_TRACES, flush_pending
from llm_gateway.storage.engine import MEMORY_DB_URL, configure_engine, dispose_engine
from tests.unit.validation.conftest import PRIMARY, no_sleep

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _trace_db():
    # M09：终态会调度 trace 落库；本文件全异步，模块级注入内存库即可（同目录
    # 的同步用例文件不触发落库，不受影响）。
    configure_engine(MEMORY_DB_URL)
    yield
    await flush_pending()
    await dispose_engine()


# 与 OrderDecision 字段结构对齐的 JSON Schema（结构关）：业务非法样本
# （approve+reject 同真）在此结构下完全合法——两层关卡的分工由此成立。
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


def _request(**overrides: Any) -> LLMRequest:
    payload: dict[str, Any] = {
        "model": PRIMARY,
        "messages": [Message(role="user", content="请给出订单决策")],
    }
    payload.update(overrides)
    return LLMRequest(**payload)


async def test_business_invalid_never_enters_response_repair_then_terminal(
    topology, fake_provider
):
    # spec 任务 5 首条：结构合法但业务非法 -> 不进入响应、走修复、终报
    # business_validation_failed；恰好 1 次修复（计数断言）。
    topology()
    adapter = fake_provider(
        ScenarioSequence(
            [
                InvalidOutput(content=BUSINESS_INVALID),
                InvalidOutput(content=BUSINESS_INVALID),
            ]
        )
    )
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(
            _request(response_schema=ORDER_SCHEMA, validation=ORDER_VALIDATION),
            sleep=no_sleep,
        )
    assert exc_info.value.code == "business_validation_failed"
    assert adapter.attempts == 2  # 原始 + 恰好 1 次修复
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.error_code == "business_validation_failed"
    assert trace.attempts == 2


async def test_business_repair_success_enters_response(topology, fake_provider):
    # 修复链的正向半边：首次业务非法、修复后合法 -> 成功响应携带修复后的输出。
    topology()
    adapter = fake_provider(
        ScenarioSequence(
            [InvalidOutput(content=BUSINESS_INVALID), Success(content=BUSINESS_VALID)]
        )
    )
    response = await inv.call_with_fallback(
        _request(response_schema=ORDER_SCHEMA, validation=ORDER_VALIDATION),
        sleep=no_sleep,
    )
    assert response.content == BUSINESS_VALID
    assert response.parsed == json.loads(BUSINESS_VALID)
    assert response.attempts == 2
    assert adapter.attempts == 2
    assert CALL_TRACES[-1].status == "success"


async def test_business_gate_runs_without_response_schema(topology, fake_provider):
    # 仅有 Profile 无 schema：json.loads + 业务关仍生效（流水线按"任一关卡存在
    # 即解析"触发），终报业务码而非结构码。
    topology()
    fake_provider(
        ScenarioSequence(
            [
                InvalidOutput(content=BUSINESS_INVALID),
                InvalidOutput(content=BUSINESS_INVALID),
            ]
        )
    )
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(
            _request(validation=ORDER_VALIDATION), sleep=no_sleep
        )
    assert exc_info.value.code == "business_validation_failed"


async def test_repair_feedback_carries_business_rule_detail(topology, monkeypatch):
    # 修复反馈携带对应错误细节（spec 任务 1）：第二次调用的消息里能看到
    # 业务违规项原文（互斥规则理由）与失败关卡码。
    topology()

    class _RecordingAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__(
                ScenarioSequence(
                    [
                        InvalidOutput(content=BUSINESS_INVALID),
                        InvalidOutput(content=BUSINESS_INVALID),
                    ]
                )
            )
            self.seen_messages: list[list[Message]] = []

        async def complete(
            self,
            config: ModelConfig,
            messages: list[Message],
            timeout_seconds: float,
            response_schema: dict[str, Any] | None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            json_mode: bool = False,
        ) -> tuple[str, Usage, str]:
            self.seen_messages.append(list(messages))
            return await super().complete(
                config,
                messages,
                timeout_seconds,
                response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )

    adapter = _RecordingAdapter()
    monkeypatch.setitem(inv.PROVIDER_REGISTRY, "fake", adapter)
    with pytest.raises(GatewayError):
        await inv.call_with_fallback(
            _request(response_schema=ORDER_SCHEMA, validation=ORDER_VALIDATION),
            sleep=no_sleep,
        )
    assert len(adapter.seen_messages) == 2
    repair_feedback = adapter.seen_messages[1][-1]
    assert repair_feedback.role == "user"
    assert "business_validation_failed" in repair_feedback.content
    assert "互斥" in repair_feedback.content  # 业务违规项原文被携带


async def test_unknown_profile_rejected_before_run_zero_upstream_zero_trace(
    topology, fake_provider
):
    # 未注册 Profile：Run 状态机启动前拒绝（请求级 400 错误），零上游请求、
    # 零 trace（与白名单/能力不符同层）。
    topology()
    adapter = fake_provider(Success(content="不应被调用"))
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(
            _request(validation={"name": "no_such", "version": "v9"}), sleep=no_sleep
        )
    assert exc_info.value.code == "unknown_validation_profile"
    assert exc_info.value.status_code == 400
    assert adapter.attempts == 0
    assert CALL_TRACES == []
