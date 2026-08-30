"""invocation 唯一状态机单测（M06 任务 9）。

全部用 Fake Adapter 剧本 + 注入时钟/睡眠（不真睡、零随机依赖测试结论）：
预算共享、fallback 链与路由理由、修复调用、流式铁律、取消传播、唯一终态。
断言的是行为边界（上游请求计数、终态、trace 字段），不是内部实现细节。

夹具要点：primary/backup 双模型拓扑 patch 两个消费点（invocation 的
MODEL_CONFIGS 与 routing 的 MODEL_CONFIGS）；同一 FakeAdapter 实例挂到注册表
的 "fake" 键上——两个模型共享实例，adapter.attempts 即"上游请求总数"。
"""

import asyncio
from typing import Any

import pytest

from llm_gateway.core.breaker import get_breaker, reset_breakers
from llm_gateway.core.errors import RATE_LIMITED, GatewayError
from llm_gateway.core.schemas import LLMRequest, Message, ModelConfig
from llm_gateway.providers import PROVIDER_REGISTRY
from llm_gateway.providers.fake import (
    ConsecutiveThenSuccess,
    FakeAdapter,
    InvalidOutput,
    Scenario,
    ScenarioSequence,
    SlowSuccess,
    StreamInterrupt,
    Success,
    Timeout,
)
from llm_gateway.services import invocation as inv
from llm_gateway.services.routing import build_chain
from llm_gateway.services.run_context import Budget, TraceDraft
from llm_gateway.services.trace_service import CALL_TRACES

pytestmark = pytest.mark.asyncio

PRIMARY = "general-primary"
BACKUP = "general-backup"

# 结构化输出用 schema：与契约测试的 ANSWER_SCHEMA 同构（answer: string）。
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _model_config(fallback: tuple[str, ...] = (), structured: bool = True) -> ModelConfig:
    return ModelConfig(
        provider_model="vendor-model",
        base_url="https://fake.test",
        api_key_env="FAKE_KEY",
        supports_structured_output=structured,
        provider="fake",
        fallback=fallback,
    )


@pytest.fixture
def topology(monkeypatch):
    """双模型拓扑（primary -> backup）：可定制备用能力与主模型链。

    工厂形态：fixture 返回安装函数，测试必须调用（默认参数即标准拓扑）。
    """

    def _install(
        *,
        backup_structured: bool = True,
        primary_fallback: tuple[str, ...] = (BACKUP,),
    ) -> dict[str, ModelConfig]:
        configs = {
            PRIMARY: _model_config(fallback=primary_fallback),
            BACKUP: _model_config(structured=backup_structured),
        }
        monkeypatch.setattr(inv, "MODEL_CONFIGS", configs)
        import llm_gateway.services.routing as routing

        monkeypatch.setattr(routing, "MODEL_CONFIGS", configs)
        return configs

    return _install


@pytest.fixture
def fake_provider(monkeypatch):
    """把指定剧本的 FakeAdapter 挂上注册表并返回实例（供 attempts 断言）。"""

    def _install(scenario: Scenario) -> FakeAdapter:
        adapter = FakeAdapter(scenario)
        monkeypatch.setitem(PROVIDER_REGISTRY, "fake", adapter)
        return adapter

    return _install


@pytest.fixture(autouse=True)
def _clean_global_state():
    # 进程内全局（trace list / 熔断注册表）测试间必须清零。
    CALL_TRACES.clear()
    reset_breakers()
    yield
    CALL_TRACES.clear()
    reset_breakers()


async def _no_sleep(_delay: float) -> None:
    # 注入睡眠：不真睡（退避节奏的时长正确性由 _backoff_sleep 专项用例覆盖）。
    return None


def _request(**overrides: Any) -> LLMRequest:
    payload: dict[str, Any] = {
        "model": PRIMARY,
        "messages": [Message(role="user", content="hi")],
    }
    payload.update(overrides)
    return LLMRequest(**payload)


async def _collect(events_iter):
    return [event async for event in events_iter]


# ---------------------------------------------------------------------------
# 预算（spec 验收：永远 timeout -> 上游请求总数 == 4 且 attempts==4）
# ---------------------------------------------------------------------------


async def test_budget_always_timeout_totals_four_attempts(topology, fake_provider):
    # spec 任务 9 首条：重试/fallback 共享同一计数器——"永远 timeout"剧本下
    # 上游请求总数 == 4（不是 4×2，也不是 4×模型数）：主 3（单模型上限）+ 备 1。
    topology()
    adapter = fake_provider(Timeout())
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(), sleep=_no_sleep)
    assert exc_info.value.code == "model_unavailable"
    assert adapter.attempts == 4
    trace = CALL_TRACES[-1]
    assert len(CALL_TRACES) == 1
    assert trace.status == "failed"
    assert trace.error_code == "model_unavailable"
    assert trace.attempts == 4
    assert trace.actual_model is None
    # 路由理由进 trace：主模型 3 次耗尽 + 预算见底（备用未再尝试）。
    assert "general-primary: 3 attempts exhausted" in (trace.route_reason or "")


async def test_budget_shared_without_fallback(topology, fake_provider):
    # 无 fallback 链时 4 次尝试全部落在主模型上：预算是全局的，不是按模型的。
    topology(primary_fallback=())
    adapter = fake_provider(Timeout())
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(), sleep=_no_sleep)
    assert exc_info.value.code == "model_unavailable"
    assert adapter.attempts == 4
    assert CALL_TRACES[-1].attempts == 4


async def test_budget_deadline_stops_retries(topology, fake_provider):
    # 墙钟 deadline（spec 任务 3）：超时即终态，即使预算未耗尽。
    topology()
    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = _Clock()

    async def _advance(_delay: float) -> None:
        clock.now += 1.0  # 每次退避推进 1s：timeout_seconds=2 -> 第 3 次 try_spend 超时

    adapter = fake_provider(Timeout())
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(timeout_seconds=2.0), sleep=_advance, clock=clock)
    assert exc_info.value.code == "model_unavailable"
    assert adapter.attempts == 2  # attempt1@0s、attempt2@1s，attempt3 前已超时
    assert CALL_TRACES[-1].attempts == 2


# ---------------------------------------------------------------------------
# fallback 链与路由理由（spec 验收：主模型 3 次失败后 backup 成功）
# ---------------------------------------------------------------------------


async def test_fallback_success_after_three_primary_failures(topology, fake_provider):
    # 主模型 3 次失败后 backup 成功：断言 trace 的 actual_model、route_reason、
    # attempts（spec 任务 9），attempts = 主 3 + 备 1 = 预算 4。
    topology()
    adapter = fake_provider(ConsecutiveThenSuccess(failures=3))
    response = await inv.call_with_fallback(_request(), caller="tester", sleep=_no_sleep)
    assert response.model == BACKUP
    assert response.attempts == 4
    assert adapter.attempts == 4
    trace = CALL_TRACES[-1]
    assert len(CALL_TRACES) == 1
    assert trace.status == "success"
    assert trace.actual_model == BACKUP
    assert trace.attempts == 4
    assert trace.route_reason == f"{PRIMARY}: 3 attempts exhausted (model_unavailable)"
    assert trace.caller == "tester"


async def test_fallback_candidate_circuit_open_is_skipped(topology, fake_provider):
    # fallback 候选熔断 open：直接跳过（circuit_open 理由进 trace），不消耗
    # 该候选的任何上游请求；链尾终态 failed model_unavailable。
    topology()
    adapter = fake_provider(Timeout())
    breaker = get_breaker(BACKUP)
    for _ in range(5):
        breaker.record_failure()  # 达到阈值，打开
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(), sleep=_no_sleep)
    assert exc_info.value.code == "model_unavailable"
    assert adapter.attempts == 3  # 只有主模型的 3 次；备用零调用
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert "general-backup: circuit_open" in (trace.route_reason or "")


async def test_fallback_candidate_structured_unsupported_is_skipped(topology, fake_provider):
    # 备用模型不支持结构化输出：不等价的 fallback 绝不发生（跳过 + 理由）。
    topology(backup_structured=False)
    adapter = fake_provider(
        ScenarioSequence([Timeout(), Timeout(), Timeout(), Success(content='{"answer": "ok"}')])
    )
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(response_schema=SCHEMA), sleep=_no_sleep)
    assert exc_info.value.code == "model_unavailable"
    assert adapter.attempts == 3
    assert "general-backup: structured_output_unsupported" in (CALL_TRACES[-1].route_reason or "")


async def test_non_retryable_error_is_terminal_without_fallback(topology, fake_provider, monkeypatch):
    # 确定性错误（gateway_misconfigured 形态）不重试不 fallback：首次失败即
    # 终态，码原样对外，trace 恰好一条 failed。
    topology()
    from llm_gateway.core.errors import GATEWAY_MISCONFIGURED

    class _Misconfigured(FakeAdapter):
        async def complete(self, *args: Any, **kwargs: Any):
            self.attempts += 1
            raise GatewayError(GATEWAY_MISCONFIGURED)

    adapter = _Misconfigured(Success())
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake", adapter)
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(), sleep=_no_sleep)
    assert exc_info.value.code == "gateway_misconfigured"
    assert adapter.attempts == 1
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.error_code == "gateway_misconfigured"


# ---------------------------------------------------------------------------
# 修复调用（spec 任务 5 / 验收：invalid JSON 修复成功、截断提高 max_tokens、
# 各恰好 1 次修复且消耗预算）
# ---------------------------------------------------------------------------


async def test_repair_invalid_json_success(topology, fake_provider):
    # 非法 JSON -> 携错误反馈重调 -> 成功：恰好 1 次修复，预算共享（attempts==2）。
    topology()
    adapter = fake_provider(
        ScenarioSequence([InvalidOutput(content='{"answer":'), Success(content='{"answer": "ok"}')])
    )
    response = await inv.call_with_fallback(_request(response_schema=SCHEMA), sleep=_no_sleep)
    assert response.parsed == {"answer": "ok"}
    assert response.attempts == 2
    assert adapter.attempts == 2  # 原始 + 恰好 1 次修复
    trace = CALL_TRACES[-1]
    assert trace.status == "success"
    assert trace.attempts == 2


async def test_repair_invalid_json_failure_reports_after_single_repair(topology, fake_provider):
    # 修复仍失败 -> 才向调用方报错：恰好 1 次修复，trace failed invalid_json。
    topology()
    adapter = fake_provider(ScenarioSequence([InvalidOutput(), InvalidOutput()]))
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(response_schema=SCHEMA), sleep=_no_sleep)
    assert exc_info.value.code == "invalid_json"
    assert adapter.attempts == 2
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.error_code == "invalid_json"
    assert trace.attempts == 2


async def test_repair_truncation_success_with_raised_max_tokens(topology, fake_provider):
    # finish_reason == length -> 提高 max_tokens 重调 -> 成功；恰好 1 次修复。
    topology()
    adapter = fake_provider(
        ScenarioSequence(
            [Success(content="half", finish_reason="length"), Success(content="full answer")]
        )
    )
    response = await inv.call_with_fallback(_request(max_tokens=16), sleep=_no_sleep)
    assert response.content == "full answer"
    assert response.finish_reason == "stop"
    assert response.attempts == 2
    assert adapter.attempts == 2
    assert CALL_TRACES[-1].status == "success"


async def test_repair_truncation_failure_reports_output_truncated(topology, fake_provider):
    # 截断修复仍截断 -> output_truncated 终态（注册表稳定码）。
    topology()
    adapter = fake_provider(
        ScenarioSequence(
            [Success(finish_reason="length"), Success(finish_reason="length")]
        )
    )
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(), sleep=_no_sleep)
    assert exc_info.value.code == "output_truncated"
    assert adapter.attempts == 2
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.error_code == "output_truncated"


async def test_repair_denied_when_budget_exhausted(topology, fake_provider):
    # 修复消耗统一预算：预算见底（第 4 次尝试才产出坏输出）时不再修复，
    # 立即终态——"重试/fallback/修复共享计数器"的直接证明。
    topology()
    adapter = fake_provider(
        ScenarioSequence(
            [Timeout(), Timeout(), Timeout(), InvalidOutput(content="not json")]
        )
    )
    with pytest.raises(GatewayError) as exc_info:
        await inv.call_with_fallback(_request(response_schema=SCHEMA), sleep=_no_sleep)
    assert exc_info.value.code == "invalid_json"
    assert adapter.attempts == 4  # 主 3 + 备 1；第 5 次（修复）未发生
    assert CALL_TRACES[-1].attempts == 4
    assert CALL_TRACES[-1].status == "failed"


# ---------------------------------------------------------------------------
# 退避节奏（ADR-0003 决策 3/4：0.5s ×2 + 抖动，Retry-After 优先）
# ---------------------------------------------------------------------------


async def test_backoff_respects_retry_after():
    delays: list[float] = []

    async def _record(delay: float) -> None:
        delays.append(delay)

    await inv._backoff_sleep(_record, GatewayError(RATE_LIMITED, retry_after=3.5), 0)
    assert delays == [3.5]  # 上游亲口说的节奏优先于指数退避


async def test_backoff_exponential_with_jitter_bounds():
    delays: list[float] = []

    async def _record(delay: float) -> None:
        delays.append(delay)

    exc = GatewayError("model_unavailable")
    for consecutive in range(3):
        await inv._backoff_sleep(_record, exc, consecutive)
    # 0.5 ×2^n + uniform(0, 0.2)：下界不含抖动、上界含满抖动。
    assert delays[0] == pytest.approx(0.5, abs=0.2)
    assert delays[1] == pytest.approx(1.0, abs=0.2)
    assert delays[2] == pytest.approx(2.0, abs=0.2)
    assert all(0.5 * 2**i <= d <= 0.5 * 2**i + 0.2 for i, d in enumerate(delays))


# ---------------------------------------------------------------------------
# 流式铁律（spec 任务 6/7：首块前可 fallback、首块后不重生成、取消传播）
# ---------------------------------------------------------------------------


async def test_stream_fallback_before_first_chunk(topology, fake_provider):
    # 首块前失败：重试 + fallback 全程可用，调用方只见备用模型的完整流。
    topology()
    adapter = fake_provider(
        ScenarioSequence(
            [Timeout(), Timeout(), Timeout(), Success(content="备份内容")]
        )
    )
    events = await _collect(inv.stream_with_fallback(_request(), sleep=_no_sleep))
    assert [event["type"] for event in events] == ["content.delta", "response.completed"]
    assert events[0]["model"] == BACKUP
    assert adapter.attempts == 4
    trace = CALL_TRACES[-1]
    assert len(CALL_TRACES) == 1
    assert trace.status == "success"
    assert trace.actual_model == BACKUP
    assert trace.attempts == 4
    assert trace.route_reason == f"{PRIMARY}: 3 attempts exhausted (model_unavailable)"


async def test_stream_no_regeneration_after_first_chunk_failure(topology, fake_provider):
    # 首块后中断（不变量 #7）：已发出的块不重复、终态错误只发一次、备用零调用。
    topology()
    adapter = fake_provider(StreamInterrupt(chunks_before_failure=2))
    events = await _collect(inv.stream_with_fallback(_request(), sleep=_no_sleep))
    assert [event["type"] for event in events] == [
        "content.delta",
        "content.delta",
        "response.failed",
    ]
    deltas = [event["delta"] for event in events if event["type"] == "content.delta"]
    assert deltas == ["chunk-0", "chunk-1"]  # 不重复（没有重新生成拼接）
    failures = [event for event in events if event["type"] == "response.failed"]
    assert len(failures) == 1  # 终态错误只发一次
    assert failures[0]["error"] == "upstream_stream_failed"
    assert adapter.attempts == 1  # 没有第二个模型的任何请求
    trace = CALL_TRACES[-1]
    assert len(CALL_TRACES) == 1
    assert trace.status == "failed"
    assert trace.error_code == "upstream_stream_failed"
    assert trace.actual_model == PRIMARY


async def test_stream_cancellation_propagates_and_finalizes_cancelled(topology, fake_provider):
    # 取消传播（不变量 #8）：客户端中途取消 -> 下游 provider 流被关闭
    # （Fake Adapter 感知）-> trace 恰好一条 cancelled（不落 failed）。
    topology()
    adapter = fake_provider(SlowSuccess(delay_seconds=5.0))
    task = asyncio.create_task(_collect(inv.stream_with_fallback(_request())))
    await asyncio.sleep(0.05)  # 等消费进入 provider 的在途 sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.cancelled_streams == 1  # 下游任务被取消（Fake 感知到）
    assert adapter.attempts == 1
    assert len(CALL_TRACES) == 1
    assert CALL_TRACES[-1].status == "cancelled"


async def test_stream_success_records_ttft_and_single_trace(topology, fake_provider):
    # TTFT 在首块记录（spec 任务 6）；成功终态恰好一条 trace。
    topology()
    fake_provider(Success(content="你好"))
    events = await _collect(inv.stream_with_fallback(_request()))
    assert events[-1]["type"] == "response.completed"
    trace = CALL_TRACES[-1]
    assert len(CALL_TRACES) == 1
    assert trace.status == "success"
    assert trace.ttft_ms is not None


async def test_stream_request_level_error_before_run(topology, fake_provider):
    # 请求级错误（白名单外）在任何上游调用与 trace 之前拒绝。
    topology()
    fake_provider(Success())
    with pytest.raises(GatewayError) as exc_info:
        await _collect(inv.stream_with_fallback(_request(model="no-such-model")))
    assert exc_info.value.code == "unknown_model"
    assert CALL_TRACES == []


# ---------------------------------------------------------------------------
# 唯一终态与基础件（spec 任务 8 / RunContext / routing）
# ---------------------------------------------------------------------------


async def test_trace_finalize_is_idempotent(topology):
    # TraceDraft.finalize 幂等：多次调用只落一条 trace（终态恰好一次的防线）。
    topology()
    draft = TraceDraft(
        request_id="req-1",
        requested_model=PRIMARY,
        prompt=None,
        caller="tester",
        started_at=0.0,
        clock=lambda: 10.0,
    )
    draft.bind_budget(Budget(30.0, clock=lambda: 10.0))
    draft.finalize("success", PRIMARY)
    draft.finalize("failed", None, error_code="model_unavailable")
    assert len(CALL_TRACES) == 1
    assert CALL_TRACES[-1].status == "success"


async def test_run_context_is_frozen_snapshot(topology):
    # RunContext 构建后只读（spec 任务 1）：frozen dataclass 拒绝改写。
    from dataclasses import FrozenInstanceError

    topology()
    ctx = inv.build_run_context(_request(response_schema=SCHEMA), caller="tester")
    with pytest.raises(FrozenInstanceError):
        ctx.__setattr__("max_tokens", 99)  # frozen dataclass 的 __setattr__ 即抛
    assert ctx.price_version  # 价格表版本快照已入上下文


async def test_build_chain_dedupes_and_keeps_declaration_order():
    configs = {
        "a": _model_config(fallback=("b", "c", "b")),
        "b": _model_config(),
        "c": _model_config(),
    }
    assert build_chain("a", configs) == ["a", "b", "c"]
