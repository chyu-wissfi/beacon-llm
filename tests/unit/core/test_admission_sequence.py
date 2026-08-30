"""准入组合单测（M05 任务 5）：序列常量、逐层拒绝、资源无泄漏。

组合断言面（spec 任务 6）：第一层拒绝时后层计数不变；任一层拒绝后已持有
资源全部释放。时钟注入不真睡；熔断器走进程内注册表（每用例前后重置）。
"""

import pytest

from llm_gateway.core.breaker import FAILURE_THRESHOLD, get_breaker, reset_breakers
from llm_gateway.core.errors import GatewayError
from llm_gateway.core.ratelimit import AdmissionGate
from llm_gateway.core.schemas import Usage
from tests.unit.core import FakeClock

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clean_breakers():
    # 进程内熔断注册表是用例间唯一的共享残留面，前后双清。
    reset_breakers()
    yield
    reset_breakers()


def _gate(**overrides) -> AdmissionGate:
    defaults: dict = {"global_concurrency": 2, "provider_concurrency": 2, "clock": FakeClock()}
    defaults.update(overrides)
    return AdmissionGate(**defaults)


def _provider_in_flight(gate: AdmissionGate, provider: str) -> int:
    # 断言面取计数而非容器存在性：闸/桶是懒创建的，"建了但计数 0"与"未建"
    # 行为等价，只有 in_flight 才是行为面。
    provider_gate = gate.provider_gates.get(provider)
    return provider_gate.in_flight if provider_gate is not None else 0


async def test_successful_admission_holds_then_releases() -> None:
    gate = _gate()
    permit = await gate.acquire("m", "p", rpm=10, tpm=None)
    assert gate.global_gate.in_flight == 1
    assert gate.provider_gates["p"].in_flight == 1
    permit.release()
    assert gate.global_gate.in_flight == 0
    assert gate.provider_gates["p"].in_flight == 0
    # 释放幂等：重复释放不得把计数打负。
    permit.release()
    assert gate.global_gate.in_flight == 0


async def test_reject_at_first_layer_leaves_later_layers_untouched() -> None:
    # 组合不变量（spec 任务 6）：全局并发满 -> 拒绝时后层计数全部不变。
    gate = _gate(global_concurrency=1)
    first = await gate.acquire("m", "p", rpm=10, tpm=None)
    with pytest.raises(GatewayError) as exc_info:
        await gate.acquire("m", "p", rpm=10, tpm=None)
    assert exc_info.value.code == "overloaded"
    assert exc_info.value.retry_after is not None
    assert gate.global_gate.in_flight == 1  # 只剩 first 的持有（被拒请求已归还）
    assert _provider_in_flight(gate, "p") == 1  # 同理：仅 first，被拒请求没走到这层
    # 被拒请求没走到 RPM 层：桶里只剩 first 消费后的令牌。
    assert gate.buckets["m"].tokens == 9.0
    assert get_breaker("m").state == "closed"
    first.release()


async def test_reject_at_rpm_releases_global_slot() -> None:
    gate = _gate()
    keeper = await gate.acquire("m", "p", rpm=1, tpm=None)  # 吃掉唯一令牌
    with pytest.raises(GatewayError) as exc_info:
        await gate.acquire("m", "p", rpm=1, tpm=None)
    assert exc_info.value.code == "rate_limited"
    assert exc_info.value.retry_after is not None and exc_info.value.retry_after > 0
    # 无泄漏：被拒请求占过的全局位已归还；供应商位只剩 keeper 的持有。
    assert gate.global_gate.in_flight == 1
    assert _provider_in_flight(gate, "p") == 1
    # 被拒请求未消费令牌：桶里只剩 keeper 吃剩的 0 枚（rpm=1）。
    assert gate.buckets["m"].tokens == 0.0
    keeper.release()


async def test_reject_at_tpm_releases_global_slot() -> None:
    gate = _gate()
    gate.record_usage("m", Usage(input_tokens=60, output_tokens=40))
    with pytest.raises(GatewayError) as exc_info:
        await gate.acquire("m", "p", rpm=10, tpm=100)
    assert exc_info.value.code == "token_budget_exceeded"
    assert exc_info.value.retry_after is not None
    assert gate.global_gate.in_flight == 0
    assert _provider_in_flight(gate, "p") == 0


async def test_reject_at_provider_concurrency_releases_earlier_layers() -> None:
    gate = _gate(provider_concurrency=1)
    keeper = await gate.acquire("m", "p", rpm=10, tpm=None)
    with pytest.raises(GatewayError) as exc_info:
        await gate.acquire("m", "p", rpm=10, tpm=None)
    assert exc_info.value.code == "provider_overloaded"
    assert gate.global_gate.in_flight == 1  # 只剩 keeper
    # 裁决记录（模块注）：令牌是消费而非持有，拒绝不退还——此处钉死该语义。
    assert gate.buckets["m"].tokens == 8.0  # 10 - keeper - 被拒请求各 1
    keeper.release()


async def test_open_circuit_rejects_without_touching_later_layers() -> None:
    gate = _gate()
    breaker = get_breaker("m")
    for _ in range(FAILURE_THRESHOLD):
        breaker.record_failure()
    with pytest.raises(GatewayError) as exc_info:
        await gate.acquire("m", "p", rpm=10, tpm=None)
    assert exc_info.value.code == "circuit_open"
    assert exc_info.value.status_code == 503
    assert gate.global_gate.in_flight == 0  # 全局位已归还
    assert gate.buckets == {}  # 令牌未消费
    assert gate.provider_gates == {}


async def test_admit_context_releases_on_success_and_on_error() -> None:
    gate = _gate()
    async with gate.admit("m", "p", rpm=10, tpm=None):
        assert gate.global_gate.in_flight == 1
    assert gate.global_gate.in_flight == 0

    with pytest.raises(RuntimeError, match="boom"):
        async with gate.admit("m", "p", rpm=10, tpm=None):
            raise RuntimeError("boom")
    # 请求体异常同样不泄漏准入资源。
    assert gate.global_gate.in_flight == 0
    assert gate.provider_gates["p"].in_flight == 0


async def test_rpm_undeclared_falls_back_to_default() -> None:
    # 全局闸放宽到 3，让第三个请求能走到 RPM 层（否则先被全局并发拒绝）。
    gate = _gate(global_concurrency=3, default_rpm=2)
    await gate.acquire("m", "p", rpm=None, tpm=None)
    await gate.acquire("m", "p", rpm=None, tpm=None)
    with pytest.raises(GatewayError) as exc_info:
        await gate.acquire("m", "p", rpm=None, tpm=None)
    assert exc_info.value.code == "rate_limited"
