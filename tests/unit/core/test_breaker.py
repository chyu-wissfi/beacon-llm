"""core/breaker.py 单测（M05 任务 4）：全状态机 + 半开单探测。

时钟注入不真睡；口径断言钉死裁决（台账已录）：计入熔断的是传输故障
（MODEL_UNAVAILABLE 的语义面），429 类不计——本文件以 record_failure 的
调用纪律表达该口径（编排层只对 MODEL_UNAVAILABLE 调它）。
"""

from llm_gateway.core.breaker import (
    FAILURE_THRESHOLD,
    OPEN_SECONDS,
    CircuitBreaker,
    get_breaker,
    reset_breakers,
)
from tests.unit.core import FakeClock


def _open_breaker(breaker: CircuitBreaker, clock: FakeClock) -> None:
    for _ in range(FAILURE_THRESHOLD):
        breaker.record_failure()
    assert breaker.state == "open"


def test_closed_allows_requests() -> None:
    assert CircuitBreaker(clock=FakeClock()).allow_request() is True


def test_below_threshold_stays_closed() -> None:
    breaker = CircuitBreaker(clock=FakeClock())
    for _ in range(FAILURE_THRESHOLD - 1):
        breaker.record_failure()
    assert breaker.state == "closed"
    assert breaker.allow_request() is True


def test_success_resets_consecutive_failures() -> None:
    breaker = CircuitBreaker(clock=FakeClock())
    for _ in range(FAILURE_THRESHOLD - 1):
        breaker.record_failure()
    breaker.record_success()
    # 清零后重新累计：再来阈值减一次仍闭合。
    for _ in range(FAILURE_THRESHOLD - 1):
        breaker.record_failure()
    assert breaker.state == "closed"


def test_threshold_failures_open_the_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(clock=clock)
    _open_breaker(breaker, clock)
    assert breaker.allow_request() is False


def test_open_rejects_until_window_expires() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(clock=clock)
    _open_breaker(breaker, clock)
    clock.advance(OPEN_SECONDS - 0.1)
    assert breaker.allow_request() is False


def test_half_open_admits_exactly_one_probe() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(clock=clock)
    _open_breaker(breaker, clock)
    clock.advance(OPEN_SECONDS)
    # 到期后第一个请求成为探测；探测在途时其余请求仍拒。
    assert breaker.allow_request() is True
    assert breaker.state == "half_open"
    assert breaker.allow_request() is False


def test_probe_success_closes_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(clock=clock)
    _open_breaker(breaker, clock)
    clock.advance(OPEN_SECONDS)
    assert breaker.allow_request() is True
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow_request() is True


def test_probe_failure_reopens_for_full_window() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(clock=clock)
    _open_breaker(breaker, clock)
    clock.advance(OPEN_SECONDS)
    assert breaker.allow_request() is True
    breaker.record_failure()
    # 重开后不需要再攒 5 次：完整窗口内拒绝，到期再给一次探测机会。
    assert breaker.state == "open"
    clock.advance(OPEN_SECONDS - 0.1)
    assert breaker.allow_request() is False
    clock.advance(0.2)
    assert breaker.allow_request() is True


def test_relinquish_probe_returns_slot_without_verdict() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(clock=clock)
    _open_breaker(breaker, clock)
    clock.advance(OPEN_SECONDS)
    assert breaker.allow_request() is True
    # 探测被准入后层拒绝：名额归还，下个请求仍可作为探测。
    breaker.relinquish_probe()
    assert breaker.allow_request() is True
    # 闭合态归还是无操作。
    breaker.record_success()
    breaker.relinquish_probe()
    assert breaker.state == "closed"


def test_breaker_registry_is_per_model_and_resettable() -> None:
    reset_breakers()
    primary = get_breaker("general-primary")
    assert get_breaker("general-primary") is primary
    assert get_breaker("general-backup") is not primary
    reset_breakers()
    assert get_breaker("general-primary") is not primary
