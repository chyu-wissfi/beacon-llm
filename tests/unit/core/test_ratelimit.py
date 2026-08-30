"""core/ratelimit.py 组件单测（M05 任务 2+3）：令牌桶 / 并发闸 / TPM 账本。

时钟注入不真睡。三个组件独立断言，组合行为在 test_admission_sequence.py。
"""

import pytest

from llm_gateway.core.ratelimit import (
    ADMISSION_ORDER,
    ConcurrencyGate,
    TokenBucket,
    TpmLedger,
)
from tests.unit.core import FakeClock


def test_admission_order_is_frozen_constant() -> None:
    # spec 任务 5：准入顺序固化为常量序列（tpm 的插入位为 Controller 裁决，
    # 台账已录——任务 3 要求准入期检查而原序列未列）。
    assert ADMISSION_ORDER == (
        "auth",
        "global_concurrency",
        "circuit_breaker",
        "rpm",
        "tpm",
        "provider_concurrency",
    )


# -- 令牌桶（RPM） -----------------------------------------------------------


def test_token_bucket_starts_full_and_drains() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=3, clock=clock)
    assert [bucket.try_acquire() for _ in range(3)] == [True] * 3
    assert bucket.try_acquire() is False


def test_token_bucket_refills_at_rpm_per_minute() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=60, clock=clock)
    for _ in range(60):
        assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False
    # rpm=60 -> 每秒 1 令牌。
    clock.advance(1.0)
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False


def test_token_bucket_never_exceeds_capacity() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, clock=clock)
    clock.advance(600.0)  # 长时间空闲不会囤积超额令牌
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False


def test_time_until_available_reports_next_token() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, clock=clock)  # rpm=2 -> 30 秒/令牌
    bucket.try_acquire()
    bucket.try_acquire()
    assert bucket.time_until_available() == pytest.approx(30.0)
    clock.advance(15.0)  # 半枚令牌
    assert bucket.time_until_available() == pytest.approx(15.0)
    clock.advance(15.0)
    assert bucket.time_until_available() == 0.0


# -- 并发闸 ------------------------------------------------------------------


def test_concurrency_gate_enforces_limit_and_releases() -> None:
    gate = ConcurrencyGate(limit=2)
    assert gate.try_acquire() is True
    assert gate.try_acquire() is True
    assert gate.in_flight == 2
    assert gate.try_acquire() is False
    gate.release()
    assert gate.in_flight == 1
    assert gate.try_acquire() is True


def test_zero_limit_gate_always_rejects() -> None:
    gate = ConcurrencyGate(limit=0)
    assert gate.try_acquire() is False


# -- TPM 账本（事后记账，60 秒滑动窗口） --------------------------------------


def test_tpm_ledger_accumulates_and_compares_budget() -> None:
    ledger = TpmLedger(clock=FakeClock())
    ledger.record(60)
    ledger.record(30)
    assert ledger.usage() == 90
    assert ledger.over_budget(100) is False
    assert ledger.over_budget(90) is True  # 达预算即超额（新请求拒绝）
    assert ledger.over_budget(50) is True


def test_tpm_ledger_entries_expire_after_window() -> None:
    clock = FakeClock()
    ledger = TpmLedger(clock=clock)
    ledger.record(80)
    clock.advance(30.0)
    ledger.record(40)
    assert ledger.usage() == 120
    clock.advance(31.0)  # 第一笔已出窗（61 秒前），第二笔仍在
    assert ledger.usage() == 40
    clock.advance(30.0)
    assert ledger.usage() == 0


def test_tpm_time_until_available_is_earliest_exit() -> None:
    clock = FakeClock()
    ledger = TpmLedger(clock=clock)
    ledger.record(100)
    assert ledger.time_until_available(100) == pytest.approx(60.0)
    clock.advance(10.0)
    assert ledger.time_until_available(100) == pytest.approx(50.0)
    clock.advance(50.0)
    assert ledger.time_until_available(100) == 0.0
