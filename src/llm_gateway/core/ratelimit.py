"""多层准入资源：令牌桶（RPM）、并发闸、TPM 账本与准入门（M05 任务 2+3+5）。

全部状态单进程内存（不变量 #17，ADR-0004）：信号量/桶/账本在进程内对
所有请求即时可见；重启清零是保护机制可接受的代价。时钟全部注入
（默认 time.monotonic）——单测不真睡（spec 任务 6）。

准入序列固化为 ADMISSION_ORDER（spec 任务 5）：
鉴权 -> 全局并发 -> 熔断 -> RPM -> TPM -> 供应商并发 ->（编排）。
auth 在端点依赖层完成（core/auth.py），其余层由 AdmissionGate.acquire
按序获取；spec 序列未列 TPM，任务 3 又要求准入期检查——Controller 裁决
TPM 紧随 RPM（同为每模型预算面）。拒绝时已持有的资源反序全部释放
（AdmissionGate.acquire 的 except 路径），无泄漏。

拒绝码映射（注册表常量，禁字面量）：
- 全局并发超限 -> OVERLOADED（429）
- 供应商并发超限 -> PROVIDER_OVERLOADED（429；与上游 429 同码，语义均为
  "供应商面过载"，注册表 message 即此意）
- RPM 超限 -> RATE_LIMITED（429）
- TPM 超额 -> TOKEN_BUDGET_EXCEEDED（429）
均携带 retry_after，api 层附 Retry-After 头。

TPM 语义（spec 任务 3）：**事后记账**，不预估扣减——调用前输出长度未知，
按 max_tokens 预留会把限流收得过死（ADR-0004 理由节）。调用完成后按实际
usage 记账（record_usage），窗口超额后新请求才被拒。
"""

import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Final

from llm_gateway.core.breaker import get_breaker
from llm_gateway.core.config import CONFIG
from llm_gateway.core.errors import (
    CIRCUIT_OPEN,
    OVERLOADED,
    PROVIDER_OVERLOADED,
    RATE_LIMITED,
    TOKEN_BUDGET_EXCEEDED,
    GatewayError,
)
from llm_gateway.core.schemas import Usage
from llm_gateway.observability.metrics import RATE_LIMITED_TOTAL

# 准入顺序的常量序列（spec 任务 5）：auth 在端点依赖层，其余在 acquire 内。
ADMISSION_ORDER: Final[tuple[str, ...]] = (
    "auth",
    "global_concurrency",
    "circuit_breaker",
    "rpm",
    "tpm",  # spec 序列未列；任务 3 要求准入期检查，裁决紧随 rpm（见模块注）
    "provider_concurrency",
)

# rpm 未声明时的默认值（design.md §3.4 表：每模型 RPM 默认 60）。
DEFAULT_RPM: Final[int] = 60
# TPM 记账窗口：1 分钟（TPM 的 M 即 minute）。
_TPM_WINDOW_SECONDS: Final[float] = 60.0
# 并发类拒绝无可预测的补充时刻（释放取决于在途请求何时结束），兜底 1 秒。
_FALLBACK_RETRY_AFTER: Final[float] = 1.0

Clock = Callable[[], float]


class TokenBucket:
    # 令牌桶（RPM 限速）：容量 = rpm，按 rpm/60 每秒匀速补充。
    # 请求消耗 1 令牌；桶空即拒，并给出下个令牌到达的秒数（Retry-After）。

    def __init__(self, capacity: int, clock: Clock = time.monotonic) -> None:
        self.capacity = float(capacity)
        self.refill_rate = capacity / 60.0
        self._clock = clock
        self.tokens = self.capacity
        self._last_refill = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self._last_refill = now

    def try_acquire(self) -> bool:
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def time_until_available(self) -> float:
        # 距下个令牌到达的秒数（拒绝时的 Retry-After 依据）。
        self._refill()
        if self.tokens >= 1:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


class ConcurrencyGate:
    # 非阻塞并发闸：计数式信号量。asyncio.Semaphore 只有阻塞式 acquire，
    # 准入拒绝要求立即失败（429），故用计数器自实现——协程单线程模型下
    # 读改写同步完成，无需锁。
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_flight = 0

    def try_acquire(self) -> bool:
        if self.in_flight < self.limit:
            self.in_flight += 1
            return True
        return False

    def release(self) -> None:
        self.in_flight -= 1


class TpmLedger:
    # 每模型 TPM 账本：60 秒滑动窗口的事后记账（(时刻, token 数) 队列）。
    # 准入查窗口内累计用量；记账只发生在调用完成后（record），不预估。
    def __init__(self, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._entries: deque[tuple[float, int]] = deque()

    def record(self, tokens: int) -> None:
        self._entries.append((self._clock(), tokens))

    def _purge(self, now: float) -> None:
        while self._entries and now - self._entries[0][0] >= _TPM_WINDOW_SECONDS:
            self._entries.popleft()

    def usage(self) -> int:
        now = self._clock()
        self._purge(now)
        return sum(tokens for _, tokens in self._entries)

    def over_budget(self, limit: int) -> bool:
        return self.usage() >= limit

    def time_until_available(self, limit: int) -> float:
        # 最早一笔记账出窗的时刻（窗口只出不进地单调推进，上限 60 秒）。
        now = self._clock()
        self._purge(now)
        if sum(tokens for _, tokens in self._entries) < limit:
            return 0.0
        if not self._entries:
            return 0.0
        return max(0.0, self._entries[0][0] + _TPM_WINDOW_SECONDS - now)


class AdmissionPermit:
    # 已获取准入资源的持有者：释放按获取的反序执行，且幂等
    # （端点异常路径与 finally 可能重叠触发）。
    def __init__(self, releases: list[Callable[[], None]]) -> None:
        self._releases = releases
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for release in reversed(self._releases):
            release()


class AdmissionGate:
    # 准入装配：按 ADMISSION_ORDER 逐层获取；任一层拒绝 -> 已持有资源
    # 反序释放后抛对应 GatewayError（无泄漏，spec 任务 5）。
    # 准入只查请求模型：编排内部 fallback 切备用模型不再过准入
    # （编排内部行为属 M06 面）。

    def __init__(
        self,
        *,
        global_concurrency: int = 20,
        provider_concurrency: int = 10,
        default_rpm: int = DEFAULT_RPM,
        clock: Clock = time.monotonic,
    ) -> None:
        # 默认值出处：ADR-0004/design.md §3.4（全局 20 / 每供应商 10）；
        # 运行时取值以 config/models.yaml 的 admission 块为准（见模块底部）。
        self.global_gate = ConcurrencyGate(global_concurrency)
        self.provider_concurrency = provider_concurrency
        self.default_rpm = default_rpm
        self._clock = clock
        self.provider_gates: dict[str, ConcurrencyGate] = {}
        self.buckets: dict[str, TokenBucket] = {}
        self.tpm_ledgers: dict[str, TpmLedger] = {}

    def _bucket(self, model: str, rpm: int | None) -> TokenBucket:
        bucket = self.buckets.get(model)
        if bucket is None:
            capacity = rpm if rpm is not None else self.default_rpm
            bucket = TokenBucket(capacity, clock=self._clock)
            self.buckets[model] = bucket
        return bucket

    def _provider_gate(self, provider: str) -> ConcurrencyGate:
        gate = self.provider_gates.get(provider)
        if gate is None:
            gate = ConcurrencyGate(self.provider_concurrency)
            self.provider_gates[provider] = gate
        return gate

    def _ledger(self, model: str) -> TpmLedger:
        ledger = self.tpm_ledgers.get(model)
        if ledger is None:
            ledger = TpmLedger(clock=self._clock)
            self.tpm_ledgers[model] = ledger
        return ledger

    async def acquire(
        self,
        model: str,
        provider: str,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
    ) -> AdmissionPermit:
        # 按序获取各层资源。返回 permit 由调用方在请求生命周期结束时释放
        # （非流式随上下文退出；流式持有到生成器结束，spec 任务 2）。
        releases: list[Callable[[], None]] = []
        try:
            if not self.global_gate.try_acquire():
                raise GatewayError(OVERLOADED, retry_after=_FALLBACK_RETRY_AFTER)
            releases.append(self.global_gate.release)

            breaker = get_breaker(model)
            if not breaker.allow_request():
                raise GatewayError(CIRCUIT_OPEN)
            # 半开探测名额也入释放栈：后层拒绝时归还，不计成败。
            releases.append(breaker.relinquish_probe)

            bucket = self._bucket(model, rpm)
            if not bucket.try_acquire():
                raise GatewayError(RATE_LIMITED, retry_after=bucket.time_until_available())

            # tpm 未声明 -> 该模型不设 TPM 预算（与 rpm 有默认值不同：
            # design.md 未给 TPM 默认值，不声明即不治理）。
            if tpm is not None and self._ledger(model).over_budget(tpm):
                raise GatewayError(
                    TOKEN_BUDGET_EXCEEDED,
                    retry_after=self._ledger(model).time_until_available(tpm),
                )

            provider_gate = self._provider_gate(provider)
            if not provider_gate.try_acquire():
                raise GatewayError(PROVIDER_OVERLOADED, retry_after=_FALLBACK_RETRY_AFTER)
            releases.append(provider_gate.release)

            return AdmissionPermit(releases)
        except BaseException as exc:
            # 拒绝路径（含非 GatewayError 的编程错误）：已持有资源反序全放。
            for release in reversed(releases):
                release()
            # 限流计数在拒绝动作点（M10 任务 4）：只计 RPM 桶拒绝（RATE_LIMITED）
            # ——全局/供应商并发超限（OVERLOADED/PROVIDER_OVERLOADED）与 TPM 超额是
            # 容量/预算面而非速率限流面；上游 429 不在此层发生（走重试口径）。
            if isinstance(exc, GatewayError) and exc.code == RATE_LIMITED:
                RATE_LIMITED_TOTAL.labels(model=model).inc()
            raise

    @asynccontextmanager
    async def admit(
        self,
        model: str,
        provider: str,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
    ) -> AsyncIterator[None]:
        # 非流式路径的便捷形态：准入资源随上下文退出释放。
        permit = await self.acquire(model, provider, rpm=rpm, tpm=tpm)
        try:
            yield
        finally:
            permit.release()

    def record_usage(self, model: str, usage: Usage) -> None:
        # TPM 事后记账：调用完成后按实际用量（input + output）入账。
        self._ledger(model).record(usage.input_tokens + usage.output_tokens)


def _build_default_gate() -> AdmissionGate:
    # 运行时阈值来自配置（ADR-0004 §5：阈值全部是配置项）；
    # CONFIG.admission 的默认值在 core/config.py 声明。
    admission = CONFIG.admission
    return AdmissionGate(
        global_concurrency=admission.global_concurrency,
        provider_concurrency=admission.provider_concurrency,
    )


# 进程内单例（不变量 #17）：端点与测试共享同一份准入状态。
# 使用方经模块属性访问（ratelimit.ADMISSION），reset 后不致持有旧实例。
ADMISSION: AdmissionGate = _build_default_gate()


def reset_admission() -> None:
    # 测试卫生面：重建单例（桶/闸/账本清零，阈值回到配置值）。
    global ADMISSION
    ADMISSION = _build_default_gate()
