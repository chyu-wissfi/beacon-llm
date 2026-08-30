"""按模型熔断器（M05 任务 4）。

状态机（ADR-0004 §3 / design.md §3.4）：
closed --连续 FAILURE_THRESHOLD 次失败--> open（OPEN_SECONDS 内全部拒绝，
503 circuit_open）--> 到期转 half_open：放行 **1** 个探测请求（其余仍拒），
探测成功闭合、失败重新打开 30s。

失败判定口径（Controller 裁决，spec 括注字面自相矛盾的自洽读法）：
计入熔断的失败 = 网关侧可重试类失败中的**传输故障**——连接/超时/上游 5xx，
经 providers 层映射后即 MODEL_UNAVAILABLE；上游 429（PROVIDER_OVERLOADED）
是限流而非模型损坏，不计失败。哪些码触发 record_failure 的决策在编排层
（services/invocation.py）做出，本状态机只收"失败/成功"事实——但口径在
此处与单测中显式钉死，防止未来把 429 误计入。

时钟注入（默认 time.monotonic）：单测不真睡。单进程内存实现（不变量 #17）：
状态对所有请求即时可见，重启即清零——保护机制非记账数据，可接受。
"""

import time
from collections.abc import Callable
from typing import Final, Literal

from llm_gateway.observability.metrics import update_breaker_state

# 状态机参数（spec 任务 4 定值）：连续 5 次失败打开，开 30 秒。
FAILURE_THRESHOLD: Final[int] = 5
OPEN_SECONDS: Final[float] = 30.0

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    # 单个模型的熔断状态机。协程单线程模型下无需锁：
    # allow_request / record_* 的读改写都在事件循环内同步完成。

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.state: BreakerState = "closed"
        self._consecutive_failures = 0
        self._opened_at = 0.0
        # half_open 的探测名额：放行时置位，探测有结果（成功/失败）或被
        # 准入后层拒绝而归还（relinquish_probe）时清除。
        self._probe_in_flight = False
        # 模型坐标：get_breaker 创建时注入，状态广播（M10）用它打标签；
        # 直建新实例（单测）缺省 None 时不广播——状态机语义零依赖观测面。
        self.model: str | None = None

    def _broadcast(self) -> None:
        # 熔断状态迁移点广播独热（M10 任务 4：机制在动作点计数）。
        if self.model is not None:
            update_breaker_state(self.model, self.state)

    def allow_request(self) -> bool:
        # 准入查询。open 到期自动转 half_open 并给出唯一探测名额。
        if self.state == "closed":
            return True
        now = self._clock()
        if self.state == "open":
            if now - self._opened_at < OPEN_SECONDS:
                return False
            self.state = "half_open"
            self._broadcast()
        # half_open：探测名额一次一个；已有探测在途则其余请求仍拒。
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def relinquish_probe(self) -> None:
        # 归还探测名额：探测请求拿到放行后未走到上游（被准入后层拒绝、
        # 端点异常等），名额归还不计成败——否则半开窗口会被幽灵探测卡死。
        if self.state == "half_open":
            self._probe_in_flight = False

    def record_success(self) -> None:
        # 成功闭合：半开探测成功或闭合态正常成功，连续失败计数清零。
        self.state = "closed"
        self._consecutive_failures = 0
        self._probe_in_flight = False
        self._broadcast()

    def record_failure(self) -> None:
        # 失败计入：半开探测失败重新打开；闭合态累计到阈值打开。
        if self.state == "half_open":
            self._open()
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= FAILURE_THRESHOLD:
            self._open()

    def _open(self) -> None:
        self.state = "open"
        self._opened_at = self._clock()
        self._probe_in_flight = False
        self._broadcast()
        # 连续失败数保留在阈值上：重新打开后若到期再探测失败，直接重开，
        # 不需要再攒 5 次（半开的语义就是"给一次机会"）。


# ---------------------------------------------------------------------------
# 进程内注册表（不变量 #17）：按模型一个状态机，准入与编排共享同一实例。
# ---------------------------------------------------------------------------

_BREAKERS: dict[str, CircuitBreaker] = {}


def get_breaker(model: str) -> CircuitBreaker:
    # 懒创建：模型名在配置白名单内，但熔断器不必在启动期预建。
    # 创建点注入模型坐标：状态迁移广播（M10）据此打标签。
    breaker = _BREAKERS.get(model)
    if breaker is None:
        breaker = CircuitBreaker()
        breaker.model = model
        _BREAKERS[model] = breaker
    return breaker


def reset_breakers() -> None:
    # 测试卫生面：进程内全局状态在测试间必须可清（契约 conftest autouse）。
    _BREAKERS.clear()
