"""Prometheus 指标注册（M10 任务 1）。

每个指标对应一个已有机制（design.md §6"没有为观测而观测"）：
- ``llm_requests_total{model, status}``：终态计数，与 trace 同一终态入口
  （trace_service.record_trace）记账——metrics 是速率、trace 是审计，
  口径不同但事件同源，天然继承"恰好一次终态"的幂等防线；
- ``llm_request_latency_seconds``：终态延迟直方图，同一出口观测；
- ``llm_tokens_total{model, direction}``：按观测到的 usage 累计——
  只在实际服务到模型（actual_model 非 None）时入账，取消/早退的缺口
  与 trace 同款语义（不伪造）；
- ``llm_retries_total{model}``：编排层退避重试点计数——修复调用是
  质量关卡的再调不是重试，不计（与 attempts 审计口径的差异由 trace 承载）；
- ``llm_rate_limited_total{model}``：准入 RPM 令牌桶拒绝点计数（上游
  429 是可重试故障，走重试口径不走限流口径）；
- ``llm_requests_in_flight``：编排入口持有、终态（含取消）释放；
- ``llm_breaker_state{model, state}``：熔断状态机迁移点广播三态独热。

重启清零是 Prometheus 进程内计数语义的本然（spec 边界）。指标接线点
全部在机制的动作点（spec 任务 4）：重试/限流/熔断各自在自己的拒绝/
迁移点计数，终态类指标集中在 record_trace 唯一出口，无遗漏无重复。
"""

from typing import Final

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

REQUESTS_TOTAL: Final = Counter(
    "llm_requests_total",
    "网关请求终态计数（model=请求模型，status=success/failed/cancelled）",
    ["model", "status"],
)

LATENCY_SECONDS: Final = Histogram(
    "llm_request_latency_seconds",
    "请求终态延迟（从 RunContext 构建到终态迁移）",
)

TOKENS_TOTAL: Final = Counter(
    "llm_tokens_total",
    "已观测 Token 累计（direction=input|output，model=实际服务模型）",
    ["model", "direction"],
)

RETRIES_TOTAL: Final = Counter(
    "llm_retries_total",
    "编排层退避重试次数（每次可重试->再试计一次，修复调用不计）",
    ["model"],
)

RATE_LIMITED_TOTAL: Final = Counter(
    "llm_rate_limited_total",
    "准入 RPM 限流拒绝次数",
    ["model"],
)

REQUESTS_IN_FLIGHT: Final = Gauge(
    "llm_requests_in_flight",
    "已进入编排状态机的在途请求数（准入拒绝不进编排，不计入）",
)

BREAKER_STATE: Final = Gauge(
    "llm_breaker_state",
    "熔断器状态独热（同一模型三态恰好一个为 1）",
    ["model", "state"],
)

# 熔断三态词表与 breaker.BreakerState 逐字一致（状态机迁移点广播）。
_BREAKER_STATES: Final[tuple[str, ...]] = ("closed", "open", "half_open")

# 本模块注册的指标全集：测试卫生面（reset_metrics）逐一清零，
# 避免用例间累计计数串扰断言。
_ALL_METRICS: Final[tuple[object, ...]] = (
    REQUESTS_TOTAL,
    LATENCY_SECONDS,
    TOKENS_TOTAL,
    RETRIES_TOTAL,
    RATE_LIMITED_TOTAL,
    REQUESTS_IN_FLIGHT,
    BREAKER_STATE,
)


def update_breaker_state(model: str, state: str) -> None:
    # 熔断状态迁移点广播独热：三态全部置值（当前态 1 其余 0），
    # 保证任何时刻每个模型都有一条可读的状态样本。
    for candidate in _BREAKER_STATES:
        BREAKER_STATE.labels(model=model, state=candidate).set(1 if candidate == state else 0)


def reset_metrics() -> None:
    # 测试卫生面（与 reset_breakers / reset_admission 同款约定）：
    # 清空带标签子项、归零无标签数值。prometheus_client 无官方 reset API，
    # 这里只触碰其文档化结构内长期稳定的成员（_metrics / _value / _sum / _buckets）。
    for metric in _ALL_METRICS:
        children = getattr(metric, "_metrics", None)
        if children is not None:
            children.clear()
        value = getattr(metric, "_value", None)
        if value is not None and hasattr(value, "set"):
            value.set(0)
        total = getattr(metric, "_sum", None)
        if total is not None and hasattr(total, "set"):
            total.set(0)
        buckets = getattr(metric, "_buckets", None)
        if buckets is not None:
            for bucket in buckets:
                bucket.set(0)
    # REGISTRY 本身无累计状态需要清：上面清完即空（本进程只注册这七个指标）。
    _ = REGISTRY  # 保留导入引用：本模块与默认注册表绑定的事实可寻址
