"""observability/metrics.py 单测：指标对象语义与熔断状态广播。

接线点的端到端计数（剧本触发后精确增量）在契约层
（tests/contract/test_metrics.py）；本文件只管注册面自身的行为：
独热广播、重置卫生、标签维度。
"""

from prometheus_client import REGISTRY

from llm_gateway.observability.metrics import (
    BREAKER_STATE,
    RATE_LIMITED_TOTAL,
    REQUESTS_IN_FLIGHT,
    REQUESTS_TOTAL,
    RETRIES_TOTAL,
    TOKENS_TOTAL,
    reset_metrics,
    update_breaker_state,
)


def _sample(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


def test_metrics_registered_with_spec_names() -> None:
    # spec 任务 1 的七个指标名逐字钉死：改名会击穿 Prometheus 抓取契约。
    # Counter 的内部 _name 不带 _total 后缀（prometheus_client 输出时自动补），
    # 故此处断言去后缀形态；文本协议面的全名在契约层 /metrics 用例断言。
    names = {metric._name for metric in (  # noqa: SLF001  # prometheus 内部名即契约面
        REQUESTS_TOTAL,
        TOKENS_TOTAL,
        RETRIES_TOTAL,
        RATE_LIMITED_TOTAL,
        BREAKER_STATE,
    )}
    assert names == {
        "llm_requests",
        "llm_tokens",
        "llm_retries",
        "llm_rate_limited",
        "llm_breaker_state",
    }
    assert REQUESTS_IN_FLIGHT._name == "llm_requests_in_flight"  # noqa: SLF001


def test_update_breaker_state_is_one_hot() -> None:
    # 独热：同一模型三态恰好一个为 1，迁移后旧态归 0。
    update_breaker_state("demo-model", "closed")
    assert _sample("llm_breaker_state", model="demo-model", state="closed") == 1
    assert _sample("llm_breaker_state", model="demo-model", state="open") == 0
    assert _sample("llm_breaker_state", model="demo-model", state="half_open") == 0

    update_breaker_state("demo-model", "open")
    assert _sample("llm_breaker_state", model="demo-model", state="open") == 1
    assert _sample("llm_breaker_state", model="demo-model", state="closed") == 0


def test_reset_metrics_clears_labeled_children_and_gauges() -> None:
    # 卫生面：累计计数与在途值全部清零（用例隔离的可靠前提）。
    REQUESTS_TOTAL.labels(model="m", status="success").inc()
    TOKENS_TOTAL.labels(model="m", direction="input").inc(5)
    RETRIES_TOTAL.labels(model="m").inc(2)
    REQUESTS_IN_FLIGHT.inc()
    update_breaker_state("m", "open")

    reset_metrics()

    assert _sample("llm_requests_total", model="m", status="success") is None
    assert _sample("llm_tokens_total", model="m", direction="input") is None
    assert _sample("llm_retries_total", model="m") is None
    assert _sample("llm_breaker_state", model="m", state="open") is None
    assert REQUESTS_IN_FLIGHT._value.get() == 0  # noqa: SLF001


def test_rate_limited_counter_labels_per_model() -> None:
    RATE_LIMITED_TOTAL.labels(model="a").inc()
    RATE_LIMITED_TOTAL.labels(model="b").inc(2)
    assert _sample("llm_rate_limited_total", model="a") == 1
    assert _sample("llm_rate_limited_total", model="b") == 2
