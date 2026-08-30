"""pytest 全局配置：自定义 marker 注册。

`live` marker 标记需要真实上游凭据与网络的验收测试（M12 的 live 验收形态）。
排除 live 用 `-m "not live"` 的 marker 过滤而非路径排除：live 用例今后可与
离线用例同目录共存，离线质量闸门（make check）与真实验收（make test-live）
只差一个 marker 表达式。

在 pytest_configure 里显式注册还消除未知 marker 告警，并让 marker 语义
在仓库内有单一事实来源（`pytest --markers` 可查）。
"""

import pytest

from llm_gateway.observability.metrics import reset_metrics


def pytest_configure(config: pytest.Config) -> None:
    # 注册 live marker：make check 排除，make test-live 单独执行。
    config.addinivalue_line(
        "markers",
        "live: 需要真实上游与凭据的验收测试；make check 用 -m 'not live' 排除，make test-live 单独执行",
    )


@pytest.fixture(autouse=True)
def _reset_metrics():
    # Prometheus 指标是进程内全局累计（不变量 #17 同款单进程面）：
    # 状态机/准入/记账类用例都会触接线点，用例前后清零避免跨用例串扰。
    reset_metrics()
    yield
    reset_metrics()
