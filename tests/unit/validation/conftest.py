"""validation 单测共享夹具（M08）：双模型拓扑 + FakeAdapter 挂表 + 全局清零。

与 tests/unit/services/test_invocation.py 的夹具同构（编排层行为边界测试的
既定形态）：本目录的流水线用例同样直调 call_with_fallback，需要拓扑与剧本
基础设施；独立一份避免跨目录共享夹具的隐式耦合。
"""

import pytest

from llm_gateway.core.breaker import reset_breakers
from llm_gateway.core.schemas import ModelConfig
from llm_gateway.providers import PROVIDER_REGISTRY
from llm_gateway.providers.fake import FakeAdapter, Scenario
from llm_gateway.services import invocation as inv
from llm_gateway.services.trace_service import CALL_TRACES

PRIMARY = "general-primary"
BACKUP = "general-backup"


def _model_config(fallback: tuple[str, ...] = ()) -> ModelConfig:
    return ModelConfig(
        provider_model="vendor-model",
        base_url="https://fake.test",
        api_key_env="FAKE_KEY",
        supports_structured_output=True,
        provider="fake",
        fallback=fallback,
    )


@pytest.fixture
def topology(monkeypatch):
    """双模型拓扑（primary -> backup）：工厂形态，测试必须调用（同 test_invocation）。"""

    def _install() -> dict[str, ModelConfig]:
        configs = {
            PRIMARY: _model_config(fallback=(BACKUP,)),
            BACKUP: _model_config(),
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


async def no_sleep(_delay: float) -> None:
    # 注入睡眠：不真睡（与 test_invocation 同款约定）。
    return None
