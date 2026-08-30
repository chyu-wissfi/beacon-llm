"""Provider 单测共享夹具。

openai 3.x 的 legacy httpx shim 复用 tests/support/httpx_shim.py（M04 从
contract conftest 提取）；provider 直调不经 FastAPI，只需假 key + respx。
anthropic 1.2.0 的 httpx2 流量同样在 respx 视野外，其单测改用 SDK 方法层
mock（见 test_anthropic_provider.py 模块注），这里只提供配置夹具。
"""

import pytest

from llm_gateway.core.schemas import ModelConfig
from tests.support.httpx_shim import openai_legacy_httpx  # noqa: F401

__all__ = ["openai_legacy_httpx", "chat_config", "responses_config", "anthropic_config"]


@pytest.fixture(autouse=True)
def _openai_legacy_httpx_auto(openai_legacy_httpx):
    # 全目录自动启用 shim：openai 3.x 的 httpx2 流量 respx 不可见，不启用
    # 会静默漏到真实网络。
    return openai_legacy_httpx


@pytest.fixture
def chat_config() -> ModelConfig:
    # provider_api=chat 的标准模型坐标（与 config/models.yaml 同构，base_url
    # 指向 respx 可拦截的假域）。
    return ModelConfig(
        provider_model="deepseek-v4-flash",
        base_url="https://upstream.test/v1",
        api_key_env="DEEPSEEK_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
        provider_api="chat",
    )


@pytest.fixture
def responses_config(chat_config: ModelConfig) -> ModelConfig:
    # 同坐标但走 Responses 传输。
    return ModelConfig(
        provider_model=chat_config.provider_model,
        base_url=chat_config.base_url,
        api_key_env=chat_config.api_key_env,
        supports_structured_output=True,
        structured_output_mode="json_schema",
        provider_api="responses",
    )


@pytest.fixture
def anthropic_config(chat_config: ModelConfig) -> ModelConfig:
    # Anthropic 模型坐标：结构化输出仅 json_object（provider 层裁定）。
    return ModelConfig(
        provider_model="claude-test",
        base_url=chat_config.base_url,
        api_key_env="ANTHROPIC_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
        provider_api="chat",
        provider="anthropic",
    )


@pytest.fixture(autouse=True)
def _provider_env(monkeypatch):
    # 与契约测试同款假 key；缺凭据场景的用例自行 monkeypatch.delenv 覆盖。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-primary-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
