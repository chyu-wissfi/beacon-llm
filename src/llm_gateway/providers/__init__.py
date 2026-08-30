"""Provider 包：配置驱动的注册表（M04 任务 5）。

provider 名 -> Adapter 实例的查表面：invocation 装配点按 ModelConfig.provider
取实例，services 层只面向 Provider Protocol（design.md §3.6 / ADR-0002）——
新增供应商只需实现 Protocol 并在此登记，编排零波及。

注册表里的 FakeAdapter 只带默认 Success 剧本：剧本由调用方构造
（`FakeAdapter(scenario=...)`），测试与后续里程碑都自建实例。
"""

from typing import Final

from llm_gateway.providers.anthropic_provider import AnthropicProvider
from llm_gateway.providers.base import Provider
from llm_gateway.providers.fake import FakeAdapter
from llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

PROVIDER_REGISTRY: Final[dict[str, Provider]] = {
    "openai_compatible": OpenAICompatibleProvider(),
    "anthropic": AnthropicProvider(),
    "fake": FakeAdapter(),
}

__all__ = ["PROVIDER_REGISTRY"]
