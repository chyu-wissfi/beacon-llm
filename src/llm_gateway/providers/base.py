"""供应商 Adapter 的统一接口协议。

编排层只面向本 Protocol 编程，不依赖具体供应商 SDK（design.md §2.2）；
新增供应商时实现本协议即可，对编排零波及。
"""

from collections.abc import AsyncIterator
from typing import Any, Protocol

from llm_gateway.core.schemas import Message, ModelConfig, Usage


class Provider(Protocol):
    # 规定供应商 Adapter 的统一接口，业务流程不依赖具体 SDK。
    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]: ...

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]: ...
