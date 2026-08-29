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

    # stream 用普通 def 声明而非 async def：Adapter 的实现是 async generator，
    # 调用即返回 AsyncIterator（不产生 awaitable）。若协议声明成 async def，
    # 类型上承诺的是 Coroutine[..., AsyncIterator[str]]，任何 async generator
    # 实现都无法结构匹配，调用方的 `async for ... in provider.stream(...)`
    # 也会被判错。纯类型声明调整：Protocol 成员从不执行，无运行期影响。
    def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]: ...
