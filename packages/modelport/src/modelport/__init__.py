"""ModelPort：LLM Gateway 的 Agent 侧客户端包（Agent 的唯一依赖）。

不变量 #2（design.md §8）：Agent 只 import modelport，不导入任何供应商
SDK——openai SDK 在本包内部是实现细节，公开面（__all__）只有本包自己的
类型。错误统一为 ModelPortError 层级（.code 是网关稳定错误码），
request_id 可从响应对象直接取到，便于与 /v1/traces 对账。

典型用法：

    import modelport

    port = modelport.ModelPort(base_url="http://localhost:8000", api_key="sk-...")
    result = await port.complete(
        model="general-primary",
        messages=[{"role": "user", "content": "你好"}],
    )
    print(result.content, result.request_id)
"""

from modelport.client import (
    Completion,
    ModelPort,
    StreamChunk,
    Usage,
    prompt_ref,
    validation_ref,
)
from modelport.errors import (
    AuthenticationError,
    CancelledRequestError,
    GatewayUnavailableError,
    ModelPortError,
    RateLimitedError,
    RequestRejectedError,
    TransportError,
    UpstreamFailedError,
)

__version__ = "0.1.0"

# 公开面是封闭清单：全部是本包自有类型，无供应商 SDK 符号再导出
# （tests/contract/test_modelport.py 的隔离测试按此清单断言）。
__all__ = [
    "ModelPort",
    "Completion",
    "StreamChunk",
    "Usage",
    "prompt_ref",
    "validation_ref",
    "ModelPortError",
    "AuthenticationError",
    "RequestRejectedError",
    "RateLimitedError",
    "UpstreamFailedError",
    "GatewayUnavailableError",
    "CancelledRequestError",
    "TransportError",
]
