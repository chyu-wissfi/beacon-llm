"""ModelPort 错误层：网关错误码 -> 统一异常层级（M12 任务 1）。

design.md §4.7（ModelPort 防腐层）：网关的 OpenAI 风格错误体
（{"error": {"message", "type", "code"}}）在这里收敛为 ModelPortError 层级——
`.code` 是网关稳定错误码（core/errors.py 的封闭集合），Agent 的 except 分支
只见本模块的类型，供应商 SDK 异常形态永不外泄。

层级划分按"调用方该怎么办"而不是按 HTTP 状态：
- AuthenticationError：换 key（401 unauthorized）；
- RequestRejectedError：改请求本身——模型名/字段/模板/档案坐标写错（400 类），
  重试无意义；
- RateLimitedError：按建议节奏退避后再试（429 类，可带 retry_after）；
- UpstreamFailedError：上游/输出质量问题（502 类 + 流内终态错误），网关已用尽
  Run 预算，调用方决定是否降级处理；
- GatewayUnavailableError：网关自身暂不可用（503：缺凭据 / 熔断中），稍后重试；
- CancelledRequestError：请求被取消（499）；
- TransportError：传输层失败（连不上网关 / 超时），没有网关响应故无 code。

未知码兜底 ModelPortError 基类：错误码是封闭集合，但客户端不应因网关新增码
而崩溃——码原样携带，分类退到基类（向前兼容姿态）。
"""

# 按网关稳定码分派的异常类映射表在 client 侧消费（见 client._map_sdk_exception）；
# 本模块只定义层级与构造形态，保持"类型定义"单一职责。


class ModelPortError(Exception):
    """ModelPort 异常层级之根：Agent 感知到的一切失败都是本族异常。

    属性：
    - code：网关稳定错误码（str）；传输层失败无网关响应，为 None；
    - status_code：网关响应的 HTTP 状态（可用时）；
    - retry_after：429 类的建议重试间隔（秒），网关未给则 None。
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.retry_after = retry_after


class AuthenticationError(ModelPortError):
    """401 unauthorized：调用方 key 未通过认证。"""


class RequestRejectedError(ModelPortError):
    """400 类：请求本身被拒（未知模型/模板/档案、白名单外字段等）。重试无意义。"""


class RateLimitedError(ModelPortError):
    """429 类：RPM 限流 / 并发过载 / Token 预算超限。按 retry_after 退避。"""


class UpstreamFailedError(ModelPortError):
    """502 类与流内终态错误：上游不可用或输出未过网关校验关卡。"""


class GatewayUnavailableError(ModelPortError):
    """503：网关自身暂不可用（凭据未配置 / 模型熔断中）。"""


class CancelledRequestError(ModelPortError):
    """499 request_cancelled：请求被取消。"""


class TransportError(ModelPortError):
    """传输层失败（连接网关失败 / 超时）：无网关响应，code 恒为 None。"""
