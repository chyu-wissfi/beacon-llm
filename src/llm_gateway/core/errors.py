"""稳定错误码注册表（单一事实来源）。

GatewayError 是内部各层向上抛出的统一错误形态：code 是对外稳定契约
（不随重构漂移），status_code 由抛出方按语义指定。上层（api/）负责把它
适配成 HTTP 响应体，core 自身不依赖任何上层（design.md §2.2）。
"""


class GatewayError(Exception):
    # 将内部错误标准化为可安全暴露给调用方的稳定错误码和 HTTP 状态。
    def __init__(self, code: str, message: str, status_code: int = 502) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)
