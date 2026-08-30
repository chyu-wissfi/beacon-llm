"""稳定错误码注册表（单一事实来源）。

GatewayError 是内部各层向上抛出的统一错误形态：code 是对外稳定契约（不随重构
漂移），message/status 的默认三元组集中在本文件注册，上层（api/）负责把
GatewayError 适配成 HTTP 响应体，core 自身不依赖任何上层（design.md §2.2）。

错误码是**封闭集合**，两层表达缺一不可：
- 类型层：ErrorCode 是 Literal 联合，调用点写注册表外的字符串字面量会被
  pyright 直接拒绝；
- 运行时层：注册表查不到的 code（动态拼出的字符串、绕过类型检查的 str）
  在 GatewayError 构造期抛 ValueError——封闭性由此可被单测稳定断言
  （M02 验收命令 `pytest -k closed_set`）。

单一事实来源的落点：错误码字符串只在常量定义处赋值一次，其他模块一律
import 常量符号、禁止再写字面量；默认 message/status 只在 ERROR_REGISTRY
登记一次，调用点不再手写（动态 message 是唯一例外，见 ErrorSpec）。
"""

from dataclasses import dataclass
from typing import Final, Literal

# ---------------------------------------------------------------------------
# 错误码封闭集合：类型层（Literal 联合）+ 调用点引用符号（模块级常量）
# ---------------------------------------------------------------------------

# 对外稳定契约的全体成员。新增/修改错误码 = 改 Literal 成员 + 对应常量 +
# ERROR_REGISTRY 条目，三处都在本文件内；Literal 与注册表互相兜底
# （只加 Literal 不登记注册表 -> 运行时构造即抛 ValueError，会被 Task 4 的
# 全量三元组单测抓住）。
ErrorCode = Literal[
    # 现有码（demo 等价迁移，(code, message, status) 三元组冻结，不得漂移）
    "unknown_model",
    "unknown_prompt_template",
    "missing_prompt_variable",
    "gateway_misconfigured",
    "invalid_json",
    "schema_validation_failed",
    "model_unavailable",
    "upstream_stream_failed",
    "use_stream_endpoint",
    "unsupported_combination",
    "structured_output_unsupported",
    # 预留码（本里程碑只注册不使用，M05 准入 / M06 预算 / M08 校验消费）
    "unauthorized",
    "rate_limited",
    "overloaded",
    "provider_overloaded",
    "token_budget_exceeded",
    "circuit_open",
    "unsupported_field",
    "output_truncated",
    "business_validation_failed",
    "unknown_validation_profile",
    "request_cancelled",
]

# 调用点的引用符号：错误码字符串只在这里赋值一次，其他模块 import 这些名字。
UNKNOWN_MODEL: Final[ErrorCode] = "unknown_model"
UNKNOWN_PROMPT_TEMPLATE: Final[ErrorCode] = "unknown_prompt_template"
MISSING_PROMPT_VARIABLE: Final[ErrorCode] = "missing_prompt_variable"
GATEWAY_MISCONFIGURED: Final[ErrorCode] = "gateway_misconfigured"
INVALID_JSON: Final[ErrorCode] = "invalid_json"
SCHEMA_VALIDATION_FAILED: Final[ErrorCode] = "schema_validation_failed"
MODEL_UNAVAILABLE: Final[ErrorCode] = "model_unavailable"
UPSTREAM_STREAM_FAILED: Final[ErrorCode] = "upstream_stream_failed"
USE_STREAM_ENDPOINT: Final[ErrorCode] = "use_stream_endpoint"
UNSUPPORTED_COMBINATION: Final[ErrorCode] = "unsupported_combination"
STRUCTURED_OUTPUT_UNSUPPORTED: Final[ErrorCode] = "structured_output_unsupported"
UNAUTHORIZED: Final[ErrorCode] = "unauthorized"
RATE_LIMITED: Final[ErrorCode] = "rate_limited"
OVERLOADED: Final[ErrorCode] = "overloaded"
PROVIDER_OVERLOADED: Final[ErrorCode] = "provider_overloaded"
TOKEN_BUDGET_EXCEEDED: Final[ErrorCode] = "token_budget_exceeded"
CIRCUIT_OPEN: Final[ErrorCode] = "circuit_open"
UNSUPPORTED_FIELD: Final[ErrorCode] = "unsupported_field"
OUTPUT_TRUNCATED: Final[ErrorCode] = "output_truncated"
BUSINESS_VALIDATION_FAILED: Final[ErrorCode] = "business_validation_failed"
UNKNOWN_VALIDATION_PROFILE: Final[ErrorCode] = "unknown_validation_profile"
REQUEST_CANCELLED: Final[ErrorCode] = "request_cancelled"


@dataclass(frozen=True)
class ErrorSpec:
    """单个错误码的默认展示形态：注册表托管的默认 message 与 HTTP status。

    调用点可覆盖 message（注册表无法预知动态上下文，如缺失的变量名），
    但默认值必须取自这里——三元组稳定有唯一出处，Task 4 的逐码三元组
    稳定性单测以此表为准。
    """

    message: str
    status_code: int


# ---------------------------------------------------------------------------
# 默认三元组映射表：登记默认 message/status 的唯一位置
# ---------------------------------------------------------------------------

ERROR_REGISTRY: Final[dict[ErrorCode, ErrorSpec]] = {
    # -- 现有码：三元组与 demo 逐字一致（行为零改动铁律）--
    UNKNOWN_MODEL: ErrorSpec("模型不在 Gateway 允许列表中", 400),
    UNKNOWN_PROMPT_TEMPLATE: ErrorSpec("Prompt 模板不存在", 400),
    # 动态 message 的静态兜底：实际响应由调用点拼接变量名后覆盖 message。
    MISSING_PROMPT_VARIABLE: ErrorSpec("缺少 Prompt 变量", 400),
    GATEWAY_MISCONFIGURED: ErrorSpec("Gateway 模型凭据未配置", 503),
    INVALID_JSON: ErrorSpec("模型没有返回合法 JSON", 502),
    SCHEMA_VALIDATION_FAILED: ErrorSpec("模型结果不符合 response_schema", 502),
    MODEL_UNAVAILABLE: ErrorSpec("主模型和备用模型均不可用", 502),
    # 不作 GatewayError 抛出：只作为 trace error_code 与 SSE 终态事件 payload
    # 使用；登记默认三元组只为注册表形态统一，message/status 当前无行为面。
    UPSTREAM_STREAM_FAILED: ErrorSpec("上游流式输出失败", 502),
    USE_STREAM_ENDPOINT: ErrorSpec("流式请求请使用 /v1/llm/stream", 400),
    UNSUPPORTED_COMBINATION: ErrorSpec("流式输出不支持 response_schema", 400),
    STRUCTURED_OUTPUT_UNSUPPORTED: ErrorSpec("模型不支持 Structured Output", 400),
    # -- 预留码：本里程碑只注册不使用。status 取自 design.md 的错误映射表
    #    （§3.1/§3.2/§3.4/§3.7），message 是占位文案，消费里程碑接入时定稿；
    #    一经 Task 4 单测冻结，再改即为显式契约变更。 --
    UNAUTHORIZED: ErrorSpec("调用方未通过认证", 401),
    RATE_LIMITED: ErrorSpec("请求触发限流", 429),
    OVERLOADED: ErrorSpec("Gateway 过载保护中", 429),
    PROVIDER_OVERLOADED: ErrorSpec("供应商并发过载", 429),
    TOKEN_BUDGET_EXCEEDED: ErrorSpec("Token 预算已超限", 429),
    CIRCUIT_OPEN: ErrorSpec("模型熔断中，暂时不可用", 503),
    UNSUPPORTED_FIELD: ErrorSpec("请求包含不支持的字段", 400),
    OUTPUT_TRUNCATED: ErrorSpec("模型输出被截断", 502),
    BUSINESS_VALIDATION_FAILED: ErrorSpec("模型输出未通过业务校验", 502),
    UNKNOWN_VALIDATION_PROFILE: ErrorSpec("未知的 Validation Profile", 400),
    # design.md 未指定状态；499 是"客户端主动断开"的业界惯例（nginx），
    # 消费里程碑（M06 终态迁移）接入时再确认。
    REQUEST_CANCELLED: ErrorSpec("请求已取消", 499),
}


class GatewayError(Exception):
    """内部各层向上抛出的统一错误形态，code 是对外稳定契约。

    构造签名保持向后兼容（message/status_code 仍可显式传参），但默认值
    改由注册表托管：调用点只给 code，不再手写 message/status——三元组
    的单一事实来源因此成立。
    """

    # 将内部错误标准化为可安全暴露给调用方的稳定错误码和 HTTP 状态。
    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        status_code: int | None = None,
    ) -> None:
        spec = ERROR_REGISTRY.get(code)
        if spec is None:
            # 运行时封闭性：动态字符串/绕过类型检查的 code 在这里被拦下。
            # ValueError 而非 KeyError：这是编程错误（契约违反），不是查表缺失。
            raise ValueError(
                f"未注册的错误码: {code!r}（错误码是封闭集合，请先在 core/errors.py 注册）"
            )
        # 对外暴露为普通 str：消费方（HTTP 适配、trace、SSE payload）只见字符串。
        self.code: str = code
        self.message = message if message is not None else spec.message
        self.status_code = status_code if status_code is not None else spec.status_code
        super().__init__(self.message)
