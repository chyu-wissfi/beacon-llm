"""core/errors.py 注册表的单元测试：22 码三元组冻结 + 错误码封闭集（closed_set）。

三元组 (code, message, status) 是对外稳定契约：本文件把 22 个码的期望值逐字
硬编码为冻结快照——11 个现有码取自 demo 等价迁移时的实际行为（task-3 决议），
11 个预留码以 M02 注册表落锤值为准。注册表任何一侧的漂移都会在这里被抓住；
消费里程碑要改预留码三元组时，必须显式改本文件的快照——"显式契约变更"的
落点就在这里，而不是改完注册表无人知晓。
"""

from typing import cast, get_args

import pytest

from llm_gateway.core.errors import (
    BUSINESS_VALIDATION_FAILED,
    CIRCUIT_OPEN,
    ERROR_REGISTRY,
    GATEWAY_MISCONFIGURED,
    INVALID_JSON,
    MISSING_PROMPT_VARIABLE,
    MODEL_UNAVAILABLE,
    OUTPUT_TRUNCATED,
    OVERLOADED,
    PROVIDER_OVERLOADED,
    RATE_LIMITED,
    REQUEST_CANCELLED,
    SCHEMA_VALIDATION_FAILED,
    STRUCTURED_OUTPUT_UNSUPPORTED,
    TOKEN_BUDGET_EXCEEDED,
    UNAUTHORIZED,
    UNKNOWN_MODEL,
    UNKNOWN_PROMPT_TEMPLATE,
    UNKNOWN_VALIDATION_PROFILE,
    UNSUPPORTED_COMBINATION,
    UNSUPPORTED_FIELD,
    UPSTREAM_STREAM_FAILED,
    USE_STREAM_ENDPOINT,
    ErrorCode,
    GatewayError,
)

# ---------------------------------------------------------------------------
# 冻结快照：22 个码逐一登记期望三元组（注册表符号, code 字面量, message, status）
# ---------------------------------------------------------------------------

_FROZEN_TRIPLES: list[tuple[ErrorCode, str, str, int]] = [
    # -- 现有码：demo 等价迁移时的实际行为（task-3 决议第 2 条冻结）--
    (UNKNOWN_MODEL, "unknown_model", "模型不在 Gateway 允许列表中", 400),
    (UNKNOWN_PROMPT_TEMPLATE, "unknown_prompt_template", "Prompt 模板不存在", 400),
    (MISSING_PROMPT_VARIABLE, "missing_prompt_variable", "缺少 Prompt 变量", 400),
    (GATEWAY_MISCONFIGURED, "gateway_misconfigured", "Gateway 模型凭据未配置", 503),
    (INVALID_JSON, "invalid_json", "模型没有返回合法 JSON", 502),
    (SCHEMA_VALIDATION_FAILED, "schema_validation_failed", "模型结果不符合 response_schema", 502),
    (MODEL_UNAVAILABLE, "model_unavailable", "主模型和备用模型均不可用", 502),
    # 不作 GatewayError 抛出（只进 trace error_code 与 SSE 终态事件 payload），
    # 默认三元组同样冻结，形态与其他码保持统一。
    (UPSTREAM_STREAM_FAILED, "upstream_stream_failed", "上游流式输出失败", 502),
    (USE_STREAM_ENDPOINT, "use_stream_endpoint", "流式请求请使用 /v1/llm/stream", 400),
    (UNSUPPORTED_COMBINATION, "unsupported_combination", "流式输出不支持 response_schema", 400),
    (STRUCTURED_OUTPUT_UNSUPPORTED, "structured_output_unsupported", "模型不支持 Structured Output", 400),
    # -- 预留码：以 M02 注册表落锤值冻结（本里程碑只注册不使用）--
    (UNAUTHORIZED, "unauthorized", "调用方未通过认证", 401),
    (RATE_LIMITED, "rate_limited", "请求触发限流", 429),
    (OVERLOADED, "overloaded", "Gateway 过载保护中", 429),
    (PROVIDER_OVERLOADED, "provider_overloaded", "供应商并发过载", 429),
    (TOKEN_BUDGET_EXCEEDED, "token_budget_exceeded", "Token 预算已超限", 429),
    (CIRCUIT_OPEN, "circuit_open", "模型熔断中，暂时不可用", 503),
    (UNSUPPORTED_FIELD, "unsupported_field", "请求包含不支持的字段", 400),
    (OUTPUT_TRUNCATED, "output_truncated", "模型输出被截断", 502),
    (BUSINESS_VALIDATION_FAILED, "business_validation_failed", "模型输出未通过业务校验", 502),
    (UNKNOWN_VALIDATION_PROFILE, "unknown_validation_profile", "未知的 Validation Profile", 400),
    (REQUEST_CANCELLED, "request_cancelled", "请求已取消", 499),
]

# 收集期自检：快照必须恰好覆盖 22 码，防止后续维护时静默漏行或重复登记。
assert len(_FROZEN_TRIPLES) == 22
assert len({entry[1] for entry in _FROZEN_TRIPLES}) == 22


# ---------------------------------------------------------------------------
# 三元组稳定：逐码断言 (code, message, status) 与冻结快照一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "code_str", "message", "status_code"),
    _FROZEN_TRIPLES,
    ids=[entry[1] for entry in _FROZEN_TRIPLES],
)
def test_registry_triple_frozen(spec: ErrorCode, code_str: str, message: str, status_code: int) -> None:
    # 用注册表符号构造（与调用点同路径），断言产物三元组与冻结快照逐字一致：
    # 既锁注册表值，也锁符号本身不被改指到别的码。
    error = GatewayError(spec)
    assert error.code == code_str
    assert error.message == message
    assert error.status_code == status_code


def test_registry_members_exactly_match_frozen_set() -> None:
    # 封闭集合的成员清单本身也是契约：注册表与快照必须逐码对齐，多一个码
    # （三元组未冻结）或少一个码都不允许；Literal 联合与注册表键完全一致
    # ——错误码的两层表达（类型层 / 运行时层）互相兜底，任何一侧单独漂移
    # 都在这里现形。
    assert set(ERROR_REGISTRY) == {entry[1] for entry in _FROZEN_TRIPLES}
    assert len(ERROR_REGISTRY) == len(_FROZEN_TRIPLES) == 22
    assert set(get_args(ErrorCode)) == set(ERROR_REGISTRY)


# ---------------------------------------------------------------------------
# 错误码封闭集（M02 验收命令按 -k closed_set 选中本组）
# ---------------------------------------------------------------------------


def test_closed_set_rejects_unregistered_code() -> None:
    # cast 是故意的：模拟"动态拼出的字符串 / 绕过类型检查的 str"这条运行时
    # 要拦的路径——封闭性的运行时防线必须在 GatewayError 构造期（而非 raise
    # 时）生效，抛 ValueError（编程错误，不是查表缺失）。
    unregistered = cast(ErrorCode, "no_such_code")
    with pytest.raises(ValueError, match="no_such_code"):
        GatewayError(unregistered)


def test_closed_set_accepts_registered_code() -> None:
    # 反面：集合内的码必须可构造，默认 message/status 由注册表托管——
    # 调用点只给 code 是唯一必填项。
    error = GatewayError(UNKNOWN_MODEL)
    assert error.code == "unknown_model"
    assert error.message == "模型不在 Gateway 允许列表中"
    assert error.status_code == 400


def test_closed_set_allows_message_override_but_code_and_status_stay_registered() -> None:
    # 动态 message（如拼接缺失的变量名）是默认值之上唯一合法的覆盖面；
    # code 不受影响，status 仍取注册表默认——三元组中只有 message 开放给调用点。
    error = GatewayError(MISSING_PROMPT_VARIABLE, message="缺少 Prompt 变量: product_name")
    assert error.code == "missing_prompt_variable"
    assert error.message == "缺少 Prompt 变量: product_name"
    assert error.status_code == 400
