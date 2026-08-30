"""业务校验包：Validation Profile（M08，ADR-0005 代码注册表）。

分层定位：结构校验（供应商约束 + 本地 jsonschema 双重）之后叠加业务校验——
"结构合法但业务非法"的 JSON 绝不进入 Agent Loop（不变量 #9/#10）。
注册表与解析入口见 registry.py，规则本体（Pydantic 模型）见 profiles.py。
"""

from llm_gateway.validation.profiles import OrderDecision
from llm_gateway.validation.registry import (
    ORDER_DECISION_V1,
    VALIDATION_PROFILES,
    ValidationProfile,
    resolve_profile,
)

__all__ = [
    "ORDER_DECISION_V1",
    "VALIDATION_PROFILES",
    "OrderDecision",
    "ValidationProfile",
    "resolve_profile",
]
