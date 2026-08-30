"""Validation Profile 注册表（ADR-0005：代码注册表，封闭集合）。

注册表 = (name, version) 坐标 -> Profile 登记项的静态查表面：新增规则改代码、
走测试、发版，不做热加载（spec 边界）。两个消费点，语义各有分工：
- api 层准入前预检：未注册 -> 400 unknown_validation_profile，上游请求数 == 0
  （spec 任务 3；请求级错误不进状态机、不产生 trace）；
- 编排层入口（run_context 构建期）：再解析一次入 RunContext——纯查表无副作用，
  直调服务层的链路同样有此关卡，且未注册在 trace 落账之前拒绝。
"""

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel

from llm_gateway.core.errors import UNKNOWN_VALIDATION_PROFILE, GatewayError
from llm_gateway.core.schemas import ValidationSelection
from llm_gateway.validation.profiles import OrderDecision


@dataclass(frozen=True)
class ValidationProfile:
    """注册表登记项：name/version 坐标 + Pydantic 模型本体。

    模型本体是业务规则的执行者：model_validate 按字段约束 + model_validator
    规则解析，违规抛 ValidationError（invocation 层收敛为
    business_validation_failed，判定与报错的分工见 profiles.py 模块注）。
    """

    name: str
    version: str
    model: type[BaseModel]


ORDER_DECISION_V1: Final[ValidationProfile] = ValidationProfile(
    name="order_decision",
    version="v1",
    model=OrderDecision,
)

# (name, version) -> 登记项：注册表的事实来源，新增 Profile 在此登记一行。
VALIDATION_PROFILES: Final[dict[tuple[str, str], ValidationProfile]] = {
    (ORDER_DECISION_V1.name, ORDER_DECISION_V1.version): ORDER_DECISION_V1,
}


def resolve_profile(selection: ValidationSelection | None) -> ValidationProfile | None:
    """选择项 -> 登记项：未指定返回 None，未注册抛 400 稳定码。

    动态 message 是注册表默认值之上的合法覆盖（同 missing_prompt_variable
    先例）：点名 name/version 坐标，调用方一次定位写错了哪个档案。
    """
    if selection is None:
        return None
    profile = VALIDATION_PROFILES.get((selection.name, selection.version))
    if profile is None:
        raise GatewayError(
            UNKNOWN_VALIDATION_PROFILE,
            message=f"未知的 Validation Profile: {selection.name}/{selection.version}",
        )
    return profile
