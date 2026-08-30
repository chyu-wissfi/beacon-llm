"""Validation Profile 的规则本体（ADR-0005：规则即代码）。

每个 Profile 本体 = 一个带业务规则的 Pydantic 模型：字段结构是第一层业务约束
（类型/必填/取值），跨字段规则（互斥、条件依赖）用 `model_validator` 写在第二层
——规则词汇表不受限（spec 任务 2：任意复杂度的业务规则），全部违规统一收敛为
Pydantic ValidationError，由 invocation 层翻译为 business_validation_failed
（错误码不在本包出现：本体只负责"判定"，"报错形态"归错误注册表与编排层）。

新增规则 = 在本文件加模型 + registry.py 登记 + 单测，随版本发布（无热加载）。
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrderDecision(BaseModel):
    """订单决策 Profile 本体（order_decision/v1）：演示两类跨字段业务规则。

    - 互斥字段：approve 与 reject 不得同时为真（二选一的决策语义）；
    - 条件依赖：escalate 为真时必须给出 escalation_reason（升级必有理由）。
    两条规则都写在 model_validator 里——结构校验（JSON Schema）表达不了的
    正是这类"每个字段单独看都合法、合起来非法"的约束。
    """

    # forbid：结构关（response_schema）放进来的字段集合之外再混入未知键，
    # 业务关同样拒绝——两层关卡宁严勿松（不变量 #10）。
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=1, max_length=100)
    approve: bool = False
    reject: bool = False
    escalate: bool = False
    escalation_reason: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _check_business_rules(self) -> Self:
        if self.approve and self.reject:
            raise ValueError("approve 与 reject 互斥，不能同时为 true")
        if self.escalate and not self.escalation_reason:
            raise ValueError("escalate=true 时必须提供 escalation_reason")
        return self
