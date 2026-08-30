"""OrderDecision Profile 本体的规则单测（M08 任务 2）。

断言的是业务规则的判定边界：互斥字段、条件依赖、字段级约束与 extra 封闭；
这些规则正是结构校验（JSON Schema）表达不了的"每个字段单独看都合法、
合起来非法"形态（ADR-0005 背景）。
"""

import pytest
from pydantic import ValidationError

from llm_gateway.validation.profiles import OrderDecision


def test_valid_approve_only_passes():
    decision = OrderDecision.model_validate({"order_id": "A-1", "approve": True})
    assert decision.approve is True
    assert decision.reject is False


def test_valid_escalate_with_reason_passes():
    decision = OrderDecision.model_validate(
        {"order_id": "A-1", "escalate": True, "escalation_reason": "金额超限"}
    )
    assert decision.escalate is True


def test_mutex_approve_and_reject_rejected():
    # 互斥字段：结构上两个布尔都合法，同时为真即业务非法。
    with pytest.raises(ValidationError) as exc_info:
        OrderDecision.model_validate(
            {"order_id": "A-1", "approve": True, "reject": True}
        )
    assert "互斥" in str(exc_info.value)


def test_conditional_escalate_requires_reason():
    # 条件依赖：escalate 为真而理由缺失/空白即违规。
    with pytest.raises(ValidationError) as exc_info:
        OrderDecision.model_validate({"order_id": "A-1", "escalate": True})
    assert "escalation_reason" in str(exc_info.value)
    with pytest.raises(ValidationError):
        OrderDecision.model_validate(
            {"order_id": "A-1", "escalate": True, "escalation_reason": None}
        )


def test_missing_order_id_rejected():
    with pytest.raises(ValidationError):
        OrderDecision.model_validate({"approve": True})


def test_unknown_extra_field_rejected():
    # extra=forbid：结构关放进来的字段集合之外再混键，业务关同样拒绝。
    with pytest.raises(ValidationError):
        OrderDecision.model_validate({"order_id": "A-1", "approve": True, "shadow": 1})
