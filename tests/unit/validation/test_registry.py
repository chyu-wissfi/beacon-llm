"""Validation Profile 注册表与解析入口单测（M08 任务 2/3）。

注册表是代码（ADR-0005）：查表、未注册 400 稳定码、坐标一致性——
"未注册在调用模型前失败"的服务层半边（HTTP 半边在契约测试）。
"""

import pytest

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import ValidationSelection
from llm_gateway.validation import (
    ORDER_DECISION_V1,
    VALIDATION_PROFILES,
    resolve_profile,
)


def test_registry_coordinates_match_profile_entry():
    # 查表键与登记项坐标必须一致：漂移即"点名不到已注册的档案"。
    assert (ORDER_DECISION_V1.name, ORDER_DECISION_V1.version) == (
        "order_decision",
        "v1",
    )
    assert VALIDATION_PROFILES[("order_decision", "v1")] is ORDER_DECISION_V1


def test_resolve_none_selection_returns_none():
    # 未指定 validation：无业务关，返回 None（流水线跳过业务校验层）。
    assert resolve_profile(None) is None


def test_resolve_registered_profile():
    selection = ValidationSelection(name="order_decision", version="v1")
    assert resolve_profile(selection) is ORDER_DECISION_V1


@pytest.mark.parametrize(
    "selection",
    [
        ValidationSelection(name="no_such_profile", version="v1"),
        ValidationSelection(name="order_decision", version="v999"),
    ],
    ids=["未注册名", "已注册名但版本未注册"],
)
def test_resolve_unregistered_raises_400_stable_code(selection):
    # 未注册 -> 400 unknown_validation_profile，message 点名坐标（spec 任务 3）。
    with pytest.raises(GatewayError) as exc_info:
        resolve_profile(selection)
    assert exc_info.value.code == "unknown_validation_profile"
    assert exc_info.value.status_code == 400
    assert selection.name in exc_info.value.message
    assert selection.version in exc_info.value.message
