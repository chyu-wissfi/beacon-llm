"""core/auth.py 单测（M05 任务 1）。

断言面：合法键返回对应调用方；一切非法形态（缺头/非 Bearer/空值/键不匹配/
空表）归一为 401 unauthorized——错误形态不区分失败原因（防探测）。
"""

import pytest

from llm_gateway.core.auth import authenticate, parse_bearer
from llm_gateway.core.config import CallerConfig
from llm_gateway.core.errors import GatewayError

_CALLERS = {
    "sk-alpha": CallerConfig(display_name="Alpha"),
    "sk-beta": CallerConfig(display_name="Beta"),
}


def test_valid_key_returns_matching_caller() -> None:
    assert authenticate("Bearer sk-beta", _CALLERS).display_name == "Beta"
    assert authenticate("Bearer sk-alpha", _CALLERS).display_name == "Alpha"


def test_bearer_scheme_is_case_insensitive() -> None:
    assert authenticate("bearer sk-alpha", _CALLERS).display_name == "Alpha"
    assert authenticate("BEARER sk-alpha", _CALLERS).display_name == "Alpha"


def test_token_surrounding_whitespace_is_stripped() -> None:
    assert parse_bearer("Bearer   sk-alpha  ") == "sk-alpha"


@pytest.mark.parametrize(
    "header",
    [
        None,  # 缺头
        "",  # 空值
        "Bearer",  # 缺 token
        "Bearer    ",  # token 全空白
        "Basic sk-alpha",  # 非 Bearer 方案
        "sk-alpha",  # 裸 token（无方案）
        "Bearer sk-wrong",  # 键不匹配
        "Bearer sk-alph",  # 前缀命中整键不命中（防前缀匹配漏洞）
    ],
)
def test_invalid_header_maps_to_unauthorized(header: str | None) -> None:
    with pytest.raises(GatewayError) as exc_info:
        authenticate(header, _CALLERS)
    assert exc_info.value.code == "unauthorized"
    assert exc_info.value.status_code == 401


def test_empty_callers_table_rejects_anything() -> None:
    with pytest.raises(GatewayError) as exc_info:
        authenticate("Bearer sk-alpha", {})
    assert exc_info.value.code == "unauthorized"
