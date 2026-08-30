"""observability/logging.py 单测：脱敏过滤器与 JSON 出口形态。

端到端（请求全链路日志无 key、无 content）在契约层
（tests/contract/test_log_scrubbing.py）；本文件直接驱动过滤器与
格式化器，钉住不变量 #13 的机制细节。
"""

import json
import logging

import pytest

from llm_gateway.observability.logging import (
    SCRUBBER,
    JsonFormatter,
    ScrubbingFilter,
    setup_logging,
)


def _record(msg: str, *args: object, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("llm_gateway", logging.INFO, __file__, 1, msg, args, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


@pytest.fixture
def fresh_filter() -> ScrubbingFilter:
    # 独立实例：避免向模块级 SCRUBBER 登记测试专用密钥造成跨用例残留。
    return ScrubbingFilter()


def test_registered_key_replaced_in_message(fresh_filter: ScrubbingFilter) -> None:
    fresh_filter.register("sk-secret-value-123")
    record = _record("upstream called with key %s", "sk-secret-value-123")
    assert fresh_filter.filter(record) is True
    assert record.getMessage() == "upstream called with key ***"
    # args 清空：格式化重放不会拼回原值。
    assert record.args == ()


def test_sk_pattern_caught_without_registration(fresh_filter: ScrubbingFilter) -> None:
    # 未登记的 sk- 形态令牌由正则兜底拦住。
    record = _record("header=Bearer sk-unknown-token-987654321")
    fresh_filter.filter(record)
    assert "sk-unknown-token-987654321" not in record.getMessage()
    assert "***" in record.getMessage()


def test_short_value_not_registered() -> None:
    # 过短字面不登记：防止把正常文本替换成筛子。
    local = ScrubbingFilter()
    local.register("ab")
    record = _record("ab normal text about ab")
    local.filter(record)
    assert record.getMessage() == "ab normal text about ab"


def test_sensitive_field_names_dropped_from_extra(fresh_filter: ScrubbingFilter) -> None:
    # 消息内容面（content/messages）与凭据面（api_key/authorization）整键抹除，
    # 其余结构化字段保留——"默认不记消息内容"是出口强制而非约定。
    record = _record(
        "request received",
        request_id="r-1",
        caller="ops",
        content="机密提示词正文",
        messages=[{"role": "user", "content": "机密"}],
        api_key="sk-whatever-000",
        authorization="Bearer x",
        status="success",
    )
    fresh_filter.filter(record)
    structured = record.structured  # type: ignore[attr-defined]
    assert set(structured) == {"request_id", "caller", "status"}


def test_key_value_inside_extra_string_scrubbed(fresh_filter: ScrubbingFilter) -> None:
    fresh_filter.register("sk-extra-secret-456")
    record = _record("forwarded", note="upstream key is sk-extra-secret-456")
    fresh_filter.filter(record)
    assert record.structured["note"] == "upstream key is ***"  # type: ignore[attr-defined]


def test_nested_sensitive_keys_stripped_recursively(fresh_filter: ScrubbingFilter) -> None:
    record = _record(
        "debug",
        payload={"outer": {"content": "秘密", "keep": 1}, "messages": ["a"]},
    )
    fresh_filter.filter(record)
    structured = record.structured  # type: ignore[attr-defined]
    assert structured == {"payload": {"outer": {"keep": 1}}}


def test_plain_message_unchanged_and_kept(fresh_filter: ScrubbingFilter) -> None:
    record = _record("trace %s persisted", "r-2")
    assert fresh_filter.filter(record) is True
    assert record.getMessage() == "trace r-2 persisted"
    assert record.args == ("r-2",)
    assert not hasattr(record, "structured")


def test_json_formatter_shape() -> None:
    formatter = JsonFormatter()
    record = _record("hello %s", "world", request_id="r-9")
    SCRUBBER.filter(record)
    payload = json.loads(formatter.format(record))
    assert payload["logger"] == "llm_gateway"
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "r-9"
    assert "ts" in payload


def test_json_formatter_includes_exception() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "llm_gateway", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
        )
    payload = json.loads(formatter.format(record))
    assert "ValueError: boom" in payload["exception"]


def test_setup_logging_idempotent() -> None:
    # 重复装配不叠加 handler：多轮导入（测试/重载）不产生重复输出。
    logger = logging.getLogger("llm_gateway")
    setup_logging()
    first = len(logger.handlers)
    setup_logging()
    setup_logging()
    assert len(logger.handlers) == first
    # 过滤器挂在 logger 级：统一出口不依赖 handler 装配位置。
    assert any(isinstance(f, ScrubbingFilter) for f in logger.filters)
