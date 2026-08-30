"""M10 契约测试（不变量 #13）：日志无 API Key、默认不记消息内容。

剧本（spec 任务 5）：构造含 API key 的请求 + key 放入日志上下文，全量捕获
日志输出断言无 key 子串；断言无消息 content 字段。脱敏在
observability/logging.py 统一出口（logger 级过滤器），任何 handler 形态
（含本测试的 caplog）都只能看到已脱敏的记录。
"""

import json
import logging

import pytest

from llm_gateway.observability.logging import SCRUBBER
from tests.contract.helpers import (
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    VALID_CALLER_KEY,
    chat_request,
    completion,
)

_GATEWAY_LOGGER = "llm_gateway"


@pytest.fixture(autouse=True)
def _capture_gateway_logs(caplog):
    # 全量捕获：DEBUG 起步，网关具名 logger 的一切输出尽收囊中。
    caplog.set_level(logging.DEBUG, logger=_GATEWAY_LOGGER)
    return caplog


@pytest.mark.asyncio
async def test_request_path_logs_no_key_and_no_content(client, mock_upstream, caplog):
    # 请求本身携带 key：Authorization 头是调用方 key 本体，消息体再内嵌
    # key 与机密内容——只要日志不记消息内容且出口脱敏，两者都泄不出去。
    secret_content = f"my key is {VALID_CALLER_KEY} and this is confidential"
    payload = chat_request(messages=[{"role": "user", "content": secret_content}])
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        json=completion("ok")
    )
    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    # 至少有一条调用日志可供断言（否则断言空集无意义）。
    trace_records = [record for record in caplog.records if "llm_call_trace=" in record.getMessage()]
    assert trace_records
    # 全量捕获面：无 key 子串、无消息内容子串。
    assert VALID_CALLER_KEY not in caplog.text
    assert "confidential" not in caplog.text
    for record in trace_records:
        trace_json = json.loads(record.getMessage().split("=", 1)[1])
        # 无消息 content 字段（也没有 messages 面）。
        assert "content" not in trace_json
        assert "messages" not in trace_json
        # spec 任务 3 要求的字段面在场：request_id / caller / status / attempts。
        assert {"request_id", "caller", "status", "attempts"} <= set(trace_json)


def test_key_in_log_context_is_scrubbed(caplog):
    # key 放入日志上下文（消息参数与结构化字段两条路）：出口过滤器就地改写，
    # caplog 捕到的已是脱敏后的记录——统一出口不依赖 handler 装配位置。
    secret = "sk-context-secret-0123456789"
    logger = logging.getLogger(_GATEWAY_LOGGER)
    logger.info("upstream configured with %s", secret)
    logger.info(
        "forwarded request",
        extra={"authorization": f"Bearer {secret}", "request_id": "r-scrub-1"},
    )
    assert secret not in caplog.text
    assert "upstream configured with ***" in caplog.text
    structured = getattr(caplog.records[-1], "structured", {})
    # authorization 整键抹除；非敏感的结构化字段保留。
    assert "authorization" not in structured
    assert structured["request_id"] == "r-scrub-1"


def test_provider_env_key_registered_and_scrubbed(monkeypatch, caplog):
    # 供应商 key（环境变量）在已知密钥登记面：重新登记后日志出现即脱敏。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-primary-key")
    SCRUBBER.register_known_keys()
    logging.getLogger(_GATEWAY_LOGGER).info("client built with %s", "test-primary-key")
    assert "test-primary-key" not in caplog.text
    assert "client built with ***" in caplog.text
