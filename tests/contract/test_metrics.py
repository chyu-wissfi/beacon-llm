"""M10 契约测试：/healthz、/metrics 文本协议与机制计数器的精确增量。

验收哲学（design.md §7）：剧本触发重试/限流后对应计数器**精确** +N——
上游请求计数与计数器互为对账面。本文件全部计数器用例名含 "counters"，
spec 验收命令 `-k counters` 精确命中。
"""

import httpx
import pytest
from prometheus_client import REGISTRY

from llm_gateway.core import ratelimit
from llm_gateway.main import app
from tests.contract.helpers import (
    BACKUP_PROVIDER_MODEL,
    BACKUP_URL,
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    SSE_HEADERS,
    chat_request,
    collect_sse_events,
    completion,
    sse_body,
)

pytestmark = pytest.mark.asyncio

# 上游默认用量（helpers.completion / sse usage 块的口径）：对账用常量。
_UPSTREAM_INPUT_TOKENS = 13
_UPSTREAM_OUTPUT_TOKENS = 5


def _sample(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


async def test_healthz_returns_ok_and_version(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": app.version}


async def test_metrics_exposes_prometheus_text_protocol(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    # 文本协议：指标定义（HELP/TYPE）与样本面都在。
    assert "# HELP llm_requests_total" in response.text
    assert "# TYPE llm_request_latency_seconds histogram" in response.text
    assert "llm_requests_in_flight" in response.text


async def test_retry_counters_exact_increment(client, mock_upstream):
    # 剧本：primary 持续连接失败、backup 接盘成功。动态留量下 primary 恰好
    # 尝试 3 次（预算 4 - 后续候选 1），每次失败后的再试各计一次重试——
    # 计数器与上游请求计数互为对账：3 次尝试 = 3 次重试（首试不算重试）。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        json=completion("ok")
    )
    response = await client.post("/v1/chat/completions", json=chat_request())
    assert response.status_code == 200
    assert primary.call_count == 3
    assert backup.call_count == 1
    assert _sample("llm_retries_total", model="general-primary") == 3
    # backup 一次成功：没有重试。
    assert _sample("llm_retries_total", model="general-backup") is None
    # 终态恰好一次计数（请求模型维度）；在途归零。
    assert _sample("llm_requests_total", model="general-primary", status="success") == 1
    assert _sample("llm_requests_in_flight") == 0


async def test_tokens_counters_booked_to_actual_model(client, mock_upstream):
    # tokens 只在实际服务到模型时入账，且记在实际服务模型上（fallback 后
    # 记 backup 而非请求模型）——与 trace 的 actual_model 语义同源。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down")
    )
    mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        json=completion("ok")
    )
    response = await client.post("/v1/chat/completions", json=chat_request())
    assert response.status_code == 200
    assert _sample("llm_tokens_total", model="general-backup", direction="input") == _UPSTREAM_INPUT_TOKENS
    assert _sample("llm_tokens_total", model="general-backup", direction="output") == _UPSTREAM_OUTPUT_TOKENS
    assert _sample("llm_tokens_total", model="general-primary", direction="input") is None


async def test_failure_terminal_counters(client, mock_upstream):
    # 主备全灭：failed 终态恰好一次计数。重试口径对账：primary 3 次尝试
    # = 3 次再试；backup 是链尾候选吃满剩余预算（2 次尝试），第二次尝试前
    # 同样计一次重试——预算耗尽发生在再试判定之后（先计后退避）。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down")
    )
    mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("backup down")
    )
    response = await client.post("/v1/chat/completions", json=chat_request())
    assert response.status_code == 502
    assert _sample("llm_requests_total", model="general-primary", status="failed") == 1
    assert _sample("llm_requests_total", model="general-primary", status="success") is None
    assert _sample("llm_retries_total", model="general-primary") == 3
    assert _sample("llm_retries_total", model="general-backup") == 1
    # 未服务到任何模型：tokens 无样本（缺口不伪造，与 trace 同款语义）。
    assert _sample("llm_tokens_total", model="general-primary", direction="input") is None
    assert _sample("llm_tokens_total", model="general-backup", direction="input") is None
    assert _sample("llm_requests_in_flight") == 0


async def test_rate_limited_counters_exact_increment(client, mock_upstream):
    # 剧本：RPM 桶抽空后下一次请求被拒——限流计数精确 +1，且请求止步于
    # 准入（assert_all_mocked 保证任何上游请求都会让测试失败；终态/在途
    # 零波及证明拒绝发生在编排之前）。
    bucket = ratelimit.ADMISSION._bucket("general-primary", 60)  # noqa: SLF001  # 测试卫生面直触
    bucket.tokens = 0.0
    response = await client.post("/v1/chat/completions", json=chat_request())
    assert response.status_code == 429
    assert _sample("llm_rate_limited_total", model="general-primary") == 1
    # 准入拒绝不进编排：终态与在途都没有样本。
    assert _sample("llm_requests_total", model="general-primary", status="failed") is None
    assert _sample("llm_requests_in_flight") == 0


async def test_breaker_state_counters_one_hot(client, mock_upstream):
    # 剧本：主模型累计 5 次传输故障（第一请求 3 次 + 第二请求前 2 次）触发
    # 熔断打开——状态独热恰好 open=1；backup 的成功不影响 primary 的状态。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down")
    )
    mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        json=completion("ok")
    )
    for _ in range(2):
        response = await client.post("/v1/chat/completions", json=chat_request())
        assert response.status_code == 200  # backup 每次都接盘
    assert _sample("llm_breaker_state", model="general-primary", state="open") == 1
    assert _sample("llm_breaker_state", model="general-primary", state="closed") == 0
    assert _sample("llm_breaker_state", model="general-primary", state="half_open") == 0


async def test_stream_counters_exact_increment(client, mock_upstream):
    # 流式与非流式同款口径：成功终态 +1、在途归零、usage 回传则 tokens 入账。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        content=sse_body(["你好"], include_usage=True), headers=SSE_HEADERS
    )
    events = await collect_sse_events(
        client, "/v1/chat/completions", chat_request(stream=True, stream_options={"include_usage": True})
    )
    assert events[-1] == "[DONE]"
    assert _sample("llm_requests_total", model="general-primary", status="success") == 1
    assert _sample("llm_tokens_total", model="general-primary", direction="input") == _UPSTREAM_INPUT_TOKENS
    assert _sample("llm_tokens_total", model="general-primary", direction="output") == _UPSTREAM_OUTPUT_TOKENS
    assert _sample("llm_requests_in_flight") == 0
