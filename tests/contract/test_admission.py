"""准入控制契约测试（M05 任务 6）：认证/限流/并发/熔断的 HTTP 面。

断言面（与 spec 验收逐条对应）：
- 401：缺头/错键 -> unauthorized，且上游请求数为 0（准入拒绝零波及）；
- 429 三态 + Retry-After 头：rate_limited（RPM）、token_budget_exceeded（TPM
  事后记账超额）、overloaded / provider_overloaded（并发，见下方用例）；
- 503：熔断打开 -> circuit_open，上游请求数为 0；
- `-k global_concurrency`：25 个慢请求（Fake Adapter 慢成功），断言到达上游
  （FakeAdapter.attempts 即上游可观测面）≤ 20（全局并发上限），其余 429。

限流/熔断是进程内状态（不变量 #17），conftest 的 autouse reset 夹具保证
用例间零残留；本文件只改被断言的模型条目（monkeypatch），不碰其他模型。
"""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
import pytest_asyncio

from llm_gateway.core import ratelimit
from llm_gateway.core.breaker import FAILURE_THRESHOLD, get_breaker
from llm_gateway.core.schemas import ModelConfig, RateLimitConfig
from llm_gateway.main import app
from llm_gateway.providers import PROVIDER_REGISTRY
from llm_gateway.providers.fake import FakeAdapter, SlowSuccess
from llm_gateway.services.catalog import MODEL_CONFIGS, PRICE_PER_MILLION
from tests.contract.helpers import (
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    chat_request,
    completion,
)

pytestmark = pytest.mark.asyncio

CHAT_PATH = "/v1/chat/completions"


@pytest_asyncio.fixture
async def client_no_auth():
    # 无默认认证头的裸客户端：401 场景专用（conftest 的 client 默认带合法头）。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
        yield c


def _error_code(response: httpx.Response) -> str:
    # OpenAI 风格错误体的稳定码位置（api/errors.py 的对外形态）。
    return response.json()["error"]["code"]


def _register_slow_fake(monkeypatch, delay_seconds: float = 0.3) -> FakeAdapter:
    # 注册"慢成功"的 fake 模型：在途占用准入资源，撑开并发断言面。
    # attempts 计数即"到达上游"的可观测面（fake 不经传输层）。
    slow = FakeAdapter(scenario=SlowSuccess(delay_seconds=delay_seconds))
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake", slow)
    monkeypatch.setitem(
        MODEL_CONFIGS,
        "slow-fake",
        ModelConfig(
            provider_model="slow",
            base_url="http://fake.test",
            api_key_env="FAKE_API_KEY",
            supports_structured_output=False,
            provider="fake",
        ),
    )
    # trace 记账按模型查价：补一条零价条目，避免成功路径 KeyError。
    monkeypatch.setitem(PRICE_PER_MILLION, "slow-fake", {"input": 0.0, "output": 0.0})
    return slow


# ---------------------------------------------------------------------------
# 认证（任务 1）：401 + 上游零波及
# ---------------------------------------------------------------------------


async def test_missing_authorization_is_401_and_upstream_untouched(client_no_auth, mock_upstream):
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )
    response = await client_no_auth.post(CHAT_PATH, json=chat_request())
    assert response.status_code == 401
    assert _error_code(response) == "unauthorized"
    # 准入拒绝时上游请求数为 0（任务 6 组合断言）。
    assert route.call_count == 0


async def test_wrong_key_is_401_unauthorized(client, mock_upstream):
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )
    response = await client.post(
        CHAT_PATH, json=chat_request(), headers={"Authorization": "Bearer sk-not-a-real-key"}
    )
    assert response.status_code == 401
    assert _error_code(response) == "unauthorized"
    assert route.call_count == 0


# ---------------------------------------------------------------------------
# RPM / TPM（任务 2+3）：429 + Retry-After + 上游零波及
# ---------------------------------------------------------------------------


async def test_rpm_exceeded_is_429_rate_limited_with_retry_after(client, mock_upstream, monkeypatch):
    # 令牌桶容量压到 1：首个请求吃掉唯一令牌，第二个请求准入期即拒。
    monkeypatch.setitem(
        MODEL_CONFIGS,
        "general-primary",
        replace(MODEL_CONFIGS["general-primary"], rate_limit=RateLimitConfig(rpm=1, tpm=None)),
    )
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )
    first = await client.post(CHAT_PATH, json=chat_request())
    assert first.status_code == 200
    second = await client.post(CHAT_PATH, json=chat_request())
    assert second.status_code == 429
    assert _error_code(second) == "rate_limited"
    assert int(second.headers["Retry-After"]) >= 1
    # 被拒请求没到上游：全程恰好 1 次上游调用。
    assert route.call_count == 1


async def test_tpm_over_budget_is_429_token_budget_exceeded(client, mock_upstream, monkeypatch):
    # TPM 事后记账：首请求成功后按实际 usage（13+5=18）入账，超过 10 的预算，
    # 第二个请求在准入期被拒——正是"先放行后记账"的滞后语义。
    monkeypatch.setitem(
        MODEL_CONFIGS,
        "general-primary",
        replace(MODEL_CONFIGS["general-primary"], rate_limit=RateLimitConfig(rpm=60, tpm=10)),
    )
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )
    first = await client.post(CHAT_PATH, json=chat_request())
    assert first.status_code == 200
    second = await client.post(CHAT_PATH, json=chat_request())
    assert second.status_code == 429
    assert _error_code(second) == "token_budget_exceeded"
    assert int(second.headers["Retry-After"]) >= 1
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# 熔断（任务 4）：503 + 上游零波及
# ---------------------------------------------------------------------------


async def test_open_circuit_is_503_circuit_open_and_upstream_untouched(client, mock_upstream):
    # 直接记满阈值失败打开熔断（失败口径的单测面在 tests/unit/core/），
    # 此处只断言 HTTP 拒绝形态与零上游波及。
    breaker = get_breaker("general-primary")
    for _ in range(FAILURE_THRESHOLD):
        breaker.record_failure()
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )
    response = await client.post(CHAT_PATH, json=chat_request())
    assert response.status_code == 503
    assert _error_code(response) == "circuit_open"
    assert route.call_count == 0


# ---------------------------------------------------------------------------
# 并发（任务 2）：全局上限 20 / 供应商上限 / 流式全程占位
# ---------------------------------------------------------------------------


async def test_global_concurrency_caps_in_flight_requests(client, monkeypatch):
    # spec 验收：异步发 25 个慢请求（Fake Adapter 慢成功），断言 ≤20 个到达上游。
    # 供应商闸放宽到 100，让全局 20 成为唯一紧约束。
    slow = _register_slow_fake(monkeypatch)
    monkeypatch.setattr(ratelimit.ADMISSION, "provider_concurrency", 100)
    responses = await asyncio.gather(
        *[client.post(CHAT_PATH, json=chat_request(model="slow-fake")) for _ in range(25)]
    )
    admitted = [r for r in responses if r.status_code == 200]
    rejected = [r for r in responses if r.status_code == 429]
    assert len(admitted) + len(rejected) == 25
    assert slow.attempts <= 20  # 到达上游（含在途）不超过全局并发上限
    assert slow.attempts == len(admitted)
    for response in rejected:
        assert _error_code(response) == "overloaded"
        assert int(response.headers["Retry-After"]) >= 1


async def test_provider_concurrency_excess_is_429_provider_overloaded(client, monkeypatch):
    # 供应商闸压到 1：两个并发慢请求只放行 1 个（全局闸不是紧约束）。
    slow = _register_slow_fake(monkeypatch, delay_seconds=0.2)
    monkeypatch.setattr(ratelimit.ADMISSION, "provider_concurrency", 1)
    responses = await asyncio.gather(
        *[client.post(CHAT_PATH, json=chat_request(model="slow-fake")) for _ in range(2)]
    )
    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 429]
    rejected = next(response for response in responses if response.status_code == 429)
    assert _error_code(rejected) == "provider_overloaded"
    assert slow.attempts == 1


async def test_stream_holds_global_concurrency_until_stream_end(client, monkeypatch):
    # 流式全程占位（任务 2）：21 个并发慢流式请求在全局 20 的上限下必有 1 个
    # 被拒——若并发位在端点返回后即释放，第 21 个会在首批流结束前被放行。
    slow = _register_slow_fake(monkeypatch)
    monkeypatch.setattr(ratelimit.ADMISSION, "provider_concurrency", 100)

    async def _one() -> tuple[int, bytes]:
        async with client.stream("POST", CHAT_PATH, json=chat_request(model="slow-fake", stream=True)) as response:
            body = await response.aread()
            return response.status_code, body

    results = await asyncio.gather(*[_one() for _ in range(21)])
    statuses = [status for status, _ in results]
    assert statuses.count(200) <= 20
    assert statuses.count(429) == 21 - statuses.count(200)
    assert slow.attempts == statuses.count(200)
    for status, body in results:
        if status == 429:
            assert json.loads(body)["error"]["code"] == "overloaded"
        else:
            # 成功流以 [DONE] 收尾（流内终态语义不在本用例展开，见 test_chat_api）。
            assert body.decode().rstrip().endswith("data: [DONE]")
