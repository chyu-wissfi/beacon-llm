"""契约测试共享夹具（tests/contract/ 全目录生效）。

从原 test_demo_semantics.py 提取：M03 任务 B 起 /v1/chat/completions 的契约
测试与既有用例共享同一套离线夹具；用例断言在各测试文件，请求/上游响应构造
器在 helpers.py。openai 3.x 的 legacy httpx shim 已提取到 tests/support/
（M04 起 unit/providers 复用），本文件只保留契约测试专属夹具。
"""

import httpx
import pytest
import pytest_asyncio
import respx

from llm_gateway.core import ratelimit
from llm_gateway.core.breaker import reset_breakers
from llm_gateway.main import app
from llm_gateway.services.trace_service import CALL_TRACES
from tests.contract.helpers import AUTH_HEADERS
from tests.support.httpx_shim import openai_legacy_httpx  # noqa: F401

__all__ = ["openai_legacy_httpx"]


@pytest.fixture(autouse=True)
def _openai_legacy_httpx_auto(request, openai_legacy_httpx):
    # 契约测试全目录自动启用 shim（autouse 薄壳包住共享 fixture，保持原
    # _openai_legacy_httpx 名称的 autouse 语义；单元测试目录自行显式引用）。
    return openai_legacy_httpx


@pytest.fixture(autouse=True)
def _gateway_env(monkeypatch):
    # create_client 在每次调用时读环境变量，缺 key 会 503 gateway_misconfigured；
    # 契约测试只关心网关行为，用假 key 即可（respx 拦截，不会真发请求）。
    # 需要缺凭据场景的用例（gateway_misconfigured）自己 monkeypatch.delenv 覆盖。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-primary-key")
    monkeypatch.setenv("DEEPSEEK_BACKUP_API_KEY", "test-backup-key")


@pytest.fixture(autouse=True)
def _clean_traces():
    # CALL_TRACES 是网关的模块级全局 list，测试间必须清空，避免相互污染导致
    # "/v1/traces 返回记录"之类的断言受其他用例残留影响。
    CALL_TRACES.clear()
    yield
    CALL_TRACES.clear()


@pytest.fixture(autouse=True)
def _reset_admission_state():
    # 准入/熔断是进程内全局状态（不变量 #17）：用例间的令牌消耗、并发计数、
    # 熔断失败数都必须清零，否则限流类断言会被前序用例的残留击穿。
    ratelimit.reset_admission()
    reset_breakers()
    yield
    ratelimit.reset_admission()
    reset_breakers()


@pytest.fixture
def mock_upstream():
    # assert_all_mocked=True：任何未注册的上游请求直接让测试失败（离线保证）。
    # assert_all_called=False：注册了但未被调用的路由不算失败——"备用模型没被
    # 调用"这类不变量由用例自己断言 call_count == 0，语义更明确。
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as mock:
        yield mock


@pytest_asyncio.fixture
async def client():
    # 用 httpx ASGI 传输直打 app，不起端口、不走网络。默认携带合法调用方头
    # （M05 起 /v1/chat/completions 鉴权）；401 用例按请求覆盖该头。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway.test", headers=AUTH_HEADERS
    ) as c:
        yield c


@pytest_asyncio.fixture
async def client_lenient():
    # 同 client，但 raise_app_exceptions=False：未捕获异常兜底（500 处理器）先
    # 发响应再由 Starlette 重新抛出原异常，严格传输会把异常直接抛进测试；宽松
    # 传输只用于断言"兜底响应体形态"的用例，其余用例保持严格以暴露真实崩溃。
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway.test", headers=AUTH_HEADERS
    ) as c:
        yield c
