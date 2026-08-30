"""契约测试共享夹具（tests/contract/ 全目录生效）。

从原 test_demo_semantics.py 提取：M03 任务 B 起 /v1/chat/completions 的契约
测试与既有用例共享同一套离线夹具；用例断言在各测试文件，请求/上游响应构造
器在 helpers.py。

openai 3.6.0 兼容 shim：让 respx 能拦截上游流量。openai 3.x 默认用
httpx2.AsyncClient 发请求，而 respx 0.23.1 只 patch httpx/httpcore 层，对
httpx2 流量完全不可见——mock 会静默漏到真实网络（已实测 DeepSeek 返回 401）。
openai 3.x 官方支持运行 legacy httpx client（http_client 参数的 is_legacy_*
分支），因此这里把 openai 的默认 client 工厂替换为 legacy httpx1 版本，使
respx 恢复拦截。该 shim 只动 openai 库自己的命名空间，不碰网关代码；升级
openai 或 respx 时需复核此 shim 是否仍必要/仍有效。
"""

from typing import Any, cast

import httpx
import pytest
import pytest_asyncio
import respx

from llm_gateway.main import app
from llm_gateway.services.trace_service import CALL_TRACES


@pytest.fixture(autouse=True)
def _openai_legacy_httpx(monkeypatch):
    # 只透传 base_url：客户端级 timeout 对离线 mock 无意义，且 openai 传入的
    # httpx2.Timeout 对象与 httpx1 不兼容，直接忽略（网关的每请求超时由
    # openai 的 legacy 归一化逻辑另行处理，与本工厂无关）。
    def _legacy_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
        # cast 是运行期恒等：仅满足 httpx base_url: URLTypes 的标注。openai 构造
        # AsyncHttpxClientWrapper 时恒传 base_url（openai/_base_client.py），不会缺键。
        return httpx.AsyncClient(
            base_url=cast("httpx.URL | str", kwargs.get("base_url")),
        )

    monkeypatch.setattr("openai._base_client.AsyncHttpxClientWrapper", _legacy_httpx_client)


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


@pytest.fixture
def mock_upstream():
    # assert_all_mocked=True：任何未注册的上游请求直接让测试失败（离线保证）。
    # assert_all_called=False：注册了但未被调用的路由不算失败——"备用模型没被
    # 调用"这类不变量由用例自己断言 call_count == 0，语义更明确。
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as mock:
        yield mock


@pytest_asyncio.fixture
async def client():
    # 用 httpx ASGI 传输直打 app，不起端口、不走网络。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
        yield c


@pytest_asyncio.fixture
async def client_lenient():
    # 同 client，但 raise_app_exceptions=False：未捕获异常兜底（500 处理器）先
    # 发响应再由 Starlette 重新抛出原异常，严格传输会把异常直接抛进测试；宽松
    # 传输只用于断言"兜底响应体形态"的用例，其余用例保持严格以暴露真实崩溃。
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
        yield c
