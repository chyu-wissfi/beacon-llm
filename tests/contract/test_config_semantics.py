"""M01 遗留契约测试：模型凭据未配置时的请求期语义（gateway_misconfigured）。

M02 把 catalog 常量改为配置加载产物后，api_key_env 来自 core/config.py 的
加载结果，但对外行为必须保持 demo 语义：DEEPSEEK_API_KEY 未设置时，
/v1/chat/completions 以 503 gateway_misconfigured 拒绝服务，且不产生任何上游
调用。（M01 最终审查 triage 指出该路径此前无契约测试覆盖；M03 任务 B 随
/v1/llm 删除把路径迁到统一 OpenAI 端点，错误体同步换 OpenAI 形态。）

本文件与 test_chat_api.py 相互独立、不共享 mock 细节：这里的用例在构造任何
openai client 之前就失败，因此不需要为 respx 拦截上游流量准备的 openai legacy
httpx shim（shim 由 tests/contract/conftest.py 提供，对本文件无害）。
"""

import httpx
import pytest
import pytest_asyncio
import respx

from llm_gateway.main import app

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def upstream_any():
    # 兜底路由 + "零上游调用"断言：本组用例的不变量是"缺凭据在调用上游
    # 之前失败"。assert_all_mocked=True 保证若有请求漏网在此失败而不是真发
    # 网络；assert_all_called=False 是因为该路由预期 0 次调用，"注册了没
    # 被调用"不算失败，零调用由用例自己断言。
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as mock:
        route = mock.route()
        yield route


@pytest_asyncio.fixture
async def client():
    # 用 httpx ASGI 传输直打 app，不起端口、不走网络。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
        yield c


async def test_missing_primary_api_key_returns_503_gateway_misconfigured(client, upstream_any, monkeypatch):
    # 不变量：DEEPSEEK_API_KEY 未设置时必须 503 gateway_misconfigured——配置
    # 错误不能伪装成上游故障（502 model_unavailable）。备用 key 在场也救不了：
    # GatewayError 不进 fallback，凭据缺失在构造 primary client 时就抛出，
    # 因此备用模型零调用（这正是 503 与 502 的语义分界）。
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_BACKUP_API_KEY", "test-backup-key")

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "general-primary", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "gateway_misconfigured"
    # message 来自注册表默认三元组（逐字冻结于 tests/unit/test_error_registry.py）。
    assert error["message"] == "Gateway 模型凭据未配置"
    # 5xx 类按 controller 裁决归 api_error。
    assert error["type"] == "api_error"
    # 拒绝发生在任何上游调用之前：主/备模型都没有被碰。
    assert upstream_any.call_count == 0
