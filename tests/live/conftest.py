"""live 冒烟测试共享夹具（tests/live/ 全目录生效）。

与契约层的分工：契约层用 Fake Adapter / respx 在 CI 内确定性证明行为边界；
本层打真实上游（DeepSeek），证明六大功能在真模型路径上端到端成立（M12 任务 4）。
离线闸门 `make check` 以 `-m "not live"` 排除本目录；凭据缺失时整目录跳过
（而不是失败）——本地 `make test-live` 是唯一消费入口。

卫生面与契约层同款约定（进程内全局状态的用例隔离）：准入/熔断/指标复位、
trace 内存库逐用例建拆（不写生产库 data/traces.db）、CALL_TRACES 清零。
"""

import httpx
import pytest
import pytest_asyncio

from llm_gateway.core import ratelimit
from llm_gateway.core.breaker import reset_breakers
from llm_gateway.main import app
from llm_gateway.observability.metrics import reset_metrics
from llm_gateway.services.trace_service import CALL_TRACES, flush_pending
from llm_gateway.storage.engine import MEMORY_DB_URL, configure_engine, dispose_engine
from tests.contract.helpers import AUTH_HEADERS


@pytest.fixture(autouse=True)
def _reset_process_state():
    # 准入/熔断/指标/内存 trace 全是进程内全局状态：用例间零残留，
    # 限流与指标计数断言才不会被前序用例击穿。
    ratelimit.reset_admission()
    reset_breakers()
    reset_metrics()
    CALL_TRACES.clear()
    yield
    CALL_TRACES.clear()


@pytest_asyncio.fixture(autouse=True)
async def _trace_db():
    # 每用例全新内存库（与契约层同款：事件循环 scope 是 function，
    # aiosqlite 连接绑定循环，引擎必须每用例建/拆）。
    configure_engine(MEMORY_DB_URL)
    yield
    await flush_pending()
    await dispose_engine()


@pytest_asyncio.fixture
async def client():
    # httpx ASGI 传输直打 app：网关进程内的 openai 客户端照常走真实网络
    # 打上游——"本地过 = 真链路过"，只是不起端口。真模型调用耗时不可控，
    # 传输超时放宽到 120s（httpx 默认 5s 会切断正常调用）。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gateway.test",
        headers=AUTH_HEADERS,
        timeout=httpx.Timeout(120.0),
    ) as c:
        yield c
