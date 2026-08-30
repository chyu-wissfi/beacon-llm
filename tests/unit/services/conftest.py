"""services 单测共享夹具（M09 起）。

编排层终态会调度 trace 落库：每用例注入全新内存库（事件循环 scope 是
function，aiosqlite 连接绑定循环，引擎必须每用例建/拆），避免写进生产
默认库 data/traces.db；teardown 对账掉未完成的写任务（消除悬空任务噪声）。
"""

import pytest_asyncio

from llm_gateway.services.trace_service import flush_pending
from llm_gateway.storage.engine import MEMORY_DB_URL, configure_engine, dispose_engine


@pytest_asyncio.fixture(autouse=True)
async def _trace_db():
    configure_engine(MEMORY_DB_URL)
    yield
    await flush_pending()
    await dispose_engine()
