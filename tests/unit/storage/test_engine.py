"""引擎生命周期单测（M09 任务 1）。

惰性初始化、建表幂等、单例复位、文件库目录自建；内存库的共享连接形态由
"写后可读"类用例隐式覆盖（默认池会让内存库各连接各持一份空库）。
"""

import pytest
from sqlalchemy import inspect, text

from llm_gateway.storage.engine import (
    MEMORY_DB_URL,
    configure_engine,
    dispose_engine,
    get_engine,
)

pytestmark = pytest.mark.asyncio


async def test_get_engine_creates_traces_table():
    engine = await get_engine()

    def _tables(conn) -> list[str]:
        return inspect(conn).get_table_names()

    async with engine.connect() as conn:
        tables = await conn.run_sync(_tables)
    assert "traces" in tables


async def test_get_engine_is_idempotent_singleton():
    # 惰性单例：重复取到同一引擎，建表不重复执行也不报错。
    first = await get_engine()
    second = await get_engine()
    assert first is second


async def test_configure_engine_resets_singleton():
    first = await get_engine()
    await dispose_engine()
    configure_engine(MEMORY_DB_URL)
    second = await get_engine()
    assert second is not first


async def test_dispose_without_engine_is_noop():
    # 无引擎时拆除不报错：夹具的防御性语义（有的用例从不建引擎）。
    await dispose_engine()
    await dispose_engine()


async def test_file_engine_creates_parent_directory(tmp_path):
    # 文件库目录不存在时自建（sqlite 不替我们 mkdir）。
    url = f"sqlite+aiosqlite:///{tmp_path}/nested/traces.db"
    configure_engine(url)
    try:
        engine = await get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        assert (tmp_path / "nested" / "traces.db").exists()
    finally:
        await dispose_engine()
