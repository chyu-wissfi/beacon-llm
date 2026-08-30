"""异步引擎生命周期（M09 任务 1）。

默认库按 spec 字面：`sqlite+aiosqlite:///data/traces.db`（相对仓库根；
CLAUDE.md 的运行约定是从仓库根启动）。引擎**惰性**初始化而非导入期创建：
建库/建表是第一次读写 trace 时才需要的副作用，导入期做会把"测试没注入
测试库"的错误升级成在仓库根落一个真 traces.db 的静默事故。

测试注入面：`configure_engine(url)` 换库并复位单例（下一次 get 重建），
`dispose_engine()` 拆连接池。两者配合顶层 tests/conftest.py 的 autouse
夹具——pytest 的 event loop scope 是 function，aiosqlite 连接绑定循环，
引擎必须每测试建/拆，跨用例复用必炸。

`:memory:` 库用 StaticPool：SQLite 内存库按连接私有，默认池的多连接会各
持一份空库，读写"消失"；单连接共享是唯一正确形态。
"""

import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from llm_gateway.storage.models import Base

# spec 任务 1 的字面 URL。
DEFAULT_DB_URL = "sqlite+aiosqlite:///data/traces.db"
# 测试注入面：每用例一份全新内存库（夹具见 tests/ 各 conftest）。
MEMORY_DB_URL = "sqlite+aiosqlite:///:memory:"

_engine: AsyncEngine | None = None
_db_url: str = DEFAULT_DB_URL
# 初始化锁：同一时刻多条 trace 并发首次落库时，建引擎+建表只能发生一次；
# 未建完就放第二个写者进去会撞 "no such table"（M09 验收踩中）。
# asyncio.Lock 首次 acquire 才绑定事件循环；测试每用例一个新循环（且换库），
# configure_engine 复位时一并重建锁。
_init_lock = asyncio.Lock()


def configure_engine(url: str) -> None:
    # 换库并复位单例：下一次 get_engine 按新 URL 重建。测试夹具专用面——
    # 生产路径不调用（吃默认 URL 的惰性初始化）。
    global _engine, _db_url, _init_lock
    _engine = None
    _db_url = url
    # 锁随循环重建：旧锁可能已绑定上一个事件循环（首次 acquire 时绑定）。
    _init_lock = asyncio.Lock()


async def get_engine() -> AsyncEngine:
    # 惰性单例：首次调用建引擎 + 建表（幂等）。建表完成前不把引擎暴露给
    # 第二个调用者（锁内双检），避免并发首写撞缺失表。
    global _engine
    if _engine is not None:
        return _engine
    async with _init_lock:
        if _engine is None:
            engine = _build_engine(_db_url)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            _engine = engine
    return _engine


async def dispose_engine() -> None:
    # 拆除引擎（连接池一并释放）：测试夹具的 teardown 面。无引擎时 no-op。
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


def _build_engine(url: str) -> AsyncEngine:
    if url == MEMORY_DB_URL:
        return create_async_engine(url, poolclass=StaticPool)
    if url.startswith("sqlite+aiosqlite:///"):
        # 文件库：先保证目录存在（sqlite 不会替我们建目录）；根目录下的
        # 裸文件名（parent 为空路径）无需建目录。
        db_path = Path(url.removeprefix("sqlite+aiosqlite:///"))
        if db_path.parent != Path(""):
            db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_async_engine(url)
