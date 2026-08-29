"""应用组装：FastAPI 实例与路由装配。

uvicorn 启动入口：`uv run uvicorn llm_gateway.main:app`。
title/version 与 demo 一致（对外 OpenAPI 文档属于行为面）。
"""

from fastapi import FastAPI

from llm_gateway.api.governance import router as governance_router
from llm_gateway.api.llm import router as llm_router

app = FastAPI(title="Agent LLM Gateway", version="0.0.1")

app.include_router(llm_router)
app.include_router(governance_router)
