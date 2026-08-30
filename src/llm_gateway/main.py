"""应用组装：FastAPI 实例与路由装配。

uvicorn 启动入口：`uv run uvicorn llm_gateway.main:app`。
title/version 与 demo 一致（对外 OpenAPI 文档属于行为面）。
"""

from fastapi import FastAPI

from llm_gateway.api.chat import router as chat_router
from llm_gateway.api.errors import register_error_handlers
from llm_gateway.api.governance import router as governance_router

app = FastAPI(title="Agent LLM Gateway", version="0.0.1")

# OpenAI 风格错误体处理器（M03 任务 4）：旧 /v1/llm 端点已随任务 5 删除，
# 全部路由无条件 OpenAI 风格（api/errors.py 模块 docstring 记载覆盖面）。
register_error_handlers(app)

app.include_router(chat_router)
app.include_router(governance_router)
