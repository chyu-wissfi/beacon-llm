"""应用组装：FastAPI 实例与路由装配。

uvicorn 启动入口：`uv run uvicorn llm_gateway.main:app`。
title/version 与 demo 一致（对外 OpenAPI 文档属于行为面）。
"""

from fastapi import FastAPI

from llm_gateway.api.errors import register_error_handlers
from llm_gateway.api.governance import router as governance_router
from llm_gateway.api.llm import router as llm_router

app = FastAPI(title="Agent LLM Gateway", version="0.0.1")

# OpenAI 风格错误体处理器（M03 任务 4）：注册全局但只作用于 OpenAI 兼容面，
# 旧 /v1/llm 端点的 422/detail 形态由处理器内部按路径回退保持原样。
register_error_handlers(app)

app.include_router(llm_router)
app.include_router(governance_router)
