"""契约测试共享的常量与构造器（原 test_demo_semantics.py 内嵌 helper 的提取）。

只放"如何构造网关请求 / 上游响应"的纯函数与派生常量；fixture 在 conftest.py，
用例与断言在各测试文件。所有上游请求由 respx 拦截，测试全程离线：凡是 respx
未注册的上游请求都会让测试失败（AllMockedAssertionError），而不是真发网络。
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.core.config import CONFIG
from llm_gateway.services.catalog import MODEL_CONFIGS

# ---------------------------------------------------------------------------
# 认证（M05）：契约测试的合法调用方身份取自配置中心的加载产物（单一事实来源，
# callers.yaml 换 key 时测试零改动）；需要 401 场景的用例按请求覆盖该头。
# ---------------------------------------------------------------------------

VALID_CALLER_KEY = next(iter(CONFIG.callers))
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_CALLER_KEY}"}

# ---------------------------------------------------------------------------
# 上游坐标：从 MODEL_CONFIGS 派生供应商模型名与 URL，避免硬编码，也保证
# PRIMARY_PROVIDER_MODEL / BACKUP_BASE_URL 等环境变量覆盖时测试仍然成立。
# ---------------------------------------------------------------------------

PRIMARY_PROVIDER_MODEL = MODEL_CONFIGS["general-primary"].provider_model
BACKUP_PROVIDER_MODEL = MODEL_CONFIGS["general-backup"].provider_model


def _chat_completions_url(config: Any) -> str:
    # openai SDK 会在 base_url 后拼接 /chat/completions。
    return config.base_url.rstrip("/") + "/chat/completions"


PRIMARY_URL = _chat_completions_url(MODEL_CONFIGS["general-primary"])
BACKUP_URL = _chat_completions_url(MODEL_CONFIGS["general-backup"])

SSE_HEADERS = {"content-type": "text/event-stream"}

# 结构化输出（服务层回归）测试用的最小 JSON Schema：要求 answer 为 string。
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# 网关请求构造器（OpenAI 兼容面）
# ---------------------------------------------------------------------------


def chat_request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "你好"}],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 上游响应构造器（OpenAI Compatible 的 chat.completion / chunk 形态）
# ---------------------------------------------------------------------------


def completion(content: str, *, prompt_tokens: int = 13, completion_tokens: int = 5) -> dict[str, Any]:
    # 构造 OpenAI Compatible 的 chat.completion 响应体（SDK 会按 Pydantic 模型解析）。
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "provider-model-ignored",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse_chunk_payload(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "provider-model-ignored",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }


def _sse_usage_chunk_payload(prompt_tokens: int = 13, completion_tokens: int = 5) -> dict[str, Any]:
    # 上游的 usage 块（include_usage 回传，OpenAI 惯例：choices 为空）；
    # 默认用量与 completion() 的默认值同口径，便于用例直接对账。
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "provider-model-ignored",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _encode_sse_chunk(text: str) -> bytes:
    return f"data: {json.dumps(_sse_chunk_payload(text))}\n\n".encode()


def sse_body(chunks: list[str], *, include_usage: bool = False) -> bytes:
    # 完整的上游 SSE 流：若干增量块 +（可选）usage 块 + [DONE] 终止符。
    # include_usage 默认 False：既有用例行为零改动（M04 新增参数）。
    parts = [_encode_sse_chunk(text) for text in chunks]
    if include_usage:
        parts.append(f"data: {json.dumps(_sse_usage_chunk_payload())}\n\n".encode())
    return b"".join(parts) + b"data: [DONE]\n\n"


def sse_stream_then_break(chunks: list[str]) -> AsyncIterator[bytes]:
    # 先正常吐出 chunks，然后模拟上游连接中断（首块之后才挂）。
    async def _gen() -> AsyncIterator[bytes]:
        for text in chunks:
            yield _encode_sse_chunk(text)
        raise httpx.ReadError("upstream stream broke after first chunk")

    return _gen()


# ---------------------------------------------------------------------------
# 网关 SSE 消费器
# ---------------------------------------------------------------------------


async def collect_sse_events(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> list[str]:
    # 以流式方式消费网关 SSE 响应，返回 data: 行的原始负载（JSON 字符串或
    # "[DONE]"）。解析留给用例：成功终态以 [DONE] 收尾，流内失败以错误事件
    # 收场、没有 [DONE]——两种形态都是契约的一部分。
    events: list[str] = []
    async with client.stream("POST", path, json=payload) as response:
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(line[len("data: "):])
    return events
