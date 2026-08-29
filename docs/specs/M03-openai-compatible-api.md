# M03 - OpenAI 兼容 API 面

## 目标

自定义 `/v1/llm` 协议升级为 OpenAI 兼容 `/v1/chat/completions`（含 SSE 流式），openai SDK 零改动可调；字段白名单在调用模型前拒绝不支持的字段。

## 前置依赖

M02。

## 任务

1. `api/schemas.py`：OpenAI 请求字段白名单（`model / messages / stream / temperature / max_tokens / response_format / stream_options`）+ 扩展字段 `prompt`、`validation`（本里程碑仅解析透传，语义在 M07/M08 落地）。白名单外字段 -> 400 `unsupported_field`。
2. `api/chat.py`：`POST /v1/chat/completions`，非流式返回 OpenAI chat.completion 结构（`id / object / created / model / choices / usage`）；流式返回 SSE chunk 流（`data: {chunk}`，终态 `data: [DONE]`），复用 demo 的流式语义（首块前可 fallback，首块后不重生成）。
3. `api/governance.py`：`GET /v1/models`（平台模型列表）；`GET /v1/traces` 保留。
4. `api/errors.py`：全部错误转 OpenAI 风格 `{"error": {"message", "type", "code"}}`。
5. **删除** `/v1/llm` 与 `/v1/llm/stream`。
6. 契约测试改用 **openai 官方 SDK 客户端**打本地 ASGI app（`AsyncOpenAI(base_url=..., http_client=自定义 httpx.AsyncClient(transport=ASGITransport))`）：非流式、流式增量、错误体可被 SDK 解析（`except openai.BadRequestError` 能捕获且 `e.code` 正确）、白名单外字段被拒且**无上游请求发生**（respx 计数为 0）。

## 验收

```bash
make check
uv run pytest tests/contract/test_chat_api.py -q
uv run pytest tests/contract/test_field_whitelist.py -q   # 不支持字段 400，上游请求数==0
uv run python - <<'EOF'                                   # 真 SDK 端到端冒烟（Fake Provider 或 respx）
from openai import AsyncOpenAI
...
EOF
```

## 覆盖的不变量

- #3（OpenAI 客户端可直接调用）
- #4（未支持字段调用模型前明确失败）

## 边界

- `validation` 扩展字段本里程碑只透传，未注册 profile 的报错在 M08。
