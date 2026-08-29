# ADR-0001: API 契约采用 OpenAI 兼容 + extra_body 扩展字段

日期：2026-08-30 · 状态：已接受

## 背景

demo `gateway.py` 使用自定义 `/v1/llm` 协议。调用方（业务 Agent）要么为网关专门写客户端，要么放弃存量 SDK。同时网关的模板选择（name/version/variables）与校验档案选择（name/version）在 OpenAI 协议中没有对应字段。

## 决策

1. 主端点改为 `POST /v1/chat/completions`（含 SSE 流式），OpenAI 兼容；错误体用 OpenAI 风格 `{"error": {message, type, code}}`；提供 `GET /v1/models`。
2. 网关能力通过**扩展 body 字段**暴露：`prompt: {name, version, variables}`、`validation: {name, version}`，走 openai SDK 官方的 `extra_body` 通道。
3. 字段白名单封闭：标准字段子集 + 两个扩展字段之外的一切字段 400 `unsupported_field`。
4. 自定义端点 `/v1/llm`、`/v1/llm/stream` 在迁移完成后删除；`/v1/traces` 保留（治理域，不属于 OpenAI 协议）。

## 备选方案

- **自定义 header 传模板/校验选择**：结构化数据（含 variables 字典）需要自造编码与解析，SDK 对自定义 header 的透传支持参差。否决。
- **模型名约定**（`knowledge_decision:v1` 当 model）：污染平台模型白名单语义，Trace 的模型字段被迫承载路由外信息。否决。
- **保留自定义协议并存**：双协议两份契约测试，长期漂移，且 OpenAI 兼容后自定义协议无增量价值。否决。

## 后果

正面：存量 openai SDK / 生态工具零改动接入；契约测试可直接用官方 SDK 客户端写。
负面：兼容协议是移动目标（OpenAI 偶尔加字段），白名单需要维护；扩展字段对非 openai SDK 的调用方需要裸 JSON 能力（curl 可，多数客户端库可）。
