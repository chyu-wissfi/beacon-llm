# ADR-0002: 多供应商深度止步于协议级双实现，不做能力映射层

日期：2026-08-30 · 状态：已接受

## 背景

需要支持多家模型供应商。demo 只有 OpenAI-compatible 单实现（DeepSeek）。"多供应商"存在三个深度：纯配置（多家 OpenAI-compatible）、协议级多实现（含协议真正不同的供应商）、全能力映射层（抹平各家 structured output / 流式 / 工具调用语义差异）。

## 决策

1. OpenAI-compatible 协议内同时支持 `chat.completions` 与 `Responses` 两种上游传输模式（`ModelConfig.provider_api` 字段选择）。
2. 增加 Anthropic Messages 原生实现作为第二个 Provider。
3. **不做**全能力映射层：各供应商能力差异在 `ModelConfig`（如 `structured_output_mode: json_schema | json_object`）按模型显式声明，不隐式抹平。
4. 增加 Fake Adapter 作为第三个"供应商"，用于剧本化故障测试。

深度止步于此（"中"）。

## 理由

多供应商的卖点是**验证 Provider 抽象的正确性**，不是穷举供应商。chat/Responses + Anthropic 已覆盖"协议真的不同"的完整光谱：Responses 与 chat 的差异在同协议内，Anthropic 的差异在协议本身。全能力映射层是真实多供应商团队的需求（数十家供应商、语义差异矩阵），在单实例轻量场景下是无限复杂度换零个用户。能力差异显式声明（ModelConfig 字段）而非隐式抹平，使得"模型不支持 Structured Output"是调用前 400 而不是运行时惊喜。

## 后果

正面：新增 OpenAI-compatible 供应商零代码（纯配置）；Provider Protocol 经两种真协议 + 一种假实现三方验证。
负面：供应商新增语义型能力（如某家特有的工具调用协议）时需要扩 ModelConfig 而非自动获得；能力声明表需要随供应商演进维护。
