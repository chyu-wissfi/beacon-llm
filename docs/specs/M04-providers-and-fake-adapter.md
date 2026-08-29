# M04 - Provider 层：双协议 + Fake Adapter

## 目标

Provider Protocol 升级（返回 finish_reason）；OpenAI-compatible 支持 chat/Responses 两种 provider_api 模式；Anthropic Messages 原生实现；Fake Adapter 剧本化复现五类故障。所有 SDK 内置重试关闭。

## 前置依赖

M03。

## 任务

1. `providers/base.py`：Protocol 定稿——`complete() -> (content, usage, finish_reason)`、`stream() -> AsyncIterator[StreamEvent]`。SDK 异常在此层映射为 `GatewayError`，**不允许任何 SDK 异常类型向上穿透**。
2. `providers/openai_compatible.py`：按 `ModelConfig.provider_api` 分派 `chat.completions.create` 或 `responses.create`；`structured_output_mode` 维持 json_schema / json_object 两模式；`AsyncOpenAI(..., max_retries=0)`。
3. `providers/anthropic_provider.py`：原生 Messages API，`max_retries=0`，usage 映射为统一 `Usage`，finish_reason 映射（max_tokens -> "length"）。
4. `providers/fake.py`：**Fake Adapter**。剧本由调用方构造：`FakeAdapter(scenario=...)`，scenario 覆盖：`success`（可控 content/usage/finish_reason）、`rate_limited`（429，可带 Retry-After）、`timeout`、`stream_interrupt`（流中途断开，已发出可配置数量的块）、`invalid_output`（非法 JSON 或合法 JSON 但违反 schema）、`consecutive_then_success`（前 N 次失败后成功）。
5. Provider 选择：配置驱动注册表（provider 名 -> 实例），services 层只面向 Protocol。
6. 测试：三个 Provider 各自单测（respx 注入 openai/anthropic 的 httpx 层）+ Fake Adapter 自身的确定性测试（同剧本跑 10 次结果完全一致）。

## 验收

```bash
make check
uv run pytest tests/unit/providers/ -q
uv run pytest tests/unit/providers/test_fake_adapter.py -q -k deterministic
# 无隐藏重试证明：剧本"连续 3 次 timeout 后成功"，断言上游恰好收到 3+1=4 次请求
uv run pytest tests/unit/providers/ -q -k no_hidden_retry
```

## 覆盖的不变量

- #16（Fake Adapter 稳定复现五类故障）
- #1（唯一 Provider 入口、错误不穿透的地基）

## 边界

- 编排逻辑（重试/fallback/预算）本里程碑不接 Provider 层改造，仍用旧编排跑通现有测试；M06 重写。
