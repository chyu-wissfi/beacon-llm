# M10 - 可观测与日志安全

## 目标

`/metrics`（Prometheus）+ `/healthz` + 结构化日志（含 caller、脱敏）。每个指标对应一个已有机制。

## 前置依赖

M09。

## 任务

1. `observability/metrics.py`（prometheus-client）：
   - `llm_requests_total{model, status}`
   - `llm_request_latency_seconds`（直方图）
   - `llm_tokens_total{model, direction=input|output}`
   - `llm_retries_total{model}`
   - `llm_rate_limited_total{model}`
   - `llm_requests_in_flight`（gauge）
   - `llm_breaker_state{model}`
2. `/healthz`：200 + 版本号。`/metrics`：文本协议输出。
3. `observability/logging.py`：结构化 JSON 日志统一出口；字段含 request_id、caller、status、attempts。**脱敏规则**：日志过滤器保证任何字段不含 API key；默认不记录 Prompt/消息内容（trace 与日志都如此）。
4. 指标接线：各机制（重试、限流、熔断、token）在动作点计数，无遗漏无重复（与 M09 记账不同口径：metrics 是速率、trace 是审计）。
5. 测试：
   - 剧本触发重试/限流后 `/metrics` 中对应计数器增长精确 +1。
   - 脱敏：构造含 API key 的请求 + key 放入日志上下文，全量捕获日志输出断言无 key 子串；断言无消息 content 字段。

## 验收

```bash
make check
uv run pytest tests/unit/observability/ tests/contract/test_metrics.py -q
uv run pytest tests/contract/test_log_scrubbing.py -q   # 日志无 key、无 prompt 内容
uv run pytest tests/contract/test_metrics.py -q -k counters
```

## 覆盖的不变量

- #13（日志不含 API Key、默认不记敏感 Prompt）

## 边界

- 无告警规则、无 dashboard json（演进项）；metrics 重启清零（Prometheus 语义本就如此）。
