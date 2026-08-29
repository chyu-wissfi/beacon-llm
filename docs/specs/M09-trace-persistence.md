# M09 - Trace 落库与聚合

## 目标

Trace 从内存 list 迁移到 SQLite（SQLAlchemy async + aiosqlite）；字段补全；用量恰好记账一次；`/v1/traces` 支持按调用方/模型/Prompt 版本聚合过滤。

## 前置依赖

M08。

## 任务

1. `storage/engine.py`：`create_async_engine("sqlite+aiosqlite:///data/traces.db")`；`storage/models.py`：`traces` 表（字段见下）。
2. **Trace 字段（demo 全量一个不漏 + 新增）**：`request_id / timestamp / caller / requested_model / actual_model / final_endpoint / route_reason / prompt_name / prompt_version / validation_profile / input_tokens / output_tokens / cost_usd / price_version / latency_ms / ttft_ms / attempts / status / error_code`。
3. `services/trace_service.py`：终态迁移时**恰好一次**落库（含 failed/cancelled）；每次上游响应的 usage 恰好记账一次（失败尝试的消耗也计入 run 总量）。
4. 价格表按 `price_version` 快照计算成本（版本来自 `config/prices.yaml`）。
5. `api/governance.py` 的 `/v1/traces`：读库，支持 `caller / model / prompt_version / status` 过滤与聚合（总量、总成本、平均 TTFT/延迟，按调用方、模型、Prompt 版本分组）。
6. 测试：
   - 记账恰好一次：多 attempt 剧本（重试+fallback+修复），断言 trace 单条、usage = 各次尝试之和、无重复。
   - 聚合：灌入多调用方多模型数据，断言分组聚合数值。
   - 崩溃安全：终态迁移前进程取消，无半条 trace。

## 验收

```bash
make check
uv run pytest tests/unit/storage/ tests/unit/services/test_trace_service.py -q
uv run pytest tests/contract/test_traces_api.py -q
uv run pytest tests/contract/test_traces_api.py -q -k "exactly_once"
```

## 覆盖的不变量

- #5（Trace 可见最终端点与路由理由）
- #12（可定位 Prompt/Schema/路由/模型/价格版本/尝试次数）
- #14（按调用方、模型、Prompt 版本聚合）
- #1（Trace 真实可靠）

## 边界

- 无 retention/轮转（单实例低流量，写演进项）；不做迁移工具（重启视为清空旧内存 trace 可接受）。
