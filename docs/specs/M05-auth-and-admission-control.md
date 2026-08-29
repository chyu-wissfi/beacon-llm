# M05 - 认证与多层准入控制

## 目标

调用方认证 + 多层准入控制 + 按模型熔断，全部在请求进入编排之前完成；限流/熔断为单进程内存实现（明示边界）。

## 前置依赖

M04。

## 任务

1. `core/auth.py`：`Authorization: Bearer <key>` 对 `config/callers.yaml`；`hmac.compare_digest` 常数时间比较；失败 401 `unauthorized`；认证结果（caller 标识）注入请求上下文。
2. `core/ratelimit.py`：令牌桶（RPM，按模型）+ 两级并发信号量（全局默认 20 / 每供应商默认 10）。超限：`rate_limited`（RPM）/ `overloaded`（全局）/ `provider_overloaded`（供应商），均 429 + `Retry-After` 头。并发计数从准入持有到流式结束。
3. TPM：`core/ratelimit.py` 内按模型记账（调用完成后按实际 usage 扣减），超额后新请求 429 `token_budget_exceeded`。**事后记账**，不预估扣减。
4. `core/breaker.py`：按模型，连续 5 次失败 -> open 30s（503 `circuit_open`）-> half-open 放行 1 个探测，成功闭合、失败重开。失败判定 = 网关侧可重试类失败（连接/超时/上游 5xx/429 不算失败）。
5. 准入顺序固化为常量序列：鉴权 -> 全局并发 -> 熔断 -> RPM -> 供应商并发 ->（编排）。拒绝时必须已持有的资源全部释放（无泄漏）。
6. 测试：每层独立单测（时间用 `freezegun` 或注入时钟，不真睡）+ 组合测试（第一层拒绝时后层计数不变）+ 准入拒绝时上游请求数为 0。

## 验收

```bash
make check
uv run pytest tests/unit/core/ -q
uv run pytest tests/contract/test_admission.py -q
# 并发上限行为：异步发 25 个慢请求（Fake Adapter 慢成功），断言 <=20 个到达上游
uv run pytest tests/contract/test_admission.py -q -k global_concurrency
```

## 覆盖的不变量

- #1（准入层真实可靠）
- #17（单进程边界：测试断言限流/熔断状态在同进程内对所有请求可见）

## 边界

- 不做按调用方限流（演进项）；多副本共享状态不做（非目标）。
