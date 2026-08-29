# M06 - Run 预算与编排状态机

## 目标

重写 `services/invocation.py` 为唯一状态机：RunContext 冻结、统一预算计数器、指数退避、声明式 fallback 链与路由理由、修复调用、截断处理、流式铁律、取消传播、唯一终态。这是全项目行为密度最高的里程碑。

## 前置依赖

M05。

## 任务

1. **RunContext**（不可变 dataclass）：渲染后 Prompt、response_format、预算实例、request_id、caller、价格表版本快照、trace 骨架。构建点 = 编排入口，构建后全链路只读。
2. `services/routing.py`：求值声明式 fallback 链，产出候选列表 + 每步路由理由（如 `"primary: 3 attempts exhausted (timeout)"`、`"primary: circuit_open"`）。理由进 trace。
3. **预算**：总尝试 4 次（重试/fallback/修复共享计数器）+ 墙钟 deadline（`timeout_seconds`，默认 30s）。超预算/超时 -> 终态 failed `model_unavailable`。
4. **退避**：0.5s ×2 + 0~0.2s 抖动（注入时钟可测）；尊重上游 `Retry-After`。可重试判定：连接/超时/429/5xx；其余 4xx 不重试。
5. **修复调用**（上限 1，消耗预算）：schema 失败 -> 携带错误反馈重调；`finish_reason == "length"` -> `output_truncated`，提高 max_tokens 重调。修复失败才向调用方报错。
6. **流式铁律**：首块前可重试/切 fallback；首块后失败只发终态错误事件，不重新生成；TTFT 在首块记录。
7. **取消**：客户端断开（`Request.is_disconnected` / asyncio 取消传播）-> 取消下游任务 -> trace 单一 `cancelled` 终态，不落 `failed`。
8. **唯一终态**：`success / failed / cancelled` 三选一恰好一次迁移；trace 在终态时写一次（暂写内存，M09 落库）。
9. 测试（全部用 Fake Adapter 剧本，注入时钟）：
   - 预算：剧本"永远 timeout"，断言上游请求总数 == 4（不是 4×2 或 4×模型数）且 attempts==4。
   - fallback：主模型 3 次失败后 backup 成功，断言 trace 的 actual_model、route_reason、attempts。
   - 首块后中断：断言已发出的块不重复、终态错误只发一次。
   - 取消：客户端中途断开，断言下游任务取消（Fake Adapter 感知到）、trace 终态 cancelled 且仅一条。
   - 修复：invalid JSON -> 修复成功；截断 -> 提高 max_tokens 成功；各恰好 1 次修复。

## 验收

```bash
make check
uv run pytest tests/unit/services/test_invocation.py -q
uv run pytest tests/unit/services/test_invocation.py -q -k "budget or fallback"
uv run pytest tests/contract/test_stream_semantics.py -q -k "no_regeneration or cancellation"
```

## 覆盖的不变量

- #6（首 Token 前可恢复错误可重试/Fallback）
- #7（已输出文本后不偷偷重新生成拼接）
- #8（取消传播、单一 cancelled 终态）
- #15（重试/Fallback/修复同一 Run 预算）

## 边界

- 修复调用的反馈提示词从简（"上次输出违反 X，请重新输出合法 JSON"），不做复杂反思链。
- 取消记账规则：`cancelled` 的 trace 只记已观测到的 usage（流式的 usage chunk 在最后，中途取消观测不到上游已消耗部分），此缺口是已知且可识别的（对账先查 cancelled/failed 条目）；取消传播的意义是把观测不到的浪费压到最小，而非分毫不差。
- 非流式请求的客户端断开 Starlette 不会自动取消 handler，需 `request.is_disconnected()` 轮询或超时兜底；本里程碑取消测试以流式为主。
