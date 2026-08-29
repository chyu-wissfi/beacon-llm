# M12 - ModelPort 与 Live 冒烟验收

## 目标

交付 `packages/modelport/`（Agent 侧唯一依赖）；真模型 live 冒烟证明六大功能；全量不变量审计收口。

## 前置依赖

M11。

## 任务

1. `packages/modelport/`：
   - 薄封装：openai SDK 客户端（base_url 指向网关）+ Bearer key 管理。
   - 错误映射：网关错误码 -> 统一异常类型（`ModelPortError` 层级，含 code），SDK 异常不外泄。
   - request_id 透传：响应对象可取 `request_id`，便于与 Trace 对账。
   - 暴露 `complete()` / `stream()` / 结构化输出（response_format）与模板/校验选择的 extra_body 透传。
2. ModelPort 自身测试（打本地 ASGI app + Fake Adapter）：错误映射完整、request_id 可取、流式可用。
3. **Agent 不依赖供应商 SDK 的证明**：测试断言 `packages/modelport/` 源码不出现对供应商 SDK 符号的再导出（ModelPort 内部使用 openai SDK 是实现细节，对 Agent 是黑盒），并断言示例 Agent 代码仅 `import modelport`。
4. `tests/live/`（`@pytest.mark.live`，需真实 key，CI 跳过，`make test-live` 运行）：真模型验证六大功能各至少一个用例：
   - 非流式调用（含 usage 正确）
   - 流式（SSE 增量 + TTFT 记录进 trace）
   - 结构化输出（response_format + 校验 + trace 的 parsed 路径）
   - 模板引用（prompt 扩展字段渲染）
   - 可观测（调用后 `/metrics` 计数变化、`/v1/traces` 新记录含 caller/route/价格版本/尝试数）
   - 重试与限流（低 RPM 配置的模型上真实触发一次限流拒绝；重试可注入 Fake 模型完成，live 侧仅验证正常路径 attempts==1）
5. 不变量审计表：对照 `docs/design.md` §8 逐条打勾，全部有对应测试文件与用例名，写入 `docs/specs/acceptance-audit.md`。

## 验收

```bash
make check
uv run pytest tests/contract/test_modelport.py -q
make test-live                 # 六大功能全部通过（需 DEEPSEEK_API_KEY）
test -f docs/specs/acceptance-audit.md && grep -c "✅" docs/specs/acceptance-audit.md  # >= 16
```

## 覆盖的不变量

- #2（Agent 只依赖 ModelPort）
- #17（边界审计）
- #1/#3/#5/#6 等 live 侧最终证据

## 边界

- ModelPort 不发版到 PyPI（本地 `uv pip install -e packages/modelport` 使用）；live 测试依赖真实 key 与网络，不进 CI。
