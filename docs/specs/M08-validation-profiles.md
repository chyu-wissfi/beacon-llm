# M08 - 双重校验与 Validation Profile

## 目标

结构校验（供应商约束 + 本地 jsonschema 双重）之后叠加业务校验（Validation Profile，Pydantic 代码注册表）；非法输出绝不进入 Agent Loop。

## 前置依赖

M07。

## 任务

1. 校验流水线定型（`services/` 内，接 M06 修复调用）：
   `json.loads` -> `output_truncated`（finish_reason=length，修复=提高 max_tokens）-> jsonschema 本地校验 -> Validation Profile 业务校验；每层失败先走修复调用（上限 1，携带对应错误反馈），修复仍失败才报错。
2. `src/llm_gateway/validation/`：注册表 + 首个示例 Profile（如 `order_decision/v1`：演示互斥字段与条件依赖，用 `model_validator` 写规则）。
3. `api/` 层解析扩展字段 `validation: {name, version}`：未注册 -> 400 `unknown_validation_profile`（**上游请求数 == 0**）。
4. 错误码落地：`invalid_json` / `output_truncated` / `schema_validation_failed` / `business_validation_failed`。
5. 测试：
   - 双重校验：Fake Adapter 返回"结构合法但业务非法"的 JSON，断言不进入响应、走修复、终报 `business_validation_failed`。
   - 结构双重：jsonschema 本地校验捕获 Fake Adapter 绕过供应商约束的输出。
   - 未注册 profile：400 且上游 0 请求。
   - 修复链：每种失败类型恰好 1 次修复调用（计数断言）。

## 验收

```bash
make check
uv run pytest tests/unit/validation/ -q
uv run pytest tests/contract/test_structured_output.py -q
uv run pytest tests/contract/test_structured_output.py -q -k "business_rule_blocked"
```

## 覆盖的不变量

- #9（供应商约束 + 本地双重校验）
- #10（JSON 合法但业务非法不进 Agent Loop）

## 边界

- Profile 注册表是代码（ADR-0005），不做热加载；业务校验失败不触发 fallback 换模型（模型没错，是输出错）。
