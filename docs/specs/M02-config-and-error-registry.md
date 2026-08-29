# M02 - 配置中心化与错误码注册表

## 目标

demo 中散落的 `MODEL_CONFIGS`、`PROMPT_TEMPLATES`、`PRICE_PER_MILLION`、调用方信息收敛为文件配置 + 统一加载器；所有错误码集中到 `core/errors.py` 注册表，成为错误码的单一事实来源。

## 前置依赖

M01。

## 任务

1. `core/config.py`：启动时加载并校验 `config/models.yaml`（平台模型 -> 供应商模型、base_url、api_key_env、provider_api、structured_output_mode、fallback 链、限流参数占位）、`config/callers.yaml`（key + 显示名）、`config/prices.yaml`（**带版本字段**）。加载失败 = 启动失败（fail-fast），错误信息指明文件与字段。
2. 提供环境变量覆盖（`PRIMARY_PROVIDER_MODEL` 等，兼容 demo 用法）。
3. `core/errors.py`：错误码注册表（dataclass 或 Literal 联合 + 映射表），现有全部错误码迁入：`unknown_model / unknown_prompt_template / missing_prompt_variable / gateway_misconfigured / invalid_json / schema_validation_failed / model_unavailable / upstream_stream_failed / use_stream_endpoint / unsupported_combination`。预留（本里程碑只注册不使用）：`unauthorized / rate_limited / overloaded / provider_overloaded / token_budget_exceeded / circuit_open / unsupported_field / output_truncated / business_validation_failed / unknown_validation_profile / request_cancelled`。
4. 单元测试：配置加载（合法/缺字段/类型错）、每个注册错误码的 code/message/status 三元组稳定。

## 验收

```bash
make check
uv run pytest tests/unit/test_config.py tests/unit/test_error_registry.py -q
# 错误码封闭性：注册表外的 code 无法构造 GatewayError（类型层面或运行时断言）
uv run pytest tests/unit/test_error_registry.py -k closed_set
```

## 覆盖的不变量

- #1（配置与错误契约真实可靠的地基）

## 边界

- 调用方表此里程碑只加载不校验（认证在 M05）。
