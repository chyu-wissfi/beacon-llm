# 不变量审计表（M12 任务 5）

对照 `docs/design.md` §8 的 17 条行为不变量与 §8.1 六大功能证据矩阵逐条审计：
每条不变量给出**契约层**（CI 内确定性证明）与 **live 层**（真模型冒烟）的
测试文件与用例名。审计口径是"行为证据"（上游请求计数、终态唯一性、字段断言），
不是"接口返回 200"。

复核方式：

```bash
make check                                            # 契约层全量
uv run pytest tests/contract/test_modelport.py -q     # ModelPort 隔离与映射
make test-live                                        # live 层（需真实 key）
```

---

## 一、行为不变量（design.md §8）

| # | 不变量 | 契约层证据（文件::用例） | live 层证据 | 状态 |
|---|---|---|---|---|
| 1 | 统一契约、唯一 Provider 入口、校验、重试、fallback、Trace 真实可靠 | `unit/services/test_invocation.py::test_budget_shared_without_fallback`、`contract/test_chat_api.py::test_retryable_error_retries_primary_then_falls_back_to_backup`、`contract/test_traces_api.py::test_trace_written_exactly_once_across_retry_and_repair` | `live/test_live_smoke.py::test_live_nonstream_call_with_usage`（attempts==1） | ✅ |
| 2 | Agent 侧只依赖 ModelPort，不导入供应商 SDK | `contract/test_modelport.py::test_public_surface_exposes_no_vendor_sdk_symbols`、`::test_source_has_no_star_reexport_of_vendor_sdks`、`::test_example_agent_imports_only_modelport` | —（隔离是可静态验收的行为） | ✅ |
| 3 | OpenAI 客户端可调 /v1/chat/completions | `contract/test_openai_sdk_contract.py::test_sdk_non_stream_parses_chat_completion`、`::test_sdk_stream_iterates_deltas_and_ends_cleanly` | live 全用例经同一端点 | ✅ |
| 4 | 未支持字段在模型调用前明确失败 | `contract/test_field_whitelist.py::test_unknown_top_level_field_rejected_400_no_upstream`、`::test_unknown_nested_field_rejected_400_no_upstream`（均断言上游请求数==0） | — | ✅ |
| 5 | Trace 可见最终端点与路由理由 | `contract/test_traces_api.py::test_trace_fields_complete_after_success`（final_endpoint / route_reason 字段断言）、`unit/services/test_invocation.py::test_fallback_success_after_three_primary_failures` | `live/test_live_smoke.py::test_live_observability_metrics_and_traces`（/v1/traces 记录在场） | ✅ |
| 6 | 首 Token 前可恢复错误可重试 / Fallback | `contract/test_chat_api.py::test_stream_falls_back_before_first_chunk`、`unit/services/test_invocation.py::test_stream_fallback_before_first_chunk`（Fake Adapter 剧本） | —（真模型失败时机不可控，见 §8.1 Why） | ✅ |
| 7 | 已输出文本后错误不偷偷重新生成拼接 | `contract/test_stream_semantics.py::test_stream_no_regeneration_after_first_chunk_failure`、`contract/test_chat_api.py::test_stream_failure_after_first_chunk_emits_error_without_regeneration`、`contract/test_openai_sdk_contract.py::test_sdk_in_stream_failure_raises_apierror_with_registry_code`（备用模型零调用） | —（同 #6） | ✅ |
| 8 | 客户端取消后下游停止、单一 cancelled 终态 | `contract/test_stream_semantics.py::test_stream_cancellation_propagates_single_cancelled_trace`、`contract/test_traces_api.py::test_cancellation_before_terminal_leaves_single_complete_row`、`unit/services/test_invocation.py::test_stream_cancellation_propagates_and_finalizes_cancelled`（Fake 感知流关闭） | — | ✅ |
| 9 | Structured Output 供应商约束 + 本地双重校验 | `contract/test_structured_output.py::test_jsonschema_local_catches_output_bypassing_vendor_constraints`、`unit/validation/test_validation_pipeline.py::test_business_gate_runs_without_response_schema` | `live/test_live_smoke.py::test_live_structured_output_with_validation_profile` | ✅ |
| 10 | JSON 合法但不满足业务规则不进 Agent Loop | `contract/test_structured_output.py::test_business_rule_blocked_after_single_repair_never_in_response`、`unit/validation/test_validation_pipeline.py::test_business_invalid_never_enters_response_repair_then_terminal`、`contract/test_modelport.py::test_business_validation_failure_maps_upstream_failed` | live 结构化用例现场复核业务规则 | ✅ |
| 11 | Prompt 缺变量在调用模型前失败 | `contract/test_prompt.py::test_missing_variable_fail_before_upstream`（上游请求数==0）、`contract/test_modelport.py::test_missing_prompt_variable_rejected_before_upstream` | — | ✅ |
| 12 | 每次调用可定位 Prompt/Schema/路由/模型/价格版本/尝试次数 | `contract/test_traces_api.py::test_trace_fields_complete_after_success`、`::test_traces_filters_by_caller_model_status_prompt_version` | `live/test_live_smoke.py::test_live_observability_metrics_and_traces`（caller / price_version / attempts） | ✅ |
| 13 | 日志不含 API Key、默认不记敏感 Prompt | `contract/test_log_scrubbing.py::test_request_path_logs_no_key_and_no_content`、`::test_key_in_log_context_is_scrubbed`、`::test_provider_env_key_registered_and_scrubbed`、`unit/observability/test_logging.py`（10 例） | — | ✅ |
| 14 | Token/Cost/TTFT/延迟按调用方、模型、Prompt 版本聚合 | `contract/test_traces_api.py::test_traces_aggregation_group_by_caller`、`::test_traces_aggregation_group_by_model_and_prompt_version`、`::test_traces_aggregation_empty_scope` | live 流式用例断言 TTFT 进 trace | ✅ |
| 15 | 重试、Fallback、修复受同一 Run 预算约束 | `unit/services/test_invocation.py::test_budget_always_timeout_totals_four_attempts`、`::test_budget_shared_without_fallback`、`::test_repair_denied_when_budget_exhausted`、`contract/test_modelport.py::test_timeout_maps_upstream_failed`（attempts==4 无隐藏放大） | —（同 #6） | ✅ |
| 16 | Fake Adapter 稳定复现五类故障 | `unit/providers/test_fake_adapter.py::test_scenario_rate_limited`、`::test_scenario_timeout_no_real_sleep`、`::test_scenario_stream_interrupt`、`::test_scenario_invalid_output_returns_bad_content_faithfully`、`::test_deterministic_complete_same_result_ten_times` 等（零随机可复现断言） | —（契约层地基） | ✅ |
| 17 | 限流/熔断单进程边界；不维护 Agent Run 状态 | 边界声明：`docs/design.md` §1 非目标 + §3.2；行为面：`unit/core/test_breaker.py`、`unit/core/test_ratelimit.py`、`contract/test_admission.py`（进程内全局状态复位夹具即单进程边界的字面证明） | `live/test_live_smoke.py::test_live_rate_limit_real_429` | ✅ |

## 二、六大功能证据矩阵（design.md §8.1）

| 功能 | 契约层证据（CI） | live 层证据（M12） | 状态 |
|---|---|---|---|
| 非流式调用 | `contract/test_openai_sdk_contract.py::test_sdk_non_stream_parses_chat_completion`、`contract/test_modelport.py::test_complete_success_and_request_id_matches_trace` | `live/test_live_smoke.py::test_live_nonstream_call_with_usage`（usage 三键自洽 + attempts==1） | ✅ |
| 流式 | `contract/test_openai_sdk_contract.py::test_sdk_stream_include_usage_emits_usage_chunk`、`contract/test_stream_semantics.py`（首块语义/不重生成/取消）、`contract/test_modelport.py::test_stream_yields_chunks_and_request_id_matches_trace` | `live/test_live_smoke.py::test_live_stream_sse_increments_and_ttft`（SSE 增量 + TTFT 进 trace） | ✅ |
| 结构化输出 | `contract/test_structured_output.py`（双重校验 + 业务阻断 + 修复计数）、`contract/test_modelport.py::test_structured_output_passthrough_with_schema_validation` | `live/test_live_smoke.py::test_live_structured_output_with_validation_profile`（response_format + 校验 + trace 档案路径） | ✅ |
| 模板引用 | `contract/test_prompt.py::test_prompt_selection_via_extra_body_renders_system_message_first`、`unit/prompt/test_loader.py`（热加载 12 例）、`contract/test_modelport.py::test_prompt_ref_passthrough_renders_template` | `live/test_live_smoke.py::test_live_prompt_template_reference` | ✅ |
| 可观测 | `contract/test_metrics.py`（计数器精确增量 8 例）、`contract/test_log_scrubbing.py` | `live/test_live_smoke.py::test_live_observability_metrics_and_traces`（/metrics 恰好 +1、/v1/traces 新记录含 caller/价格版本/尝试数） | ✅ |
| 重试 | `unit/services/test_invocation.py`（预算/退避/修复剧本，上游请求计数断言）、`contract/test_modelport.py::test_provider_rate_limit_exhausts_budget_as_model_unavailable`（attempts==4） | 正常路径 attempts==1（`test_live_nonstream_call_with_usage`）——完整重试链由契约层证明（真模型失败不可控） | ✅ |
| 限流 | `contract/test_admission.py`（25 慢请求 ≤20 到上游、429 三态 + Retry-After）、`contract/test_modelport.py::test_rate_limited_maps_rate_limited_error_with_retry_after` | `live/test_live_smoke.py::test_live_rate_limit_real_429`（低 RPM 配置真实触发一次 429，零上游波及） | ✅ |

## 三、审计结论

- 17 条不变量 + 7 行功能矩阵全部有可执行测试证据（上文逐条点名文件与用例）；
- 契约层证据在 `make check` 内确定性复现，live 层证据经 `make test-live`
  真模型复现；两层的分工与弱化理由（重试/限流的不对称）见 `docs/design.md` §8.1；
- ModelPort 交付后，"Agent 不依赖供应商 SDK"从文档口号变为三条静态可验收断言
  （#2 行），错误码映射表与网关错误注册表全集对齐由
  `test_error_mapping_covers_full_gateway_registry` 钉死。
