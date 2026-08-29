# LLM Gateway 设计文档

> 阅读指引：本文按**一次真实模型调用的主链路**组织（§3），每个环节回答三件事——发生什么、落在哪个文件、为什么这么设计。跨主链路的核心决策集中 in §4。§8 是行为不变量与里程碑的映射总表。行级实现契约见 `docs/specs/`。

---

## 1. 定位与范围

**定位**：作品集项目——按生产工程标准实现，轻量规模。不承载真实公网流量，但所有机制都按"接真流量时会怎么死"来设计。

**规模假设**：1~5 个调用方（业务 Agent + 个人助手），单实例，峰值 QPS < 10，无多租户。聚合维度全部指**调用方**（见 CONTEXT.md，"租户"一词废弃）。

**非目标（刻意边界，写进 spec 的验收边界而不是含糊带过）**：

1. 多副本部署：限流、熔断、并发计数均为单进程内存实现，水平扩展需迁移到 Redis 等共享存储。
2. 网关只负责模型执行，不维护 Agent Run 状态、不判断任务是否完成——那是 Harness 的职责。

**Why**：定位决定完备的标准。真实生产会逼出多租户、高可用、审计合规，与"轻量"矛盾；纯练习又撑不起"工程化完备"。规模假设是让"轻量"成立的契约：不需要分布式，但调用方多于一个，所以认证、配额、审计必须是真的。

---

## 2. 总体架构

### 2.1 布局

```
src/llm_gateway/
  api/                     # HTTP 入口层：只做协议适配与准入
    chat.py                #   /v1/chat/completions（OpenAI 兼容 + SSE）
    governance.py          #   /v1/models、/v1/traces
    schemas.py             #   请求字段白名单校验
    errors.py              #   OpenAI 风格错误体适配
  services/                # 编排层：业务核心
    invocation.py          #   Run 预算 + 重试 + fallback + 修复 的状态机
    routing.py             #   Fallback 链求值 + 路由理由
    prompt_service.py      #   模板渲染
    trace_service.py       #   记账与 Trace 落库
  core/                    # 横切能力：被依赖，不依赖任何人
    auth.py / ratelimit.py / breaker.py / budget.py
    errors.py              #   错误码注册表（单一事实来源）
    config.py             #   配置加载
  providers/               # 供应商适配层
    base.py                #   Provider Protocol
    openai_compatible.py   #   chat + responses 两种 provider_api 模式
    anthropic_provider.py
    fake.py                #   Fake Adapter（剧本化故障）
  prompt/
    loader.py              #   模板 mtime 惰性热加载
  storage/                 #   SQLAlchemy async + aiosqlite
  observability/           #   metrics 注册 + 日志脱敏
  validation/              #   Validation Profile 注册表（Pydantic 业务规则）
templates/                 # 模板资产 yaml（数据不是代码）
packages/modelport/        # Agent 侧客户端包
tests/                     # unit / contract / live 三层
```

### 2.2 依赖方向（本设计的第一条硬规则）

```
api  ->  services  ->  providers
              \->  core / prompt / storage / observability
```

严格单向。具体禁止：`services/` 不得 import `fastapi`，不得 import 任何供应商 SDK；`providers/` 不知道编排层的存在；`core/` 不依赖任何上层。

**Why**：这是"各模块可独立测试"的落点，不是口号——services 层可以不启动 HTTP、注入 Fake Adapter 直接单测重试/fallback/预算这些最核心的逻辑；新增供应商只动 `providers/` + 配置，对编排零波及。如果这条规则被破（比如 services 里出现 `from fastapi import`），可测试性立刻塌回"必须起整个服务才能测"。

---

## 3. 主链路：一次模型调用的生命周期

```
调用方提交 OpenAI-compatible 请求
                ↓
API Layer 完成鉴权、限流与请求校验        ← review 锚点 ①
                ↓
规范化为内部 ModelRequest
                ↓
解析并冻结 Prompt / Schema / Budget / Trace 上下文   ← review 锚点 ②
                ↓
Router 生成可解释的候选端点               ← review 锚点 ③
                ↓
Adapter 调用具体 Provider
                ↓
普通结果校验，或把上游事件转成统一事件流
                ↓
记录 Token / Cost / TTFT / Latency / Retry / Fallback
                ↓
ModelResponse 或 Stream Terminal Event 回到 Agent Loop  ← review 锚点 ④⑤
```

### 3.1 调用方提交 OpenAI-compatible 请求

- 端点 `POST /v1/chat/completions`（非流式 + SSE 流式），**取代** demo 的 `/v1/llm` 与 `/v1/llm/stream`（迁移完成后删除）。`GET /v1/models` 返回平台模型列表。`GET /v1/traces` 保留——审计是网关自己的域，不属于 OpenAI 协议。
- 字段白名单：`model / messages / stream / temperature / max_tokens / response_format / stream_options`，加两个扩展字段 `prompt: {name, version, variables}` 与 `validation: {name, version}`（openai SDK 的 `extra_body` 通道）。白名单外任何字段 → 400 `unsupported_field`，**在调用模型之前失败**。
- 错误体统一 OpenAI 风格 `{"error": {"message", "type", "code"}}`，SDK 端可正常解析。

**Why**：OpenAI 协议已是事实标准，兼容它意味着存量 SDK / Agent / curl 零改动接入，这是网关存在的第一价值。扩展字段走 `extra_body` 而不是自定义 header（要自造编码）或模型名约定（污染白名单语义与 trace 的模型字段）。

**验收锚点**：M03。

### 3.2 API Layer 完成鉴权、限流与请求校验

多层准入控制，全部在进入编排之前完成：

| 层 | 拦什么 | 时机 | 超限行为 |
|---|---|---|---|
| 鉴权 | 调用方身份（`Authorization: Bearer sk-...` 对调用方表，`hmac.compare_digest` 常数时间比较） | 每请求 | 401 `unauthorized` |
| 全局并发（默认 20 in-flight） | 进程总负载 | 准入时（信号量） | 429 `overloaded` + Retry-After |
| 每供应商并发（默认 10） | 单供应商连接堆积 | 准入时 | 429 `provider_overloaded` |
| 每模型 RPM（令牌桶，默认 60） | 请求速率 | 准入时 | 429 `rate_limited` + Retry-After |
| 每模型 TPM | Token 消耗速率 | **调用后按实际 usage 记账** | 429 `token_budget_exceeded` |
| 熔断（连续 5 次失败 → 开 30s → 半开放行 1 探测） | 故障模型快速失败 | 准入时 | 503 `circuit_open` |

并发计数从准入持有到流式结束（流式请求全程占位）。所有阈值是配置项不是代码常量。

**Why**：LLM 请求是长耗时操作，纯 QPS 限流挡不住连接堆积——100 QPS 限制下，每个请求跑 30 秒，照样堆出 3000 个并发连接把进程和供应商配额同时打穿。TPM 用事后记账而非预估扣减：调用前输出长度未知，按 `max_tokens` 预留会把限流收得过死。

**review 锚点 ①**：外部请求在 `api/schemas.py` 完成白名单校验与规范化，鉴权/准入在 `core/` 的对应模块——外部方言到此为止。

**验收锚点**：M05。

### 3.3 规范化为内部 ModelRequest

OpenAI 方言（camelCase 字段、SSE chunk 结构、错误体）在 API 层一次性翻译为内部 `ModelRequest`。此后全链路只见内部类型。

**Why**：内部类型单一，所有下游（编排、Provider、测试）只需构造一种请求；换 API 方言（比如明天要兼容 Anthropic 风格入口）只改 `api/` 一个目录。

### 3.4 解析并冻结 Prompt / Schema / Budget / Trace 上下文

准入通过后一次性构建不可变的 `RunContext`：

- 渲染完成的 Prompt（`prompt_service` 用 `string.Template` 从模板资产渲染；缺变量 → 400 `missing_prompt_variable`）
- 选定的 `response_format`（结构性 Schema）
- 选定的 Validation Profile（未注册的 name/version → 400 `unknown_validation_profile`）
- Run 预算实例（总尝试上限 4 + 墙钟 timeout，默认 30s）
- request_id、Trace 骨架、价格表版本快照

**Why（本设计最重要的决策之一）**：没有冻结点，"Trace 能定位每次调用用的 Prompt/Schema/价格版本"就是谎话——模板热加载发生在在途回合中间，你永远说不清这次调用用的是旧版还是新版。冻结让每次调用可解释、可复现：所有再尝试（重试/fallback/修复）都在同一个 RunContext 上进行，"修复调用带上原始 Prompt 与失败原因"才有明确所指。

**review 锚点 ②**：`services/invocation.py` 入口处构建，构建后全链路只读。

**验收锚点**：M06（预算）、M07（模板）、M08（校验）。

### 3.5 Router 生成可解释的候选端点

`services/routing.py` 读取平台模型的声明式 Fallback 链（如 `general-primary -> general-backup`），返回候选 `ModelConfig` 列表，并为每个决定生成路由理由（如 `"primary: circuit_open"`、`"primary: retried 3 times, failed with timeout"`），写入 Trace 的 `route_reason`。

**Why**：demo 硬编码 `general-backup` 是玩具写法；更关键的是"指定逻辑模型后能从 Trace 看到最终端点和路由理由"这条不变量——路由不只是行为，还是**可审计的决策记录**。出问题时，"为什么降级了"必须能从 Trace 回答，而不是靠猜。

**review 锚点 ③**：`routing.py` 返回 `(候选列表, 理由列表)`，理由进 Trace。

**验收锚点**：M06、M09。

### 3.6 Adapter 调用具体 Provider

- `providers/base.py` 定义 Protocol：`complete(config, messages, timeout, response_format) -> (content, usage, finish_reason)` 与 `stream(...) -> AsyncIterator[StreamEvent]`。**finish_reason 必须返回**（demo 丢弃了它，导致截断无法识别——这是真实盲点）。
- `openai_compatible.py` 支持 `provider_api: chat | responses` 两种上游传输模式（ModelConfig 字段），结构化输出模式按模型配置：`json_schema`（原生）或 `json_object`（schema 注入 system）。
- `anthropic_provider.py` 原生 Anthropic Messages 实现。
- **所有 SDK 内置重试关闭**（openai: `max_retries=0`，anthropic 同理）。网关是重试的唯一权威。
- `fake.py`：Fake Adapter，按剧本（scenario）复现成功 / 限流 / 超时 / 流中断 / 非法输出。

**Why（多供应商止步于此）**：多供应商的卖点是验证 Provider 抽象的正确性，不是穷举供应商。OpenAI-compatible 内的 chat/Responses 两模式 + Anthropic 原生，已覆盖"协议真的不同"的完整光谱；更深的"全能力映射层"（抹平各家 structured output、工具调用的语义差异）是真实多供应商团队的需求，超出轻量边界。**Why（SDK 重试关闭）**：SDK 自己也重试的话，预算计数器数的是网关视角的尝试，真实上游请求数是 N×倍，预算控制失效。

**review 锚点**：Provider 层是供应商协议差异的唯一墓地。SDK 异常在此映射为错误码注册表中的 `GatewayError`，**永不穿透到响应体**。

**验收锚点**：M04。

### 3.7 校验，或把上游事件转成统一事件流

**非流式 + 结构化输出**，三重关卡：

1. `json.loads` 失败 → `invalid_json`
2. `finish_reason == "length"` → `output_truncated`（截断与"不会写 JSON"是两种病，治法不同）
3. 结构校验：供应商约束（response_format）+ 本地 `jsonschema` 双重 → `schema_validation_failed`
4. 业务校验：Validation Profile（Pydantic `model_validator`，代码注册表）→ `business_validation_failed`

**修复调用**（上限 1 次，消耗 Run 预算）：schema/业务失败 → 携带错误反馈重调同一端点；截断 → 提高 `max_tokens` 重调。仍失败才向调用方报错。**"JSON 合法但不满足业务规则"绝不进入 Agent Loop**。

**Why（双重校验）**：上游约束不可信（供应商可能静默降级），出口必须自己再验一遍。**Why（业务校验是 Pydantic 代码注册表）**：业务规则的形态是开放的（互斥、条件依赖、跨字段计算），声明式规则词汇表迟早不够用然后逼你在 YAML 里发明一门小语言；代码即规则，天然可单测。详见 ADR-0005。

**流式**：上游事件转换为 OpenAI chunk 格式转发；TTFT 在首个内容块时记录。语义铁律（demo 已有，保留）：**首块前**出现可恢复错误可重试或切 fallback；**首块后**失败只发终态错误事件，**绝不重新生成并拼接**——防文本重复。客户端取消：传播取消到下游任务，Trace 落单一 `cancelled` 终态。

**review 锚点 ④**：`invocation.py` 是唯一的状态机——attempt 计数、deadline、终态迁移只发生在一处，保证 `success/failed/cancelled` 三选一恰好一次。

**验收锚点**：M06、M08。

### 3.8 记账：Token / Cost / TTFT / Latency / Retry / Fallback

- 每次上游响应的 usage **恰好记账一次**（含失败尝试的真实消耗——钱花了就要记）。
- 成本按价格表版本计算，Trace 记录 `price_version`。
- Trace 在**终态迁移时落库一次**（SQLite + SQLAlchemy async + aiosqlite），字段 = demo 全量（request_id / timestamp / requested_model / actual_model / prompt_name+version / input+output_tokens / cost_usd / latency_ms / attempts / status / error_code）**新增** caller / final_endpoint / route_reason / ttft_ms / price_version / validation_profile。
- 聚合维度：调用方、模型、Prompt 版本（`/v1/traces` 支持过滤）。

**review 锚点 ⑤**：错误不穿透（provider 层映射 + 错误码注册表封闭集合）、用量不重复记账（每次响应一次、Trace 一次落库）——这两条全部由 Fake Adapter 剧本测试证明。

**验收锚点**：M09。

### 3.9 返回

- 非流式：`ModelResponse`（OpenAI 兼容 chat.completion 结构，含 `id / model / choices / usage`）。
- 流式：SSE，OpenAI chunk 格式，终态后发 `[DONE]`；流内失败发 OpenAI 风格错误事件后终止。
- 一切错误走错误码注册表 → OpenAI 错误体。响应体永远不出现 SDK 异常类名、堆栈、供应商内部信息。

---

## 4. 跨切面核心决策（设计亮点）

1. **Run 预算统一计数器**：重试、Fallback、修复调用共享同一个尝试计数与墙钟 deadline，一个数据结构管住一切"再来一次"。网关是重试唯一权威，SDK 重试全部关闭。→ ADR-0003
2. **RunContext 冻结**：准入时构建不可变上下文，Trace 的可解释性、热加载的安全性、修复调用的确定性都建立在它上面。
3. **可解释路由**：route_reason 随 Trace 落库，降级决策可审计。
4. **唯一终态状态机**：`invocation.py` 单点控制 attempt/deadline/终态，"恰好一次终态迁移"是可断言的不变量，不是祈祷。
5. **错误码注册表**：`core/errors.py` 是错误码的单一事实来源，封闭集合、稳定契约；供应商异常在 provider 层终结。
6. **可测试性三支柱**：单向依赖（services 脱离 HTTP 可测）、Fake Adapter（剧本化故障复现）、respx（SDK 层故障注入）+ live 冒烟（真模型六功能证据）。
7. **ModelPort 防腐层**："Agent 不导入供应商 SDK"是可验收行为（测试断言 agent 代码无 `import openai/anthropic`），不是文档口号。

---

## 5. 数据与配置

| 资产 | 形态 | 说明 |
|---|---|---|
| 平台模型表 | `config/models.yaml` | 逻辑模型 → 供应商模型、base_url、key 环境变量、provider_api、structured_output_mode、fallback 链、限流参数 |
| 调用方表 | `config/callers.yaml` | key + 显示名 + 元数据（不发动态签发，改配置重启生效） |
| 价格表 | `config/prices.yaml` | 带版本号，成本计算快照版本进 Trace |
| Prompt 模板 | `templates/<name>/<version>.yaml` | 版本即文件，mtime 惰性热加载，加载失败保留旧版 |
| Validation Profile | `src/llm_gateway/validation/` 代码注册表 | Pydantic 模型 + name/version |

**Why（模板热加载语义）**：模板是运行资产不是启动资产——线上正在服务时一个坏文件不应该放大为故障，所以"加载失败保留旧版 + error 日志"。

---

## 6. 可观测与安全

**`/metrics`（Prometheus）**：`llm_requests_total{model,status}`、`llm_request_latency_seconds`（直方图）、`llm_tokens_total{model,direction}`、`llm_retries_total{model}`、`llm_rate_limited_total{model}`、`llm_requests_in_flight`、`llm_breaker_state{model}`。每个指标对应一个已有机制，没有为观测而观测。

**`/healthz`**：200 + 版本号。

**日志**：结构化 JSON，含 request_id 与 caller；**永不包含 API Key**（脱敏在 `observability/` 统一出口做），默认不记录 Prompt/消息内容（demo 已如此，保留）。

---

## 7. 工程化

- **工具链**：uv（锁定依赖）、pytest + pytest-asyncio、respx（mock openai SDK 的 httpx 层）、ruff + pyright（`make check` = lint + type + test 一条命令）。
- **测试三层**：unit（services/core，Fake Adapter 直连）→ contract（API 层，httpx ASGI + respx/Fake Adapter 剧本）→ live（`@pytest.mark.live`，真模型冒烟，`make test-live`，CI 默认跳过）。
- **验收哲学**：每个里程碑的验收是**可执行命令证明行为边界**--上游请求计数（如"恰好 4 次而非 4×2"证明无隐藏重试）、前置失败（如"400 且上游请求数==0"证明失败发生在模型调用之前）、终态唯一性（如"终态错误事件只发一次"）--而不是"接口返回 200"。具体命令见 `docs/specs/` 各里程碑。
- **部署**：Dockerfile + compose；SQLite 数据卷挂载；healthcheck 打 `/healthz`。
- **CI**：GitHub Actions，跑同一条 `make check`——保证"本地过 = CI 过"。

---

## 8. 不变量 → 里程碑映射总表

| # | 不变量 | 里程碑 | 验收形式 |
|---|---|---|---|
| 1 | 统一契约、唯一 Provider 入口、校验、重试、fallback、Trace 真实可靠 | M01/M04/M06/M09 | 全量测试套件 |
| 2 | Agent 侧只依赖 ModelPort，不导入供应商 SDK | M12 | import 检查测试 |
| 3 | OpenAI 客户端可调 /v1/chat/completions | M03 | 真 openai SDK 契约测试 |
| 4 | 未支持字段在模型调用前明确失败 | M03 | contract 断言 |
| 5 | Trace 可见最终端点与路由理由 | M06+M09 | trace 字段断言 |
| 6 | 首 Token 前可恢复错误可重试/Fallback | M06 | Fake Adapter 剧本 |
| 7 | 已输出文本后错误不偷偷重新生成拼接 | M06 | Fake Adapter 流中断剧本 |
| 8 | 客户端取消后下游停止、单一 cancelled 终态 | M06 | 取消传播测试 |
| 9 | Structured Output 供应商约束 + 本地双重校验 | M08 | 双重校验测试 |
| 10 | JSON 合法但不满足业务规则不进 Agent Loop | M08 | Profile 测试 |
| 11 | Prompt 缺变量在调用模型前失败 | M07 | contract 断言 |
| 12 | 每次调用可定位 Prompt/Schema/路由/模型/价格版本/尝试次数 | M09 | trace 字段断言 |
| 13 | 日志不含 API Key、默认不记敏感 Prompt | M10 | 日志脱敏测试 |
| 14 | Token/Cost/TTFT/延迟按调用方、模型、Prompt 版本聚合 | M09 | 聚合查询断言 |
| 15 | 重试、Fallback、修复受同一 Run 预算约束 | M06 | 预算剧本测试 |
| 16 | Fake Adapter 稳定复现五类故障 | M04 | Fake Adapter 自身测试 |
| 17 | 限流/熔断单进程边界；不维护 Agent Run 状态 | 边界 | 文档 + 非目标声明 |

### 8.1 六大功能证据矩阵（验收口径：行为证据，不是接口 200）

用户侧验收要求六大功能各有可验证证据，且流式/非流式两种调用形态均正常工作。证据分两层：**契约层**（CI 内确定性证明，Fake Adapter 剧本 + 真 openai SDK 打 ASGI app）与 **live 层**（M12 真模型冒烟）。

| 功能 | 契约层证据（CI） | live 层证据（M12） |
|---|---|---|
| 非流式调用 | M03：真 openai SDK 非流式契约测试 | 非流式调用 + usage 正确 |
| 流式 | M03 SSE 增量；M06 首块语义/不重生成/取消传播 | SSE 增量 + TTFT 进 trace |
| 结构化输出 | M08 双重校验 + 业务规则阻断 + 修复计数 | response_format + 校验 + trace parsed 路径 |
| 模板引用 | M07 渲染/缺变量前置失败/热加载 | prompt 扩展字段渲染 |
| 可观测 | M10 指标精确计数 + 日志脱敏 | /metrics 计数变化 + /v1/traces 新记录 |
| 重试 | M06 预算/退避/fallback 剧本（上游请求计数） | attempts==1（重试语义由契约层剧本证明，真模型失败不可控） |
| 限流 | M05 多层准入行为测试（25 慢请求 ≤20 到上游） | 低 RPM 模型真实触发一次 429 |

**Why（重试的 live 证据弱化）**：真模型的失败时机不可控，无法确定性观测一次完整重试链；重试是网关自己的行为，用 Fake Adapter 剧本反而能给出"上游恰好收到 N 次请求"这种比 live 更强的断言。限流则相反，必须 live 真实触发一次才能证明令牌桶在真请求路径上生效。
