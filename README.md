# Beacon LLM Gateway

[![CI](https://github.com/chyu-wissfi/beacon-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/chyu-wissfi/beacon-llm/actions/workflows/ci.yml)

面向业务 Agent 的 LLM 网关：调用方（Caller）不直连模型供应商，一切模型调用经
网关完成——OpenAI 兼容协议、多供应商路由与降级、准入控制、Run 预算、结构化
输出双重校验、Prompt 模板、Trace 审计与可观测，按生产工程标准实现的轻量单实例。

设计与决策的完整论述见 [`docs/design.md`](docs/design.md)；分里程碑的执行契约
见 [`docs/specs/`](docs/specs/)。本页只做启动指引。

## 快速开始（本地）

前置：[uv](https://docs.astral.sh/uv/)、Python 3.13（`uv` 按 `.python-version` 自动就位）。

```bash
uv sync                                    # 依赖 + 项目本体（按 uv.lock 冻结）
export DEEPSEEK_API_KEY=sk-...             # 主模型上游凭据
export DEEPSEEK_BACKUP_API_KEY=sk-...      # 备用模型上游凭据
export VVEAI_API_KEY=sk-...                # 双协议组合凭据（vve-* 两模型共用）
make run                                   # uvicorn 起服务，端口 8000
```

缺 key 不影响启动——调用到对应模型时返回 503 `gateway_misconfigured`。

验证：

```bash
curl -sf localhost:8000/healthz | jq -e '.version'
curl -sf localhost:8000/v1/models | jq '.data[].id'
curl -sf localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <调用方 key，见 config/callers.yaml>" \
  -H "Content-Type: application/json" \
  -d '{"model": "general-primary", "messages": [{"role": "user", "content": "你好"}]}'
curl -sf localhost:8000/metrics | head            # Prometheus 指标（按调用方/模型/状态计数）
curl -sf "localhost:8000/v1/traces?group_by=caller" | jq  # 审计 Trace 按调用方聚合（也可按 model/prompt_version，支持 status 过滤）
```

## 快速开始（Docker）

```bash
make docker                                # 构建镜像并后台起服务（compose）
curl -sf localhost:8000/healthz | jq -e '.version'
docker compose exec gateway curl -sf localhost:8000/healthz   # 容器内健康
docker compose down
```

- 上游凭据从宿主环境变量透传（`DEEPSEEK_API_KEY` / `DEEPSEEK_BACKUP_API_KEY`）；
- SQLite trace 库落在宿主 `data/`（卷挂载），重启不丢审计；
- healthcheck 打 `/healthz`，容器内进程以非 root 用户运行。

## 双协议组合：OpenAI Responses API 与 Anthropic Messages API

同一套 `POST /v1/chat/completions` 入口，网关按请求里的 `model` 字段路由到
不同协议适配器，调用方无需感知底层鉴权、请求体结构与返回格式差异：

| 平台模型名 | 上游模型 | 协议 | 结构化输出模式 |
|---|---|---|---|
| `vve-gpt-responses` | `gpt-5.6-luna` | OpenAI Responses API（`client.responses.create`） | `json_schema`（原生 `text.format` 约束） |
| `vve-claude-anthropic` | `claude-sonnet-5` | Anthropic Messages API（`client.messages.create`） | `json_object`（schema 注入 system） |

curl 示例（`$KEY` 为 `config/callers.yaml` 里的调用方 key）：

```bash
# 非流式（Responses API 路径）
curl -sf localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "vve-gpt-responses", "messages": [{"role": "user", "content": "你好"}]}'

# 流式 SSE（Anthropic Messages API 路径）：逐块输出，末尾 data: [DONE]
curl -N localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "vve-claude-anthropic", "stream": true, "messages": [{"role": "user", "content": "你好"}]}'

# 结构化输出（Responses API：原生 json_schema 约束 + 本地校验）
curl -sf localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "vve-gpt-responses", "messages": [{"role": "user", "content": "给我订单 D-1001 的决策"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "order", "strict": true,
          "schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "approve": {"type": "boolean"}},
                     "required": ["order_id", "approve"], "additionalProperties": false}}}}'

# 模板引用（渲染在网关侧完成，trace 记录模板坐标）
curl -sf localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "vve-gpt-responses", "messages": [{"role": "user", "content": "只回复一个词：ok"}],
        "prompt": {"name": "knowledge_decision", "version": "v1", "variables": {"product_name": "Beacon"}}}'

# 可观测：token 分类统计 / 延迟 / TTFT / 成本 / 尝试数 / 调用方
curl -sf "localhost:8000/v1/traces?model=vve-gpt-responses" | jq
curl -sf localhost:8000/metrics | grep -E 'llm_(requests|tokens)_total'
```

六大功能点的一键取证脚本（含重试退避与 429 限流的主动演示）：

```bash
uv run python scripts/verify_acceptance.py   # 需 VVEAI_API_KEY；消耗少量真实额度
```

## 配置文件

| 文件 | 作用 |
|---|---|
| `config/models.yaml` | 平台模型表：逻辑模型名 → 供应商坐标、fallback 链、限流参数、准入阈值 |
| `config/callers.yaml` | 调用方表：API Key → 显示名（`Authorization: Bearer` 鉴权） |
| `config/prices.yaml` | 价格表（带版本号）：成本计算快照版本进 Trace |
| `templates/<name>/<version>.yaml` | Prompt 模板资产：版本即文件，运行期 mtime 惰性热加载 |

改配置重启生效；模板是唯一例外（热加载，加载失败保留旧版）。

## Agent 侧接入（ModelPort）

Agent 只依赖 `packages/modelport/`（本地可编辑安装），不导入任何供应商 SDK：

```bash
uv pip install -e packages/modelport
```

```python
import modelport

port = modelport.ModelPort(base_url="http://localhost:8000", api_key="sk-...")

result = await port.complete(
    model="general-primary",
    messages=[{"role": "user", "content": "你好"}],
)
print(result.content, result.request_id)   # request_id 可与 /v1/traces 对账

async for chunk in port.stream(model="general-primary", messages=[...]):
    print(chunk.delta, end="")
```

网关错误码映射为 `ModelPortError` 异常层级（`.code` 是网关稳定错误码），
SDK 异常不外泄；模板（`prompt=`）与校验档案（`validation=`）选择项经
`extra_body` 通道透传。详见 `packages/modelport/README.md`。

## 质量闸门与测试

```bash
make check        # ruff + pyright + pytest（排除 live），一条命令一个退出码
make test-live    # 真模型冒烟（需真实上游 key，CI 不跑）
```

测试三层：`tests/unit`（服务/核心逻辑直测）→ `tests/contract`（API 层，
离线 Fake/拦截）→ `tests/live`（真模型冒烟）。

`make check` 完全离线可复现：上游全部经 Fake Adapter / 请求拦截模拟，
克隆后无需任何真实 key 即可跑通（CI 与本地跑的是同一条命令，"本地过 =
CI 过"）。真模型行为（usage 自洽、TTFT、真实 429）只由 `make test-live`
覆盖，两层的分工理由见 `docs/design.md` §8.1。

## 交付证据与验收

设计 §8 的 17 条行为不变量与六大功能（非流式/流式/结构化输出/模板/
可观测/重试/限流）逐条对应到可执行测试，证据矩阵见
[`docs/specs/acceptance-audit.md`](docs/specs/acceptance-audit.md)
（每条不变量给出契约层与 live 层的测试文件::用例名）。复核方式：

```bash
make check          # 契约层全量（含 ModelPort 隔离证明与错误码映射全集对齐）
make test-live      # live 层真模型冒烟
```

## 文档地图

- [`docs/design.md`](docs/design.md)：设计全文（主链路 + 决策 why + 不变量总表）
- [`docs/adr/`](docs/adr/)：已落锤的硬决策（ADR-0001~0005）
- [`docs/specs/`](docs/specs/)：M01-M12 执行契约与验收命令
- [`docs/specs/acceptance-audit.md`](docs/specs/acceptance-audit.md)：不变量审计表（交付证据矩阵）
- [`CONTEXT.md`](CONTEXT.md)：术语表（项目统一语言的唯一权威）
