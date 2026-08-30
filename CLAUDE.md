# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An LLM Gateway for business-facing Agents: callers hit the gateway instead of model vendors directly. Currently a working single-file demo (`gateway.py`) that is being restructured into an engineering-complete service (OpenAI-compatible API, multi-provider, admission control, Run budget, persistent traces) per the design docs below.

## Documentation map (read these before non-trivial work)

| 文档 | 作用 | 何时读 |
|---|---|---|
| `CONTEXT.md` | 术语表，项目统一语言的唯一权威（代码/文档/spec 术语冲突时以它为准） | 任何工作开始前；使用或新增术语时 |
| `docs/design.md` | 设计文档：按一次模型调用主链路组织，含每个决策的 why、§8 不变量->里程碑映射 | 修改行为语义、架构、错误码前 |
| `docs/adr/0001-0005` | 已落锤的硬决策（OpenAI 兼容+extra_body、多供应商深度、Run 预算、准入控制、Validation Profile） | 想改这些领域的设计前，先读对应 ADR |
| `docs/specs/M01-M12` | 分步执行契约：每个 milestone 含目标/任务/可执行验收命令 | 执行开发或验收时；按序执行，勿跳步 |
| `docs/specs/acceptance-audit.md` | （M12 产出）不变量审计表 | 最终验收 |

**策略：本文件不重复这些文档的内容。** 设计细节改动应改 design.md/ADR/spec 原文，而不是只在 CLAUDE.md 里记一份会漂移的副本。

## Current state

`gateway.py` 是迁移前的 demo，行为以 `docs/specs/M01` 的契约测试为准（等价迁移先写测试）。迁移（M01）完成后 `gateway.py` 删除，代码进入 `src/llm_gateway/`。在此之前的改动需保持 demo 与 M01 契约测试可写。

## Running (demo, until M01)

```bash
uvicorn gateway:app --reload
```

Required env vars: `DEEPSEEK_API_KEY`, `DEEPSEEK_BACKUP_API_KEY` (missing keys -> `gateway_misconfigured`). Optional: `PRIMARY_PROVIDER_MODEL`, `PRIMARY_BASE_URL`, `BACKUP_PROVIDER_MODEL`, `BACKUP_BASE_URL`.

Dependencies (no lockfile yet): fastapi, uvicorn, openai, pydantic, jsonschema. No test suite/linter/build step yet - all introduced by the specs.

## Conventions

- 执行 specs 时验收标准是**可执行命令的输出**，不是"看起来对了"；每完成一个 milestone 必须实际跑验收命令。
- Pydantic models use `ConfigDict(extra="forbid")` throughout - adding fields requires updating both sides intentionally.
- 术语与 CONTEXT.md 冲突时当场修正，不引入同义词（尤其：聚合维度叫"调用方 Caller"，禁用"租户"）。
- 注释用中文并解释设计动机（why），保持 demo 现有风格。


## 提速策略

编码任务尽量派给subagent，你重点做好任务调度与统筹；但也要做任务分级，不然太慢了。以下任务由你直接实现，不派 subagent，不走brief/报告/审查流程
- (a) 10 行以内且不改变行为的注释/文案/配置修正；
- (b) 你刚审查过的文件上的微调。直接做+跑 make check 即可。其余任务照旧走实现+审查双流程。纯测试文件用轻量审查（只审断言边界与既有用例零改动）。