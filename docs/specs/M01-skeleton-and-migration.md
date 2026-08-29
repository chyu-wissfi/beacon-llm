# M01 - 工程骨架与等价迁移

## 目标

把单文件 `gateway.py` 迁移为 `src/llm_gateway/` 包结构（布局 v2 的骨架），行为完全等价；建立 uv + pytest + make 工具链；删除 `gateway.py`。

**本里程碑不改任何行为**--OpenAI 兼容改造是 M03。等价性靠迁移前写好的契约测试证明。

## 前置依赖

无（第一个里程碑）。

## 任务

1. `uv init`：`pyproject.toml`（依赖：fastapi、uvicorn、openai、anthropic、pydantic、jsonschema、sqlalchemy[asyncio]、aiosqlite、prometheus-client、respx、pytest、pytest-asyncio、ruff、pyright；dev 依赖分组）。
2. 创建目录骨架（`api/ services/ core/ providers/ prompt/ storage/ observability/ validation/`，`tests/unit tests/contract tests/live`），各层先放空 `__init__.py`。
3. **先写契约测试**（`tests/contract/test_demo_semantics.py`，用 respx mock 上游）：覆盖 demo 全部可测语义--白名单拒绝 `unknown_model`；`stream+response_schema` 拒绝；`invalid_json` / `schema_validation_failed` 不触发 fallback；可重试错误 -> 重试后切 `general-backup`；流式首块后失败发 `response.failed` 不重生成；模板缺变量 400；`/v1/traces` 返回记录。
4. 迁移代码到包内（`api/ chat.py` 暂名 `llm.py` 保留 `/v1/llm` 端点，`providers/openai_compatible.py`、`services/`、`core/errors.py` 等），保持行为。
5. `Makefile`：`make check` = ruff + pyright + pytest（排除 live）。`tests/conftest.py` 注册 `live` marker。
6. 确认全绿后 `git rm gateway.py`，`git init` + 首次提交。

## 验收（全部必须实际执行通过）

```bash
make check                                    # 退出码 0
uv run pytest tests/contract -q              # demo 语义契约测试全绿
test ! -f gateway.py && echo "demo 已删除"    # 输出"demo 已删除"
uv run uvicorn llm_gateway.main:app &        # 服务可启动
curl -s localhost:8000/v1/traces | jq 'type' # "array"
```

## 覆盖的不变量

- #1（部分：契约与 Provider 入口等价迁移）

## 边界

- 不引入新功能；不改错误码；不接 SQLite（CALL_TRACES 仍是内存 list，M09 才落库）。
