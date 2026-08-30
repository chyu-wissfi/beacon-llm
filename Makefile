# M01 任务 5：质量闸门。
# why：本仓库的验收标准是"可执行命令的输出"（CLAUDE.md 约定），make check 把
# ruff / pyright / pytest 串成一条命令、一个退出码——三关全过才退出 0，任何人
# 与 CI 用同一条命令即可复验，避免"各自跑一半"造成的绿灯错觉。

.PHONY: check test-live run docker

# ruff 检查面 = src + tests + packages，与 pyright 的 include 对齐：
# src 是网关、packages/modelport 是 Agent 侧交付包（M12 起入闸门），
# tests 是全部测试源——覆盖这三处即覆盖全部源码面，不存在工具链检查盲区。
check:
	uv run ruff check src tests packages
	uv run pyright
	uv run pytest -m "not live"

# live 测试依赖真实上游凭据，不能进离线闸门（make check 已用 -m "not live" 排除），
# 这里提供单独入口（M12 起有真模型冒烟用例）。
test-live:
	uv run pytest -m live

# 本地起服务（make run）：默认端口 8000，上游凭据从环境变量读取
# （DEEPSEEK_API_KEY / DEEPSEEK_BACKUP_API_KEY，缺 key 是运行时 503 不是启动失败）。
run:
	uv run uvicorn llm_gateway.main:app --host 0.0.0.0 --port 8000

# Docker 部署入口（M11 任务 4）：构建并后台起服务，健康检查与数据卷见
# compose.yaml；验收命令序列见 docs/specs/M11。
docker:
	docker compose up -d --build
