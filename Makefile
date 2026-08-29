# M01 任务 5：质量闸门。
# why：本仓库的验收标准是"可执行命令的输出"（CLAUDE.md 约定），make check 把
# ruff / pyright / pytest 串成一条命令、一个退出码——三关全过才退出 0，任何人
# 与 CI 用同一条命令即可复验，避免"各自跑一半"造成的绿灯错觉。

.PHONY: check test-live

# ruff 检查面 = src + tests，与 pyright 的 include 对齐：
# 根目录已无历史 demo（gateway.py 已删除），src + tests 即全部源码，
# 覆盖这两处即覆盖全部源码面，不存在工具链检查盲区。
check:
	uv run ruff check src tests
	uv run pyright
	uv run pytest -m "not live"

# live 测试依赖真实上游凭据，不能进离线闸门（make check 已用 -m "not live" 排除），
# 这里提供单独入口；当前暂无 live 用例，规则先行存在（design.md 提及 make test-live）。
test-live:
	uv run pytest -m live
