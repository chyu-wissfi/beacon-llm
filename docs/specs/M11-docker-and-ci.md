# M11 - Docker、CI 与工程收口

## 目标

Docker 化部署（SQLite 卷 + healthcheck）；GitHub Actions CI 跑同一条 `make check`；依赖与工具链收口。

## 前置依赖

M10。

## 任务

1. `Dockerfile`：uv 安装依赖、非 root 用户运行、`CMD` 起 uvicorn。
2. `compose.yaml`：服务 + `data/` 卷挂载（SQLite）+ healthcheck 打 `/healthz`。
3. `.github/workflows/ci.yml`：push/PR 触发，跑 `make check`（live 排除）。保证"本地过 = CI 过"。
4. `Makefile` 收口：`make check`（lint + type + test）、`make test-live`、`make run`、`make docker`。
5. `.gitignore`：`data/`、`.venv`、`__pycache__`、`config/local/`。
6. README：一页启动指引（本地 + Docker）、配置文件说明、指向 `docs/design.md`。

## 验收

```bash
make check
docker compose up -d
curl -sf localhost:8000/healthz | jq -e '.version'    # 退出码 0
docker compose exec gateway curl -sf localhost:8000/healthz  # 容器内健康
docker compose down
git ls-files | grep -v '^$' >/dev/null && echo "仓库干净"
```

CI 验收：GitHub Actions 首跑通过（绿勾）。

## 覆盖的不变量

- #1（工程化完备的收口）

## 边界

- 单 stage 部署（无蓝绿/滚动）；无镜像发布流水线（演进项）。
