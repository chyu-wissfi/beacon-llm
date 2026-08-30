# M11 任务 1：部署镜像。
#
# why（uv 安装依赖）：依赖面由 uv.lock 冻结，`uv sync --frozen` 保证镜像内的
# 依赖与本地开发、CI 三方同锁，不存在"各自解析一遍"的版本漂移。
# why（非 root）：容器内进程以普通用户运行是部署侧的底线卫生要求。
# why（curl）：M11 验收命令 `docker compose exec gateway curl -sf .../healthz`
# 要求容器内有 curl；slim 基础镜像不带，显式安装（健康检查本身见 compose.yaml）。
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 gateway \
    && useradd --uid 1000 --gid gateway --create-home gateway

# 先拷依赖清单与锁文件：依赖不变时命中层缓存，源码改动不触发重装。
# --no-install-project：此层只装第三方依赖，项目本体等源码就位后再装。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 运行面三件套：代码（src）、启动资产（config）、运行资产（templates）。
# tests/docs/packages 是开发与交付侧资产，不进运行镜像（.dockerignore 兜底）。
COPY src ./src
COPY config ./config
COPY templates ./templates
RUN uv sync --frozen --no-dev

# data/ 是 SQLite trace 库的落点（compose 挂载卷）：预先建好并交给运行用户，
# 未挂载卷时 storage/engine.py 也会自建目录，行为不因部署形态分叉。
RUN mkdir -p /app/data && chown -R gateway:gateway /app

USER gateway
EXPOSE 8000

# uvicorn 直起 ASGI app：main.py 导入期完成日志装配与路由注册（M10）。
CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
