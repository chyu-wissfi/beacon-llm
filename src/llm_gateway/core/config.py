"""配置中心：启动时加载并校验 config/*.yaml，配置坏了进程就起不来（fail-fast）。

设计动机：
- demo 把模型/价格常量散在代码里，改拓扑要改代码重发布；M02 起它们是仓库里的
  YAML 资产，本模块是唯一加载点（单一事实来源），services/catalog 只保留旧导入面。
- 校验用 Pydantic（extra=forbid 拦字段名拼写错误），错误文本必须指明文件与字段
  ——启动失败的排障入口就是这一行异常，不能让人再翻源码对照。
- api_key_env 只存环境变量"名字"，不在此解析值：key 缺失是运行时 503
  gateway_misconfigured（provider 层，demo 语义），fail-fast 只针对配置文件本身
 （文件缺失、缺必填字段、类型错、文件间引用失配）。
- 环境变量覆盖写在 loader 里（PRIMARY_*/BACKUP_*，兼容 demo 用法）：demo 中 env
  覆盖代码默认值，现在默认值来自 YAML，覆盖语义逐字等价（按平台模型逐个映射）。

依赖边界：本模块属 core 层（design.md §2.2），只允许 stdlib + pydantic + yaml。
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_gateway.core.schemas import ModelConfig, RateLimitConfig

# 默认配置目录：仓库根的 config/（CLAUDE.md 的运行约定是从仓库根启动；
# 单测通过参数传入临时目录，不依赖 CWD）。
DEFAULT_CONFIG_DIR = Path("config")


class ConfigError(RuntimeError):
    # 配置加载失败的载体。不复用 GatewayError：那是请求期的稳定错误码契约，
    # 配置错误发生在任何请求之前，没有 HTTP 语义，混用会污染错误码注册表。
    pass


# ---------------------------------------------------------------------------
# 文件契约：YAML -> Pydantic（extra=forbid，字段拼错/多写立即失败）
# ---------------------------------------------------------------------------


class RateLimitEntry(BaseModel):
    # 限流参数占位的文件形态：M02 只校验类型（正整数或未声明），M05 才消费。
    model_config = ConfigDict(extra="forbid")

    rpm: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)


class ModelEntry(BaseModel):
    # 单个平台模型的文件形态：前五个字段与 M01 的 MODEL_CONFIGS 等价迁移，
    # 后几个是 spec 要求新增的声明式占位（provider_api/provider/fallback/限流）。
    model_config = ConfigDict(extra="forbid")

    provider_model: str
    base_url: str
    api_key_env: str
    supports_structured_output: bool
    structured_output_mode: Literal["json_schema", "json_object"]
    # M04 起 openai_compatible 按 provider_api 分派 chat / responses 两族传输。
    provider_api: Literal["chat", "responses"] = "chat"
    # Provider 选择（M04 任务 5）：注册表查表键，Literal 在配置层拦下拼写错误，
    # 不留到请求期才 KeyError。
    provider: Literal["openai_compatible", "anthropic", "fake"] = "openai_compatible"
    fallback: list[str] = Field(default_factory=list)
    rate_limit: RateLimitEntry = Field(default_factory=RateLimitEntry)


class CallerEntry(BaseModel):
    # 调用方条目：M02 只加载（认证在 M05），除结构校验外不做任何语义校验。
    model_config = ConfigDict(extra="forbid")

    display_name: str


class PriceEntry(BaseModel):
    # 单模型牌价：每百万 Token 的美元单价，input/output 都必填。
    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class ModelsFile(BaseModel):
    # models.yaml 顶层：包一层 models 键，与 callers/prices 的文件结构保持同构，
    # 也给未来文件级元数据（如版本字段）留出空间。
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelEntry]


class CallersFile(BaseModel):
    # callers.yaml 顶层：键就是调用方 API Key 本体（design.md §5：key + 显示名）。
    model_config = ConfigDict(extra="forbid")

    callers: dict[str, CallerEntry]


class PricesFile(BaseModel):
    # prices.yaml 顶层：带版本字段（version 进 Trace 属 M09，当前只加载）。
    model_config = ConfigDict(extra="forbid")

    version: str
    prices: dict[str, PriceEntry]


# ---------------------------------------------------------------------------
# 运行时形态：加载产物（与文件契约解耦，供 catalog 接缝与未来里程碑消费）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallerConfig:
    # 运行时的调用方信息；M05 接入鉴权时在此扩展元数据字段。
    display_name: str


@dataclass(frozen=True)
class GatewayConfig:
    # 一次加载出的全部启动配置。models 的值是共享契约 ModelConfig（providers
    # 与编排层的既有签名），新字段随 ModelConfig 流转，模型配置只有一个带载者。
    models: dict[str, ModelConfig]
    callers: dict[str, CallerConfig]
    prices: dict[str, PriceEntry]
    price_version: str


# ---------------------------------------------------------------------------
# 环境变量覆盖：demo 兼容语义，按平台模型逐个映射（不是全局命名约定）
# ---------------------------------------------------------------------------

# (环境变量名, 被覆盖的 ModelConfig 字段)：未设置的环境变量保持 YAML 值。
_ENV_OVERRIDES: dict[str, tuple[tuple[str, str], ...]] = {
    "general-primary": (
        ("PRIMARY_PROVIDER_MODEL", "provider_model"),
        ("PRIMARY_BASE_URL", "base_url"),
    ),
    "general-backup": (
        ("BACKUP_PROVIDER_MODEL", "provider_model"),
        ("BACKUP_BASE_URL", "base_url"),
    ),
}


def _apply_env_overrides(models: dict[str, ModelConfig]) -> None:
    # 用 os.environ.get + is not None 判定"设置过"，与 demo 的 os.getenv(name,
    # default) 语义逐字等价：设置成空串也照样覆盖（等价迁移，不额外加防御）。
    for model_name, pairs in _ENV_OVERRIDES.items():
        if model_name not in models:
            continue
        for env_name, field_name in pairs:
            value = os.environ.get(env_name)
            if value is not None:
                # 必须每次从 dict 里重取最新对象再 replace：frozen dataclass 的
                # replace 返回新对象，若基于循环外缓存的旧对象，同一模型的多个
                # 覆盖会相互丢弃（只留下最后一个）。
                models[model_name] = replace(models[model_name], **{field_name: value})


# ---------------------------------------------------------------------------
# 加载与校验
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(f"{path}: 配置文件不存在（应从仓库根启动，或传入正确的配置目录）")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: 不是合法 YAML: {exc}") from exc


def _config_error(path: Path, exc: ValidationError) -> ConfigError:
    # 把 Pydantic 的英文校验细节翻译成"指明文件与字段"的启动错误文本：
    # 缺字段用 brief 规定句式，未识别字段点名 extra=forbid，其余按类型错误汇报。
    lines: list[str] = []
    for error in exc.errors():
        loc = [str(item) for item in error["loc"]]
        etype = error["type"]
        if etype == "missing" and loc:
            field = loc.pop()
            target = ".".join(loc)
            prefix = f"{path}: {target}" if target else f"{path}:"
            lines.append(f"{prefix} 缺少必填字段 {field}")
        elif etype == "extra_forbidden" and loc:
            field = loc.pop()
            target = ".".join(loc)
            prefix = f"{path}: {target}" if target else f"{path}:"
            lines.append(f"{prefix} 含未识别字段 {field}（extra=forbid，请检查字段名拼写）")
        else:
            target = ".".join(loc)
            body = f"{target} 类型错误: {error['msg']}" if target else f"类型错误: {error['msg']}"
            lines.append(f"{path}: {body}")
    return ConfigError("\n".join(lines))


def _load_models(path: Path) -> dict[str, ModelConfig]:
    try:
        file = ModelsFile.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise _config_error(path, exc) from exc
    models: dict[str, ModelConfig] = {}
    for name, entry in file.models.items():
        rate = entry.rate_limit
        models[name] = ModelConfig(
            provider_model=entry.provider_model,
            base_url=entry.base_url,
            api_key_env=entry.api_key_env,
            supports_structured_output=entry.supports_structured_output,
            structured_output_mode=entry.structured_output_mode,
            provider_api=entry.provider_api,
            provider=entry.provider,
            fallback=tuple(entry.fallback),
            rate_limit=RateLimitConfig(rpm=rate.rpm, tpm=rate.tpm, concurrency=rate.concurrency),
        )
    # 环境变量覆盖发生在结构校验之后：YAML 本身坏了先报 YAML 的错，env 只是取值来源。
    _apply_env_overrides(models)
    # fallback 链的引用完整性：占位字段也要在启动期把"引用了不存在的平台模型"
    # 拦下来，否则问题会拖到 M06 消费时才在请求期爆炸。
    for name, config in models.items():
        for target in config.fallback:
            if target not in models:
                raise ConfigError(f"{path}: {name}.fallback 引用了未定义的平台模型 {target}")
    return models


def _load_callers(path: Path) -> dict[str, CallerConfig]:
    try:
        file = CallersFile.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise _config_error(path, exc) from exc
    return {key: CallerConfig(display_name=entry.display_name) for key, entry in file.callers.items()}


def _load_prices(path: Path) -> tuple[dict[str, PriceEntry], str]:
    try:
        file = PricesFile.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise _config_error(path, exc) from exc
    return file.prices, file.version


def load_config(root: Path | str = DEFAULT_CONFIG_DIR) -> GatewayConfig:
    # 加载并校验三个配置文件，任一失败抛 ConfigError（启动 fail-fast）。
    # 独立成可传 root 的函数是为了可单测：Task 4 的用例直接传临时目录构造
    # 合法/缺字段/类型错的各种配置，不碰真实的仓库 config/。
    root = Path(root)
    models = _load_models(root / "models.yaml")
    callers = _load_callers(root / "callers.yaml")
    prices_path = root / "prices.yaml"
    prices, price_version = _load_prices(prices_path)
    # 交叉校验 models ⊆ prices：M01 时代模型表与价格表硬编码在同一文件、两集合
    # 必然一致；M02 拆成两个 YAML 后"加模型忘补价"成为可配置状态，而 prices 在
    # trace_service.calculate_cost 里今天就按模型名索引消费——失配会在请求期
    # KeyError，主模型的成功调用记不上 trace（花费凭空消失、备用静默接盘）甚至
    # 对外误报 502。必须启动期拦下，别拖到请求期才爆炸（与 fallback 悬空引用
    # 同一标准）。反方向的多余价格条目无行为危害，留待 M05 锚定模型集时裁决。
    for name in models:
        if name not in prices:
            raise ConfigError(f"{prices_path}: 缺少模型 {name} 的价格条目")
    return GatewayConfig(models=models, callers=callers, prices=prices, price_version=price_version)


# 导入即加载：uvicorn 导入 app 的链条必然经过本模块，配置损坏在进程启动阶段
# 就炸出来（fail-fast），而不是等第一个请求才暴露。
CONFIG = load_config()
