"""core/config.py 的单元测试：配置加载（合法 / 缺字段 / 类型错 / env 覆盖）。

全部用例在临时目录构造 YAML，不依赖仓库真实 config/：loader 的行为契约
（fail-fast、错误文本指明文件与字段、env 覆盖语义）与某一版具体取值是
两回事——后者是 catalog 等价性验证的职责，单元测试必须能在仓库 config/
损坏或取值演进时依然成立。

loader 在模块导入时已跑过一次（CONFIG = load_config()），所以这里的用例
一律直接调用 load_config(临时目录)，配合 monkeypatch 隔离环境变量，互不
污染真实进程状态。
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_gateway.core.config import (
    CONFIG,
    ConfigError,
    GatewayConfig,
    load_config,
)
from llm_gateway.core.schemas import ModelConfig, RateLimitConfig

# ---------------------------------------------------------------------------
# 测试配置的构造：内容是测试自有的取值（与仓库 config/ 刻意不同值），
# 断言这些值即可证明"读到的是临时目录的文件"，而不是碰巧与仓库配置同值。
# ---------------------------------------------------------------------------

# _write_config 的内容哨兵：表示"写入该文件的合法默认内容"。
_WRITE_VALID = object()


def _valid_models() -> dict[str, Any]:
    # 合法 models.yaml 的测试形态：primary 带全字段（含 fallback 与限流），
    # backup 不带 rate_limit——同时覆盖"显式声明"与"缺省默认"两条加载路径。
    return {
        "models": {
            "general-primary": {
                "provider_model": "test-primary-provider",
                "base_url": "https://primary.test/v1",
                "api_key_env": "TEST_PRIMARY_KEY",
                "provider_api": "chat",
                "structured_output_mode": "json_object",
                "supports_structured_output": True,
                "fallback": ["general-backup"],
                "rate_limit": {"rpm": 60, "tpm": 200000, "concurrency": 8},
            },
            "general-backup": {
                "provider_model": "test-backup-provider",
                "base_url": "https://backup.test/v1",
                "api_key_env": "TEST_BACKUP_KEY",
                "provider_api": "chat",
                "structured_output_mode": "json_object",
                "supports_structured_output": True,
                "fallback": [],
            },
        }
    }


def _valid_callers() -> dict[str, Any]:
    return {"callers": {"test-caller-key": {"display_name": "测试调用方"}}}


def _valid_prices() -> dict[str, Any]:
    return {
        "version": "2026-01-01",
        "prices": {
            "general-primary": {"input": 2.0, "output": 5.0},
            "general-backup": {"input": 1.5, "output": 4.5},
        },
    }


def _valid_content(filename: str) -> dict[str, Any]:
    if filename == "models.yaml":
        return _valid_models()
    if filename == "callers.yaml":
        return _valid_callers()
    return _valid_prices()


def _write_config(
    root: Path,
    *,
    models: Any = _WRITE_VALID,
    callers: Any = _WRITE_VALID,
    prices: Any = _WRITE_VALID,
) -> None:
    # 内容形态：_WRITE_VALID（合法默认）/ dict（safe_dump 落盘）/ str（原文
    # 写入，用于构造非法 YAML）/ None（不写文件，构造"文件缺失"形态）。
    sources = {"models.yaml": models, "callers.yaml": callers, "prices.yaml": prices}
    for filename, content in sources.items():
        if content is _WRITE_VALID:
            content = _valid_content(filename)
        if content is None:
            continue
        text = content if isinstance(content, str) else yaml.safe_dump(content, allow_unicode=True)
        (root / filename).write_text(text, encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolate_override_env(monkeypatch):
    # 隔离 shell 环境：PRIMARY_*/BACKUP_* 是 loader 的取值来源之一，开发者
    # shell 若恰好设了这些变量，"合法加载"与"未设回落 YAML"的断言就会被
    # 环境污染；统一删掉，让每个用例显式声明自己需要的环境。
    for name in ("PRIMARY_PROVIDER_MODEL", "PRIMARY_BASE_URL", "BACKUP_PROVIDER_MODEL", "BACKUP_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# 合法加载
# ---------------------------------------------------------------------------


def test_load_valid_config_full_shape(tmp_path):
    # 不变量：三个文件全部加载为运行时产物，字段取值逐项落位。
    root = tmp_path / "config"
    root.mkdir()
    _write_config(root)

    config = load_config(root)

    assert set(config.models) == {"general-primary", "general-backup"}
    primary = config.models["general-primary"]
    assert isinstance(primary, ModelConfig)
    assert primary.provider_model == "test-primary-provider"
    assert primary.base_url == "https://primary.test/v1"
    assert primary.api_key_env == "TEST_PRIMARY_KEY"
    assert primary.supports_structured_output is True
    assert primary.structured_output_mode == "json_object"
    assert primary.provider_api == "chat"
    assert primary.fallback == ("general-backup",)  # 文件层 list -> 运行时 tuple
    assert primary.rate_limit == RateLimitConfig(rpm=60, tpm=200000, concurrency=8)

    backup = config.models["general-backup"]
    assert backup.fallback == ()
    assert backup.rate_limit == RateLimitConfig()  # 未声明 -> 全 None，不预设治理默认值

    assert set(config.callers) == {"test-caller-key"}
    assert config.callers["test-caller-key"].display_name == "测试调用方"

    assert config.prices["general-primary"].input == 2.0
    assert config.prices["general-primary"].output == 5.0
    assert config.prices["general-backup"].input == 1.5
    assert config.price_version == "2026-01-01"


def test_import_time_config_loaded_from_default_dir():
    # 不变量：模块导入即完成默认目录加载（fail-fast 正常路径畅通）。测试进程
    # 能 import 到 CONFIG 本身就证明加载成功；只断言产物形态与平台模型名
    # （结构性契约），不断言具体供应商取值，避免与仓库 config/ 内容漂移耦合。
    assert isinstance(CONFIG, GatewayConfig)
    assert set(CONFIG.models) == {"general-primary", "general-backup"}


# ---------------------------------------------------------------------------
# 环境变量覆盖（demo 兼容语义：设置即覆盖，未设回落 YAML）
# ---------------------------------------------------------------------------


def test_env_override_single_field_unset_falls_back_to_yaml(tmp_path, monkeypatch):
    root = tmp_path / "config"
    root.mkdir()
    _write_config(root)
    monkeypatch.setenv("PRIMARY_PROVIDER_MODEL", "env-override-model")

    config = load_config(root)

    # 设置过的字段被覆盖；未设的 BACKUP_* 回落 YAML 值。
    assert config.models["general-primary"].provider_model == "env-override-model"
    assert config.models["general-backup"].provider_model == "test-backup-provider"


def test_env_override_stacks_both_fields_on_same_model(tmp_path, monkeypatch):
    # 回归护栏（Task 1 曾在此翻车）：同一模型的两个覆盖若基于旧对象 replace
    # 会相互丢弃、只留最后一个——两个环境变量必须同时生效。
    root = tmp_path / "config"
    root.mkdir()
    _write_config(root)
    monkeypatch.setenv("PRIMARY_PROVIDER_MODEL", "env-primary-model")
    monkeypatch.setenv("PRIMARY_BASE_URL", "https://env-primary.test/v1")

    config = load_config(root)

    assert config.models["general-primary"].provider_model == "env-primary-model"
    assert config.models["general-primary"].base_url == "https://env-primary.test/v1"


def test_env_override_applies_per_model(tmp_path, monkeypatch):
    # 覆盖按平台模型逐个映射：改 backup 不得波及 primary。
    root = tmp_path / "config"
    root.mkdir()
    _write_config(root)
    monkeypatch.setenv("BACKUP_PROVIDER_MODEL", "env-backup-model")
    monkeypatch.setenv("BACKUP_BASE_URL", "https://env-backup.test/v1")

    config = load_config(root)

    assert config.models["general-backup"].provider_model == "env-backup-model"
    assert config.models["general-backup"].base_url == "https://env-backup.test/v1"
    assert config.models["general-primary"].provider_model == "test-primary-provider"


def test_env_override_empty_string_still_counts_as_set(tmp_path, monkeypatch):
    # demo 语义逐字等价：os.getenv(name, default) 下设置成空串也照样覆盖
    # （Task 1 已实测并定为等价迁移语义），不得"好心"改成 falsy 判定。
    root = tmp_path / "config"
    root.mkdir()
    _write_config(root)
    monkeypatch.setenv("PRIMARY_PROVIDER_MODEL", "")

    config = load_config(root)

    assert config.models["general-primary"].provider_model == ""


# ---------------------------------------------------------------------------
# 坏配置 fail-fast：九种破坏形态（清单复用 Task 1 实测报告第三节），
# 每种都断言错误文本指明文件与字段——启动失败的排障入口就是这一行异常。
# ---------------------------------------------------------------------------


def _without_required_field() -> dict[str, Any]:
    models = _valid_models()
    del models["models"]["general-primary"]["api_key_env"]
    return models


def _with_wrong_type_bool() -> dict[str, Any]:
    # "sure" 不是 Pydantic 宽松模式可收编的布尔串（true/false/yes/no... 才是），
    # 稳定触发"类型错误: Input should be a valid boolean"。
    models = _valid_models()
    models["models"]["general-primary"]["supports_structured_output"] = "sure"
    return models


def _with_unknown_field() -> dict[str, Any]:
    # 字段名拼写错误（provder_model）：extra=forbid 必须在启动期点名拦截。
    models = _valid_models()
    backup = models["models"]["general-backup"]
    backup["provder_model"] = backup.pop("provider_model")
    return models


def _with_dangling_fallback() -> dict[str, Any]:
    models = _valid_models()
    models["models"]["general-primary"]["fallback"] = ["no-such-model"]
    return models


def _with_wrong_type_rate_limit() -> dict[str, Any]:
    models = _valid_models()
    models["models"]["general-primary"]["rate_limit"]["rpm"] = "sixty"
    return models


@pytest.mark.parametrize(
    ("kwargs", "fragments"),
    [
        pytest.param({"models": None}, ["models.yaml", "配置文件不存在"], id="file-missing"),
        pytest.param(
            {"models": _without_required_field()},
            ["models.yaml", "general-primary", "缺少必填字段 api_key_env"],
            id="missing-required-field",
        ),
        pytest.param(
            {"models": _with_wrong_type_bool()},
            ["models.yaml", "general-primary.supports_structured_output", "类型错误"],
            id="wrong-type-bool",
        ),
        pytest.param(
            {"models": _with_unknown_field()},
            ["models.yaml", "general-backup", "含未识别字段 provder_model", "extra=forbid"],
            id="unknown-field",
        ),
        pytest.param(
            {"models": _with_dangling_fallback()},
            ["models.yaml", "general-primary.fallback", "no-such-model"],
            id="dangling-fallback-ref",
        ),
        pytest.param(
            {"models": "models: [unclosed"},
            ["models.yaml", "不是合法 YAML"],
            id="invalid-yaml-syntax",
        ),
        pytest.param(
            {"callers": {"callers": {"key-1": {}}}},
            ["callers.yaml", "key-1", "缺少必填字段 display_name"],
            id="callers-missing-display-name",
        ),
        pytest.param(
            {"prices": {"prices": {"general-primary": {"input": 2.0, "output": 5.0}}}},
            ["prices.yaml", "缺少必填字段 version"],
            id="prices-missing-version",
        ),
        pytest.param(
            {"models": _with_wrong_type_rate_limit()},
            ["models.yaml", "general-primary.rate_limit.rpm", "类型错误"],
            id="wrong-type-rate-limit",
        ),
    ],
)
def test_bad_config_fails_fast_naming_file_and_field(tmp_path, kwargs, fragments):
    # 不变量：任何一种坏配置都让 load_config 抛 ConfigError（启动 fail-fast），
    # 且错误文本同时指明文件与字段/目标。
    root = tmp_path / "config"
    root.mkdir()
    _write_config(root, **kwargs)

    with pytest.raises(ConfigError) as excinfo:
        load_config(root)

    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, message
