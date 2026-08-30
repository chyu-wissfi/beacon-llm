"""prompt/loader.py 单测：模板资产加载与 mtime 惰性热加载（M07）。

断言的是行为边界——查询结果、坏文件保留旧版、判变触发条件、日志可观测面；
不触碰内部快照结构。mtime 显式前推（os.utime）：部分文件系统时间戳粒度粗，
同一秒内重写文件若无 mtime 差异就探测不到变更，前推保证用例确定性。
"""

import logging
import os
import time
from pathlib import Path

import pytest

from llm_gateway.prompt.loader import PromptTemplateLoader


def write_template(
    root: Path,
    name: str,
    version: str,
    system_template: str = "你是${product_name}的助手。",
    *,
    raw: str | None = None,
    bump_seconds: int = 10,
) -> Path:
    # 在临时目录落一个模板文件（版本即文件）；raw 非 None 时写入任意（可非法）内容。
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version}.yaml"
    content = raw if raw is not None else (
        f"name: {name}\nversion: {version}\nsystem_template: {system_template}\n"
    )
    # 旧 mtime 必须在写入前读：write_text 本身会重置时间戳。
    previous = path.stat().st_mtime if path.exists() else 0.0
    path.write_text(content, encoding="utf-8")
    # mtime 严格递增：重写同一路径时基于旧值前推，规避同一秒内两次写入
    # mtime 相同导致快照误判未变（判变口径就是 mtime，测试必须比它严格）。
    mtime = max(previous + bump_seconds, time.time() + bump_seconds)
    os.utime(path, (mtime, mtime))
    return path


def test_startup_loads_all_templates(tmp_path: Path) -> None:
    write_template(tmp_path, "alpha", "v1")
    write_template(tmp_path, "beta", "v2")
    loader = PromptTemplateLoader(tmp_path)
    assert loader.get("alpha", "v1") is not None
    assert loader.get("beta", "v2") is not None
    assert loader.get("alpha", "v2") is None


def test_missing_directory_is_empty_registry(tmp_path: Path) -> None:
    # 模板是运行资产：目录缺失不起不来进程，只是查无模板（请求期 400）。
    loader = PromptTemplateLoader(tmp_path / "not_there")
    assert loader.get("anything", "v1") is None


def test_new_version_available_on_next_get(tmp_path: Path) -> None:
    # 热加载：启动后新增文件，下一次查询立即可用，无需重启。
    write_template(tmp_path, "alpha", "v1")
    loader = PromptTemplateLoader(tmp_path)
    assert loader.get("alpha", "v2") is None
    write_template(tmp_path, "alpha", "v2", system_template="第二版${product_name}。")
    template = loader.get("alpha", "v2")
    assert template is not None
    assert template.system_template == "第二版${product_name}。"


def test_content_change_detected_via_mtime(tmp_path: Path) -> None:
    # 改写既有文件正文不改目录项，判变靠文件 mtime（写入时显式前推）。
    write_template(tmp_path, "alpha", "v1", system_template="旧版正文。")
    loader = PromptTemplateLoader(tmp_path)
    write_template(tmp_path, "alpha", "v1", system_template="新版正文。")
    template = loader.get("alpha", "v1")
    assert template is not None
    assert template.system_template == "新版正文。"


def test_no_reload_when_snapshot_unchanged(tmp_path: Path) -> None:
    # 惰性：快照未变则不重载。把文件改成非法内容但把 mtime 还原，
    # 快照判为未变 → 不重解析，继续返回内存里的旧模板。
    path = write_template(tmp_path, "alpha", "v1")
    loader = PromptTemplateLoader(tmp_path)
    original_mtime = path.stat().st_mtime
    path.write_text("::: definitely not a template :::", encoding="utf-8")
    os.utime(path, (original_mtime, original_mtime))
    assert loader.get("alpha", "v1") is not None


def test_bad_file_keeps_old_version_and_logs_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # 坏文件：解析失败保留内存旧版继续服务 + error 日志（含路径与原因）。
    write_template(tmp_path, "alpha", "v1", system_template="旧版正文。")
    loader = PromptTemplateLoader(tmp_path)
    caplog.set_level(logging.ERROR, logger="llm_gateway")
    write_template(tmp_path, "alpha", "v1", raw="name: [unclosed bracket")
    template = loader.get("alpha", "v1")
    assert template is not None
    assert template.system_template == "旧版正文。"
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "alpha" in errors[0].getMessage()


def test_bad_file_logged_once_until_next_change(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # 快照判变后才重载：同一坏文件不随每次查询重复记日志（变更才触发）。
    write_template(tmp_path, "alpha", "v1")
    loader = PromptTemplateLoader(tmp_path)
    caplog.set_level(logging.ERROR, logger="llm_gateway")
    write_template(tmp_path, "alpha", "v1", raw="not: [valid")
    for _ in range(5):
        loader.get("alpha", "v1")
    assert sum(1 for record in caplog.records if record.levelno == logging.ERROR) == 1


def test_bad_file_does_not_block_other_templates(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # 单文件失败不放大为全局故障：其余模板正常加载。
    write_template(tmp_path, "good", "v1")
    loader = PromptTemplateLoader(tmp_path)
    caplog.set_level(logging.ERROR, logger="llm_gateway")
    write_template(tmp_path, "bad", "v1", raw="name: [unclosed")
    assert loader.get("good", "v1") is not None
    assert loader.get("bad", "v1") is None


def test_startup_bad_file_skipped_with_error_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # 启动期同口径：坏文件记 error 日志并跳过，无旧版可保留则缺位。
    write_template(tmp_path, "bad", "v1", raw="name: [unclosed")
    caplog.set_level(logging.ERROR, logger="llm_gateway")
    loader = PromptTemplateLoader(tmp_path)
    assert loader.get("bad", "v1") is None
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_metadata_path_mismatch_treated_as_load_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # 治理：文件元数据与路径坐标不一致按解析失败处理（保留旧版语义同坏文件）。
    write_template(tmp_path, "alpha", "v1")
    loader = PromptTemplateLoader(tmp_path)
    caplog.set_level(logging.ERROR, logger="llm_gateway")
    # 复制文件忘改元数据：路径是 alpha/v2，正文自称 alpha/v1。
    write_template(tmp_path, "alpha", "v2", raw="name: alpha\nversion: v1\nsystem_template: 冒名。\n")
    assert loader.get("alpha", "v2") is None
    assert loader.get("alpha", "v1") is not None  # 旧版不受牵连
    assert any("不一致" in record.getMessage() for record in caplog.records)


def test_extra_field_in_file_rejected(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # PromptTemplate extra=forbid：模板文件多写字段按解析失败，不放行。
    caplog.set_level(logging.ERROR, logger="llm_gateway")
    write_template(
        tmp_path,
        "alpha",
        "v1",
        raw="name: alpha\nversion: v1\nsystem_template: 正文。\nevil_field: 注入\n",
    )
    loader = PromptTemplateLoader(tmp_path)
    assert loader.get("alpha", "v1") is None


def test_deleted_file_drops_template(tmp_path: Path) -> None:
    # 删除文件 = 模板下线：下一次查询即不可用。
    path = write_template(tmp_path, "alpha", "v1")
    loader = PromptTemplateLoader(tmp_path)
    assert loader.get("alpha", "v1") is not None
    path.unlink()
    assert loader.get("alpha", "v1") is None
