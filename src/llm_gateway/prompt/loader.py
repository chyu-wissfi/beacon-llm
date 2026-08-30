"""Prompt 模板资产加载与 mtime 惰性热加载（M07）。

模板是运行资产不是启动资产（design.md §5）：
- 形态：`templates/<name>/<version>.yaml`，版本即文件；路径是模板坐标的唯一
  事实来源，文件内 name/version 元数据必须与路径一致（不一致按解析失败处理，
  资产治理不变量 #1）。
- 热加载：每次 `get()` 先比对目录快照（目录树形态 + 各文件 mtime），变更才
  全量重载——新增/删除/改写三类变更都逃不掉，且未变更时每请求只做 O(文件数)
  的 stat，不重复读盘解析。
- 坏文件语义：单文件解析失败 → 该 (name, version) 保留内存旧版继续服务 +
  error 日志（含文件路径与原因）。线上一个坏文件不应放大为故障；快照已更新，
  同一个坏文件不会每次请求重复记日志，直到文件再次变化。
- 启动期不做 fail-fast：坏文件同样按"记日志 + 跳过"处理（此时无旧版可保留），
  与 config/（启动资产，坏了进程起不来）刻意区分。

依赖边界：本模块属 prompt 层（design.md §2.1 与 core 同层），只允许
stdlib + pydantic + yaml；services/prompt_service 是它的唯一消费方。
"""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from llm_gateway.core.schemas import PromptTemplate

# 与其他模块同名的具名 logger：logging.getLogger 按名单例，日志行为一致。
logger = logging.getLogger("llm_gateway")

# 默认模板目录：仓库根的 templates/（与 config/ 同款约定，从仓库根启动；
# 单测通过构造参数传入临时目录，不依赖 CWD）。
DEFAULT_TEMPLATES_DIR = Path("templates")

TemplateKey = tuple[str, str]

# 目录快照：((相对路径, mtime), ...) 按路径排序，任一变化即触发重载。
_Snapshot = tuple[tuple[str, float], ...]


class PromptTemplateLoader:
    """模板目录的加载、判变与查询入口（每进程一个默认实例：TEMPLATES）。"""

    def __init__(self, root: Path | str = DEFAULT_TEMPLATES_DIR) -> None:
        self.root = Path(root)
        self._templates: dict[TemplateKey, PromptTemplate] = {}
        # 启动全量加载（spec 任务 2）；快照先置空再重载，保证首次必然执行。
        self._snapshot: _Snapshot = ()
        self._reload()

    def get(self, name: str, version: str) -> PromptTemplate | None:
        # 唯一查询入口：判变检查内联在这里，调用方（prompt_service）不感知
        # 热加载的存在——"每次请求检查"由渲染路径天然保证。
        self._maybe_reload()
        return self._templates.get((name, version))

    # ------------------------------------------------------------------
    # 判变与重载
    # ------------------------------------------------------------------

    def _scan(self) -> _Snapshot:
        # 目录树形态 + 各文件 mtime 的有序快照。扫描窗口内文件消失等竞态
        # 直接跳过该条目（下一次请求快照再变化时自然补上），不让 stat 异常
        # 打穿请求路径。
        entries: list[tuple[str, float]] = []
        if self.root.is_dir():
            for path in sorted(self.root.glob("*/*.yaml")):
                try:
                    entries.append((str(path.relative_to(self.root)), path.stat().st_mtime))
                except OSError:
                    continue
        return tuple(entries)

    def _maybe_reload(self) -> None:
        snapshot = self._scan()
        if snapshot == self._snapshot:
            return
        self._snapshot = snapshot
        self._reload()

    def _reload(self) -> None:
        # 全量重载：逐文件解析，坏文件保留该坐标的内存旧版（首次加载时无旧版
        # 则缺位）并记 error 日志；其余文件正常生效。快照在调用侧已更新，
        # 即使解析失败也不回滚——不变更不重试，避免同一坏文件逐请求刷屏。
        loaded: dict[TemplateKey, PromptTemplate] = {}
        if self.root.is_dir():
            for path in sorted(self.root.glob("*/*.yaml")):
                key: TemplateKey = (path.parent.name, path.stem)
                try:
                    loaded[key] = self._parse(path, key)
                except (OSError, ValueError, yaml.YAMLError, ValidationError) as exc:
                    logger.error("Prompt 模板加载失败 %s，保留旧版继续服务: %s", path, exc)
                    old = self._templates.get(key)
                    if old is not None:
                        loaded[key] = old
        self._templates = loaded

    @staticmethod
    def _parse(path: Path, key: TemplateKey) -> PromptTemplate:
        # 文件契约：YAML -> PromptTemplate（extra=forbid 拦多余字段）。
        # 元数据与路径坐标一致性是治理校验：写错版本目录或复制文件忘改元数据，
        # 都在这里按解析失败拦下，不留到请求期造成"查得到的坐标、对不上的正文"。
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        template = PromptTemplate.model_validate(data)
        if (template.name, template.version) != key:
            raise ValueError(
                f"文件元数据 ({template.name}, {template.version}) 与路径坐标 {key} 不一致"
            )
        return template


# 导入即加载：与 core/config.py 的 CONFIG 同款惯例。prompt_service 从这里取
# 默认实例；测试按需构造临时目录实例并 monkeypatch 到 prompt_service。
TEMPLATES = PromptTemplateLoader()
