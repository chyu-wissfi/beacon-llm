"""结构化 JSON 日志与脱敏统一出口（M10 任务 3）。

design.md §6：结构化 JSON、含 request_id 与 caller、**永不包含 API Key**、
默认不记录 Prompt/消息内容。不变量 #13 的防线全部收口在本模块：

- ``ScrubbingFilter``：日志过滤器（统一出口），两道保证——
  1. 已知密钥子串替换：调用方 key（callers.yaml 键即 key 本体）与供应商
     key（模型表 api_key_env 指向的环境变量值）启动期登记；运行期可用
     ``register`` 补登记（如密钥轮换前的旧值）。任何字段出现登记值或
     ``sk-`` 形态令牌一律替换为 ``***``；
  2. 敏感字段抹除：消息体/结构化附加字段里出现 ``content`` / ``messages``
     / ``api_key`` / ``authorization`` 键名的，整键抹除——"默认不记消息
     内容"从约定升级为出口强制（现有 trace 日志结构上就不含这些键，
     本层是防未来回归的防线）。
     注意词表不收 "prompt"：trace 的 prompt_name / prompt_version 是
     可定位治理元数据（不变量 #12），不是 Prompt 内容。
- ``JsonFormatter``：单行 JSON（ts / level / logger / message + 结构化
  附加字段），机器可读；异常信息折进 exception 键。
- ``setup_logging``：幂等装配——具名 logger "llm_gateway" 挂本模块
  handler，main 导入期调用一次。

字段面（request_id / caller / status / attempts）不是新发明：调用日志的
唯一消息体是 CallTrace JSON（trace_service.record_trace），上述字段天然
在其中；本模块不重复搬运字段，只保证出口形态安全。
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Final

from llm_gateway.core.config import CONFIG

# 出口强制抹除的键名词表（见模块注的 "prompt" 豁免说明）。
_SENSITIVE_FIELDS: Final[frozenset[str]] = frozenset(
    {"content", "messages", "api_key", "authorization"}
)

# sk- 形态令牌的兜底正则：未登记进注册表的调用方/上游 key 也拦得住。
# 长度下限 8 防止误伤普通文本（如 "sk-learn" 类词汇长度不足不命中）。
_TOKEN_PATTERN: Final = re.compile(r"sk-[A-Za-z0-9_-]{8,}")

# LogRecord 的标准属性面：收集结构化附加字段时排除这些键，其余视为
# 调用方显式注入的结构化字段（logging 惯例：extra={...}）。
_STANDARD_RECORD_ATTRS: Final = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys() | {"message", "asctime"}
)

_MASKED: Final = "***"


class ScrubbingFilter(logging.Filter):
    # 日志脱敏过滤器：挂在网关日志的统一出口（见 setup_logging）。
    # filter 返回恒真（不丢日志），只在原地改写 record 的消息面。

    def __init__(self) -> None:
        super().__init__()
        self._keys: set[str] = set()
        self.register_known_keys()

    def register_known_keys(self) -> None:
        # 启动期登记已知密钥：调用方 key + 环境变量里的供应商 key。
        # 供应商 key 缺失（未配置的供应商）不登记——没有值可泄。
        self._keys.update(CONFIG.callers.keys())
        for model_config in CONFIG.models.values():
            value = os.environ.get(model_config.api_key_env)
            if value:
                self.register(value)

    def register(self, value: str) -> None:
        # 补登记：非空且具区分度（长度 >= 4）才收——过短的字面（如 "a"）
        # 会把正常文本替换成筛子，与其误伤不如让该值走正则兜底。
        if len(value) >= 4:
            self._keys.add(value)

    def filter(self, record: logging.LogRecord) -> bool:
        text = self._scrub_text(record.getMessage())
        structured = self._scrub_mapping(record.__dict__)
        if structured:
            record.structured = structured
        if text != record.getMessage():
            # 消息已被脱敏：就地替换，args 清空防止格式化时再拼回原值。
            record.msg = text
            record.args = ()
        return True

    def _scrub_text(self, text: str) -> str:
        for key in self._keys:
            text = text.replace(key, _MASKED)
        return _TOKEN_PATTERN.sub(_MASKED, text)

    def _scrub_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        # 结构化附加字段（extra=...）的递归脱敏：敏感键整键抹除，
        # 其余值走密钥子串替换。仅当确有附加字段时返回非空结果。
        extra = {key: value for key, value in mapping.items() if key not in _STANDARD_RECORD_ATTRS}
        if not extra:
            return {}
        return _strip_sensitive(extra, self._keys)


def _strip_sensitive(value: Any, keys: set[str]) -> Any:
    # 递归脱敏：字典里敏感键名整键丢弃；字符串做密钥替换；
    # 列表/元组逐项递归。其余类型原样保留（数值、None 无泄密面）。
    if isinstance(value, dict):
        return {
            key: _strip_sensitive(item, keys)
            for key, item in value.items()
            if key not in _SENSITIVE_FIELDS
        }
    if isinstance(value, str):
        for key in keys:
            value = value.replace(key, _MASKED)
        return _TOKEN_PATTERN.sub(_MASKED, value)
    if isinstance(value, list):
        return [_strip_sensitive(item, keys) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_sensitive(item, keys) for item in value)
    return value


class JsonFormatter(logging.Formatter):
    # 单行 JSON 出口：固定骨架 + 结构化附加字段（过滤器已脱敏）。
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        structured = getattr(record, "structured", None)
        if structured:
            payload.update(structured)
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# 模块级单例过滤器：测试可向其登记临时密钥（与网关配置同源语义）。
SCRUBBER: Final = ScrubbingFilter()

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    # 幂等装配：重复调用（测试多轮导入）不叠加 handler。
    # 只挂具名 logger "llm_gateway"：uvicorn/access 等第三方 logger 不受
    # 本网关日志形态约束（它们的敏感面不在本网关责任域）。
    # 过滤器挂在 **logger 级**（而非 handler）：过滤在传播前就地改写
    # record，任何下游 handler（含测试的 caplog）都只能看到已脱敏的记录——
    # "统一出口"的语义就是不依赖 handler 装配位置。
    global _configured
    if _configured:
        return
    gateway_logger = logging.getLogger("llm_gateway")
    gateway_logger.addFilter(SCRUBBER)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    gateway_logger.addHandler(handler)
    gateway_logger.setLevel(level)
    _configured = True
