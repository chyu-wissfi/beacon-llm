"""OpenAI 兼容 API 面的请求 schema（M03 任务 1）：字段白名单 + 扩展字段。

design.md §2：OpenAI 方言（字段、错误体、SSE chunk）在 API 层一次性翻译为内部
类型，此后全链路只见内部类型。本模块只承载"接受哪些字段"的白名单职责——
不变量 #4 要求不支持的字段在调用模型之前就明确失败；字段语义翻译
（response_format -> response_schema、temperature/max_tokens 进入 ModelRequest
等）是任务 B 端点适配与 services 层的边界，不在此处预做。

白名单的执行机制：ConfigDict(extra="forbid") 让白名单外字段在解析期即报
extra_forbidden，api/errors.py 的 RequestValidationError 处理器再把它翻译成
400 unsupported_field（M03 spec 任务 1），从而不落在 FastAPI 默认的 422。
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_gateway.core.schemas import Message, PromptSelection, ValidationSelection


class StreamOptions(BaseModel):
    # OpenAI stream_options 目前只有 include_usage 一个键；建模成严格子模型
    # （而非宽松 dict）以延续全库 extra="forbid" 约定——已放行的字段内部混入
    # 未知键同样按"白名单外"拒绝，而不是静默吞掉。语义消费（向上游请求流式
    # usage 回传、[DONE] 前附 usage chunk）在 M04 落地（api/chat.py）。
    model_config = ConfigDict(extra="forbid")

    include_usage: bool | None = None


class ChatCompletionRequest(BaseModel):
    # OpenAI /v1/chat/completions 的请求白名单：7 个 OpenAI 标准字段 + prompt /
    # validation 两个扩展字段（经 openai SDK 的 extra_body 通道提交）。白名单
    # 是封闭集合——未来要支持新字段必须在这里显式加行，而不是放宽 forbid。
    # messages 直接复用 core.Message：当前支持的子集（system/user/assistant +
    # 文本 content）与 OpenAI 基本消息形态一致，无需 API 层另造一份再翻译；
    # 更丰富的 OpenAI 消息形态（content part 列表、tool 角色）随对应能力
    # （M07/M08+）一起进入白名单。
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=100)
    messages: list[Message] = Field(min_length=1, max_length=100)
    stream: bool = False
    # 取值约束沿用 OpenAI 文档面（temperature 0..2、max_tokens >= 1），与
    # OpenAI SDK 客户端侧不设防不同：网关是契约的第一道防线，越界值在
    # 白名单层就 400，而不是透传给上游后由供应商报错。
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    # response_format 的两形态语义在 M04 落地（api/chat._translate_response_format）：
    # json_object 开 JSON 模式（无 schema、无本地校验）；json_schema 提取内层
    # schema 走 response_schema 既有链路。未知 type / 缺 schema 报 400
    # unsupported_field；深层形态校验（字段级约束等）仍后置，白名单层保持宽松 dict。
    response_format: dict[str, Any] | None = None
    stream_options: StreamOptions | None = None
    prompt: PromptSelection | None = None
    validation: ValidationSelection | None = None
