"""跨层共享的协议类型：请求/响应/trace 的 Pydantic 契约与模型配置。

这些类型被 api、services、providers 三层共同引用（Provider 方法签名要用
Message/Usage/ModelConfig，端点签名要用 LLMRequest/LLMResponse/CallTrace），
因此统一放在依赖链最底端的 core/：任何上层都可以 import，而 core 自身只
依赖 pydantic/stdlib（design.md §2.2 依赖方向）。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    # 定义跨模型通用的单条对话消息，隔离供应商消息格式差异。
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class PromptSelection(BaseModel):
    # 只允许调用方选择受控模板及变量，不能提交或覆盖模板正文。
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    variables: dict[str, str] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    # 统一 Gateway 请求协议，并在 HTTP 入口拦截不合法组合和字段。
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=100)
    messages: list[Message] = Field(min_length=1, max_length=100)
    # stream 字段已删除（M04 deferred 清理）：分流只在端点层（api/chat.py）
    # 发生，内部协议从未消费过它；stream + response_schema 互斥检查同属端点层，
    # 原 validator 随字段一并移除。
    response_schema: dict[str, Any] | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    prompt: PromptSelection | None = None
    # M03 白名单放行字段的内部承载（M04 接线，api 层已做过取值约束）：
    # None 表示调用方未指定——provider 不传参，不伪造上游默认值。
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    # stream_options.include_usage 的内部形态：是否向上游请求流式 usage 回传，
    # 并决定 api 层是否在 [DONE] 前附 usage chunk（OpenAI 惯例）。
    include_usage: bool = False
    # response_format={"type":"json_object"}（无 schema）的内部形态：仅开
    # JSON 模式，本地校验无从谈起；json_schema 形态走 response_schema 既有链路。
    json_mode: bool = False


class Usage(BaseModel):
    # 统一输入与输出 Token 统计口径，用于成本和用量治理。
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class LLMResponse(BaseModel):
    # 统一模型调用结果，并作为 FastAPI 响应出口的 Pydantic 校验契约。
    model_config = ConfigDict(extra="forbid")

    request_id: str
    model: str
    content: str
    parsed: dict[str, Any] | list[Any] | None = None
    usage: Usage
    latency_ms: int = Field(ge=0)
    attempts: int = Field(ge=1)
    # M04 起 provider 协议回传的上游终态原因（内部词表 stop/length）。
    # None 仅出现在旧调用路径；output_truncated 关卡的消费在 M06/M08。
    finish_reason: str | None = None


class PromptTemplate(BaseModel):
    # 表示由 Gateway 发布和版本化管理的系统 Prompt 模板资产。
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    system_template: str


class CallTrace(BaseModel):
    # 保存单次调用的模型、Token、成本、延迟与状态，默认不记录文本内容。
    model_config = ConfigDict(extra="forbid")

    request_id: str
    timestamp: datetime
    requested_model: str
    actual_model: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    attempts: int = Field(ge=0)
    status: Literal["success", "failed"]
    error_code: str | None = None


@dataclass(frozen=True)
class RateLimitConfig:
    # 模型级限流参数的运行时载体：M02 只随配置加载流转（core/config.py 负责校验），
    # M05 准入控制才消费；None 表示该维度未声明，不预设任何治理默认值。
    rpm: int | None = None
    tpm: int | None = None
    concurrency: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    # 将平台模型名映射为供应商模型、地址、密钥与能力配置。
    # 前五个字段是 M01 就被 provider/编排层消费的原有面，取值语义不得漂移；
    # 后三个是 M02 配置中心化新增的声明式字段（provider_api/fallback/限流），
    # 随 ModelConfig 一并流转让模型配置只有一个带载者，M02 内无人消费。
    provider_model: str
    base_url: str
    api_key_env: str
    supports_structured_output: bool
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    # provider_api 的取值约束在配置层用 Literal 收口（M04 起 chat / responses
    # 两族）；运行时面放宽为 str，避免新增协议时要同时改两处类型定义。
    provider_api: str = "chat"
    # provider 选择（M04 任务 5）：注册表查表键，配置层 Literal 校验
    # （openai_compatible / anthropic / fake），运行时面同为 str 放宽。
    provider: str = "openai_compatible"
    fallback: tuple[str, ...] = ()
    rate_limit: RateLimitConfig | None = None
