"""OpenAI Compatible Adapter：集中处理供应商协议与认证细节（M04 任务 2）。

SDK 异常（openai）只允许出现在本模块——provider 层将其映射为 GatewayError
（providers/base.py 的 map_provider_error），编排层不再感知 SDK 类型（M04 起
is_retryable SDK 谓词删除，可重试性由错误码承载）。

按 ModelConfig.provider_api 分派两种上游传输（ADR-0002）：
- "chat"：chat.completions.create（demo 等价迁移 + M04 语义参数接线）
- "responses"：responses.create（Responses API，max_tokens 映射为
  max_output_tokens，结构化输出走 text.format，system 注入走 instructions）

AsyncOpenAI(..., max_retries=0)：网关是重试的唯一权威，SDK 内置重试必须关闭
（design.md §3.6：SDK 自己重试会让真实上游请求数是网关视角的 N 倍）。
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_gateway.core.errors import GATEWAY_MISCONFIGURED, GatewayError
from llm_gateway.core.schemas import Message, ModelConfig, Usage
from llm_gateway.providers.base import (
    FINISH_REASON_LENGTH,
    FINISH_REASON_STOP,
    ContentDelta,
    StreamCompleted,
    map_provider_error,
    normalize_finish_reason,
)

# json_object 模式的 system 注入模板：schema 由网关侧拼进 Prompt，供应商约束
# 缺位时（上游静默降级）出口还有本地 jsonschema 校验兜底（design.md §3.7）。
_JSON_OBJECT_INSTRUCTION = (
    "只返回一个合法 JSON 对象，必须严格符合下列 JSON Schema，"
    "不要返回 Markdown 或额外文字："
)

# Responses 流式事件里网关消费的三类（SDK 3.6.0 实测类型名，其余事件忽略）。
_RESPONSES_DELTA_EVENT = "response.output_text.delta"
_RESPONSES_COMPLETED_EVENT = "response.completed"
_RESPONSES_INCOMPLETE_EVENT = "response.incomplete"


class OpenAICompatibleProvider:
    # 实现 OpenAI Compatible Adapter，集中处理供应商协议与认证细节。
    # 将 API Key 保留在 Gateway 内，业务 Agent 无需接触供应商密钥。
    def create_client(self, config: ModelConfig) -> AsyncOpenAI:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            # 错误码与默认三元组取自注册表（core/errors.py），调用点不写字面量。
            raise GatewayError(GATEWAY_MISCONFIGURED)
        return AsyncOpenAI(api_key=api_key, base_url=config.base_url, max_retries=0)

    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> tuple[str, Usage, str]:
        # 将统一请求转换为 OpenAI Compatible 调用，隔离厂商协议差异。
        # GatewayError 直接放行（含 create_client 的缺凭据），其余异常一律
        # 在此映射，SDK 类型不出本模块。
        client = self.create_client(config)
        try:
            if config.provider_api == "responses":
                return await self._complete_responses(
                    client, config, messages, timeout_seconds, response_schema,
                    temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                )
            return await self._complete_chat(
                client, config, messages, timeout_seconds, response_schema,
                temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
            )
        except GatewayError:
            raise
        except Exception as exc:
            raise map_provider_error(exc) from exc

    async def _complete_chat(
        self,
        client: AsyncOpenAI,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
        *,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> tuple[str, Usage, str]:
        request_data: dict[str, Any] = {
            "model": config.provider_model,
            "messages": [message.model_dump() for message in messages],
            "timeout": timeout_seconds,
        }
        response_format, system_prefix = self._structured_output(
            config, response_schema, json_mode
        )
        if system_prefix is not None:
            request_data["messages"] = [
                {"role": "system", "content": system_prefix},
                *request_data["messages"],
            ]
        if response_format is not None:
            request_data["response_format"] = response_format
        # 白名单接线字段：None 不传参，保持上游默认（不伪造语义）。
        if temperature is not None:
            request_data["temperature"] = temperature
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        completion = await client.chat.completions.create(**request_data)
        choice = completion.choices[0] if completion.choices else None
        content = (choice.message.content if choice else None) or ""
        usage = completion.usage
        finish_reason = normalize_finish_reason(choice.finish_reason if choice else None)
        return content, self._usage(prompt=usage.prompt_tokens if usage else 0, completion=usage.completion_tokens if usage else 0), finish_reason

    async def _complete_responses(
        self,
        client: AsyncOpenAI,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
        *,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> tuple[str, Usage, str]:
        # Responses API：input 是 role/content 平面消息（system 角色合法），
        # json_object 的 schema 注入走 instructions；max_tokens 的 Responses
        # 对应物是 max_output_tokens（语义同为输出上限）。
        params: dict[str, Any] = {
            "model": config.provider_model,
            "input": [{"role": message.role, "content": message.content} for message in messages],
            "timeout": timeout_seconds,
        }
        text_format, system_prefix = self._responses_text_format(
            config, response_schema, json_mode
        )
        if text_format is not None:
            params["text"] = {"format": text_format}
        if system_prefix is not None:
            params["instructions"] = system_prefix
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        response = await client.responses.create(**params)
        content = ""
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []):
                    if getattr(part, "type", None) == "output_text":
                        content += part.text
        usage = response.usage
        finish_reason = self._responses_finish_reason(response.status, response.incomplete_details)
        return (
            content,
            self._usage(prompt=usage.input_tokens if usage else 0, completion=usage.output_tokens if usage else 0),
            finish_reason,
        )

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        include_usage: bool = False,
    ) -> AsyncIterator[ContentDelta | StreamCompleted]:
        # 逐事件读取上游响应，产出内部 tagged 事件流（design.md §3.6 定稿）。
        # 异常映射包住整个迭代：流中途断开（APITimeoutError/APIConnectionError
        # 等）同样不穿透 SDK 类型；GatewayError（含缺凭据）直接放行。
        client = self.create_client(config)
        try:
            if config.provider_api == "responses":
                async for event in self._stream_responses(client, config, messages, timeout_seconds):
                    yield event
            else:
                async for event in self._stream_chat(client, config, messages, timeout_seconds, include_usage=include_usage):
                    yield event
        except GatewayError:
            raise
        except Exception as exc:
            raise map_provider_error(exc) from exc

    async def _stream_chat(
        self,
        client: AsyncOpenAI,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        *,
        include_usage: bool,
    ) -> AsyncIterator[ContentDelta | StreamCompleted]:
        # cast 是运行期恒等：model_dump() 产出 dict[str, Any]，SDK 形参是
        # TypedDict 联合，两者结构等价但类型系统无法证明，用 cast 桥接。
        kwargs: dict[str, Any] = {
            "model": config.provider_model,
            "messages": cast(
                "list[ChatCompletionMessageParam]",
                [message.model_dump() for message in messages],
            ),
            "stream": True,
            "timeout": timeout_seconds,
        }
        # include_usage 是 openai chat 协议的 usage 回传开关；不传时上游不发
        # usage 块，StreamCompleted.usage 保持 None（不向调用方谎报 0）。
        if include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        response = await client.chat.completions.create(**kwargs)
        finish_reason: str | None = None
        usage: Usage | None = None
        async for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                # usage 块（choices 为空）或任一带 usage 的 chunk 都可回填；
                # 后到者覆盖先到者，最终以最后一次为准。
                usage = self._usage(
                    prompt=chunk.usage.prompt_tokens or 0,
                    completion=chunk.usage.completion_tokens or 0,
                )
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            if choice.delta.content:
                yield ContentDelta(choice.delta.content)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        # 上游流自然结束但未给终态块（异常服务器行为）时按 stop 收口：
        # 传输层已完整走完，语义上等价于正常结束。
        yield StreamCompleted(finish_reason=normalize_finish_reason(finish_reason), usage=usage)

    async def _stream_responses(
        self,
        client: AsyncOpenAI,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[ContentDelta | StreamCompleted]:
        params: dict[str, Any] = {
            "model": config.provider_model,
            "input": [{"role": message.role, "content": message.content} for message in messages],
            "timeout": timeout_seconds,
            "stream": True,
        }
        response = await client.responses.create(**params)
        finish_reason: str | None = None
        usage: Usage | None = None
        async for event in response:
            event_type = getattr(event, "type", "")
            if event_type == _RESPONSES_DELTA_EVENT:
                yield ContentDelta(event.delta)
            elif event_type in (_RESPONSES_COMPLETED_EVENT, _RESPONSES_INCOMPLETE_EVENT):
                # usage 在终态事件内嵌的 response 上（Responses 协议原生回传，
                # 无需 include_usage 开关）。
                inner = event.response
                if inner.usage is not None:
                    usage = self._usage(
                        prompt=inner.usage.input_tokens or 0,
                        completion=inner.usage.output_tokens or 0,
                    )
                if event_type == _RESPONSES_INCOMPLETE_EVENT:
                    finish_reason = self._responses_finish_reason("incomplete", inner.incomplete_details)
                else:
                    finish_reason = FINISH_REASON_STOP
        yield StreamCompleted(finish_reason=finish_reason or normalize_finish_reason(None), usage=usage)

    def _structured_output(
        self,
        config: ModelConfig,
        response_schema: dict[str, Any] | None,
        json_mode: bool,
    ) -> tuple[dict[str, Any] | None, str | None]:
        # 结构化输出 -> (chat response_format, system 注入文案)。
        # json_schema：原生约束；json_object：供应商只承诺合法 JSON，schema
        # 注入 system 补齐约束；json_mode（response_format=json_object，无
        # schema）：仅要求合法 JSON，本地校验留给编排层（M06-M08 接线）。
        if response_schema is not None:
            if config.structured_output_mode == "json_schema":
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }, None
            return {"type": "json_object"}, _JSON_OBJECT_INSTRUCTION + json.dumps(
                response_schema, ensure_ascii=False
            )
        if json_mode:
            return {"type": "json_object"}, None
        return None, None

    def _responses_text_format(
        self,
        config: ModelConfig,
        response_schema: dict[str, Any] | None,
        json_mode: bool,
    ) -> tuple[dict[str, Any] | None, str | None]:
        # Responses 的 text.format 形态（SDK 3.6.0 实测参数面）：
        # json_schema = {"type":"json_schema","name","strict","schema"}；
        # json_object = {"type":"json_object"}。system 注入走 instructions
        # （返回值第二位），与 chat 的首条 system 消息注入等价。
        if response_schema is not None:
            if config.structured_output_mode == "json_schema":
                return {
                    "type": "json_schema",
                    "name": "agent_response",
                    "strict": True,
                    "schema": response_schema,
                }, None
            return {"type": "json_object"}, _JSON_OBJECT_INSTRUCTION + json.dumps(
                response_schema, ensure_ascii=False
            )
        if json_mode:
            return {"type": "json_object"}, None
        return None, None

    def _responses_finish_reason(self, status: str | None, incomplete_details: Any) -> str:
        # Responses 终态映射：completed -> stop；incomplete 且原因为
        # max_output_tokens -> length（与 §3.7 截断识别对齐）；其余 incomplete
        # 原因词表外，归一为 stop（normalize 会告警）。
        if status == "incomplete":
            reason = getattr(incomplete_details, "reason", None)
            if reason == "max_output_tokens":
                return FINISH_REASON_LENGTH
            return normalize_finish_reason(reason)
        return FINISH_REASON_STOP

    @staticmethod
    def _usage(*, prompt: int, completion: int) -> Usage:
        # 统一 Usage 口径（pydantic 校验 ge=0）的集中构造点。
        return Usage(input_tokens=max(prompt, 0), output_tokens=max(completion, 0))
