"""Anthropic Adapter：原生 Messages API（M04 任务 3）。

与 openai_compatible 同构的边界约定：
- SDK 异常（anthropic）只允许出现在本模块——映射为 GatewayError 的唯一出口是
  providers/base.py 的 map_provider_error（429 由 status_code 鸭子类型识别，
  RateLimitError 携带该属性；APITimeoutError/APIConnectionError 归传输故障）。
- `AsyncAnthropic(..., max_retries=0)`：网关是重试的唯一权威，SDK 内置重试
  必须关闭（与 openai_compatible 同一不变量）。

与 OpenAI 族的协议差异集中在 _build_params：
- system 角色不是消息而是顶层参数（messages 里的 role=system 提取合并）；
- **max_tokens 是 Anthropic 必填参数**：调用方未指定时网关默认 4096
  （Controller 裁决值）——不传会让上游直接 400，伪造默认好过调用失败；
- 结构化输出仅支持 json_object（system 注入 schema）；配置了
  structured_output_mode=json_schema 的模型在请求面显式报
  STRUCTURED_OUTPUT_UNSUPPORTED（不静默抹平，ADR-0002 原则）。
"""

import json
import os
from collections.abc import AsyncGenerator
from typing import Any, Final

from anthropic import AsyncAnthropic

from llm_gateway.core.errors import (
    GATEWAY_MISCONFIGURED,
    STRUCTURED_OUTPUT_UNSUPPORTED,
    GatewayError,
)
from llm_gateway.core.schemas import Message, ModelConfig, Usage
from llm_gateway.providers.base import (
    FINISH_REASON_LENGTH,
    FINISH_REASON_STOP,
    ContentDelta,
    StreamCompleted,
    map_provider_error,
    normalize_finish_reason,
)
from llm_gateway.providers.openai_compatible import _JSON_OBJECT_INSTRUCTION

# Anthropic 必填参数 max_tokens 的网关默认值（Controller 裁决）：调用方未指定
# 时的兜底上限。取值不代表治理决策，只是让必填参数有确定出处。
_DEFAULT_MAX_TOKENS: Final[int] = 4096

# json_mode（无 schema）时的 system 注入文案：仅要求合法 JSON 对象，
# 本地校验留给编排层（与 openai_compatible 的 json_mode 分支同语义）。
_JSON_MODE_INSTRUCTION = "只返回一个合法 JSON 对象，不要返回 Markdown 或额外文字："


class AnthropicProvider:
    # 实现 Anthropic Adapter：Messages API 原生传输，接口形态与
    # OpenAICompatibleProvider 对齐（Provider Protocol）。
    def create_client(self, config: ModelConfig) -> AsyncAnthropic:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            # 与 openai_compatible.create_client 同构：缺凭据是配置问题，
            # 错误码与默认三元组取自注册表（core/errors.py）。
            raise GatewayError(GATEWAY_MISCONFIGURED)
        return AsyncAnthropic(api_key=api_key, base_url=config.base_url, max_retries=0)

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
        self._check_structured_output(config, response_schema)
        client = self.create_client(config)
        try:
            params = self._build_params(
                config, messages, timeout_seconds, response_schema,
                temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
            )
            message = await client.messages.create(**params)
        except GatewayError:
            raise
        except Exception as exc:
            raise map_provider_error(exc) from exc
        content = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        usage = message.usage
        return (
            content,
            self._usage(prompt=usage.input_tokens or 0, completion=usage.output_tokens or 0),
            self._map_finish_reason(message.stop_reason),
        )

    def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        # include_usage 是 openai chat 协议的开关；Anthropic 原生回传 usage，
        # 形参收下只为结构匹配 Protocol，实现不消费。
        include_usage: bool = False,
    ) -> AsyncGenerator[ContentDelta | StreamCompleted, None]:
        # 普通 def 返回 async generator（与 Protocol 声明的返回类型对齐，
        # 见 base.py 注释）。流式不携带 response_schema，结构化裁定只在
        # complete 面生效。
        return self._stream(config, messages, timeout_seconds)

    async def _stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncGenerator[ContentDelta | StreamCompleted, None]:
        client = self.create_client(config)
        try:
            # 流式不携带语义参数（Protocol 的 stream 签名无此面），缺省值对齐。
            params = self._build_params(
                config, messages, timeout_seconds, None,
                temperature=None, max_tokens=None, json_mode=False,
            )
            params["stream"] = True
            input_tokens = 0
            output_tokens = 0
            stop_reason: str | None = None
            async for event in await client.messages.create(**params):
                event_type = getattr(event, "type", "")
                if event_type == "message_start":
                    input_tokens = event.message.usage.input_tokens or 0
                elif event_type == "content_block_delta":
                    delta = event.delta
                    if getattr(delta, "type", None) == "text_delta":
                        yield ContentDelta(delta.text)
                elif event_type == "message_delta":
                    # 终态原因与输出用量都在 message_delta 上累积到达。
                    if event.delta.stop_reason is not None:
                        stop_reason = event.delta.stop_reason
                    if event.usage is not None and event.usage.output_tokens is not None:
                        output_tokens = event.usage.output_tokens
            yield StreamCompleted(
                finish_reason=self._map_finish_reason(stop_reason),
                usage=self._usage(prompt=input_tokens, completion=output_tokens),
            )
        except GatewayError:
            raise
        except Exception as exc:
            raise map_provider_error(exc) from exc

    def _check_structured_output(
        self, config: ModelConfig, response_schema: dict[str, Any] | None
    ) -> None:
        # Anthropic 结构化输出裁定：仅 json_object（system 注入）。配置了
        # json_schema 的模型显式拒绝——静默降级会让调用方误以为约束生效。
        if response_schema is not None and config.structured_output_mode == "json_schema":
            raise GatewayError(STRUCTURED_OUTPUT_UNSUPPORTED)

    def _build_params(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
        *,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        system_texts = [m.content for m in messages if m.role == "system"]
        instruction = self._system_instruction(config, response_schema, json_mode)
        if instruction is not None:
            system_texts.append(instruction)
        params: dict[str, Any] = {
            "model": config.provider_model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages if m.role != "system"
            ],
            # Anthropic 必填：未指定时用网关裁决默认值（见 _DEFAULT_MAX_TOKENS）。
            "max_tokens": max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS,
            "timeout": timeout_seconds,
        }
        if system_texts:
            params["system"] = "\n\n".join(system_texts)
        if temperature is not None:
            params["temperature"] = temperature
        return params

    def _system_instruction(
        self,
        config: ModelConfig,
        response_schema: dict[str, Any] | None,
        json_mode: bool,
    ) -> str | None:
        # json_object 结构化：schema 注入 system（供应商约束缺位时的补丁，
        # 与 openai_compatible 的 json_object 分支同思路）；json_mode（无
        # schema）只要求合法 JSON，本地校验留给编排层。
        if response_schema is not None:
            return _JSON_OBJECT_INSTRUCTION + json.dumps(response_schema, ensure_ascii=False)
        if json_mode:
            return _JSON_MODE_INSTRUCTION
        return None

    def _map_finish_reason(self, stop_reason: str | None) -> str:
        # Anthropic 终态映射：end_turn/stop_sequence -> stop；max_tokens ->
        # length（spec 明确，截断识别依据）；其余（tool_use 等）词表外，
        # 归一为 stop（normalize 会告警）。
        if stop_reason in ("end_turn", "stop_sequence"):
            return FINISH_REASON_STOP
        if stop_reason == "max_tokens":
            return FINISH_REASON_LENGTH
        return normalize_finish_reason(stop_reason)

    @staticmethod
    def _usage(*, prompt: int, completion: int) -> Usage:
        # 统一 Usage 口径的集中构造点（与 openai_compatible 同款）。
        return Usage(input_tokens=max(prompt, 0), output_tokens=max(completion, 0))
