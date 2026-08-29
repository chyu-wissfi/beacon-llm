"""OpenAI Compatible Adapter：集中处理供应商协议与认证细节。

SDK 异常（openai）只允许出现在本模块——这是供应商细节，编排层通过
is_retryable 谓词询问"该不该重试"，自己不感知 SDK 类型。
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import Message, ModelConfig, Usage


class OpenAICompatibleProvider:
    # 实现 OpenAI Compatible Adapter，集中处理供应商协议与认证细节。
    # 将 API Key 保留在 Gateway 内，业务 Agent 无需接触供应商密钥。
    def create_client(self, config: ModelConfig) -> AsyncOpenAI:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise GatewayError("gateway_misconfigured", "Gateway 模型凭据未配置", 503)
        return AsyncOpenAI(api_key=api_key, base_url=config.base_url, max_retries=0)

    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]:
        # 将统一请求转换为 OpenAI Compatible 调用，隔离厂商协议差异。
        request_data: dict[str, Any] = {
            "model": config.provider_model,
            "messages": [message.model_dump() for message in messages],
            "timeout": timeout_seconds,
        }
        if response_schema is not None:
            if config.structured_output_mode == "json_schema":
                request_data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
            else:
                request_data["response_format"] = {"type": "json_object"}
                request_data["messages"] = [
                    {
                        "role": "system",
                        "content": (
                            "只返回一个合法 JSON 对象，必须严格符合下列 JSON Schema，"
                            "不要返回 Markdown 或额外文字："
                            f"{json.dumps(response_schema, ensure_ascii=False)}"
                        ),
                    },
                    *request_data["messages"],
                ]
        completion = await self.create_client(config).chat.completions.create(**request_data)
        content = completion.choices[0].message.content or ""
        usage = completion.usage
        return content, Usage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        # 逐块读取上游响应，为 Gateway 的实时流式代理提供标准增量文本。
        response = await self.create_client(config).chat.completions.create(
            model=config.provider_model,
            messages=[message.model_dump() for message in messages],
            stream=True,
            timeout=timeout_seconds,
        )
        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta.content:
                yield choice.delta.content


def is_retryable(exc: Exception) -> bool:
    # 判定异常是否为可重试的临时故障。openai SDK 异常的 isinstance 属供应商
    # 细节，与内置 TimeoutError/ConnectionError 一并留在此处：重试节奏（次数、
    # 退避、换模型）由编排层决定，这里只回答"这类异常是否值得重试"。
    return isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError, ConnectionError))
