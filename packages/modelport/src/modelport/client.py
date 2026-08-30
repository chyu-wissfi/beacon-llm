"""ModelPort 客户端：网关 OpenAI 兼容面的薄封装（M12 任务 1）。

定位（design.md §4.7）：Agent 侧唯一依赖。内部用 openai SDK 说 OpenAI 兼容
协议是实现细节——公开面只有本包自己的类型（Completion / StreamChunk / Usage
与 errors 层级），SDK 符号不再导出、SDK 异常不外泄（异常统一映射为
ModelPortError 层级，`raise ... from None` 连异常链也不携带 SDK 类型）。

关键语义：
- max_retries=0：网关是重试的唯一权威（ADR-0003），SDK 内置重试必须关闭，
  否则网关视角的 Run 预算与真实上游请求数脱钩；
- request_id 透传：非流式取响应 id、流式取 chunk id——网关保证两者与
  Trace 的 request_id 同源（api/chat.py 的 completion_id 传入编排层），
  调用方凭 result.request_id 即可与 /v1/traces 对账；
- prompt / validation 选择项走 extra_body 扩展字段（ADR-0001 通道）。
"""

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import openai

from modelport.errors import (
    AuthenticationError,
    CancelledRequestError,
    GatewayUnavailableError,
    ModelPortError,
    RateLimitedError,
    RequestRejectedError,
    TransportError,
    UpstreamFailedError,
)

# 网关稳定码 -> 异常类（码的语义与分类理由见 errors.py 模块注）。
# 表与 core/errors.py 的封闭集合对齐：新增码未登记时兜底基类，不崩溃。
_ERROR_CLASS_BY_CODE: dict[str, type[ModelPortError]] = {
    "unauthorized": AuthenticationError,
    # 400 类：请求本身的问题，重试无意义。
    "unknown_model": RequestRejectedError,
    "unknown_prompt_template": RequestRejectedError,
    "missing_prompt_variable": RequestRejectedError,
    "unsupported_field": RequestRejectedError,
    "unsupported_combination": RequestRejectedError,
    "structured_output_unsupported": RequestRejectedError,
    "unknown_validation_profile": RequestRejectedError,
    # 历史码（M03 起运行时不再抛出，注册表双冻结保留）：映射照常登记，
    # 保证映射表与网关错误码封闭集合全集对齐。
    "use_stream_endpoint": RequestRejectedError,
    # 429 类：准入拒绝，按建议节奏退避。
    "rate_limited": RateLimitedError,
    "overloaded": RateLimitedError,
    "provider_overloaded": RateLimitedError,
    "token_budget_exceeded": RateLimitedError,
    # 503：网关自身暂不可用。
    "gateway_misconfigured": GatewayUnavailableError,
    "circuit_open": GatewayUnavailableError,
    # 502 类与流内终态错误：上游/输出质量问题。
    "model_unavailable": UpstreamFailedError,
    "invalid_json": UpstreamFailedError,
    "schema_validation_failed": UpstreamFailedError,
    "output_truncated": UpstreamFailedError,
    "business_validation_failed": UpstreamFailedError,
    "upstream_stream_failed": UpstreamFailedError,
    # 499：取消。
    "request_cancelled": CancelledRequestError,
}

# 码缺失/未登记时按 HTTP 状态兜底分类（网关错误体的 status 是稳定口径）。
_ERROR_CLASS_BY_STATUS: dict[int, type[ModelPortError]] = {
    400: RequestRejectedError,
    401: AuthenticationError,
    429: RateLimitedError,
    499: CancelledRequestError,
    502: UpstreamFailedError,
    503: GatewayUnavailableError,
}


@dataclass(frozen=True)
class Usage:
    """用量三键（OpenAI 口径的对外命名）：网关响应的 usage 原样搬运。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class Completion:
    """非流式调用结果：网关 chat.completion 的 ModelPort 视角。

    request_id 与 /v1/traces 的 request_id 同源（网关用响应 id 承载），
    是与 Trace 对账的唯一钥匙。
    """

    request_id: str
    model: str
    content: str
    finish_reason: str
    usage: Usage

    def json(self) -> Any:
        # 结构化输出的便捷入口：content 是已过网关双重校验的 JSON 文本。
        return json.loads(self.content)


@dataclass(frozen=True)
class StreamChunk:
    """流式增量块：chunk 的 id 与 Trace 的 request_id 同源（网关保证）。"""

    request_id: str
    model: str
    delta: str
    finish_reason: str | None


def prompt_ref(name: str, version: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """构造模板选择项（网关 extra_body 的 prompt 扩展字段形态）。"""
    ref: dict[str, Any] = {"name": name, "version": version}
    if variables is not None:
        ref["variables"] = dict(variables)
    return ref


def validation_ref(name: str, version: str) -> dict[str, Any]:
    """构造 Validation Profile 选择项（网关 extra_body 的 validation 扩展字段形态）。"""
    return {"name": name, "version": version}


def _map_sdk_exception(exc: Exception) -> ModelPortError:
    """openai SDK 异常 -> ModelPortError 层级：SDK 形态到此为止。

    取码顺序：SDK 解析出的 code 属性 -> 错误体 error.code（兜底）；
    分类顺序：码映射表 -> HTTP 状态兜底表 -> 基类。流内终态错误事件是
    HTTP 200 上的 openai.APIError（非 APIStatusError），码映射表同样覆盖。
    """
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc)
    status_code = getattr(exc, "status_code", None)
    retry_after: float | None = None
    # SDK 异常对象的 code 缺失时翻错误体（形态 {"error": {...}} 或 error 本体）。
    if code is None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            inner = body.get("error")
            error = inner if isinstance(inner, dict) else body
            if isinstance(error, dict):
                value = error.get("code")
                if isinstance(value, str):
                    code = value
    # 429 的建议重试间隔在响应头上（网关准入拒绝时附带）。
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after")
            if raw is not None:
                retry_after = float(raw)
        except (TypeError, ValueError):
            retry_after = None
    error_class = _ERROR_CLASS_BY_CODE.get(code) if code else None
    if error_class is None and isinstance(status_code, int):
        error_class = _ERROR_CLASS_BY_STATUS.get(status_code)
    if error_class is None:
        # 传输层失败（连接/超时）没有 HTTP 状态；其余未分类形态兜底基类。
        error_class = TransportError if isinstance(exc, openai.APIConnectionError) else ModelPortError
    return error_class(
        message, code=code, status_code=status_code, retry_after=retry_after
    )


class ModelPort:
    """网关接入客户端：Agent 只通过本类调用模型。

    base_url 指向网关根（如 "http://localhost:8000"），ModelPort 自动补齐
    /v1 前缀；api_key 是 callers.yaml 签发的调用方 key（Bearer 鉴权）。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float | None = None,
        # http_client 是测试注入口（ASGI 直连等）：Agent 生产路径不需要传。
        http_client: object | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        kwargs: dict[str, Any] = {
            "base_url": base,
            "api_key": api_key,
            # 网关是重试唯一权威（ADR-0003）：SDK 内置重试一律关闭。
            "max_retries": 0,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = openai.AsyncOpenAI(**kwargs)

    async def aclose(self) -> None:
        """关闭底层连接池（长生命周期客户端退出时调用）。"""
        await self._client.close()

    def _request_kwargs(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        response_format: Mapping[str, Any] | None,
        prompt: Mapping[str, Any] | None,
        validation: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        # OpenAI 标准字段按需透传（None 不传，吃网关默认）；扩展字段走
        # extra_body（ADR-0001）——模板与校验档案的选择项是网关私有协议。
        kwargs: dict[str, Any] = {"model": model, "messages": list(messages), "stream": stream}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = dict(response_format)
        extra_body: dict[str, Any] = {}
        if prompt is not None:
            extra_body["prompt"] = dict(prompt)
        if validation is not None:
            extra_body["validation"] = dict(validation)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        prompt: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Completion:
        """非流式调用：返回 Completion；失败抛 ModelPortError 层级。"""
        kwargs = self._request_kwargs(
            model,
            messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            prompt=prompt,
            validation=validation,
        )
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            raw = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # asyncio.CancelledError 是 BaseException，不受影响
            # from None：SDK 异常类型连异常链都不外泄（防腐层的字面含义）。
            raise _map_sdk_exception(exc) from None
        choice = raw.choices[0]
        usage = raw.usage
        return Completion(
            # 响应 id = 网关的 request_id（api/chat.py：id 复用 request_id）。
            request_id=raw.id,
            model=raw.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage is not None else 0,
                completion_tokens=usage.completion_tokens if usage is not None else 0,
                total_tokens=usage.total_tokens if usage is not None else 0,
            ),
        )

    async def stream(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        prompt: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用：逐块产出 StreamChunk；流内失败抛 ModelPortError 层级。

        与网关语义对齐：首块后的失败是终态（网关不重生成），已收到的增量
        保留在调用方手里，异常只描述"流在这里断了"。
        """
        kwargs = self._request_kwargs(
            model,
            messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,  # 网关铁律：stream 与结构化输出互斥
            prompt=prompt,
            validation=None,  # 同款互斥：业务校验需要完整输出
        )
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            stream_obj = await self._client.chat.completions.create(**kwargs)
            async for raw in stream_obj:
                # usage 块（include_usage）choices 为空：无增量可交付，跳过。
                if not raw.choices:
                    continue
                choice = raw.choices[0]
                delta = choice.delta.content or ""
                if not delta and choice.finish_reason is None:
                    # 空增量且非终态块：没有可交付的语义，不产出噪声。
                    continue
                yield StreamChunk(
                    # chunk id 与 Trace 的 request_id 同源（网关 M12 起保证）。
                    request_id=raw.id,
                    model=raw.model,
                    delta=delta,
                    finish_reason=choice.finish_reason,
                )
        except Exception as exc:
            raise _map_sdk_exception(exc) from None
