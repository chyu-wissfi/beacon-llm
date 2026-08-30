"""调用编排：唯一状态机（M06 重写）。

spec M06 目标原文：RunContext 冻结、统一预算计数器、指数退避、声明式 fallback
链与路由理由、修复调用、截断处理、流式铁律、取消传播、唯一终态。

本层面向 Provider 协议编程（providers/base.py），不 import 任何供应商 SDK；
可重试与否由 provider 层映射后的错误码判定（PROVIDER_RETRYABLE_CODES），
重试节奏（次数、退避、换模型）留在本模块。

统一预算（ADR-0003）：总尝试上限 RUN_BUDGET_ATTEMPTS（4 次），重试 / fallback /
修复共享同一 Budget 计数器与同一墙钟 deadline——所有再尝试形式消耗同一预算，
终态迁移时 attempts 字段就是它的审计值。单模型尝试上限按候选链动态留量：
预算减去本候选之后的候选数（保证链上后续候选至少拿到一次机会，否则
"永远 timeout"会把预算全耗在主模型上，fallback 形同虚设）；无后续候选时
吃满剩余预算（无 fallback 的链也能用完 4 次）。

熔断反馈（M05 口径沿用）：逐尝试、按实际模型向 core/breaker 报告成败——计入失败
的只有 MODEL_UNAVAILABLE（连接/超时/上游 5xx 的映射码）；上游 429（限流非
损坏）不计。熔断查询归属：主模型的放行与否由 M05 准入把关（acquire 持有半开
探测名额），编排层再查会误判 _probe_in_flight——编排层只对 fallback 候选担
allow_request，凡未走到成败记账的路径必须 relinquish_probe（防幽灵探测）。

唯一终态（任务 8）：success / failed / cancelled 三选一恰好一次迁移；trace 由
TraceDraft.finalize 幂等写入（暂写内存，M09 落库）。请求级错误（白名单/能力
不符，400 类）在 RunContext 构建前拒绝，不进状态机、不产生 trace。
"""

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any, Final

from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate
from pydantic import ValidationError as PydanticValidationError

from llm_gateway.core.breaker import get_breaker
from llm_gateway.core.errors import (
    BUSINESS_VALIDATION_FAILED,
    INVALID_JSON,
    MODEL_UNAVAILABLE,
    OUTPUT_TRUNCATED,
    SCHEMA_VALIDATION_FAILED,
    STRUCTURED_OUTPUT_UNSUPPORTED,
    UNKNOWN_MODEL,
    UPSTREAM_STREAM_FAILED,
    ErrorCode,
    GatewayError,
)
from llm_gateway.core.schemas import (
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    Usage,
)
from llm_gateway.providers import PROVIDER_REGISTRY
from llm_gateway.providers.base import (
    FINISH_REASON_LENGTH,
    PROVIDER_RETRYABLE_CODES,
    ContentDelta,
    Provider,
    StreamCompleted,
)
from llm_gateway.services.catalog import MODEL_CONFIGS
from llm_gateway.services.routing import (
    build_chain,
    circuit_open_reason,
    exhausted_reason,
    unsupported_reason,
)
from llm_gateway.services.run_context import RunContext, build_run_context
from llm_gateway.validation.registry import ValidationProfile

# 与 demo 同名的具名 logger：logging.getLogger 按名单例，日志行为等价。
logger = logging.getLogger("llm_gateway")

# 单模型尝试上限的动态留量规则：预算 - 后续候选数（见模块注），无独立常量。

# ADR-0003 决策 3：退避 0.5s 起、每次 ×2、加 0~0.2s 随机抖动（防惊群）。
_BACKOFF_BASE_SECONDS: Final[float] = 0.5
_BACKOFF_JITTER_SECONDS: Final[float] = 0.2
# 截断修复的 max_tokens 翻倍基线：调用方未指定 max_tokens 时的基准（spec 任务 5
# 只说"提高 max_tokens 重调"，未定基准值；翻倍是"提高"的最小解释）。
_TRUNCATION_REPAIR_BASE_TOKENS: Final[int] = 1024

# 编排层面向协议编程，具体 Adapter 按配置驱动的注册表查表装配（M04 任务 5）。


def _get_provider(config: ModelConfig) -> Provider:
    # provider 取值在配置层已被 Literal 收窄（core/config.py），越界值在启动期
    # fail-fast；走到这里的 KeyError 只能是绕过配置中心的编程错误。
    return PROVIDER_REGISTRY[config.provider]


def validate_model(model: str, response_schema: dict[str, Any] | None) -> ModelConfig:
    # 校验模型白名单和结构化能力，阻止不等价的 fallback。
    config = MODEL_CONFIGS.get(model)
    if config is None:
        # 错误码与默认三元组取自注册表（core/errors.py），调用点不写字面量。
        raise GatewayError(UNKNOWN_MODEL)
    if response_schema is not None and not config.supports_structured_output:
        raise GatewayError(STRUCTURED_OUTPUT_UNSUPPORTED)
    return config


async def _backoff_sleep(
    sleep: Callable[[float], Awaitable[None]],
    exc: GatewayError,
    consecutive_failures: int,
) -> None:
    # ADR-0003 决策 3/4：0.5s ×2 + 抖动；上游 Retry-After 有值则优先——那是
    # 上游亲口说的节奏。consecutive_failures 从 0 起计（首次重试睡 0.5s 档）。
    if exc.retry_after is not None:
        delay = exc.retry_after
    else:
        delay = _BACKOFF_BASE_SECONDS * (2**consecutive_failures) + random.uniform(
            0, _BACKOFF_JITTER_SECONDS
        )
    await sleep(delay)


def _format_business_failure(exc: PydanticValidationError) -> str:
    # 业务校验违规项的拼接：逐条规则理由携带进修复反馈（spec 任务 1），
    # 不做更多加工——修复提示词从简是 M06 既有边界，不做复杂反思链。
    return "; ".join(error["msg"] for error in exc.errors())


def _judge_output(
    response_schema: dict[str, Any] | None,
    profile: ValidationProfile | None,
    content: str,
    finish_reason: str,
) -> tuple[ErrorCode | None, dict[str, Any] | list[Any] | None, str]:
    # 校验流水线四层关卡（M08 spec 任务 1）：json.loads -> output_truncated
    # （finish_reason=length）-> jsonschema 本地校验 -> Validation Profile 业务
    # 校验。截断关在两道本地校验关之前：截断输出即便恰好可解析，结构也不完整，
    # 语义准确的修复动作是提高 max_tokens 而非内容反馈。返回 (问题码 | None,
    # parsed, 失败细节)——细节携带进修复反馈（点名违反项）。无 schema 且无
    # profile 时仅截断关生效（M06 既有语义；json_mode 无 schema 不做本地校验）。
    if response_schema is None and profile is None:
        if finish_reason == FINISH_REASON_LENGTH:
            return OUTPUT_TRUNCATED, None, ""
        return None, None, ""
    parsed: dict[str, Any] | list[Any] | None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return INVALID_JSON, None, str(exc)
    if finish_reason == FINISH_REASON_LENGTH:
        return OUTPUT_TRUNCATED, parsed, ""
    if response_schema is not None:
        try:
            validate(instance=parsed, schema=response_schema)
        except JsonSchemaError as exc:
            return SCHEMA_VALIDATION_FAILED, None, exc.message
    if profile is not None:
        # 业务关在结构关之后：能走到这里的输出结构已双重合法，但"每个字段都
        # 对"不等于"合起来对"（ADR-0005 背景）——非法输出绝不进入 Agent Loop。
        try:
            profile.model.model_validate(parsed)
        except PydanticValidationError as exc:
            return BUSINESS_VALIDATION_FAILED, None, _format_business_failure(exc)
    return None, parsed, ""


async def _quality_gates(
    provider: Provider,
    config: ModelConfig,
    ctx: RunContext,
    model_name: str,
    content: str,
    usage: Usage,
    finish_reason: str,
) -> tuple[str, Usage, str, dict[str, Any] | list[Any] | None]:
    # 修复调用（spec 任务 5）：每层关卡失败 -> 携带对应错误反馈重调；
    # finish_reason == length -> 提高 max_tokens 重调。上限 1 次、消耗统一预算（先查预算，预算
    # 见底时不再修复、直接终态）。修复失败才向调用方报错。
    problem, parsed, detail = _judge_output(
        ctx.response_schema, ctx.validation_profile, content, finish_reason
    )
    if problem is None:
        return content, usage, finish_reason, parsed
    if not ctx.budget.try_spend():
        # 预算见底：无修复机会，按原问题终态。
        ctx.trace.finalize("failed", model_name, error_code=problem)
        raise GatewayError(problem)
    # 反馈提示词从简（spec 边界：不做复杂反思链），点名关卡与违反项（M08：
    # 每层失败携带对应错误反馈）。
    if problem == OUTPUT_TRUNCATED:
        feedback = Message(role="user", content="上次输出被截断，请重新输出完整内容")
        repair_max_tokens = (ctx.max_tokens or _TRUNCATION_REPAIR_BASE_TOKENS) * 2
    else:
        feedback = Message(
            role="user",
            content=f"上次输出被网关拒绝（{problem}）：{detail}。请按约束重新输出",
        )
        repair_max_tokens = ctx.max_tokens
    repair_messages = [*ctx.messages, Message(role="assistant", content=content), feedback]
    try:
        content, usage, finish_reason = await provider.complete(
            config,
            repair_messages,
            ctx.timeout_seconds,
            ctx.response_schema,
            temperature=ctx.temperature,
            max_tokens=repair_max_tokens,
            json_mode=ctx.json_mode,
        )
    except GatewayError as exc:
        # 修复自身的传输失败同样是终态（修复调用不重试——上限 1 已含此义）。
        ctx.trace.finalize("failed", model_name, error_code=exc.code)
        raise
    ctx.trace.observe_usage(usage)
    problem, parsed, _detail = _judge_output(
        ctx.response_schema, ctx.validation_profile, content, finish_reason
    )
    if problem is not None:
        # 修复后仍不过关：按原问题终态（修复失败才向调用方报错）。
        ctx.trace.finalize("failed", model_name, error_code=problem)
        raise GatewayError(problem)
    return content, usage, finish_reason, parsed


async def call_with_fallback(
    request: LLMRequest,
    caller: str = "anonymous",
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> LLMResponse:
    # 请求级校验在 RunContext 构建之前：白名单 / 能力不符是 400 类请求错误，
    # Run 状态机尚未开始，不产生 trace（与 M01 以来"请求问题走 HTTP 错误"一致）。
    validate_model(request.model, request.response_schema)
    # 构建点 = 编排入口：之后全链路只读（spec 任务 1）。
    ctx = build_run_context(request, caller, clock)
    try:
        return await _run_chain(ctx, sleep)
    except asyncio.CancelledError:
        # 取消传播（spec 任务 7）：终态 cancelled 而非 failed，然后原样上抛
        # ——取消语义绝不吞。finalize 幂等，内层已落终态时这里是 no-op。
        ctx.trace.finalize("cancelled", None)
        raise


async def _run_chain(
    ctx: RunContext,
    sleep: Callable[[float], Awaitable[None]],
) -> LLMResponse:
    # 非流式状态机主体：候选链 -> 逐候选有限尝试 -> 质量关卡/修复 -> 唯一终态。
    chain = build_chain(ctx.requested_model)
    last_error: GatewayError | None = None
    for index, model_name in enumerate(chain):
        try:
            config = validate_model(model_name, ctx.response_schema)
        except GatewayError:
            # 入口已校验主模型；走到这里的只能是 fallback 候选能力不符
            # （如备用模型不支持 response_schema）——跳过并记理由。
            ctx.trace.route_reasons.append(unsupported_reason(model_name))
            continue
        provider = _get_provider(config)
        breaker = get_breaker(model_name)
        # 主模型不查熔断（准入已把关，见模块注）；fallback 候选担 allow_request。
        if index > 0 and not breaker.allow_request():
            breaker.relinquish_probe()  # 未走到成败记账，归还半开探测名额
            ctx.trace.route_reasons.append(circuit_open_reason(model_name))
            continue
        attempts_on_model = 0
        # 动态留量：本候选最多花掉"预算 - 后续候选数"次尝试。
        attempts_cap = ctx.budget.max_attempts - (len(chain) - index - 1)
        while attempts_on_model < attempts_cap:
            if not ctx.budget.try_spend():
                # 预算/超时见底：全局终态（spec 任务 3，failed model_unavailable）。
                ctx.trace.finalize("failed", None, error_code=MODEL_UNAVAILABLE)
                raise GatewayError(MODEL_UNAVAILABLE) from last_error
            attempts_on_model += 1
            try:
                content, usage, finish_reason = await provider.complete(
                    config,
                    ctx.messages,
                    ctx.timeout_seconds,
                    ctx.response_schema,
                    temperature=ctx.temperature,
                    max_tokens=ctx.max_tokens,
                    json_mode=ctx.json_mode,
                )
            except GatewayError as exc:
                # M04 起 provider 层把 SDK 异常映射为 GatewayError（错误不穿透），
                # 可重试性由错误码承载（PROVIDER_RETRYABLE_CODES：连接/超时/
                # 上游 5xx 的 MODEL_UNAVAILABLE 与 429 的 PROVIDER_OVERLOADED）；
                # 请求/配置类错误（unknown_model / gateway_misconfigured 等）
                # 不在集合内，首次失败即终态，码原样对外（对外码不漂移）。
                last_error = exc
                if exc.code == MODEL_UNAVAILABLE:
                    # 熔断反馈（M05 口径）：传输故障逐尝试计入；429 不计。
                    breaker.record_failure()
                else:
                    # 未走到成败记账（确定性错误不进熔断），归还探测名额。
                    breaker.relinquish_probe()
                if exc.code not in PROVIDER_RETRYABLE_CODES:
                    ctx.trace.finalize("failed", None, error_code=exc.code)
                    raise
                # 可重试：退避后进入下一轮（预算/deadline 由 try_spend 判定）。
                await _backoff_sleep(sleep, exc, attempts_on_model - 1)
                continue
            except asyncio.CancelledError:
                # 取消传播交给入口统一落 cancelled 终态。
                raise
            except Exception:
                # 网关自身的编程错误（provider 层已收编一切上游异常形态）：
                # 不重试直接终态。error_code 记 None——不伪造稳定码，
                # api 层按未捕获异常兜底 500。
                ctx.trace.finalize("failed", None)
                raise
            # 传输成功：熔断闭合/复位；观测到的 usage 入账。
            breaker.record_success()
            ctx.trace.observe_usage(usage)
            # 质量关卡 + 修复调用（上限 1，消耗统一预算）。
            content, usage, finish_reason, parsed = await _quality_gates(
                provider, config, ctx, model_name, content, usage, finish_reason
            )
            response = LLMResponse(
                request_id=ctx.request_id,
                model=model_name,
                content=content,
                parsed=parsed,
                usage=usage,
                latency_ms=ctx.trace.latency_ms(),
                attempts=ctx.budget.spent,
                finish_reason=finish_reason,
            )
            ctx.trace.finalize("success", model_name)
            return response
        # 单模型尝试耗尽但预算尚存：记路由理由，切下一候选。
        ctx.trace.route_reasons.append(
            exhausted_reason(
                model_name,
                attempts_on_model,
                last_error.code if last_error is not None else MODEL_UNAVAILABLE,
            )
        )
    # 链尾仍未成功：统一 model_unavailable 终态（确定性码在上面原码终态）。
    ctx.trace.finalize("failed", None, error_code=MODEL_UNAVAILABLE)
    raise GatewayError(MODEL_UNAVAILABLE) from last_error


async def stream_with_fallback(
    request: LLMRequest,
    caller: str = "anonymous",
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[dict[str, Any]]:
    # 流式状态机（spec 任务 6/7 的铁律载体）：
    # - 首块前：与非流式同款重试/退避/fallback 节奏，调用方无感知；
    # - 首块后：失败只发一个流内终态错误事件，不重新生成（不变量 #7，
    #   已输出的文本绝不偷偷重拼）；备用模型不被调用；
    # - TTFT 在首个内容块用注入时钟记录；
    # - 取消（客户端断开 / asyncio 取消传播）：下游 provider 流随本生成器
    #   关闭而终止（aclosing 保证），trace 单一 cancelled 终态，不落 failed。
    # 产出内部事件流（非线格式），三类事件：
    #   {"type": "content.delta", "delta": <文本增量>, "model": <实际服务模型>}
    #   {"type": "response.completed", "model": <实际服务模型>,
    #    "finish_reason": <词表值>, "usage": <Usage | None>}
    #   {"type": "response.failed", "error": <注册表错误码>}
    # SSE / OpenAI chunk / [DONE] 等线格式由 api 层一次性翻译（design.md §3.3）。
    validate_model(request.model, None)
    ctx = build_run_context(request, caller, clock)
    try:
        async for event in _stream_chain(ctx, sleep):
            yield event
    except (asyncio.CancelledError, GeneratorExit):
        # 终态 cancelled（spec 任务 7）：actual_model 记 None——中途取消的
        # 在途模型与已观测用量同属"观测不到"的缺口（spec 边界）。GeneratorExit
        # 处理中不得 await（finalize 是同步内存写，安全），处理后必须原样上抛。
        ctx.trace.finalize("cancelled", None)
        raise


async def _stream_chain(
    ctx: RunContext,
    sleep: Callable[[float], Awaitable[None]],
) -> AsyncIterator[dict[str, Any]]:
    chain = build_chain(ctx.requested_model)
    last_error: GatewayError | None = None
    emitted = False
    completed: StreamCompleted | None = None
    for index, model_name in enumerate(chain):
        try:
            config = validate_model(model_name, None)
        except GatewayError:
            ctx.trace.route_reasons.append(unsupported_reason(model_name))
            continue
        breaker = get_breaker(model_name)
        if index > 0 and not breaker.allow_request():
            breaker.relinquish_probe()
            ctx.trace.route_reasons.append(circuit_open_reason(model_name))
            continue
        attempts_on_model = 0
        attempts_cap = ctx.budget.max_attempts - (len(chain) - index - 1)
        while attempts_on_model < attempts_cap:
            if not ctx.budget.try_spend():
                break
            attempts_on_model += 1
            try:
                # aclosing 保证取消/异常时关闭 provider 生成器（下游任务取消
                # 的可观测面：Fake Adapter 感知 GeneratorExit/CancelledError）。
                async with aclosing(
                    _get_provider(config).stream(
                        config, ctx.messages, ctx.timeout_seconds, include_usage=ctx.include_usage
                    )
                ) as stream_iter:
                    async for event in stream_iter:
                        if isinstance(event, ContentDelta):
                            if ctx.trace.ttft_ms is None:
                                # TTFT 在首块记录（spec 任务 6）。
                                ctx.trace.ttft_ms = ctx.trace.latency_ms()
                            emitted = True
                            yield {"type": "content.delta", "delta": event.text, "model": model_name}
                        else:
                            completed = event
            except GatewayError as exc:
                last_error = exc
                if exc.code == MODEL_UNAVAILABLE:
                    breaker.record_failure()
                else:
                    breaker.relinquish_probe()
                if emitted:
                    # 首块后失败（不变量 #7）：不重新生成、不切 fallback，
                    # 只发一个流内终态错误事件后终止（M03 约定：错误事件后
                    # 不发 [DONE]，api 层负责翻译）。
                    ctx.trace.finalize("failed", model_name, error_code=UPSTREAM_STREAM_FAILED)
                    yield {"type": "response.failed", "error": UPSTREAM_STREAM_FAILED}
                    return
                if exc.code not in PROVIDER_RETRYABLE_CODES:
                    break
                await _backoff_sleep(sleep, exc, attempts_on_model - 1)
                continue
            except asyncio.CancelledError:
                # 取消由 stream_with_fallback 外层统一落 cancelled 终态。
                raise
            except Exception:
                # 编程错误：与可重试故障同款流内终态（不重试、不切候选）。
                logger.exception("upstream stream failed")
                ctx.trace.finalize("failed", None, error_code=UPSTREAM_STREAM_FAILED)
                yield {"type": "response.failed", "error": UPSTREAM_STREAM_FAILED}
                return
            # 流正常走完：熔断闭合；观测到的 usage 才入账（spec 边界：流式
            # usage chunk 在最后，中途取消/失败观测不到上游已消耗部分，缺口
            # 是已知且可识别的）。
            breaker.record_success()
            ctx.trace.observe_usage(completed.usage if completed is not None else None)
            ctx.trace.finalize("success", model_name)
            yield {
                "type": "response.completed",
                "model": model_name,
                "finish_reason": completed.finish_reason if completed is not None else "stop",
                "usage": completed.usage if completed is not None else None,
            }
            return
        if emitted:
            # 理论不可达（emitted 失败路径在上面已 return）；防御性直接终态。
            ctx.trace.finalize("failed", model_name, error_code=UPSTREAM_STREAM_FAILED)
            yield {"type": "response.failed", "error": UPSTREAM_STREAM_FAILED}
            return
        ctx.trace.route_reasons.append(
            exhausted_reason(
                model_name,
                attempts_on_model,
                last_error.code if last_error is not None else MODEL_UNAVAILABLE,
            )
        )
    # 首块前仍未成功：流内终态错误事件（调用方尚未拿到任何文本，语义安全）。
    logger.exception("upstream stream failed", exc_info=last_error)
    ctx.trace.finalize("failed", None, error_code=UPSTREAM_STREAM_FAILED)
    yield {"type": "response.failed", "error": UPSTREAM_STREAM_FAILED}
