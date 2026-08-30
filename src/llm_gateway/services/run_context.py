"""RunContext：一次调用的不可变上下文与统一预算（M06 任务 1、3、8）。

ADR-0003 的单一数据结构落点：重试、fallback、修复共享**同一个** Budget 计数器
与同一个墙钟 deadline——"所有再尝试形式消耗同一预算"只有落成一个对象才可
验收（终态迁移时 attempts 字段就是它的审计值）。

RunContext 本体 frozen：构建点 = 编排入口（invocation 的 call_with_fallback /
stream_with_fallback），构建后全链路只读。可变部分被刻意隔离在两个内聚对象里：
- Budget：计数器与 deadline 的唯一持有者（时钟注入，默认 time.monotonic，与
  breaker/ratelimit 同款约定——单测不真睡）；
- TraceDraft：终态前累积的记账草稿（已观测 usage、TTFT、路由理由），
  finalize 幂等保证"trace 在终态时写一次"（M06 任务 8）。

cancelled/failed 的 usage 缺口是已知且可识别的（spec 边界：流式 usage chunk 在
最后，中途取消观测不到上游已消耗部分）；TraceDraft 只累计**观测到**的 usage，
不伪造。
"""

import time
from dataclasses import dataclass
from typing import Callable, Final
from uuid import uuid4

from llm_gateway.core.schemas import LLMRequest, Message, PromptSelection, Usage
from llm_gateway.services.catalog import PRICE_VERSION
from llm_gateway.services.prompt_service import build_messages
from llm_gateway.services.trace_service import record_trace
from llm_gateway.validation.registry import ValidationProfile, resolve_profile

# ADR-0003 决策 1：总尝试上限 4 次。重试、fallback、修复调用共享此计数器。
RUN_BUDGET_ATTEMPTS: Final = 4

TraceClock = Callable[[], float]


class Budget:
    # 统一预算：一个尝试计数器 + 一个墙钟 deadline（ADR-0003 决策 1）。
    # 协程单线程模型下读改写都在事件循环内同步完成，无需锁。
    def __init__(self, timeout_seconds: float, clock: TraceClock = time.monotonic) -> None:
        self.max_attempts = RUN_BUDGET_ATTEMPTS
        self.spent = 0
        self.clock = clock
        self.deadline = clock() + timeout_seconds

    def expired(self) -> bool:
        # 墙钟判定：deadline 一到，任何"再试一次"都不再发生。
        return self.clock() >= self.deadline

    def try_spend(self) -> bool:
        # 消耗一次尝试：预算耗尽或已超时返回 False（调用方据此进入终态）。
        # 超时检查在这里而非只靠调用方：预算与 deadline 是同一份预算的两面，
        # 判定口径只此一处。
        if self.expired() or self.spent >= self.max_attempts:
            return False
        self.spent += 1
        return True


class TraceDraft:
    # 终态前的记账草稿：usage/TTFT/路由理由随 run 推进累积，finalize 在终态
    # 迁移时调用一次（幂等旗标是"恰好一次"的最后防线——状态机的任何出口都
    # 可能与异常路径重叠，靠约定不靠旗标必然漂移）。
    def __init__(
        self,
        request_id: str,
        requested_model: str,
        prompt: PromptSelection | None,
        caller: str,
        started_at: float,
        clock: TraceClock,
    ) -> None:
        self.request_id = request_id
        self.requested_model = requested_model
        self.prompt = prompt
        self.caller = caller
        self.started_at = started_at
        self.clock = clock
        # 已观测用量累计：成功的尝试 + 修复调用；失败/取消的缺口不伪造。
        self.usage = Usage(input_tokens=0, output_tokens=0)
        self.ttft_ms: int | None = None
        self.route_reasons: list[str] = []
        self.finalized = False

    def observe_usage(self, usage: Usage | None) -> None:
        # 观测到的 usage 才入账：None（上游未回传）静默跳过，不伪造用量。
        if usage is None:
            return
        self.usage = Usage(
            input_tokens=self.usage.input_tokens + usage.input_tokens,
            output_tokens=self.usage.output_tokens + usage.output_tokens,
        )

    def latency_ms(self) -> int:
        return int((self.clock() - self.started_at) * 1000)

    def finalize(
        self,
        status: str,
        actual_model: str | None,
        error_code: str | None = None,
    ) -> None:
        # 唯一终态写入口：success / failed / cancelled 三选一恰好一次（M06 任务 8）。
        # attempts 取 Budget 实时值（计数器终值即审计值，ADR-0003 后果节）。
        if self.finalized:
            return
        self.finalized = True
        record_trace(
            request_id=self.request_id,
            requested_model=self.requested_model,
            actual_model=actual_model,
            prompt=self.prompt,
            usage=self.usage,
            latency_ms=self.latency_ms(),
            attempts=self.budget.spent,
            status=status,  # type: ignore[arg-type]  # 调用点只传三终态字面量
            error_code=error_code,
            caller=self.caller,
            route_reason="; ".join(self.route_reasons) if self.route_reasons else None,
            ttft_ms=self.ttft_ms,
        )

    def bind_budget(self, budget: Budget) -> None:
        # 构建期接线：finalize 要读预算终值。RunContext 构建点负责绑定一次。
        self.budget = budget


@dataclass(frozen=True)
class RunContext:
    # 一次调用的全链路只读上下文（spec 任务 1 的七要素 + 编排需要的语义参数）。
    # budget / trace 是刻意豁免 frozen 的可变内聚体（见模块注）。
    request_id: str
    caller: str
    requested_model: str
    prompt: PromptSelection | None
    messages: list[Message]
    response_schema: dict | None
    json_mode: bool
    temperature: float | None
    max_tokens: int | None
    include_usage: bool
    timeout_seconds: float
    price_version: str
    # M08：业务校验 Profile 登记项（未指定为 None）；质量关卡的业务关消费。
    validation_profile: ValidationProfile | None
    budget: Budget
    trace: TraceDraft


def build_run_context(
    request: LLMRequest,
    caller: str,
    clock: TraceClock = time.monotonic,
) -> RunContext:
    # 构建点 = 编排入口：request_id 在此生成一次（响应 id 与 trace.request_id
    # 可对账），之后全链路只读。渲染后 Prompt（系统消息注入，prompt_service
    # 的唯一渲染出口）在此固化为 messages——构建后全链路只读。
    # Profile 解析在 Run 状态机启动之前：未注册名是 400 类请求错误（与白名单/
    # 能力不符同层），不产生 trace（M08 spec 任务 3）。
    validation_profile = resolve_profile(request.validation)
    started_at = clock()
    request_id = str(uuid4())
    budget = Budget(request.timeout_seconds, clock=clock)
    trace = TraceDraft(
        request_id=request_id,
        requested_model=request.model,
        prompt=request.prompt,
        caller=caller,
        started_at=started_at,
        clock=clock,
    )
    trace.bind_budget(budget)
    return RunContext(
        request_id=request_id,
        caller=caller,
        requested_model=request.model,
        prompt=request.prompt,
        messages=build_messages(request),
        response_schema=request.response_schema,
        json_mode=request.json_mode,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        include_usage=request.include_usage,
        timeout_seconds=request.timeout_seconds,
        price_version=PRICE_VERSION,
        validation_profile=validation_profile,
        budget=budget,
        trace=trace,
    )
