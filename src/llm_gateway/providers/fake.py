"""Fake Adapter：剧本化复现五类故障的确定性 Provider（M04 任务 4）。

不变量 #16 的载体：同一剧本的行为完全可复现——零随机、零时钟依赖
（timeout 剧本直接抛 GatewayError，不真 sleep），同剧本跑任意多次结果一致。
剧本由调用方构造（`FakeAdapter(scenario=...)`）；注册表里的实例只带默认
Success 剧本（providers/__init__.py），真实使用场景都自建实例。

剧本 -> 行为映射（与 progress.md Controller 裁决逐条对应）：
- success：可控 content/usage/finish_reason；
- rate_limited：PROVIDER_OVERLOADED（retry_after 数据留在剧本对象上供测试
  断言——GatewayError 形态不为此扩字段，裁决明确）；
- timeout：MODEL_UNAVAILABLE（不 sleep、不消耗真实时间）；
- stream_interrupt：已发可配置数量的 ContentDelta 后抛 MODEL_UNAVAILABLE；
- invalid_output：complete 忠实返回坏内容——非法 JSON / 违反 schema 的识别
  是 invocation 三重关卡的职责，provider 不代劳；
- consecutive_then_success：前 N 次抛 MODEL_UNAVAILABLE，第 N+1 次成功
  （计数器是剧本状态，单测单协程无需并发防护）。

`attempts` 计数是 no_hidden_retry 验收的可观测面：每次 complete / stream
调用记一次"上游请求"，断言 3 次失败 + 1 次成功恰好 4 次——provider 层
不存在隐藏重试放大或吞减请求的任何路径。
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llm_gateway.core.errors import MODEL_UNAVAILABLE, PROVIDER_OVERLOADED, GatewayError
from llm_gateway.core.schemas import Message, ModelConfig, Usage
from llm_gateway.providers.base import ContentDelta, StreamCompleted


@dataclass
class Success:
    # 可控成功剧本：三元组的每个分量都由调用方指定。
    content: str = "ok"
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=3, output_tokens=2))
    finish_reason: str = "stop"


@dataclass(frozen=True)
class RateLimited:
    # 上游 429 剧本：映射为 PROVIDER_OVERLOADED。retry_after 留在剧本对象上
    # 供测试断言（裁决：不进异常链、不改 GatewayError 形态）。
    retry_after: float | None = None


@dataclass(frozen=True)
class Timeout:
    # 超时剧本：直接抛 MODEL_UNAVAILABLE，不真 sleep（零时钟依赖）。
    # 无状态字段；dataclass 体必须有语句，用常量承载裁决注释的可寻址锚点。
    stateless: bool = True


@dataclass(frozen=True)
class StreamInterrupt:
    # 流中途断开剧本：先发出指定数量的块，再抛 MODEL_UNAVAILABLE。
    chunks_before_failure: int = 2


@dataclass(frozen=True)
class InvalidOutput:
    # 坏输出剧本：非法 JSON 或违反 schema 的内容原样返回，校验留给编排层。
    content: str = "{not-valid-json"


@dataclass
class ConsecutiveThenSuccess:
    # 前 failures 次失败（模拟 timeout）后成功；raised 记录已失败次数。
    failures: int
    success: Success = field(default_factory=Success)
    raised: int = 0


Scenario = Success | RateLimited | Timeout | StreamInterrupt | InvalidOutput | ConsecutiveThenSuccess

# 坏输出剧本的兜底 usage：内容既然无效，用量语义无从谈起，记 0 不伪造。
_EMPTY_USAGE = Usage(input_tokens=0, output_tokens=0)


class FakeAdapter:
    # 完整实现 Provider Protocol 的确定性 Adapter。
    def __init__(self, scenario: Scenario | None = None) -> None:
        self.scenario: Scenario = scenario if scenario is not None else Success()
        # 收到的请求计数（complete 与 stream 共用）：no_hidden_retry 的断言面。
        self.attempts = 0

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
        # 语义参数（temperature/max_tokens/json_mode 等）对 Fake 无行为含义：
        # 剧本决定一切，收下形参只为结构匹配 Protocol。
        self.attempts += 1
        scenario = self.scenario
        if isinstance(scenario, Success):
            return scenario.content, scenario.usage, scenario.finish_reason
        if isinstance(scenario, InvalidOutput):
            # 忠实返回坏内容：三重关卡（json/schema/截断）在 invocation 触发，
            # provider 代劳校验会让故障复现失真。
            return scenario.content, _EMPTY_USAGE, "stop"
        if isinstance(scenario, RateLimited):
            raise GatewayError(PROVIDER_OVERLOADED)
        if isinstance(scenario, ConsecutiveThenSuccess):
            if scenario.raised < scenario.failures:
                scenario.raised += 1
                raise GatewayError(MODEL_UNAVAILABLE)
            return scenario.success.content, scenario.success.usage, scenario.success.finish_reason
        # Timeout / StreamInterrupt：非流式调用同样是直接失败。
        raise GatewayError(MODEL_UNAVAILABLE)

    def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        include_usage: bool = False,
    ) -> AsyncIterator[ContentDelta | StreamCompleted]:
        # 与 Protocol 声明一致：普通 def 返回 async generator。
        return self._stream()

    async def _stream(self) -> AsyncIterator[ContentDelta | StreamCompleted]:
        self.attempts += 1
        scenario = self.scenario
        if isinstance(scenario, StreamInterrupt):
            for index in range(scenario.chunks_before_failure):
                yield ContentDelta(f"chunk-{index}")
            raise GatewayError(MODEL_UNAVAILABLE)
        if isinstance(scenario, ConsecutiveThenSuccess):
            if scenario.raised < scenario.failures:
                scenario.raised += 1
                raise GatewayError(MODEL_UNAVAILABLE)
            scenario = scenario.success
        if isinstance(scenario, Success):
            if scenario.content:
                yield ContentDelta(scenario.content)
            yield StreamCompleted(scenario.finish_reason, scenario.usage)
            return
        if isinstance(scenario, InvalidOutput):
            yield ContentDelta(scenario.content)
            yield StreamCompleted("stop", _EMPTY_USAGE)
            return
        if isinstance(scenario, RateLimited):
            raise GatewayError(PROVIDER_OVERLOADED)
        raise GatewayError(MODEL_UNAVAILABLE)
