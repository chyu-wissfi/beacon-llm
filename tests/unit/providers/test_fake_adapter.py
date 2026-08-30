"""FakeAdapter 单测（M04 任务 6）：剧本化故障复现的确定性验收。

三类断言面：
1. 六剧本各一：行为与 progress.md 裁决的剧本->异常映射逐条对应；
2. `-k deterministic`：同一剧本跑 10 次结果完全一致（不变量 #16 的验收命令）；
3. `-k no_hidden_retry`：剧本"连续 3 次失败后成功"，断言上游恰好收到
   3+1=4 次请求——provider 层不存在隐藏重试放大/吞减请求的任何路径。

Fake 不经任何传输层：剧本即行为，断言面是 GatewayError.code / 事件序列 /
计数器，不钉实现细节。
"""

from dataclasses import replace

import pytest

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import Message, ModelConfig, Usage
from llm_gateway.providers.base import ContentDelta, StreamCompleted
from llm_gateway.providers.fake import (
    ConsecutiveThenSuccess,
    FakeAdapter,
    InvalidOutput,
    RateLimited,
    StreamInterrupt,
    Success,
    Timeout,
)

pytestmark = pytest.mark.asyncio

_MESSAGES = [Message(role="user", content="你好")]


@pytest.fixture
def fake_config() -> ModelConfig:
    # Fake 不读凭据与坐标，任意合法 ModelConfig 即可。
    return ModelConfig(
        provider_model="fake-model",
        base_url="http://fake.test",
        api_key_env="FAKE_API_KEY",
        supports_structured_output=True,
        provider="fake",
    )


async def _collect(stream) -> list:
    return [event async for event in stream]


# -- 六剧本各一 --


async def test_scenario_success_controllable(fake_config):
    scenario = Success(
        content="可控内容",
        usage=Usage(input_tokens=7, output_tokens=11),
        finish_reason="length",
    )
    adapter = FakeAdapter(scenario=scenario)
    content, usage, finish_reason = await adapter.complete(fake_config, _MESSAGES, 30.0, None)
    assert (content, finish_reason) == ("可控内容", "length")
    assert (usage.input_tokens, usage.output_tokens) == (7, 11)

    events = await _collect(adapter.stream(fake_config, _MESSAGES, 30.0))
    assert events == [ContentDelta("可控内容"), StreamCompleted("length", scenario.usage)]


async def test_scenario_rate_limited(fake_config):
    # retry_after 留在剧本对象上供断言（裁决：不进异常链、不改 GatewayError）。
    scenario = RateLimited(retry_after=1.5)
    adapter = FakeAdapter(scenario=scenario)
    with pytest.raises(GatewayError) as exc_info:
        await adapter.complete(fake_config, _MESSAGES, 30.0, None)
    assert exc_info.value.code == "provider_overloaded"
    assert exc_info.value.status_code == 429
    assert scenario.retry_after == 1.5

    with pytest.raises(GatewayError):
        await _collect(adapter.stream(fake_config, _MESSAGES, 30.0))


async def test_scenario_timeout_no_real_sleep(fake_config):
    adapter = FakeAdapter(scenario=Timeout())
    with pytest.raises(GatewayError) as exc_info:
        await adapter.complete(fake_config, _MESSAGES, 30.0, None)
    assert exc_info.value.code == "model_unavailable"


async def test_scenario_stream_interrupt(fake_config):
    scenario = StreamInterrupt(chunks_before_failure=3)
    adapter = FakeAdapter(scenario=scenario)
    with pytest.raises(GatewayError) as exc_info:
        await _collect(adapter.stream(fake_config, _MESSAGES, 30.0))
    assert exc_info.value.code == "model_unavailable"

    # 已发出的块数恰好等于剧本配置：不多不少（重放需要确定边界）。
    adapter = FakeAdapter(scenario=StreamInterrupt(chunks_before_failure=3))
    received: list = []
    with pytest.raises(GatewayError):
        async for event in adapter.stream(fake_config, _MESSAGES, 30.0):
            received.append(event)
    assert received == [ContentDelta("chunk-0"), ContentDelta("chunk-1"), ContentDelta("chunk-2")]


async def test_scenario_invalid_output_returns_bad_content_faithfully(fake_config):
    # provider 不代劳校验：坏内容原样返回，三重关卡在 invocation 触发。
    adapter = FakeAdapter(scenario=InvalidOutput(content="{not-valid-json"))
    content, usage, finish_reason = await adapter.complete(fake_config, _MESSAGES, 30.0, None)
    assert content == "{not-valid-json"
    assert finish_reason == "stop"
    assert (usage.input_tokens, usage.output_tokens) == (0, 0)


async def test_scenario_consecutive_then_success(fake_config):
    scenario = ConsecutiveThenSuccess(failures=2, success=Success(content="终于成功"))
    adapter = FakeAdapter(scenario=scenario)
    for _ in range(2):
        with pytest.raises(GatewayError) as exc_info:
            await adapter.complete(fake_config, _MESSAGES, 30.0, None)
        assert exc_info.value.code == "model_unavailable"
    content, _, _ = await adapter.complete(fake_config, _MESSAGES, 30.0, None)
    assert content == "终于成功"


# -- 确定性验收：同剧本跑 10 次结果完全一致（-k deterministic）--


async def test_deterministic_complete_same_result_ten_times(fake_config):
    scenario = Success(
        content="确定性内容",
        usage=Usage(input_tokens=4, output_tokens=6),
        finish_reason="stop",
    )
    results = []
    for _ in range(10):
        # 每次新建 Adapter、复用同一剧本：行为不得因实例或历史而异。
        adapter = FakeAdapter(scenario=replace(scenario))
        results.append(await adapter.complete(fake_config, _MESSAGES, 30.0, None))
    assert all(result == results[0] for result in results)


async def test_deterministic_stream_same_events_ten_times(fake_config):
    scenario = Success(content="确定性流", usage=Usage(input_tokens=1, output_tokens=2))
    event_sequences = []
    for _ in range(10):
        adapter = FakeAdapter(scenario=replace(scenario))
        event_sequences.append(tuple(await _collect(adapter.stream(fake_config, _MESSAGES, 30.0))))
    assert all(sequence == event_sequences[0] for sequence in event_sequences)


async def test_deterministic_failure_script_same_error_ten_times(fake_config):
    scenario = Timeout()
    codes = []
    for _ in range(10):
        adapter = FakeAdapter(scenario=scenario)
        with pytest.raises(GatewayError) as exc_info:
            await adapter.complete(fake_config, _MESSAGES, 30.0, None)
        codes.append(exc_info.value.code)
    assert codes == ["model_unavailable"] * 10


# -- 无隐藏重试证明（-k no_hidden_retry）--


async def test_no_hidden_retry_three_failures_then_success_exactly_four_requests(fake_config):
    # 剧本"连续 3 次失败后成功"：每次 complete 调用恰好消耗一次剧本状态，
    # adapter.attempts（收到的请求计数）恰好 3+1=4——若 provider 内部存在
    # 任何隐藏重试路径，计数会放大；若吞减请求，计数会缩小。
    scenario = ConsecutiveThenSuccess(failures=3, success=Success(content="第 4 次成功"))
    adapter = FakeAdapter(scenario=scenario)
    failure_count = 0
    final_content = None
    while final_content is None:
        try:
            final_content, _, _ = await adapter.complete(fake_config, _MESSAGES, 30.0, None)
        except GatewayError as exc:
            assert exc.code == "model_unavailable"
            failure_count += 1
    assert failure_count == 3
    assert final_content == "第 4 次成功"
    assert adapter.attempts == 4  # 上游恰好收到 3+1=4 次请求
