"""/v1/traces 落库与聚合契约（M09 任务 3/5/6）。

覆盖不变量 #1（Trace 真实可靠）、#5（可见最终端点与路由理由）、
#12（可定位 Prompt/路由/价格版本/尝试次数）、#14（按调用方/模型/
Prompt 版本聚合）：
- 记账恰好一次：多 attempt 剧本（重试+修复、重试+fallback），库里单条、
  usage = 各次尝试观测值之和、无重复；
- 字段补全：19 字段一个不漏（final_endpoint / validation_profile /
  price_version 随成功终态可断言）；
- 数据源是库不是内存：清空进程内缓存后仍可读回；
- 过滤与聚合：四过滤 + 三维度分组的数值断言；
- 崩溃安全：终态前取消 -> 库里恰好一条完整 cancelled 行，无半条。

Fake Adapter 剧本用例把实例挂到 openai_compatible 键（与 M08 契约同款）；
聚合用例直接经 record_trace 灌库（过滤/聚合是读路径语义，与编排解耦）。
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from llm_gateway.core.config import CONFIG
from llm_gateway.core.schemas import PromptSelection, Usage
from llm_gateway.providers import PROVIDER_REGISTRY
from llm_gateway.providers.fake import (
    FakeAdapter,
    InvalidOutput,
    Scenario,
    ScenarioSequence,
    Success,
    Timeout,
)
from llm_gateway.services.catalog import MODEL_CONFIGS, PRICE_VERSION
from llm_gateway.services.trace_service import CALL_TRACES, flush_pending, record_trace
from tests.contract.helpers import (
    ANSWER_SCHEMA,
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    collect_sse_events,
)

pytestmark = pytest.mark.asyncio

CHAT_PATH = "/v1/chat/completions"
TRACES_PATH = "/v1/traces"

# 19 字段面（spec 任务 2）：聚合/过滤之外的全字段断言基线。
TRACE_FIELDS = {
    "request_id",
    "timestamp",
    "caller",
    "requested_model",
    "actual_model",
    "final_endpoint",
    "route_reason",
    "prompt_name",
    "prompt_version",
    "validation_profile",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "price_version",
    "latency_ms",
    "ttft_ms",
    "attempts",
    "status",
    "error_code",
}

CALLER_DISPLAY_NAME = next(iter(CONFIG.callers.values())).display_name
PRIMARY_BASE_URL = MODEL_CONFIGS["general-primary"].base_url
BACKUP_BASE_URL = MODEL_CONFIGS["general-backup"].base_url


@pytest.fixture
def fake_provider(monkeypatch):
    """把剧本化 FakeAdapter 挂到 openai_compatible 键并返回实例（M08 同款）。"""

    def _install(scenario: Scenario) -> FakeAdapter:
        adapter = FakeAdapter(scenario)
        monkeypatch.setitem(PROVIDER_REGISTRY, "openai_compatible", adapter)
        return adapter

    return _install


def _chat_request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "你好"}],
    }
    payload.update(overrides)
    return payload


async def _traces(client) -> list[dict[str, Any]]:
    response = await client.get(TRACES_PATH)
    assert response.status_code == 200
    return response.json()


# ---------------------------------------------------------------------------
# 记账恰好一次：多 attempt 剧本 -> 单条 trace、usage 为各次尝试之和
# ---------------------------------------------------------------------------


async def test_trace_written_exactly_once_across_retry_and_repair(client, fake_provider):
    # 剧本：两次超时重试 + 一次坏 JSON 触发修复（带 schema 关卡才会判非法）
    # + 修复成功（预算恰好 4 次）。断言：库里恰好一条、usage = 各次尝试观测值之和（坏输出记 0 不伪造）、
    # 无重复行（恰好一次落库）。
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer_shape", "strict": True, "schema": ANSWER_SCHEMA},
    }
    # 修复调用的输出必须过 schema 关：否则修复失败、走不到成功终态。
    repair_success = Success(content='{"answer": "修好了"}', usage=Usage(input_tokens=10, output_tokens=4))
    fake_provider(
        ScenarioSequence([Timeout(), Timeout(), InvalidOutput(content="{坏"), repair_success])
    )
    response = await client.post(CHAT_PATH, json=_chat_request(response_format=response_format))
    assert response.status_code == 200

    traces = await _traces(client)
    assert len(traces) == 1  # 恰好一条：重试+修复不产生额外记录
    trace = traces[0]
    assert trace["attempts"] == 4
    assert trace["status"] == "success"
    # usage = 各次尝试之和：前两次超时观测不到、坏输出记 0、修复成功 10/4。
    assert trace["input_tokens"] == 10
    assert trace["output_tokens"] == 4
    assert trace["actual_model"] == "general-primary"
    assert CALL_TRACES.__len__() == 1  # 进程内缓存面同样恰好一条


async def test_trace_written_exactly_once_across_fallback(client, fake_provider):
    # 剧本：主模型吃满单模型上限（3 次超时）-> fallback 备用成功（第 4 次）。
    # 断言：恰好一条、实际模型与最终端点记备用、路由理由可定位离开原因。
    backup_success = Success(content="备用接住了", usage=Usage(input_tokens=7, output_tokens=3))
    fake_provider(
        ScenarioSequence([Timeout(), Timeout(), Timeout(), backup_success])
    )
    response = await client.post(CHAT_PATH, json=_chat_request())
    assert response.status_code == 200

    traces = await _traces(client)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["attempts"] == 4
    assert trace["actual_model"] == "general-backup"
    assert trace["final_endpoint"] == BACKUP_BASE_URL  # 最终端点 = 实际服务方
    assert "general-primary" in trace["route_reason"]  # 离开主模型的理由可定位
    assert trace["input_tokens"] == 7 and trace["output_tokens"] == 3


# ---------------------------------------------------------------------------
# 字段补全（19 字段一个不漏）与数据源（库而非内存）
# ---------------------------------------------------------------------------


async def test_trace_fields_complete_after_success(client, fake_provider):
    # 成功终态的字段面断言：新增三字段（final_endpoint / validation_profile /
    # price_version）随 19 字段全量可断言（不变量 #5、#12）。
    # 带 validation 的请求要求输出是业务合法 JSON（否则走修复/失败终态）。
    business_valid = json.dumps({"order_id": "A-1", "approve": True, "reject": False})
    fake_provider(Success(content=business_valid, usage=Usage(input_tokens=13, output_tokens=5)))
    response = await client.post(
        CHAT_PATH,
        json=_chat_request(validation={"name": "order_decision", "version": "v1"}),
    )
    assert response.status_code == 200

    traces = await _traces(client)
    assert len(traces) == 1
    trace = traces[0]
    assert set(trace.keys()) == TRACE_FIELDS  # 字段面一个不多一个不少
    assert trace["caller"] == CALLER_DISPLAY_NAME
    assert trace["final_endpoint"] == PRIMARY_BASE_URL
    assert trace["validation_profile"] == "order_decision/v1"
    assert trace["price_version"] == PRICE_VERSION
    assert trace["cost_usd"] == pytest.approx((13 * 1.0 + 5 * 4.0) / 1_000_000)


async def test_traces_served_from_db_not_memory(client, fake_provider):
    # 数据源是库：清空进程内缓存（M09 降级为缓存面）后仍可读回同一行。
    fake_provider(Success())
    response = await client.post(CHAT_PATH, json=_chat_request())
    assert response.status_code == 200
    assert await _traces(client)  # 首次读（内部对账落库）

    CALL_TRACES.clear()  # 缓存面清零：若仍从内存读，下面将返回空
    traces = await _traces(client)
    assert len(traces) == 1
    assert traces[0]["status"] == "success"


# ---------------------------------------------------------------------------
# 过滤与聚合（spec 任务 5）：灌库走 record_trace，读路径语义与编排解耦
# ---------------------------------------------------------------------------


def _seed(
    request_id: str,
    *,
    caller: str,
    requested: str = "general-primary",
    actual: str | None = "general-primary",
    status: str = "success",
    tokens: tuple[int, int] = (10, 5),
    latency: int = 100,
    ttft: int | None = None,
    prompt: PromptSelection | None = None,
) -> None:
    # 直接灌一条：聚合数值用例的成本口径由 calculate_cost 按 actual 牌价算。
    record_trace(
        request_id=request_id,
        requested_model=requested,
        actual_model=actual,
        prompt=prompt,
        usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1]),
        latency_ms=latency,
        attempts=1,
        status=status,  # type: ignore[arg-type]
        caller=caller,
        ttft_ms=ttft,
        price_version=PRICE_VERSION,
    )


async def _aggregation(client, group_by: str, **filters: str) -> dict[str, Any]:
    response = await client.get(TRACES_PATH, params={"group_by": group_by, **filters})
    assert response.status_code == 200
    return response.json()


async def test_traces_filters_by_caller_model_status_prompt_version(client):
    # 四过滤各自命中预期子集（灌 4 条差异化记录）。
    _seed("a", caller="团队甲", prompt=PromptSelection(name="knowledge_decision", version="v1"))
    _seed("b", caller="团队甲", actual="general-backup")
    _seed(
        "c",
        caller="团队乙",
        actual="general-backup",
        status="failed",
        tokens=(0, 0),
        prompt=PromptSelection(name="knowledge_decision", version="v2"),
    )
    _seed("d", caller="团队乙", requested="general-backup", actual=None, status="cancelled", tokens=(0, 0))
    await flush_pending()

    by_caller = await _traces_with(client, caller="团队甲")
    assert {t["request_id"] for t in by_caller} == {"a", "b"}
    # model 维度 = 实际服务模型；未服务到归因请求模型（d 归 backup）。
    by_model = await _traces_with(client, model="general-backup")
    assert {t["request_id"] for t in by_model} == {"b", "c", "d"}
    by_status = await _traces_with(client, status="failed")
    assert {t["request_id"] for t in by_status} == {"c"}
    by_prompt = await _traces_with(client, prompt_version="v1")
    assert {t["request_id"] for t in by_prompt} == {"a"}
    # 组合过滤：团队乙 + backup 维度。
    combo = await _traces_with(client, caller="团队乙", model="general-backup")
    assert {t["request_id"] for t in combo} == {"c", "d"}


async def _traces_with(client, **params: str) -> list[dict[str, Any]]:
    response = await client.get(TRACES_PATH, params=params)
    assert response.status_code == 200
    return response.json()


async def test_traces_aggregation_group_by_caller(client):
    # 按调用方分组：总量 / 总成本 / 平均延迟 / 平均 TTFT（ttft 仅非空）。
    # 成本口径：primary (10*1.0+5*4.0)/1e6=30e-6；backup (20*0.8+10*3.2)/1e6=48e-6。
    _seed("a", caller="团队甲", tokens=(10, 5), latency=100, ttft=20)
    _seed("b", caller="团队甲", actual="general-backup", tokens=(20, 10), latency=200)
    _seed("c", caller="团队乙", tokens=(5, 0), latency=300, ttft=40)
    await flush_pending()

    body = await _aggregation(client, "caller")
    assert body["summary"]["count"] == 3
    assert body["summary"]["cost_usd"] == pytest.approx((30 + 48 + 5) / 1_000_000)
    groups = {g["key"]: g for g in body["groups"]}
    assert set(groups) == {"团队甲", "团队乙"}
    jia = groups["团队甲"]
    assert jia["count"] == 2
    assert jia["cost_usd"] == pytest.approx(78 / 1_000_000)
    assert jia["input_tokens"] == 30 and jia["output_tokens"] == 15
    assert jia["avg_latency_ms"] == pytest.approx(150.0)
    assert jia["avg_ttft_ms"] == pytest.approx(20.0)  # 仅一条观测到 TTFT
    yi = groups["团队乙"]
    assert yi["count"] == 1 and yi["cost_usd"] == pytest.approx(5 / 1_000_000)


async def test_traces_aggregation_group_by_model_and_prompt_version(client):
    # 按模型分组用派生维度（实际服务方；未服务到归因请求模型）；
    # 按 Prompt 版本分组：未用模板的调用独立成 None 组。
    _seed("a", caller="甲", prompt=PromptSelection(name="knowledge_decision", version="v1"))
    _seed("b", caller="甲", actual="general-backup")
    _seed("c", caller="乙", requested="general-backup", actual=None, status="failed", tokens=(0, 0))
    await flush_pending()

    by_model = await _aggregation(client, "model")
    keys = {g["key"]: g["count"] for g in by_model["groups"]}
    assert keys == {"general-primary": 1, "general-backup": 2}  # c 归因到请求模型

    by_prompt = await _aggregation(client, "prompt_version")
    counts = {g["key"]: g["count"] for g in by_prompt["groups"]}
    assert counts == {"v1": 1, None: 2}


async def test_traces_aggregation_empty_scope(client):
    # 空过滤范围：总量 0、合计 0、均值 None（无样本不伪造 0）、无分组。
    _seed("a", caller="团队甲")
    await flush_pending()
    body = await _aggregation(client, "caller", caller="不存在的团队")
    assert body["summary"] == {
        "count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "avg_latency_ms": None,
        "avg_ttft_ms": None,
    }
    assert body["groups"] == []


async def test_traces_aggregation_rejects_unknown_dimension(client):
    # 非法分组维度：400（Literal 查询参数经 RequestValidationError，
    # api/errors 统一翻成 OpenAI 风格 400，零新增错误码）。
    response = await client.get(TRACES_PATH, params={"group_by": "error_code"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# 崩溃安全（spec 任务 6）：终态前取消 -> 无半条 trace
# ---------------------------------------------------------------------------


async def test_cancellation_before_terminal_leaves_single_complete_row(client, mock_upstream):
    # 首块前客户端断开：库里的终态行恰好一条且 19 字段完整（cancelled），
    # 不存在任何半条/中间态记录——落库只发生在终态迁移那一刻。
    async def _hanging_stream(request: httpx.Request) -> Any:
        await asyncio.sleep(30)
        yield httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})

    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=_hanging_stream
    )
    task = asyncio.create_task(collect_sse_events(client, CHAT_PATH, _chat_request(stream=True)))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    traces = await _traces(client)
    assert len(traces) == 1  # 恰好一条：无半条、无重复
    trace = traces[0]
    assert set(trace.keys()) == TRACE_FIELDS  # 行形完整：19 字段一个不缺
    assert trace["status"] == "cancelled"
    assert trace["actual_model"] is None and trace["final_endpoint"] is None
    assert trace["input_tokens"] == 0 and trace["output_tokens"] == 0
