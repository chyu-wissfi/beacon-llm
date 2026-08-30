"""真模型 live 冒烟（M12 任务 4）：六大功能在真实上游路径上的最终证据。

六大功能与断言面（design.md §8.1 live 层，与 spec M12 任务 4 逐条对应）：
1. 非流式调用：200 + 非空内容 + usage 三键正确 + trace attempts==1（正常路径）；
2. 流式：SSE 增量拼接非空、[DONE] 收尾、TTFT 进 trace；
3. 结构化输出：response_format + Validation Profile 双关卡通过、
   trace 记录校验档案坐标（parsed 路径的档案面）；
4. 模板引用：prompt 扩展字段渲染成功、trace 记录模板坐标；
5. 可观测：调用后 /metrics 的 llm_requests_total 恰好 +1、
   /v1/traces 新记录含 caller / 价格版本 / 尝试数；
6. 限流：低 RPM 配置真实触发一次 429（准入层令牌桶在真请求路径上生效）；
   重试的完整语义由契约层 Fake 剧本证明（真模型失败时机不可控，
   design.md §8.1 Why），live 侧只验正常路径 attempts==1（用例 1）。

约束：需真实 key（DEEPSEEK_API_KEY / DEEPSEEK_BACKUP_API_KEY）+ 网络；
凭据缺失整目录跳过。每次运行消耗少量真实额度——冒烟请求都收到最短回复。
"""

import json
import os
from dataclasses import replace

import pytest

from llm_gateway.core.schemas import RateLimitConfig
from llm_gateway.services.catalog import MODEL_CONFIGS
from llm_gateway.services.trace_service import CALL_TRACES
from tests.contract.helpers import collect_sse_events

pytestmark = [
    pytest.mark.live,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not (os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("DEEPSEEK_BACKUP_API_KEY")),
        reason="live 冒烟需要真实上游凭据（DEEPSEEK_API_KEY / DEEPSEEK_BACKUP_API_KEY）",
    ),
]

CHAT_PATH = "/v1/chat/completions"

# 冒烟消息一律收到最短回复：控制真实额度消耗，同时足以断言行为面。
_TINY_MESSAGES = [{"role": "user", "content": "只回复一个词：ok"}]


def _last_trace():
    assert len(CALL_TRACES) >= 1, "调用未产生 trace"
    return CALL_TRACES[-1]


# ---------------------------------------------------------------------------
# 1. 非流式调用（含 usage 正确）
# ---------------------------------------------------------------------------


async def test_live_nonstream_call_with_usage(client):
    response = await client.post(
        CHAT_PATH, json={"model": "general-primary", "messages": _TINY_MESSAGES, "max_tokens": 32}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    assert content.strip(), "真模型返回了空内容"
    # usage 三键口径自洽：total = prompt + completion，且均真实非零。
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    # 正常路径 attempts==1：真模型失败不可控，重试完整语义在契约层剧本
    # （design.md §8.1），live 只钉住"无隐藏重试"这一半。
    trace = _last_trace()
    assert trace.request_id == body["id"]  # 响应 id 与 Trace 对账
    assert trace.status == "success"
    assert trace.attempts == 1
    assert trace.actual_model == "general-primary"
    assert trace.input_tokens == usage["prompt_tokens"]
    assert trace.output_tokens == usage["completion_tokens"]


# ---------------------------------------------------------------------------
# 2. 流式（SSE 增量 + TTFT 进 trace）
# ---------------------------------------------------------------------------


async def test_live_stream_sse_increments_and_ttft(client):
    events = await collect_sse_events(
        client, CHAT_PATH, {"model": "general-primary", "messages": _TINY_MESSAGES, "stream": True}
    )
    assert events[-1] == "[DONE]"  # 成功终态以 [DONE] 收尾
    chunks = [json.loads(event) for event in events if event != "[DONE]"]
    deltas = "".join(
        choice["delta"].get("content", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert deltas.strip(), "流式未交付任何增量"
    # 终态块恰好一个：finish_reason 非 None（消费方据此识别流结束）。
    terminal = [
        chunk
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("finish_reason") is not None
    ]
    assert len(terminal) == 1
    trace = _last_trace()
    assert trace.status == "success"
    assert trace.ttft_ms is not None  # TTFT 在首个内容块记录并进 Trace
    assert trace.ttft_ms >= 0
    assert trace.attempts == 1


# ---------------------------------------------------------------------------
# 3. 结构化输出（response_format + 校验 + trace 的档案路径）
# ---------------------------------------------------------------------------


async def test_live_structured_output_with_validation_profile(client):
    # json_object 模式 + order_decision/v1 业务关卡：真模型输出必须同时过
    # 结构关（JSON）与业务关（Pydantic 规则）才到调用方。
    messages = [
        {
            "role": "user",
            "content": (
                "你是订单决策器。只输出一个 JSON 对象，不要任何其他文字："
                '{"order_id": "D-1001", "approve": true, "reject": false, "escalate": false}'
            ),
        }
    ]
    response = await client.post(
        CHAT_PATH,
        json={
            "model": "general-primary",
            "messages": messages,
            "response_format": {"type": "json_object"},
            "validation": {"name": "order_decision", "version": "v1"},
            "max_tokens": 256,
        },
    )
    assert response.status_code == 200, response.text
    parsed = json.loads(response.json()["choices"][0]["message"]["content"])
    # 业务规则现场复核：到达调用方的输出一定已满足 Profile（不变量 #10）。
    assert parsed["order_id"] == "D-1001"
    assert parsed["approve"] is True
    assert not (parsed.get("approve") and parsed.get("reject"))
    trace = _last_trace()
    assert trace.status == "success"
    # trace 的校验档案路径：哪套业务规则放行了这次输出可定位（不变量 #12）。
    assert trace.validation_profile == "order_decision/v1"


# ---------------------------------------------------------------------------
# 4. 模板引用（prompt 扩展字段渲染）
# ---------------------------------------------------------------------------


async def test_live_prompt_template_reference(client):
    # 模板渲染在网关侧完成：调用方只提交坐标与变量（不变量 #11 的正向面）。
    response = await client.post(
        CHAT_PATH,
        json={
            "model": "general-primary",
            "messages": [{"role": "user", "content": "只回复一个词：ok"}],
            "prompt": {
                "name": "knowledge_decision",
                "version": "v1",
                "variables": {"product_name": "Beacon"},
            },
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200, response.text
    trace = _last_trace()
    assert trace.status == "success"
    assert trace.prompt_name == "knowledge_decision"
    assert trace.prompt_version == "v1"


# ---------------------------------------------------------------------------
# 5. 可观测（/metrics 计数变化 + /v1/traces 新记录）
# ---------------------------------------------------------------------------


def _requests_total(metrics_text: str) -> float:
    # Prometheus 文本协议：累加 llm_requests_total 全部标签组合的样本值。
    return sum(
        float(line.split()[-1])
        for line in metrics_text.splitlines()
        if line.startswith("llm_requests_total{")
    )


async def test_live_observability_metrics_and_traces(client):
    before = _requests_total((await client.get("/metrics")).text)
    response = await client.post(
        CHAT_PATH, json={"model": "general-primary", "messages": _TINY_MESSAGES, "max_tokens": 32}
    )
    assert response.status_code == 200, response.text
    request_id = response.json()["id"]
    after = _requests_total((await client.get("/metrics")).text)
    assert after == before + 1  # 恰好一次终态记账（不变量 #1 的可观测面）
    # /v1/traces 新记录：治理字段面齐备（调用方 / 价格版本 / 尝试数）。
    traces = (await client.get("/v1/traces", params={"caller": "Beacon 演示 Agent"})).json()
    record = next((item for item in traces if item["request_id"] == request_id), None)
    assert record is not None, "调用后 /v1/traces 未出现对应记录"
    assert record["status"] == "success"
    assert record["caller"] == "Beacon 演示 Agent"
    assert record["price_version"]  # 成本按哪个价格版本计价可定位
    assert record["attempts"] == 1
    assert record["requested_model"] == "general-primary"


# ---------------------------------------------------------------------------
# 6. 限流（低 RPM 配置真实触发一次 429）
# ---------------------------------------------------------------------------


async def test_live_rate_limit_real_429(client, monkeypatch):
    # 令牌桶容量压到 1：首个真请求吃掉唯一令牌，第二个准入期即拒——
    # 证明 RPM 限流在真请求路径上生效（不是 mock 出来的行为）。
    monkeypatch.setitem(
        MODEL_CONFIGS,
        "general-primary",
        replace(MODEL_CONFIGS["general-primary"], rate_limit=RateLimitConfig(rpm=1, tpm=None)),
    )
    first = await client.post(
        CHAT_PATH, json={"model": "general-primary", "messages": _TINY_MESSAGES, "max_tokens": 32}
    )
    assert first.status_code == 200, first.text
    second = await client.post(
        CHAT_PATH, json={"model": "general-primary", "messages": _TINY_MESSAGES, "max_tokens": 32}
    )
    assert second.status_code == 429
    error = second.json()["error"]
    assert error["code"] == "rate_limited"
    assert second.headers.get("retry-after")  # 建议重试间隔随 429 下发
    # 限流拒绝零上游波及：只有首个请求产生 trace（第二个未进编排）。
    assert len(CALL_TRACES) == 1
