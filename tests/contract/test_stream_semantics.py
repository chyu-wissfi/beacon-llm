"""/v1/chat/completions 流式语义契约（M06）。

验收命令聚焦的两类不变量（design.md §3.9 / 不变量 #7、#8）：
- no_regeneration：首块后失败绝不重新生成——已发出的块不重复、流内终态错误
  事件只发一个、备用模型零调用；
- cancellation：客户端中途断开 -> 取消传播到下游 -> trace 恰好一条
  cancelled 终态（不落 failed）。

与 test_chat_api.py 的流式段互补：那边钉 chunk 线格式与 [DONE] 位置，这边钉
M06 的取消/重生成铁律与 trace 终态。全部离线（respx 拦截上游）。
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from llm_gateway.services.trace_service import CALL_TRACES
from tests.contract.helpers import (
    BACKUP_PROVIDER_MODEL,
    BACKUP_URL,
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    SSE_HEADERS,
    chat_request,
    collect_sse_events,
    sse_body,
    sse_stream_then_break,
)

pytestmark = pytest.mark.asyncio

CHAT_PATH = "/v1/chat/completions"


def _parse_sse(events: list[str]) -> list[dict[str, Any]]:
    # [DONE] 不是 JSON，由用例单独断言其位置。
    return [json.loads(event) for event in events if event != "[DONE]"]


# ---------------------------------------------------------------------------
# 不变量 #7：首块后失败不重新生成
# ---------------------------------------------------------------------------


async def test_stream_no_regeneration_after_first_chunk_failure(client, mock_upstream):
    # 上游吐出首块后断流：已发出的块不重复（没有重新生成拼接）、流内终态
    # 错误事件恰好一个（错误后不发 [DONE]）、备用模型零调用。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        return_value=httpx.Response(200, content=sse_stream_then_break(["首块"]), headers=SSE_HEADERS)
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, content=sse_body(["备用模型的文本"]), headers=SSE_HEADERS
    )
    events = await collect_sse_events(client, CHAT_PATH, chat_request(stream=True))
    assert events[-1] != "[DONE]"  # 失败终态不是 [DONE]
    chunks = _parse_sse(events)
    contents = [
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0]["delta"]["content"]
    ]
    assert contents == ["首块"]  # 已发出的块不重复
    errors = [chunk for chunk in chunks if "error" in chunk]
    assert len(errors) == 1  # 终态错误只发一次
    assert errors[0]["error"]["code"] == "upstream_stream_failed"
    assert errors[0]["error"]["type"] == "api_error"
    assert primary.call_count == 1
    assert backup.call_count == 0  # 没有第二个模型的任何请求 -> 不可能产生重复文本
    # 唯一终态：恰好一条 failed trace（actual = 首块的实际服务方）。
    assert len(CALL_TRACES) == 1
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.error_code == "upstream_stream_failed"
    assert trace.actual_model == "general-primary"


# ---------------------------------------------------------------------------
# 不变量 #8：取消传播、单一 cancelled 终态
# ---------------------------------------------------------------------------


async def test_stream_cancellation_propagates_single_cancelled_trace(client, mock_upstream):
    # 上游慢流（首块迟迟不到）：客户端中途断开 -> 网关取消传播到下游 ->
    # trace 恰好一条 cancelled 终态（不落 failed，M06 任务 7/8）。
    async def _hanging_stream(request: httpx.Request) -> Any:
        # 首块前挂起：模拟上游 TTFT 极慢，给客户端留出断开窗口。
        await asyncio.sleep(30)
        yield httpx.Response(200, content=sse_body(["迟到的块"]), headers=SSE_HEADERS)

    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=_hanging_stream
    )
    task = asyncio.create_task(
        collect_sse_events(client, CHAT_PATH, chat_request(stream=True))
    )
    await asyncio.sleep(0.2)  # 等流建立、网关已进入等待上游首块的状态
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # 取消终态：恰好一条、cancelled、非 failed；usage 缺口可识别（全零记账）。
    assert len(CALL_TRACES) == 1
    trace = CALL_TRACES[-1]
    assert trace.status == "cancelled"
    assert trace.error_code is None
    assert trace.input_tokens == 0 and trace.output_tokens == 0
