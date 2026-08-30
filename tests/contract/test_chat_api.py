"""/v1/chat/completions 契约测试（M03 任务 2 + 任务 5）。

前身是 test_demo_semantics.py（demo /v1/llm 契约）：旧端点已删除，全部语义
移植到统一 OpenAI 端点上，断言的仍是**行为边界**——HTTP 状态码、稳定错误码、
上游请求计数、SSE 事件序列、trace 记录字段——不断言内部实现细节。与 demo 期
的差异是线格式：响应体为 OpenAI chat.completion / chunk / OpenAI 风格错误体。

任务 C 将以 openai 官方 SDK 客户端重写本文件的同等语义（错误体可被 SDK 解析、
增量可被 SDK 迭代）；本文件先用 raw httpx + respx 钉住行为。

结构化输出三例（invalid_json / schema_validation_failed / parsed 透传）降为
**服务层回归**（直调 call_with_fallback）：M03 起 OpenAI 面的 response_format
只过白名单、内部结构在 M06-M08 消费（见 api/chat.py 边界注释），结构化分支
暂无 HTTP 面；保留服务层覆盖防止编排层在 M04/M06 改造前失去回归。
"""

import json
from typing import Any

import httpx
import pytest

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import LLMRequest, Message
from llm_gateway.services.catalog import MODEL_CONFIGS
from llm_gateway.services.invocation import call_with_fallback
from llm_gateway.services.trace_service import CALL_TRACES
from tests.contract.helpers import (
    ANSWER_SCHEMA,
    BACKUP_PROVIDER_MODEL,
    BACKUP_URL,
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    SSE_HEADERS,
    chat_request,
    collect_sse_events,
    completion,
    sse_body,
    sse_stream_then_break,
)

pytestmark = pytest.mark.asyncio

CHAT_PATH = "/v1/chat/completions"


def _error_body(response_json: dict[str, Any]) -> dict[str, Any]:
    return response_json["error"]


# ---------------------------------------------------------------------------
# 非流式：OpenAI chat.completion 形态
# ---------------------------------------------------------------------------


async def test_non_stream_returns_openai_chat_completion_shape(client, mock_upstream):
    # 不变量 #3：非流式响应是标准 chat.completion 结构（id / object / created /
    # model / choices / usage），openai SDK 可直接解析；不夹带 demo 的
    # request_id/content/parsed/attempts 扩展键——治理信息走 /v1/traces。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok", prompt_tokens=13, completion_tokens=5)
    )
    response = await client.post(CHAT_PATH, json=chat_request())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == "general-primary"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert set(choice) == {"index", "message", "finish_reason"}
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "ok"}
    # usage 换名到 OpenAI 口径：input/output_tokens -> prompt/completion_tokens，
    # total 由网关补齐。
    assert body["usage"] == {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}


# ---------------------------------------------------------------------------
# fallback 链语义（非流式）
# ---------------------------------------------------------------------------


async def test_retryable_error_retries_primary_then_falls_back_to_backup(client, mock_upstream):
    # 不变量：可重试的临时故障按"主模型最多 2 次尝试 -> 切备用"的链路执行；
    # 上游请求计数是这条链路的可观测证明（2 次主模型 + 1 次备用 = 3）。
    # 响应的 model 字段暴露实际服务方（platform 模型名，provider_model 不出网关）。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary connection refused")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, json=completion("backup ok")
    )
    response = await client.post(CHAT_PATH, json=chat_request())
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "general-backup"  # 实际服务方是备用模型
    assert body["choices"][0]["message"]["content"] == "backup ok"
    assert primary.call_count == 2  # 主模型重试 1 次（共 2 次尝试）
    assert backup.call_count == 1
    # 尝试计数不再出现在响应体（OpenAI 形态无此字段），从 trace 对账。
    assert len(CALL_TRACES) == 1
    assert CALL_TRACES[0].attempts == 3


async def test_retry_exhaustion_returns_model_unavailable(client, mock_upstream):
    # 不变量：主备都耗尽重试后，统一为 502 model_unavailable（OpenAI 体：
    # 5xx -> type=api_error），并留下 failed 的调用 trace（actual_model 为空）。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("backup down")
    )
    response = await client.post(CHAT_PATH, json=chat_request())
    assert response.status_code == 502
    error = _error_body(response.json())
    assert error["code"] == "model_unavailable"
    assert error["type"] == "api_error"
    assert primary.call_count == 2
    assert backup.call_count == 2

    assert len(CALL_TRACES) == 1
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.error_code == "model_unavailable"
    assert trace.requested_model == "general-primary"
    assert trace.actual_model is None
    assert trace.attempts == 4


# ---------------------------------------------------------------------------
# 结构化输出（服务层回归：M03 无 HTTP 面，见文件 docstring）
# ---------------------------------------------------------------------------


def _structured_request() -> LLMRequest:
    return LLMRequest(
        model="general-primary",
        messages=[Message(role="user", content="你好")],
        response_schema=ANSWER_SCHEMA,
    )


async def test_invalid_json_does_not_trigger_fallback(mock_upstream):
    # 不变量：invalid_json 是 GatewayError（内容质量问题），换模型也解决不了，
    # 必须直接失败——绝不消耗备用模型的调用。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("这不是JSON{{{")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, json=completion("backup ok")
    )
    with pytest.raises(GatewayError) as exc_info:
        await call_with_fallback(_structured_request())
    assert exc_info.value.code == "invalid_json"
    assert primary.call_count == 1
    assert backup.call_count == 0


async def test_schema_validation_failed_does_not_trigger_fallback(mock_upstream):
    # 不变量：schema_validation_failed 同样是 GatewayError，不进入 fallback。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion(json.dumps({"answer": 123}))  # answer 不是 string
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, json=completion("backup ok")
    )
    with pytest.raises(GatewayError) as exc_info:
        await call_with_fallback(_structured_request())
    assert exc_info.value.code == "schema_validation_failed"
    assert primary.call_count == 1
    assert backup.call_count == 0


async def test_structured_output_success_passthrough(mock_upstream):
    # 不变量：response_schema 走 json_object 模式转发上游（附加系统约束消息），
    # 返回内容经过 JSON 解析与 schema 校验后放进 parsed，usage 透传。
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion(json.dumps({"answer": "ok"}), prompt_tokens=13, completion_tokens=5)
    )
    llm_response = await call_with_fallback(_structured_request())
    assert llm_response.model == "general-primary"
    assert llm_response.parsed == {"answer": "ok"}
    assert llm_response.content == json.dumps({"answer": "ok"})
    assert llm_response.usage.model_dump() == {"input_tokens": 13, "output_tokens": 5}
    assert llm_response.attempts == 1

    # 上游协议契约：json_object 模式 + 注入的 schema 约束系统消息。
    upstream_body = json.loads(route.calls.last.request.content)
    assert upstream_body["response_format"] == {"type": "json_object"}
    assert upstream_body["messages"][0]["role"] == "system"
    assert "JSON Schema" in upstream_body["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Prompt 模板语义
# ---------------------------------------------------------------------------


async def test_prompt_template_renders_system_message_upstream(client, mock_upstream):
    # 不变量：调用方只能选择受控模板并传变量，Gateway 渲染后注入为系统消息，
    # 模板正文不经调用方之手。
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )
    response = await client.post(
        CHAT_PATH,
        json=chat_request(prompt={"name": "knowledge_decision", "version": "v1", "variables": {"product_name": "Beacon"}}),
    )
    assert response.status_code == 200

    upstream_body = json.loads(route.calls.last.request.content)
    system_message = upstream_body["messages"][0]
    assert system_message["role"] == "system"
    assert "Beacon" in system_message["content"]
    # 调用方的原始消息跟在渲染后的系统消息之后，保持顺序不变。
    assert upstream_body["messages"][1] == {"role": "user", "content": "你好"}


async def test_unknown_prompt_template_rejected_400(client, mock_upstream):
    # 不变量：模板不在受控库中时 400 拒绝，且不消耗任何上游调用。
    upstream_any = mock_upstream.route()
    response = await client.post(
        CHAT_PATH,
        json=chat_request(prompt={"name": "no_such_template", "version": "v1", "variables": {}}),
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unknown_prompt_template"
    assert error["type"] == "invalid_request_error"
    assert upstream_any.call_count == 0


async def test_missing_prompt_variable_rejected_before_upstream(client, mock_upstream):
    # 不变量：模板存在但缺变量（knowledge_decision/v1 需要 product_name）时，
    # 400 missing_prompt_variable，且在调用模型之前失败——错误在渲染期而非调用期。
    upstream_any = mock_upstream.route()
    response = await client.post(
        CHAT_PATH,
        json=chat_request(prompt={"name": "knowledge_decision", "version": "v1", "variables": {}}),
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "missing_prompt_variable"
    assert "product_name" in error["message"]
    assert upstream_any.call_count == 0


# ---------------------------------------------------------------------------
# 流式语义（SSE：OpenAI chunk + [DONE] 终态）
# ---------------------------------------------------------------------------


def _parse_sse(events: list[str]) -> list[dict[str, Any]]:
    # 终态前的 data: 行都是 JSON chunk/错误事件；[DONE] 不是 JSON，由用例单独
    # 断言其位置。
    return [json.loads(event) for event in events if event != "[DONE]"]


async def test_stream_success_emits_chunks_then_done(client, mock_upstream):
    # 不变量：上游增量翻译为 OpenAI chat.completion.chunk 流，成功终态是
    # [DONE]；每个 chunk 的 model 标注实际服务方，同一流的 id/created 一致。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, content=sse_body(["你", "好"]), headers=SSE_HEADERS
    )
    events = await collect_sse_events(client, CHAT_PATH, chat_request(stream=True))
    assert events[-1] == "[DONE]"
    chunks = _parse_sse(events)
    assert len(chunks) == 2
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == ["你", "好"]
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "general-primary"
        assert chunk["choices"][0]["finish_reason"] is None
        assert chunk["choices"][0]["index"] == 0
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert len({chunk["created"] for chunk in chunks}) == 1


async def test_stream_falls_back_before_first_chunk(client, mock_upstream):
    # 不变量：上游首块之前失败（尚未向调用方吐出任何文本）时，可以安全地
    # 切换备用模型重新生成，调用方无感知。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down before first chunk")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, content=sse_body(["备", "份"]), headers=SSE_HEADERS
    )
    events = await collect_sse_events(client, CHAT_PATH, chat_request(stream=True))
    assert events[-1] == "[DONE]"
    chunks = _parse_sse(events)
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == ["备", "份"]
    # 实际服务方逐块标注为备用模型。
    assert {chunk["model"] for chunk in chunks} == {"general-backup"}
    assert primary.call_count == 1
    assert backup.call_count == 1


async def test_stream_failure_after_first_chunk_emits_error_without_regeneration(client, mock_upstream):
    # 不变量：首块之后失败绝不能重新生成（调用方已经拿到一半文本，重生成会
    # 造成文本重复）——只能以 OpenAI 风格错误事件收场（design.md §3.9：错误
    # 事件后终止，不发 [DONE]）；备用模型不被调用。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        return_value=httpx.Response(200, content=sse_stream_then_break(["首块"]), headers=SSE_HEADERS)
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, content=sse_body(["备用模型的文本"]), headers=SSE_HEADERS
    )
    events = await collect_sse_events(client, CHAT_PATH, chat_request(stream=True))
    # 终态是错误事件而非 [DONE]。
    assert events[-1] != "[DONE]"
    chunks = _parse_sse(events)
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "首块"
    error = chunks[-1]["error"]
    assert error["code"] == "upstream_stream_failed"
    assert error["type"] == "api_error"
    assert primary.call_count == 1
    assert backup.call_count == 0  # 没有第二个模型的任何请求 -> 不可能产生重复文本


# ---------------------------------------------------------------------------
# 请求级组合与校验错误（HTTP 错误，非流内事件）
# ---------------------------------------------------------------------------


async def test_stream_with_response_format_rejected_400(client, mock_upstream):
    # 不变量（controller 裁决）：stream + response_format 互斥沿用 demo 的
    # unsupported_combination（400），在端点层以稳定码抛出，且错误 message
    # 指向 OpenAI 面的字段名 response_format。
    upstream_any = mock_upstream.route()
    response = await client.post(
        CHAT_PATH, json=chat_request(stream=True, response_format={"type": "json_object"})
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unsupported_combination"
    assert error["type"] == "invalid_request_error"
    assert "response_format" in error["message"]
    assert upstream_any.call_count == 0


async def test_stream_unknown_model_rejected_400_before_streaming(client, mock_upstream):
    # 不变量：请求级校验（模型白名单）保持在流开始之前——HTTP 400 而非流内
    # 错误事件；拒绝发生在任何上游调用之前。
    upstream_any = mock_upstream.route()
    response = await client.post(
        CHAT_PATH, json=chat_request(stream=True, model="mega-ultra-model")
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unknown_model"
    assert upstream_any.call_count == 0


async def test_unknown_model_rejected_400(client, mock_upstream):
    # 不变量：Gateway 只暴露白名单模型，未知模型在调用任何上游之前就被拒绝。
    upstream_any = mock_upstream.route()
    response = await client.post(CHAT_PATH, json=chat_request(model="mega-ultra-model"))
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unknown_model"
    assert error["type"] == "invalid_request_error"
    # 证明拒绝发生在上游调用之前。
    assert upstream_any.call_count == 0


# ---------------------------------------------------------------------------
# 错误面收口：404 / 405 / 500 / 503 全部 OpenAI 风格（controller 裁决补缺口）
# ---------------------------------------------------------------------------


async def test_unknown_path_returns_openai_404(client):
    # 不变量：未知路径 404 也走 OpenAI 错误体（code=None，注册表不扩员）。
    response = await client.get("/v1/nonexistent")
    assert response.status_code == 404
    error = _error_body(response.json())
    assert set(error) == {"message", "type", "code"}
    assert error["code"] is None
    assert error["type"] == "invalid_request_error"
    assert error["message"] == "Not Found"


async def test_wrong_method_returns_openai_405(client):
    # 不变量：方法不匹配 405 同样是 OpenAI 错误体。
    response = await client.get(CHAT_PATH)
    assert response.status_code == 405
    error = _error_body(response.json())
    assert error["code"] is None
    assert error["type"] == "invalid_request_error"
    assert error["message"] == "Method Not Allowed"


async def test_unhandled_exception_returns_openai_500(client_lenient, mock_upstream, monkeypatch):
    # 不变量：未捕获异常兜底为 500 + type=api_error + code=None；响应体固定
    # 文案，不回显异常内容（design.md §3.9：响应体不出现异常类名/堆栈/供应商
    # 内部信息）。用宽松传输：兜底响应发出后 Starlette 会重新抛出原异常。
    async def _boom(request: Any) -> Any:
        raise RuntimeError("内部炸弹（不应出现在响应体）")

    monkeypatch.setattr("llm_gateway.api.chat.call_with_fallback", _boom)
    response = await client_lenient.post(CHAT_PATH, json=chat_request())
    assert response.status_code == 500
    error = _error_body(response.json())
    assert set(error) == {"message", "type", "code"}
    assert error["type"] == "api_error"
    assert error["code"] is None
    assert "内部炸弹" not in error["message"]


async def test_missing_credentials_503_gateway_misconfigured(client, mock_upstream, monkeypatch):
    # 不变量：凭据缺失是 503 gateway_misconfigured（配置错误不伪装成上游故障
    # 502），OpenAI 体 type 按 5xx 归 api_error，且不产生任何上游调用。
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BACKUP_API_KEY", raising=False)
    upstream_any = mock_upstream.route()
    response = await client.post(CHAT_PATH, json=chat_request())
    assert response.status_code == 503
    error = _error_body(response.json())
    assert error["code"] == "gateway_misconfigured"
    assert error["type"] == "api_error"
    # message 来自注册表默认三元组（逐字冻结于 tests/unit/test_error_registry.py）。
    assert error["message"] == "Gateway 模型凭据未配置"
    assert upstream_any.call_count == 0


# ---------------------------------------------------------------------------
# 调用 trace 语义
# ---------------------------------------------------------------------------


async def test_traces_record_successful_call(client, mock_upstream):
    # 不变量：成功调用在 /v1/traces 留下一条记录：响应 id（chat.completion 的
    # id）与 trace 的 request_id 可对账，记录请求/实际模型、token、成本、尝试
    # 次数与状态，默认不含文本内容。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok", prompt_tokens=13, completion_tokens=5)
    )
    completion_response = await client.post(CHAT_PATH, json=chat_request())
    assert completion_response.status_code == 200
    completion_id = completion_response.json()["id"]

    traces_response = await client.get("/v1/traces")
    assert traces_response.status_code == 200
    traces = traces_response.json()
    assert len(traces) == 1

    trace = traces[0]
    assert trace["request_id"] == completion_id
    assert trace["requested_model"] == "general-primary"
    assert trace["actual_model"] == "general-primary"
    assert trace["status"] == "success"
    assert trace["error_code"] is None
    assert trace["attempts"] == 1
    assert trace["input_tokens"] == 13
    assert trace["output_tokens"] == 5
    # 成本按 general-primary 牌价计算：13*1.0 + 5*4.0 = 33 微美元级别。
    assert trace["cost_usd"] == pytest.approx((13 * 1.0 + 5 * 4.0) / 1_000_000)
    # trace 只存元数据，不记录文本内容。
    assert "content" not in trace
    assert "messages" not in trace

# ---------------------------------------------------------------------------
# 平台模型列表（M03 任务 3）
# ---------------------------------------------------------------------------


async def test_models_endpoint_lists_platform_models(client):
    # 不变量：GET /v1/models 返回 OpenAI 风格 {"object": "list", "data":
    # [{"id", "object": "model", ...}]}；数据源是配置中心加载产物 MODEL_CONFIGS
    # （controller 裁决：不得绕过配置中心另建模型表），id 即平台模型名——
    # 调用方请求 model 字段用的就是它；provider_model 不出网关。
    response = await client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert {item["id"] for item in body["data"]} == set(MODEL_CONFIGS)
    for item in body["data"]:
        # 形态即契约：不含 provider_model / api_key_env 等内部坐标。
        assert set(item) == {"id", "object", "created", "owned_by"}
        assert item["object"] == "model"
        assert isinstance(item["created"], int)
        assert item["owned_by"] == "beacon-llm"
