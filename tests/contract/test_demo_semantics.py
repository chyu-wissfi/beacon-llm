"""M01 任务 3：demo `gateway.py` 的契约测试（迁移前先写先绿）。

等价迁移的证明方式：这些用例现在对根目录 demo（FastAPI app）跑绿；任务 4 迁移后，
只需把文件顶部唯一的集中 import 从 `gateway` 换成包内路径，全部用例必须原样跑绿。

测试只断言**行为边界**——HTTP 状态码、稳定错误码、上游请求计数、SSE 事件序列、
trace 记录字段——不断言 demo 内部实现细节。所有上游请求由 respx 拦截，测试全程离线：
凡是 respx 未注册的上游请求都会让测试失败（AllMockedAssertionError），而不是真发网络。
"""

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio
import respx

# 集中 import：这是迁移任务唯一需要改动的地方（换成 llm_gateway 包内的对应符号）。
# 不在测试函数里散落 import demo 内部符号。
from llm_gateway.main import app
from llm_gateway.services.catalog import MODEL_CONFIGS
from llm_gateway.services.trace_service import CALL_TRACES

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# openai 3.6.0 兼容 shim：让 respx 能拦截上游流量
# ---------------------------------------------------------------------------
# openai 3.x 默认用 httpx2.AsyncClient 发请求，而 respx 0.23.1 只 patch httpx/httpcore
# 层，对 httpx2 流量完全不可见——mock 会静默漏到真实网络（已实测 DeepSeek 返回 401）。
# openai 3.x 官方支持运行 legacy httpx client（http_client 参数的 is_legacy_* 分支），
# 因此这里把 openai 的默认 client 工厂替换为 legacy httpx1 版本，使 respx 恢复拦截。
# 该 shim 只动 openai 库自己的命名空间，不碰 demo 代码，迁移后无需改动；
# 升级 openai 或 respx 时需复核此 shim 是否仍必要/仍有效。


@pytest.fixture(autouse=True)
def _openai_legacy_httpx(monkeypatch):
    # 只透传 base_url：客户端级 timeout 对离线 mock 无意义，且 openai 传入的
    # httpx2.Timeout 对象与 httpx1 不兼容，直接忽略（demo 的每请求超时由
    # openai 的 legacy 归一化逻辑另行处理，与本工厂无关）。
    def _legacy_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
        # cast 是运行期恒等：仅满足 httpx base_url: URLTypes 的标注。openai 构造
        # AsyncHttpxClientWrapper 时恒传 base_url（openai/_base_client.py），不会缺键。
        return httpx.AsyncClient(
            base_url=cast("httpx.URL | str", kwargs.get("base_url")),
        )

    monkeypatch.setattr("openai._base_client.AsyncHttpxClientWrapper", _legacy_httpx_client)


# ---------------------------------------------------------------------------
# 测试环境与公共夹具
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _gateway_env(monkeypatch):
    # create_client 在每次调用时读环境变量，缺 key 会 503 gateway_misconfigured；
    # 契约测试只关心网关行为，用假 key 即可（respx 拦截，不会真发请求）。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-primary-key")
    monkeypatch.setenv("DEEPSEEK_BACKUP_API_KEY", "test-backup-key")


@pytest.fixture(autouse=True)
def _clean_traces():
    # CALL_TRACES 是 demo 的模块级全局 list，测试间必须清空，避免相互污染导致
    # "/v1/traces 返回记录"之类的断言受其他用例残留影响。
    CALL_TRACES.clear()
    yield
    CALL_TRACES.clear()


@pytest.fixture(autouse=True)
def mock_upstream():
    # assert_all_mocked=True：任何未注册的上游请求直接让测试失败（离线保证）。
    # assert_all_called=False：注册了但未被调用的路由不算失败——"备用模型没被
    # 调用"这类不变量由用例自己断言 call_count == 0，语义更明确。
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as mock:
        yield mock


@pytest_asyncio.fixture
async def client():
    # 用 httpx ASGI 传输直打 app，不起端口、不走网络。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
        yield c


# ---------------------------------------------------------------------------
# 常量与构造器：从 demo 的 MODEL_CONFIGS 派生上游坐标，避免硬编码供应商模型名，
# 也保证 PRIMARY_PROVIDER_MODEL / BACKUP_BASE_URL 等环境变量覆盖时测试仍然成立。
# ---------------------------------------------------------------------------

PRIMARY_PROVIDER_MODEL = MODEL_CONFIGS["general-primary"].provider_model
BACKUP_PROVIDER_MODEL = MODEL_CONFIGS["general-backup"].provider_model


def _chat_completions_url(config) -> str:
    # openai SDK 会在 base_url 后拼接 /chat/completions。
    return config.base_url.rstrip("/") + "/chat/completions"


PRIMARY_URL = _chat_completions_url(MODEL_CONFIGS["general-primary"])
BACKUP_URL = _chat_completions_url(MODEL_CONFIGS["general-backup"])

SSE_HEADERS = {"content-type": "text/event-stream"}

# 结构化输出测试用的最小 JSON Schema：要求 answer 为 string。
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
    "additionalProperties": False,
}


def _llm_request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "你好"}],
    }
    payload.update(overrides)
    return payload


def _completion(content: str, *, prompt_tokens: int = 13, completion_tokens: int = 5) -> dict[str, Any]:
    # 构造 OpenAI Compatible 的 chat.completion 响应体（SDK 会按 Pydantic 模型解析）。
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "provider-model-ignored",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse_chunk_payload(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "provider-model-ignored",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }


def _encode_sse_chunk(text: str) -> bytes:
    return f"data: {json.dumps(_sse_chunk_payload(text))}\n\n".encode()


def _sse_body(chunks: list[str]) -> bytes:
    # 完整的 SSE 流：若干增量块 + [DONE] 终止符（openai AsyncStream 据此结束迭代）。
    return b"".join(_encode_sse_chunk(text) for text in chunks) + b"data: [DONE]\n\n"


def _sse_stream_then_break(chunks: list[str]) -> AsyncIterator[bytes]:
    # 先正常吐出 chunks，然后模拟上游连接中断（首块之后才挂）。
    async def _gen() -> AsyncIterator[bytes]:
        for text in chunks:
            yield _encode_sse_chunk(text)
        raise httpx.ReadError("upstream stream broke after first chunk")

    return _gen()


async def _collect_sse_events(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    # 以流式方式消费 SSE 响应，返回事件对象列表（每个事件是 data: 行里的 JSON）。
    events: list[dict[str, Any]] = []
    async with client.stream("POST", path, json=payload) as response:
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


# ---------------------------------------------------------------------------
# 白名单与入口组合校验
# ---------------------------------------------------------------------------

async def test_unknown_model_rejected_400(client, mock_upstream):
    # 不变量：Gateway 只暴露白名单模型，未知模型在调用任何上游之前就被拒绝。
    upstream_any = mock_upstream.route()  # 兜底路由：匹配任何上游请求
    response = await client.post("/v1/llm", json=_llm_request(model="mega-ultra-model"))
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unknown_model"
    # 证明拒绝发生在上游调用之前。
    assert upstream_any.call_count == 0


async def test_stream_with_response_schema_rejected_422(client):
    # 不变量：stream 与 response_schema 互斥，由请求模型（Pydantic validator）拒绝，
    # 表现为 FastAPI 的 422 请求校验错误，而非运行期错误码。
    response = await client.post(
        "/v1/llm", json=_llm_request(stream=True, response_schema=ANSWER_SCHEMA)
    )
    assert response.status_code == 422
    # 断言错误确实来自"组合校验"这条 validator，而不是其他字段校验。
    assert "不能同时使用" in response.text


async def test_stream_flag_on_json_endpoint_rejected_400(client):
    # 不变量：/v1/llm 是非流式端点，stream=true 必须被引导到专用流式端点。
    response = await client.post("/v1/llm", json=_llm_request(stream=True))
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "use_stream_endpoint"


async def test_stream_endpoint_rejects_response_schema_400(client):
    # 不变量：流式端点明确禁止 Structured Output 混用。
    response = await client.post(
        "/v1/llm/stream", json=_llm_request(response_schema=ANSWER_SCHEMA)
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_combination"


# ---------------------------------------------------------------------------
# fallback 链语义（非流式）
# ---------------------------------------------------------------------------

async def test_invalid_json_does_not_trigger_fallback(client, mock_upstream):
    # 不变量：invalid_json 是 GatewayError（内容质量问题），换模型也解决不了，
    # 必须直接失败——绝不消耗备用模型的调用。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=_completion("这不是JSON{{{")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, json=_completion("backup ok")
    )
    response = await client.post("/v1/llm", json=_llm_request(response_schema=ANSWER_SCHEMA))
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_json"
    assert primary.call_count == 1
    assert backup.call_count == 0


async def test_schema_validation_failed_does_not_trigger_fallback(client, mock_upstream):
    # 不变量：schema_validation_failed 同样是 GatewayError，不进入 fallback。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=_completion(json.dumps({"answer": 123}))  # answer 不是 string
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, json=_completion("backup ok")
    )
    response = await client.post("/v1/llm", json=_llm_request(response_schema=ANSWER_SCHEMA))
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "schema_validation_failed"
    assert primary.call_count == 1
    assert backup.call_count == 0


async def test_retryable_error_retries_primary_then_falls_back_to_backup(client, mock_upstream):
    # 不变量：可重试的临时故障按"主模型最多 2 次尝试 -> 切备用"的链路执行；
    # 上游请求计数是这条链路的可观测证明（2 次主模型 + 1 次备用 = 3）。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary connection refused")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, json=_completion("backup ok")
    )
    response = await client.post("/v1/llm", json=_llm_request())
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "general-backup"  # 实际服务方是备用模型
    assert body["content"] == "backup ok"
    assert body["attempts"] == 3
    assert primary.call_count == 2  # 主模型重试 1 次（共 2 次尝试）
    assert backup.call_count == 1


async def test_retry_exhaustion_returns_model_unavailable(client, mock_upstream):
    # 不变量：主备都耗尽重试后，统一为 502 model_unavailable，并留下 failed 的
    # 调用 trace（actual_model 为空），供治理与排障使用。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("backup down")
    )
    response = await client.post("/v1/llm", json=_llm_request())
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "model_unavailable"
    assert primary.call_count == 2
    assert backup.call_count == 2

    assert len(CALL_TRACES) == 1
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.error_code == "model_unavailable"
    assert trace.requested_model == "general-primary"
    assert trace.actual_model is None
    assert trace.attempts == 4


async def test_structured_output_success_passthrough(client, mock_upstream):
    # 不变量：response_schema 走 json_object 模式转发上游（附加系统约束消息），
    # 返回内容经过 JSON 解析与 schema 校验后放进 parsed，usage 透传给调用方。
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=_completion(json.dumps({"answer": "ok"}), prompt_tokens=13, completion_tokens=5)
    )
    response = await client.post("/v1/llm", json=_llm_request(response_schema=ANSWER_SCHEMA))
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "general-primary"
    assert body["parsed"] == {"answer": "ok"}
    assert body["content"] == json.dumps({"answer": "ok"})
    assert body["usage"] == {"input_tokens": 13, "output_tokens": 5}
    assert body["attempts"] == 1

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
        200, json=_completion("ok")
    )
    response = await client.post(
        "/v1/llm",
        json=_llm_request(
            prompt={"name": "knowledge_decision", "version": "v1", "variables": {"product_name": "Beacon"}}
        ),
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
        "/v1/llm",
        json=_llm_request(prompt={"name": "no_such_template", "version": "v1", "variables": {}}),
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unknown_prompt_template"
    assert upstream_any.call_count == 0


async def test_missing_prompt_variable_rejected_before_upstream(client, mock_upstream):
    # 不变量：模板存在但缺变量（knowledge_decision/v1 需要 product_name）时，
    # 400 missing_prompt_variable，且在调用模型之前失败——错误在渲染期而非调用期。
    upstream_any = mock_upstream.route()
    response = await client.post(
        "/v1/llm",
        json=_llm_request(prompt={"name": "knowledge_decision", "version": "v1", "variables": {}}),
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "missing_prompt_variable"
    assert "product_name" in detail["message"]
    assert upstream_any.call_count == 0


# ---------------------------------------------------------------------------
# 流式语义（SSE）
# ---------------------------------------------------------------------------

async def test_stream_success_emits_deltas_and_completed(client, mock_upstream):
    # 不变量：流式端点把上游增量转成 content.delta 事件，正常结束后发
    # response.completed 并标注实际服务的模型。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, content=_sse_body(["你", "好"]), headers=SSE_HEADERS
    )
    events = await _collect_sse_events(client, "/v1/llm/stream", _llm_request())
    assert [event["type"] for event in events] == ["content.delta", "content.delta", "response.completed"]
    assert events[0]["delta"] == "你"
    assert events[1]["delta"] == "好"
    assert events[-1]["model"] == "general-primary"


async def test_stream_falls_back_before_first_chunk(client, mock_upstream):
    # 不变量：上游首块之前失败（尚未向调用方吐出任何文本）时，可以安全地
    # 切换备用模型重新生成，调用方无感知。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down before first chunk")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, content=_sse_body(["备", "份"]), headers=SSE_HEADERS
    )
    events = await _collect_sse_events(client, "/v1/llm/stream", _llm_request())
    assert [event["type"] for event in events] == [
        "content.delta",
        "content.delta",
        "response.completed",
    ]
    assert events[-1]["model"] == "general-backup"
    assert primary.call_count == 1
    assert backup.call_count == 1


async def test_stream_failure_after_first_chunk_emits_failed_without_regeneration(client, mock_upstream):
    # 不变量：首块之后失败绝不能重新生成（调用方已经拿到一半文本，重生成会
    # 造成文本重复）——只能发 response.failed 流内错误收场；备用模型不被调用。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        return_value=httpx.Response(200, content=_sse_stream_then_break(["首块"]), headers=SSE_HEADERS)
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, content=_sse_body(["备用模型的文本"]), headers=SSE_HEADERS
    )
    events = await _collect_sse_events(client, "/v1/llm/stream", _llm_request())
    assert [event["type"] for event in events] == ["content.delta", "response.failed"]
    assert events[0]["delta"] == "首块"
    assert events[-1]["error"] == "upstream_stream_failed"
    assert primary.call_count == 1
    assert backup.call_count == 0  # 没有第二个模型的任何请求 -> 不可能产生重复文本


# ---------------------------------------------------------------------------
# 调用 trace 语义
# ---------------------------------------------------------------------------

async def test_traces_record_successful_call(client, mock_upstream):
    # 不变量：成功调用在 /v1/traces 留下一条记录：request_id 与响应一致，
    # 记录请求/实际模型、token、成本、尝试次数与状态，默认不含文本内容。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=_completion("ok", prompt_tokens=13, completion_tokens=5)
    )
    llm_response = await client.post("/v1/llm", json=_llm_request())
    assert llm_response.status_code == 200
    request_id = llm_response.json()["request_id"]

    traces_response = await client.get("/v1/traces")
    assert traces_response.status_code == 200
    traces = traces_response.json()
    assert len(traces) == 1

    trace = traces[0]
    assert trace["request_id"] == request_id
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
