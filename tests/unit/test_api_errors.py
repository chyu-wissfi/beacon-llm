"""api/errors.py 的单元测试：OpenAI 风格错误体（M03 任务 4 + 任务 B 收口）。

三层覆盖：
1. 纯函数层——type 映射与错误体形态（controller 裁决的 4xx/5xx 归类）；
2. 处理器直调层——GatewayError / RequestValidationError 到 OpenAI 体的一手
   翻译（不经过 HTTP，错误码三元组期望值直接取自 GatewayError 实例，不在
   测试里重复冻结 message——那是 test_error_registry.py 的职责）；
3. 应用装配层——register_error_handlers 注册后的探针 app 上走真实 HTTP：
   400 + unsupported_field，404/405/500 兜底（controller 裁决补的缺口）。

M03 任务 5 删除旧 /v1/llm 端点后全部路由无条件 OpenAI 风格：原"非 OpenAI 面
回退 FastAPI 默认 422"与旧端点钉住用例随收口移除，替换为 404/405/500 覆盖。
"""

import json
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.testclient import TestClient

from llm_gateway.api.errors import (
    API_ERROR,
    INVALID_REQUEST_ERROR,
    error_type_for_status,
    gateway_error_handler,
    openai_error_body,
    register_error_handlers,
    request_validation_handler,
)
from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.core.errors import (
    GATEWAY_MISCONFIGURED,
    MISSING_PROMPT_VARIABLE,
    UNKNOWN_MODEL,
    UNSUPPORTED_FIELD,
    GatewayError,
)
from llm_gateway.main import app as real_app

# ---------------------------------------------------------------------------
# 公共构造：最小 Request（处理器只读 scope 里的 path）与校验错误条目
# ---------------------------------------------------------------------------


def _request_for(path: str) -> Request:
    return Request(
        scope={"type": "http", "method": "POST", "path": path, "headers": [], "query_string": b""}
    )


def _extra_field_error(field: str = "top_k") -> dict[str, Any]:
    return {
        "type": "extra_forbidden",
        "loc": ("body", field),
        "msg": "Extra inputs are not permitted",
        "input": 5,
    }


def _missing_field_error(field: str = "model") -> dict[str, Any]:
    return {"type": "missing", "loc": ("body", field), "msg": "Field required", "input": {}}


def _body(response_json: dict[str, Any]) -> dict[str, Any]:
    return response_json["error"]


def _response_json(response: Response) -> dict[str, Any]:
    # JSONResponse.body 的声明类型是 bytes | memoryview，先归一成 bytes 再解析。
    return json.loads(bytes(response.body))


# ---------------------------------------------------------------------------
# 1. 纯函数：type 映射与错误体形态
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [
        (400, INVALID_REQUEST_ERROR),
        (401, INVALID_REQUEST_ERROR),
        (429, INVALID_REQUEST_ERROR),
        (499, INVALID_REQUEST_ERROR),  # 非 4/5xx 的注册表状态按 controller 的 4xx 归类
        (500, API_ERROR),
        (502, API_ERROR),
        (503, API_ERROR),
    ],
)
def test_error_type_for_status_mapping(status_code: int, expected_type: str) -> None:
    assert error_type_for_status(status_code) == expected_type


def test_openai_error_body_shape_is_message_type_code() -> None:
    # 三键齐全且 code 透传 None（缺键会让 SDK 端 e.code 的行为依赖 SDK 实现）。
    body = openai_error_body(None, "请求校验失败", 400)
    assert set(body["error"]) == {"message", "type", "code"}
    assert body["error"]["message"] == "请求校验失败"
    assert body["error"]["type"] == INVALID_REQUEST_ERROR
    assert body["error"]["code"] is None


# ---------------------------------------------------------------------------
# 2. 处理器直调：GatewayError -> OpenAI 体
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_error_renders_registered_triple_for_4xx() -> None:
    error = GatewayError(UNKNOWN_MODEL)
    response = await gateway_error_handler(_request_for("/v1/chat/completions"), error)
    assert response.status_code == error.status_code
    body = _body(_response_json(response))
    assert body["code"] == UNKNOWN_MODEL
    assert body["message"] == error.message  # 期望值取自实例，不在测试里重复冻结
    assert body["type"] == INVALID_REQUEST_ERROR


@pytest.mark.asyncio
async def test_gateway_error_renders_api_error_type_for_5xx() -> None:
    error = GatewayError(GATEWAY_MISCONFIGURED)
    response = await gateway_error_handler(_request_for("/v1/chat/completions"), error)
    assert response.status_code == error.status_code
    body = _body(_response_json(response))
    assert body["type"] == API_ERROR
    assert body["code"] == GATEWAY_MISCONFIGURED


@pytest.mark.asyncio
async def test_gateway_error_keeps_dynamic_message_but_stable_code() -> None:
    # message 动态覆盖是注册表默认值之上唯一合法的覆盖面（快照测试同款先例），
    # 适配层不得把它译回默认值，也不得让 code 随之漂移。
    error = GatewayError(MISSING_PROMPT_VARIABLE, message="缺少 Prompt 变量: product_name")
    response = await gateway_error_handler(_request_for("/v1/chat/completions"), error)
    body = _body(_response_json(response))
    assert body == {
        "message": "缺少 Prompt 变量: product_name",
        "type": INVALID_REQUEST_ERROR,
        "code": MISSING_PROMPT_VARIABLE,
    }


# ---------------------------------------------------------------------------
# 2. 处理器直调：RequestValidationError -> OpenAI 体
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_field_on_openai_surface_maps_to_unsupported_field() -> None:
    exc = RequestValidationError(errors=[_extra_field_error()], body={})
    response = await request_validation_handler(_request_for("/v1/chat/completions"), exc)
    assert response.status_code == 400
    body = _body(_response_json(response))
    assert body["code"] == UNSUPPORTED_FIELD
    assert body["type"] == INVALID_REQUEST_ERROR
    assert "top_k" in body["message"]


@pytest.mark.asyncio
async def test_extra_field_message_names_the_offending_field_path() -> None:
    # 嵌套白名单外字段报完整路径（去掉 FastAPI 的 body 前缀），去重后拼接。
    exc = RequestValidationError(
        errors=[_extra_field_error("stream_options.foo"), _extra_field_error("stream_options.foo")],
        body={},
    )
    response = await request_validation_handler(_request_for("/v1/chat/completions"), exc)
    body = _body(_response_json(response))
    assert body["message"] == "请求包含不支持的字段: stream_options.foo"


@pytest.mark.asyncio
async def test_missing_field_on_openai_surface_is_400_without_registered_code() -> None:
    # 缺字段/类型错没有稳定注册码（注册表封闭、M02 双冻结禁止扩员），code=None；
    # 形态仍为 OpenAI 三键，type 归 invalid_request_error（controller 裁决）。
    exc = RequestValidationError(errors=[_missing_field_error()], body={})
    response = await request_validation_handler(_request_for("/v1/chat/completions"), exc)
    assert response.status_code == 400
    body = _body(_response_json(response))
    assert set(body) == {"message", "type", "code"}
    assert body["code"] is None
    assert body["type"] == INVALID_REQUEST_ERROR
    assert "model" in body["message"]


@pytest.mark.asyncio
async def test_extra_field_takes_precedence_over_other_validation_errors() -> None:
    # 白名单与其他校验错并存时报 unsupported_field：不变量 #4 的语义是
    # "不接受白名单外的请求"，字段层面的问题先告知。
    exc = RequestValidationError(errors=[_missing_field_error(), _extra_field_error()], body={})
    response = await request_validation_handler(_request_for("/v1/chat/completions"), exc)
    body = _body(_response_json(response))
    assert body["code"] == UNSUPPORTED_FIELD


# ---------------------------------------------------------------------------
# 3. 应用装配：register_error_handlers 之后的真实 HTTP 行为
# ---------------------------------------------------------------------------


def _probe_app() -> FastAPI:
    # 探针 app：最小路由复刻 OpenAI 面的行为面，验证注册动作本身（含任务 B
    # 补齐的 404/405/500 兜底；真实端点的端到端覆盖在 tests/contract/）。
    app = FastAPI()
    register_error_handlers(app)

    @app.post("/v1/chat/completions")
    async def _chat(request: ChatCompletionRequest) -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/chat/completions/fail")
    async def _chat_fail(request: ChatCompletionRequest) -> dict[str, bool]:
        # 模拟真实端点直接 raise GatewayError 的形态：处理器应把注册表
        # 三元组渲染成 OpenAI 体（而非 FastAPI 默认 500）。
        raise GatewayError(UNKNOWN_MODEL)

    @app.post("/v1/chat/completions/explode")
    async def _explode(request: ChatCompletionRequest) -> dict[str, bool]:
        # 未捕获异常：兜底处理器应渲染 500 OpenAI 体而非 FastAPI 默认
        # plain text，且响应先于异常重新抛出返回。
        raise RuntimeError("内部炸弹（不应出现在响应体）")

    return app


@pytest.fixture()
def probe_client() -> TestClient:
    # raise_server_exceptions=False：Exception 兜底处理器发出响应后 Starlette
    # 会重新抛出原异常，宽松客户端才能断言已发出的兜底响应体。
    return TestClient(_probe_app(), raise_server_exceptions=False)


def test_probe_openai_surface_rejects_unknown_field_400(probe_client: TestClient) -> None:
    response = probe_client.post(
        "/v1/chat/completions",
        json={"model": "general-primary", "messages": [{"role": "user", "content": "hi"}], "top_k": 5},
    )
    assert response.status_code == 400
    body = _body(response.json())
    assert body["code"] == UNSUPPORTED_FIELD
    assert body["type"] == INVALID_REQUEST_ERROR


def test_probe_openai_surface_missing_model_400_without_code(probe_client: TestClient) -> None:
    response = probe_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    body = _body(response.json())
    assert body["code"] is None
    assert body["type"] == INVALID_REQUEST_ERROR


def test_probe_openai_surface_accepts_whitelisted_request(probe_client: TestClient) -> None:
    # 白名单内请求不受处理器影响，正常进入路由处理。
    response = probe_client.post(
        "/v1/chat/completions",
        json={
            "model": "general-primary",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt": {"name": "support", "version": "v1"},
        },
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_probe_openai_surface_renders_gateway_error_as_openai_body(probe_client: TestClient) -> None:
    # 端点 raise GatewayError 时的 HTTP 级翻译：code/status 取注册表，
    # type 按 4xx 归 invalid_request_error——真实端点可直接依赖此路径。
    response = probe_client.post(
        "/v1/chat/completions/fail",
        json={"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    body = _body(response.json())
    assert body["code"] == UNKNOWN_MODEL
    assert body["type"] == INVALID_REQUEST_ERROR


def test_probe_unknown_path_returns_openai_404(probe_client: TestClient) -> None:
    # 任务 B 补缺口：404 走 OpenAI 体（code=None，注册表不扩员），type 按
    # 4xx 归 invalid_request_error。
    response = probe_client.get("/v1/nonexistent")
    assert response.status_code == 404
    body = _body(response.json())
    assert set(body) == {"message", "type", "code"}
    assert body["code"] is None
    assert body["type"] == INVALID_REQUEST_ERROR
    assert body["message"] == "Not Found"


def test_probe_wrong_method_returns_openai_405(probe_client: TestClient) -> None:
    # 任务 B 补缺口：405（Starlette 对已匹配路径方法不匹配的 HTTPException）
    # 同样是 OpenAI 体。
    response = probe_client.get("/v1/chat/completions")
    assert response.status_code == 405
    body = _body(response.json())
    assert body["code"] is None
    assert body["type"] == INVALID_REQUEST_ERROR
    assert body["message"] == "Method Not Allowed"


def test_probe_unhandled_exception_returns_openai_500(probe_client: TestClient) -> None:
    # 任务 B 补缺口：未捕获异常兜底 500 + api_error + code=None，message 固定
    # 文案、不回显异常内容（design.md §3.9：响应体不出现异常细节）。
    response = probe_client.post(
        "/v1/chat/completions/explode",
        json={"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    body = _body(response.json())
    assert set(body) == {"message", "type", "code"}
    assert body["type"] == API_ERROR
    assert body["code"] is None
    assert "内部炸弹" not in body["message"]


# ---------------------------------------------------------------------------
# 4. 真实 app：注册处理器后的真实端点行为
# ---------------------------------------------------------------------------


def test_real_app_extra_field_maps_to_unsupported_field_400() -> None:
    # 真实 app 端到端钉住（旧端点删除前本用例钉的是"旧端点保持 422 detail"，
    # 收口后 OpenAI 面即全量面）：/v1/chat/completions 白名单外字段 400
    # unsupported_field，validation 失败发生在任何凭据/上游依赖之前。
    client = TestClient(real_app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "general-primary", "messages": [{"role": "user", "content": "hi"}], "top_k": 5},
    )
    assert response.status_code == 400
    body = _body(response.json())
    assert body["code"] == UNSUPPORTED_FIELD
    assert body["type"] == INVALID_REQUEST_ERROR
