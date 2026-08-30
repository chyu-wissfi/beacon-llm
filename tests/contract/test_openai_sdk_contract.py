"""openai 官方 SDK 视角的契约测试（M03 任务 6）。

test_chat_api.py / test_field_whitelist.py 已用 raw httpx 钉住线格式与行为边界
（响应键集合、SSE 原始行、错误体三元组）；本文件补上 spec 任务 6 要求的另一半：
**调用方用 openai 官方 SDK 零改动接入**（不变量 #3）的端到端证明——
`AsyncOpenAI(base_url=..., http_client=自定义 httpx.AsyncClient(ASGITransport))`
直打本地 app，断言 SDK 消费方实际感知的产物：chat.completion / chunk 能否被
解析成 SDK 模型、HTTP 错误体能否映射为 SDK 异常且 e.code 正确、流内错误事件
在 SDK 侧的真实表现。与 raw 测试的分工：线格式细节不在本文件重复断言（同一
契约不两处维护），既有 raw 用例零改动、语义继续有效。

实现要点（M01 实测坑的延续）：
- openai 3.x 默认走 httpx2，respx 0.23.1 只 patch httpx1 层。本文件的
  AsyncOpenAI 显式传 httpx1 AsyncClient（ASGITransport）直打 app，天然在
  httpx1 层；网关内部的 openai 客户端由 conftest 的 _openai_legacy_httpx
  autouse shim 归一到 httpx1，respx 拦截上游才生效——两层缺一不可。
- max_retries=0：SDK 默认对 5xx 自动重试，会把"一次网关调用"放大成三次，
  上游请求计数的断言（fallback 链 = 2 主 + 2 备）随之失真；关掉后计数才
  对齐"SDK 一次调用 <-> 网关一条完整 fallback 链"。
"""

import json
from typing import Any, cast

import httpx
import httpx2
import openai
import pytest
import pytest_asyncio
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llm_gateway.main import app
from tests.contract.helpers import (
    BACKUP_PROVIDER_MODEL,
    BACKUP_URL,
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    SSE_HEADERS,
    completion,
    sse_body,
    sse_stream_then_break,
)

pytestmark = pytest.mark.asyncio

# 消息参数用 dict 字面量：openai SDK 对 messages 收 TypedDict 联合类型，dict
# 形态即合法载荷，无须引入 SDK 模型类型；Any 标注避免 pyright 对 TypedDict
# 协变（dict[str, str] -> TypedDict 联合）报不兼容。
_MESSAGES: Any = [{"role": "user", "content": "你好"}]


@pytest_asyncio.fixture
async def sdk():
    # AsyncOpenAI 不另行建连：传自定义 httpx1 AsyncClient（ASGITransport）后，
    # SDK 的请求全部经 ASGI 直达 app，不起端口、不走网络。base_url 只需提供
    # SDK 拼接 /chat/completions 的锚点，host 部分被 ASGITransport 忽略。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as http_client:
        client = openai.AsyncOpenAI(
            base_url="http://gateway.test/v1",
            api_key="test-sdk-key",  # 网关不校验调用方身份，取值仅为满足 SDK 构造
            # cast 是运行期恒等：legacy httpx1 分支在运行期被官方支持
            # （_base_client.is_legacy_httpx_async_client 显式识别），仅类型标注
            # 只声明 httpx2.AsyncClient（与 conftest 的 legacy shim 同一缺口）。
            http_client=cast("httpx2.AsyncClient", http_client),
            max_retries=0,  # 见模块 docstring：SDK 级重试会放大上游计数，必须关闭
        )
        yield client


# ---------------------------------------------------------------------------
# 非流式：chat.completion 被 SDK 解析
# ---------------------------------------------------------------------------


async def test_sdk_non_stream_parses_chat_completion(sdk, mock_upstream):
    # spec 任务 6 非流式：SDK 发标准请求 -> 网关 OpenAI 体 -> SDK 解析成
    # ChatCompletion，字段口径与 raw 测试一致；isinstance 即"SDK 可零改动消费"。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok", prompt_tokens=13, completion_tokens=5)
    )
    result = await sdk.chat.completions.create(model="general-primary", messages=_MESSAGES)
    assert isinstance(result, ChatCompletion)
    assert result.object == "chat.completion"
    assert result.model == "general-primary"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].message.content == "ok"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage == CompletionUsage(prompt_tokens=13, completion_tokens=5, total_tokens=18)
    # SDK 模型对未知字段是 extra="allow"（静默收进 model_extra）：网关若夹带
    # 扩展字段，SDK 消费方不会报错但也不应看到治理信息——attempts/request_id
    # 等"治理走 /v1/traces"不变量在 SDK 视角的直接体现。
    assert not result.model_extra


# ---------------------------------------------------------------------------
# 流式：chunk 增量被 SDK 迭代
# ---------------------------------------------------------------------------


async def test_sdk_stream_iterates_deltas_and_ends_cleanly(sdk, mock_upstream):
    # spec 任务 6 流式：上游增量被 SDK 迭代为 ChatCompletionChunk，[DONE] 处
    # 干净收尾。M04 收紧（M03 账本预告的唯一改动点）：内容块 finish_reason 全
    # None，[DONE] 前新增终态块（finish_reason=="stop"、delta 空串）——依赖
    # 终态原因判断截断/终止原因的调用方从这里拿到信号。
    mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, content=sse_body(["你", "好"]), headers=SSE_HEADERS
    )
    stream = await sdk.chat.completions.create(model="general-primary", messages=_MESSAGES, stream=True)
    chunks: list[ChatCompletionChunk] = [chunk async for chunk in stream]
    content_chunks = chunks[:-1]
    assert [chunk.choices[0].delta.content for chunk in content_chunks] == ["你", "好"]
    for chunk in content_chunks:
        assert isinstance(chunk, ChatCompletionChunk)
        assert chunk.object == "chat.completion.chunk"
        assert chunk.model == "general-primary"
        # 内容块全程无终态信号。
        assert chunk.choices[0].finish_reason is None
    # 终态块：SDK 消费方可直接读到 finish_reason。
    terminal = chunks[-1]
    assert terminal.choices[0].finish_reason == "stop"
    assert terminal.choices[0].delta.content == ""
    # 同一流内 id/created 一致（SDK 消费方靠它聚合增量，终态块含在内）。
    assert len({chunk.id for chunk in chunks}) == 1
    assert len({chunk.created for chunk in chunks}) == 1


async def test_sdk_stream_include_usage_emits_usage_chunk(sdk, mock_upstream):
    # spec 任务 6 流式（M04 新增）：include_usage 时 [DONE] 前附 usage chunk，
    # 形态取 OpenAI 惯例：choices 为空、usage 三键口径；开关同时到达上游。
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, content=sse_body(["你", "好"], include_usage=True), headers=SSE_HEADERS
    )
    stream = await sdk.chat.completions.create(
        model="general-primary",
        messages=_MESSAGES,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks: list[ChatCompletionChunk] = [chunk async for chunk in stream]
    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].choices == []
    assert usage_chunks[0].usage == CompletionUsage(
        prompt_tokens=13, completion_tokens=5, total_tokens=18
    )
    # 其余块（内容块 + 终态块）不带用量。
    assert all(chunk.usage is None for chunk in chunks if chunk not in usage_chunks)
    upstream_body = json.loads(route.calls.last.request.content)
    assert upstream_body["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# 流内失败：错误事件在 SDK 侧的真实行为（任务 B 移交的专门验证点）
# ---------------------------------------------------------------------------


async def test_sdk_in_stream_failure_raises_apierror_with_registry_code(sdk, mock_upstream):
    # 不变量：首块之后失败绝不重生成（调用方已拿到一半文本）；错误以 OpenAI
    # 风格错误事件收场、不发 [DONE]。
    # 实测口径（openai 3.6.0 _streaming.py）：SDK 不静默吞掉流内错误事件——
    # 迭代中抛 openai.APIError（基类：流内失败时 HTTP 状态恒 200，不走
    # BadRequestError/InternalServerError 等 APIStatusError 子类的状态码映射），
    # e.code/e.type/e.message 从错误事件的 error 对象解析。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        return_value=httpx.Response(200, content=sse_stream_then_break(["首块"]), headers=SSE_HEADERS)
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).respond(
        200, content=sse_body(["备用模型的文本"]), headers=SSE_HEADERS
    )
    stream = await sdk.chat.completions.create(model="general-primary", messages=_MESSAGES, stream=True)
    received: list[ChatCompletionChunk] = []
    with pytest.raises(openai.APIError) as exc_info:
        async for chunk in stream:
            received.append(chunk)
    # 错误前已交付的增量完好保留在调用方手里（半截文本是消费方必须自己处理的现实）。
    assert [chunk.choices[0].delta.content for chunk in received] == ["首块"]
    assert exc_info.value.code == "upstream_stream_failed"  # 注册表稳定码（双冻结）
    assert exc_info.value.type == "api_error"  # 注册表 type 口径经 SDK 解析
    assert exc_info.value.message == "上游流式输出失败"  # 注册表 message（双冻结）
    # 非 APIStatusError（实测：无 status_code 属性）——调用方按 HTTP 状态分类的
    # except 分支（BadRequestError 等）捕获不到流内失败，只能兜 openai.APIError。
    assert not isinstance(exc_info.value, openai.APIStatusError)
    # 备用模型零调用：SDK 消费方视角与 raw 断言一致，重生成防线对 SDK 同样成立。
    assert backup.call_count == 0
    assert primary.call_count == 1


# ---------------------------------------------------------------------------
# HTTP 错误体：SDK 异常映射与 e.code（spec 任务 6"错误体可被 SDK 解析"）
# ---------------------------------------------------------------------------


async def test_sdk_unsupported_field_maps_to_bad_request_error_no_upstream(sdk, mock_upstream):
    # spec 任务 6：白名单外字段被拒（400）且无上游请求发生（respx 计数为 0）。
    # 调用方经由 SDK 的官方扩展通道 extra_body 传入白名单外字段——这正是
    # ADR-0001（OpenAI 兼容 + extra_body）里网关必须防住的接入形态。
    upstream = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("不应被调用")
    )
    with pytest.raises(openai.BadRequestError) as exc_info:
        await sdk.chat.completions.create(
            model="general-primary", messages=_MESSAGES, extra_body={"top_k": 5}
        )
    # HTTP 400 走 APIStatusError 子类映射：e.code 从错误体的 error.code 解析。
    assert exc_info.value.code == "unsupported_field"
    assert exc_info.value.status_code == 400
    assert exc_info.value.type == "invalid_request_error"
    # SDK 把错误体保持为 dict（实测口径）；e.message 是 "Error code: 400 - {…}"
    # 概要格式，注册表文案在 body["message"]，要拿人类可读信息需从 body 取。
    error_body = exc_info.value.body
    assert isinstance(error_body, dict)
    assert "top_k" in error_body["message"]
    assert upstream.call_count == 0  # 拒绝发生在任何上游调用之前


async def test_sdk_fallback_exhaustion_maps_to_internal_server_error(sdk, mock_upstream):
    # 5xx 半边：主备全耗尽 -> 502 model_unavailable -> SDK InternalServerError
    # （>=500 的映射），e.code 仍从注册表口径解析。
    # max_retries=0 在此承担语义职责：SDK 一次调用 = 网关一条完整 fallback 链
    # （2 主 + 2 备），若 SDK 默认重试 5xx，计数会被放大成三倍。
    primary = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("primary down")
    )
    backup = mock_upstream.post(BACKUP_URL, json__model=BACKUP_PROVIDER_MODEL).mock(
        side_effect=httpx.ConnectError("backup down")
    )
    with pytest.raises(openai.InternalServerError) as exc_info:
        await sdk.chat.completions.create(model="general-primary", messages=_MESSAGES)
    assert exc_info.value.code == "model_unavailable"
    assert exc_info.value.status_code == 502
    assert exc_info.value.type == "api_error"
    # fallback 链在 SDK 消费方身后完整走完：2 次主模型 + 2 次备用。
    assert primary.call_count == 2
    assert backup.call_count == 2
