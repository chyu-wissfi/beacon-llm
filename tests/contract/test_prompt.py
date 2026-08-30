"""Prompt 模板资产化与热加载契约测试（M07）。

覆盖 spec 任务 4 的五个行为面：
1. `prompt: {name, version, variables}` 经 openai SDK 的 extra_body 通道到达，
   渲染正确、系统消息注入首位；
2. 缺变量 → 400 且上游请求数 == 0（不变量 #11：调用模型前失败）；
3. 热加载：启动后新增模板文件，下一次请求立即可用；
4. 坏文件：写入非法 yaml，下一次请求仍用旧版成功 + 日志含 error；
5. 调用方不能提交模板正文（extra=forbid 的断言测试）。

模板目录经 tmp_templates 夹具注入临时目录的 loader 实例（"启动"时刻 =
夹具构建时刻），其后对磁盘的改动即"启动后变更"；mtime 显式前推规避文件
系统时间戳粒度问题（详见 tests/unit/prompt/test_loader.py 同款说明）。
"""

import json
import logging
from pathlib import Path
from typing import Any, cast

import httpx
import httpx2
import openai
import pytest
import pytest_asyncio

from llm_gateway.main import app
from llm_gateway.prompt.loader import PromptTemplateLoader
from llm_gateway.services import prompt_service
from tests.contract.helpers import (
    PRIMARY_PROVIDER_MODEL,
    PRIMARY_URL,
    VALID_CALLER_KEY,
    chat_request,
    completion,
)
from tests.unit.prompt.test_loader import write_template

pytestmark = pytest.mark.asyncio

CHAT_PATH = "/v1/chat/completions"

V1_BODY = "你是${product_name}的知识库决策器。"
V2_BODY = "第二版${product_name}决策器。"


def _error_body(response_json: dict[str, Any]) -> dict[str, Any]:
    return response_json["error"]


@pytest.fixture
def tmp_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # "启动"：先落 v1 再构建 loader；monkeypatch 到 prompt_service 的模块级
    # TEMPLATES（渲染期查全局名，注入由此生效）。
    write_template(tmp_path, "knowledge_decision", "v1", system_template=V1_BODY)
    monkeypatch.setattr(prompt_service, "TEMPLATES", PromptTemplateLoader(tmp_path))
    return tmp_path


@pytest_asyncio.fixture
async def sdk():
    # openai SDK 直打 app（同 test_openai_sdk_contract 的构造）：验证
    # extra_body 通道——prompt 扩展字段正是经它提交的（ADR-0001）。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as http_client:
        client = openai.AsyncOpenAI(
            base_url="http://gateway.test/v1",
            api_key=VALID_CALLER_KEY,
            http_client=cast("httpx2.AsyncClient", http_client),
            max_retries=0,  # SDK 级重试会放大上游计数，必须关闭
        )
        yield client


# ---------------------------------------------------------------------------
# extra_body 到达 + 渲染正确 + 系统消息首位
# ---------------------------------------------------------------------------


async def test_prompt_selection_via_extra_body_renders_system_message_first(sdk, mock_upstream, tmp_templates):
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )
    response = await sdk.chat.completions.create(
        model="general-primary",
        messages=[{"role": "user", "content": "你好"}],
        extra_body={"prompt": {"name": "knowledge_decision", "version": "v1", "variables": {"product_name": "Beacon"}}},
    )
    assert response.choices[0].message.content == "ok"

    upstream_body = json.loads(route.calls.last.request.content)
    system_message = upstream_body["messages"][0]
    assert system_message["role"] == "system"
    assert system_message["content"] == "你是Beacon的知识库决策器。"
    # 调用方消息原样跟在渲染后的系统消息之后。
    assert upstream_body["messages"][1] == {"role": "user", "content": "你好"}


# ---------------------------------------------------------------------------
# 缺变量：调用模型前失败（不变量 #11）
# ---------------------------------------------------------------------------


async def test_missing_variable_fail_before_upstream(client, mock_upstream, tmp_templates):
    # 证明口径：注册一条兜底路由统计任何上游请求——400 发生时计数必须为 0。
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
# 缺变量 / 热加载 / 坏文件保留旧版（验收 -k "missing_variable_or_hot_reload"）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["missing_variable", "hot_reload", "bad_file_keeps_old"])
async def test_missing_variable_or_hot_reload(scenario, client, mock_upstream, tmp_templates, caplog):
    caplog.set_level(logging.ERROR, logger="llm_gateway")
    route = mock_upstream.post(PRIMARY_URL, json__model=PRIMARY_PROVIDER_MODEL).respond(
        200, json=completion("ok")
    )

    async def _post(version: str, variables: dict[str, str]) -> httpx.Response:
        return await client.post(CHAT_PATH, json=chat_request(prompt={
            "name": "knowledge_decision", "version": version, "variables": variables,
        }))

    if scenario == "missing_variable":
        response = await _post("v1", {})
        assert response.status_code == 400
        assert _error_body(response.json())["code"] == "missing_prompt_variable"
        assert route.call_count == 0

    elif scenario == "hot_reload":
        # 启动后新增 v2 文件：下一次请求立即可用，无需重启。
        write_template(tmp_templates, "knowledge_decision", "v2", system_template=V2_BODY)
        response = await _post("v2", {"product_name": "Beacon"})
        assert response.status_code == 200
        upstream_body = json.loads(route.calls.last.request.content)
        assert upstream_body["messages"][0]["content"] == "第二版Beacon决策器。"

    else:  # bad_file_keeps_old
        # 先让 v2 正常加载，再用非法 yaml 覆盖：下一次请求仍用旧版成功。
        write_template(tmp_templates, "knowledge_decision", "v2", system_template=V2_BODY)
        first = await _post("v2", {"product_name": "Beacon"})
        assert first.status_code == 200
        write_template(tmp_templates, "knowledge_decision", "v2", raw="name: [unclosed bracket")
        second = await _post("v2", {"product_name": "Beacon"})
        assert second.status_code == 200
        upstream_body = json.loads(route.calls.last.request.content)
        assert upstream_body["messages"][0]["content"] == "第二版Beacon决策器。"
        # 可观测面：加载失败有 error 日志，含文件路径与原因。
        errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
        assert any("v2.yaml" in message for message in errors)


# ---------------------------------------------------------------------------
# 调用方不能提交模板正文
# ---------------------------------------------------------------------------


async def test_caller_cannot_submit_template_body(client, mock_upstream, tmp_templates):
    # prompt 选择体里混入白名单外键（模板正文）：400 unsupported_field，
    # 且失败在上游之前——正文注入无从谈起。
    upstream_any = mock_upstream.route()
    response = await client.post(
        CHAT_PATH,
        json=chat_request(prompt={
            "name": "knowledge_decision",
            "version": "v1",
            "variables": {"product_name": "Beacon"},
            "system_template": "调用方注入的正文",
        }),
    )
    assert response.status_code == 400
    error = _error_body(response.json())
    assert error["code"] == "unsupported_field"
    assert upstream_any.call_count == 0
