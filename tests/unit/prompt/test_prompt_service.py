"""services/prompt_service.py 单测：渲染与错误码（M07）。

模板来源经 monkeypatch 注入临时目录的 loader 实例（与契约测试同一接缝）；
断言行为边界：渲染结果、错误码三元组、系统消息注入位置、调用方不可提交
模板正文（extra=forbid）。
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import LLMRequest, Message, PromptSelection
from llm_gateway.prompt.loader import PromptTemplateLoader
from llm_gateway.services import prompt_service
from llm_gateway.services.prompt_service import build_messages, render_prompt
from tests.unit.prompt.test_loader import write_template


@pytest.fixture
def templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # 唯一渲染出口消费模块级 TEMPLATES：注入临时目录实例，不碰仓库真实模板。
    write_template(tmp_path, "knowledge_decision", "v1", system_template="你是${product_name}的知识库决策器。")
    monkeypatch.setattr(prompt_service, "TEMPLATES", PromptTemplateLoader(tmp_path))
    return tmp_path


def _selection(**overrides: object) -> PromptSelection:
    payload: dict[str, object] = {"name": "knowledge_decision", "version": "v1", "variables": {}}
    payload.update(overrides)
    return PromptSelection.model_validate(payload)


def test_render_prompt_substitutes_variables(templates: Path) -> None:
    message = render_prompt(_selection(variables={"product_name": "Beacon"}))
    assert message.role == "system"
    assert message.content == "你是Beacon的知识库决策器。"


def test_unknown_template_raises_registry_error(templates: Path) -> None:
    with pytest.raises(GatewayError) as excinfo:
        render_prompt(_selection(name="no_such_template"))
    assert excinfo.value.code == "unknown_prompt_template"
    assert excinfo.value.status_code == 400


def test_missing_variable_raises_with_variable_name(templates: Path) -> None:
    # 动态 message：注册表默认文案之上拼接缺失变量名（唯一允许自带 message 的场景）。
    with pytest.raises(GatewayError) as excinfo:
        render_prompt(_selection(variables={}))
    assert excinfo.value.code == "missing_prompt_variable"
    assert excinfo.value.status_code == 400
    assert "product_name" in excinfo.value.message


def test_build_messages_injects_system_message_first(templates: Path) -> None:
    request = LLMRequest(
        model="general-primary",
        messages=[Message(role="user", content="你好")],
        prompt=_selection(variables={"product_name": "Beacon"}),
    )
    messages = build_messages(request)
    assert messages[0].role == "system"
    assert "Beacon" in messages[0].content
    assert messages[1:] == request.messages  # 调用方消息顺序原样跟在后面


def test_build_messages_without_prompt_is_passthrough(templates: Path) -> None:
    request = LLMRequest(model="general-primary", messages=[Message(role="user", content="你好")])
    assert build_messages(request) is request.messages


def test_caller_cannot_submit_template_body(templates: Path) -> None:
    # 不变量：调用方只能选择受控模板并传变量——提交模板正文（或任何白名单外
    # 键）被 extra=forbid 在解析期拦下，模板正文永远不出 Gateway 资产。
    with pytest.raises(ValidationError):
        PromptSelection.model_validate(
            {
                "name": "knowledge_decision",
                "version": "v1",
                "variables": {},
                "system_template": "调用方注入的正文",
            }
        )
