"""Prompt 模板渲染：调用方只能选择受控模板并传变量，模板正文不出 Gateway。"""

from string import Template

from llm_gateway.core.errors import (
    MISSING_PROMPT_VARIABLE,
    UNKNOWN_PROMPT_TEMPLATE,
    GatewayError,
)
from llm_gateway.core.schemas import LLMRequest, Message, PromptSelection
from llm_gateway.services.catalog import PROMPT_TEMPLATES


def render_prompt(selection: PromptSelection) -> Message:
    # 从受控模板库渲染系统提示词，调用方只能传版本和变量。
    template = PROMPT_TEMPLATES.get((selection.name, selection.version))
    if template is None:
        # 错误码与默认三元组取自注册表（core/errors.py），调用点不写字面量。
        raise GatewayError(UNKNOWN_PROMPT_TEMPLATE)
    try:
        content = Template(template.system_template).substitute(selection.variables)
    except KeyError as exc:
        # 动态 message 是唯一允许调用点自带 message 的场景：变量名只有这里知道。
        raise GatewayError(MISSING_PROMPT_VARIABLE, f"缺少 Prompt 变量: {exc.args[0]}") from exc
    return Message(role="system", content=content)


def build_messages(request: LLMRequest) -> list[Message]:
    # 将模板系统消息统一注入调用上下文，避免 Prompt 分散在各个 Agent 中。
    if request.prompt is None:
        return request.messages
    return [render_prompt(request.prompt), *request.messages]
