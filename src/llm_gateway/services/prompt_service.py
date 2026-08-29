"""Prompt 模板渲染：调用方只能选择受控模板并传变量，模板正文不出 Gateway。"""

from string import Template

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import LLMRequest, Message, PromptSelection
from llm_gateway.services.catalog import PROMPT_TEMPLATES


def render_prompt(selection: PromptSelection) -> Message:
    # 从受控模板库渲染系统提示词，调用方只能传版本和变量。
    template = PROMPT_TEMPLATES.get((selection.name, selection.version))
    if template is None:
        raise GatewayError("unknown_prompt_template", "Prompt 模板不存在", 400)
    try:
        content = Template(template.system_template).substitute(selection.variables)
    except KeyError as exc:
        raise GatewayError("missing_prompt_variable", f"缺少 Prompt 变量: {exc.args[0]}", 400) from exc
    return Message(role="system", content=content)


def build_messages(request: LLMRequest) -> list[Message]:
    # 将模板系统消息统一注入调用上下文，避免 Prompt 分散在各个 Agent 中。
    if request.prompt is None:
        return request.messages
    return [render_prompt(request.prompt), *request.messages]
