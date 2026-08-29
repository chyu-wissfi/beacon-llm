"""模型目录 / Prompt 模板库 / 价格表：M02 配置中心化的对外接缝。

模型拓扑与价格已迁至 config/*.yaml（loader 见 core/config.py，导入期 fail-fast）；
本模块保留 M01 的导入面——编排层（invocation/trace_service/prompt_service）与
契约测试仍从这里 import MODEL_CONFIGS / PROMPT_TEMPLATES / PRICE_PER_MILLION，
既有消费字段取值与迁移前逐字等价。PROMPT_TEMPLATES 按用户裁决留在代码，
M07 再做文件化（模板是运行资产，热加载语义见 design.md §5）。
"""

from llm_gateway.core.config import CONFIG
from llm_gateway.core.schemas import PromptTemplate

MODEL_CONFIGS = CONFIG.models

PROMPT_TEMPLATES = {
    ("knowledge_decision", "v1"): PromptTemplate(
        name="knowledge_decision",
        version="v1",
        system_template="你是${product_name}的知识库决策器。资料不足时搜索，资料充分时结束回答。不得编造制度内容。",
    )
}

# trace_service 以 price["input"] 的字典形态取价：这是 M01 确定的消费面，
# 这里把校验后的 PriceEntry 摊平回字典，编排层零改动。
PRICE_PER_MILLION = {
    name: {"input": entry.input, "output": entry.output} for name, entry in CONFIG.prices.items()
}
