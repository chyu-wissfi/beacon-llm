"""模型目录 / Prompt 模板库 / 价格表：静态数据源。

常量集中在本模块是为了给 M02/M07 留出替换缝：届时把这里换成配置/文件
加载，编排层（invocation/prompt_service/trace_service）零改动。语义约束：
MODEL_CONFIGS 保持 import 时读环境变量（M01 与 demo 行为等价）。
"""

import os

from llm_gateway.core.schemas import ModelConfig, PromptTemplate

MODEL_CONFIGS = {
    "general-primary": ModelConfig(
        provider_model=os.getenv("PRIMARY_PROVIDER_MODEL", "deepseek-v4-flash"),
        base_url=os.getenv("PRIMARY_BASE_URL", "https://api.deepseek.com"),
        api_key_env="DEEPSEEK_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
    ),
    "general-backup": ModelConfig(
        provider_model=os.getenv("BACKUP_PROVIDER_MODEL", "deepseek-chat"),
        base_url=os.getenv("BACKUP_BASE_URL", "https://api.deepseek.com"),
        api_key_env="DEEPSEEK_BACKUP_API_KEY",
        supports_structured_output=True,
        structured_output_mode="json_object",
    ),
}

PROMPT_TEMPLATES = {
    ("knowledge_decision", "v1"): PromptTemplate(
        name="knowledge_decision",
        version="v1",
        system_template="你是${product_name}的知识库决策器。资料不足时搜索，资料充分时结束回答。不得编造制度内容。",
    )
}

PRICE_PER_MILLION = {
    "general-primary": {"input": 1.0, "output": 4.0},
    "general-backup": {"input": 0.8, "output": 3.2},
}
