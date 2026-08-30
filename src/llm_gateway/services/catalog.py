"""模型目录 / 价格表：M02 配置中心化的对外接缝。

模型拓扑与价格已迁至 config/*.yaml（loader 见 core/config.py，导入期 fail-fast）；
本模块保留 M01 的导入面——编排层（invocation/trace_service）与契约测试仍从
这里 import MODEL_CONFIGS / PRICE_PER_MILLION，既有消费字段取值与迁移前逐字等价。
PROMPT_TEMPLATES 已随 M07 迁至文件资产（templates/<name>/<version>.yaml，
prompt/loader.py 热加载），唯一消费方 prompt_service 改从 loader 取，
catalog 不再保留模板面。
"""

from llm_gateway.core.config import CONFIG

MODEL_CONFIGS = CONFIG.models

# 价格表版本快照的导出面（M06）：RunContext 构建时快照进 Run 上下文，
# 与 MODEL_CONFIGS 同理——catalog 是 M02 配置中心化的对外接缝，编排层
# 不绕过它直接读 CONFIG。
PRICE_VERSION = CONFIG.price_version

# trace_service 以 price["input"] 的字典形态取价：这是 M01 确定的消费面，
# 这里把校验后的 PriceEntry 摊平回字典，编排层零改动。
PRICE_PER_MILLION = {
    name: {"input": entry.input, "output": entry.output} for name, entry in CONFIG.prices.items()
}
