"""声明式 fallback 链的路由（M06 任务 2）。

候选列表来自 ModelConfig.fallback（M02 的声明式占位，本里程碑起消费；
config 层已在启动期校验链上引用的模型必然存在）。路由理由是"为什么离开
上一个候选"的解释，随编排推进逐条追加，终态时整体进 trace 的 route_reason：

- 次数耗尽：``general-primary: 3 attempts exhausted (model_unavailable)``
  ——括号里是注册表稳定码而非"timeout"字面：provider 层把连接/超时/5xx
  一律映射为 MODEL_UNAVAILABLE，编排层无从区分，不杜撰更细的语义；
- 熔断打开：``general-primary: circuit_open``（候选模型熔断中，直接跳过）；
- 能力不符：``general-backup: structured_output_unsupported``（fallback 候选
  不支持 response_schema，不等价的 fallback 绝不发生）。

本模块只求值链与理由字符串，不碰状态（无 IO、无时钟）——理由的"何时追加"
由编排状态机决定，便于单测穷举。
"""

from llm_gateway.core.schemas import ModelConfig
from llm_gateway.services.catalog import MODEL_CONFIGS


def build_chain(primary: str, configs: dict[str, ModelConfig] | None = None) -> list[str]:
    # 求值声明式 fallback 链：primary + 声明顺序的 fallback，去重保序——
    # 声明顺序即优先级语义，同一模型重复出现没有第二次尝试的含义（预算里
    # 的重试 already 覆盖"同一模型再来一次"）。
    model_configs = MODEL_CONFIGS if configs is None else configs
    chain = [primary]
    for target in model_configs[primary].fallback:
        if target not in chain:
            chain.append(target)
    return chain


def exhausted_reason(model: str, attempts: int, code: str) -> str:
    return f"{model}: {attempts} attempts exhausted ({code})"


def circuit_open_reason(model: str) -> str:
    return f"{model}: circuit_open"


def unsupported_reason(model: str) -> str:
    return f"{model}: structured_output_unsupported"
