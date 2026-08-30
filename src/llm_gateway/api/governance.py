"""治理类只读端点：/v1/models（OpenAI 面）与 /v1/traces（网关审计域）。

design.md §3.1：/v1/models 返回平台模型列表，供调用方发现可用模型；/v1/traces
保留内部 CallTrace 形态——审计是网关自己的域，不属于 OpenAI 协议，故不套
OpenAI 包装。
"""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from llm_gateway.core.schemas import CallTrace
from llm_gateway.services.catalog import MODEL_CONFIGS
from llm_gateway.services.trace_service import CALL_TRACES

router = APIRouter()

# 平台自营模型的 owned_by 标识：模型表内的全部条目都由本网关运营，M03 无
# 多运营主体区分，统一标注（OpenAI Model 对象要求该键，SDK 模型必填）。
_OWNED_BY: Literal["beacon-llm"] = "beacon-llm"


class ModelObject(BaseModel):
    # OpenAI Model 对象的最小标准形态。created 取 0 占位：模型配置（config/
    # models.yaml）无时间戳概念，语义是"未知"，客户端不应据此排序；SDK 端
    # 该键必填，缺键会让 openai SDK 解析直接失败。
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["model"]
    created: int
    owned_by: str


class ModelList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"]
    data: list[ModelObject]


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    # 数据源必须是配置中心的加载产物（services/catalog.py 转发的 MODEL_CONFIGS），
    # 不得另建模型表绕过 core/config.py——模型白名单的唯一事实来源（controller
    # 裁决）。id 即平台模型名：调用方请求里的 model 字段用的就是它（CONTEXT.md：
    # 平台模型表即白名单）；provider_model 是网关内部坐标，不出网关。
    return ModelList(
        object="list",
        data=[
            ModelObject(id=name, object="model", created=0, owned_by=_OWNED_BY)
            for name in MODEL_CONFIGS
        ],
    )


@router.get("/v1/traces", response_model=list[CallTrace])
async def list_traces() -> list[CallTrace]:
    # 暴露调用审计记录，供成本分析与故障排查使用。
    return CALL_TRACES
