"""治理类只读端点。design.md 布局中 /v1/models 与 /v1/traces 同属本文件；
M01 暂只迁移 /v1/traces。"""

from fastapi import APIRouter

from llm_gateway.core.schemas import CallTrace
from llm_gateway.services.trace_service import CALL_TRACES

router = APIRouter()


@router.get("/v1/traces", response_model=list[CallTrace])
async def list_traces() -> list[CallTrace]:
    # 暴露调用审计记录，供成本分析与故障排查使用。
    return CALL_TRACES
