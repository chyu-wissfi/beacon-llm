"""/v1/llm 与 /v1/llm/stream 端点：只做协议适配，业务语义在 services/ 层。"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from llm_gateway.core.errors import (
    UNSUPPORTED_COMBINATION,
    USE_STREAM_ENDPOINT,
    GatewayError,
)
from llm_gateway.core.schemas import LLMRequest, LLMResponse
from llm_gateway.services.invocation import (
    call_with_fallback,
    stream_with_fallback,
    validate_model,
)
from llm_gateway.services.prompt_service import build_messages

router = APIRouter()


@router.post("/v1/llm", response_model=LLMResponse)
async def create_llm_response(request: LLMRequest) -> LLMResponse:
    # FastAPI 在入口校验请求、在 response_model 校验统一响应出口。
    try:
        # 流式请求引导到专用端点：统一走 GatewayError -> 下方 handler 的适配
        # 路径（原为手工拼 detail 的 HTTPException，与 handler 重复）。
        if request.stream:
            raise GatewayError(USE_STREAM_ENDPOINT)
        return await call_with_fallback(request)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/v1/llm/stream")
async def create_stream(request: LLMRequest) -> StreamingResponse:
    # 提供独立流式入口，明确禁止与 Structured Output 混用。
    try:
        # 与 Structured Output 互斥的校验同样统一走 GatewayError 路径。
        if request.response_schema is not None:
            raise GatewayError(UNSUPPORTED_COMBINATION)
        validate_model(request.model, None)
        build_messages(request)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc
    return StreamingResponse(stream_with_fallback(request), media_type="text/event-stream")
