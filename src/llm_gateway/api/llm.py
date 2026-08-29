"""/v1/llm 与 /v1/llm/stream 端点：只做协议适配，业务语义在 services/ 层。"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from llm_gateway.core.errors import GatewayError
from llm_gateway.core.schemas import LLMRequest, LLMResponse
from llm_gateway.services.invocation import call_with_fallback, stream_with_fallback, validate_model
from llm_gateway.services.prompt_service import build_messages

router = APIRouter()


@router.post("/v1/llm", response_model=LLMResponse)
async def create_llm_response(request: LLMRequest) -> LLMResponse:
    # FastAPI 在入口校验请求、在 response_model 校验统一响应出口。
    if request.stream:
        raise HTTPException(status_code=400, detail={"code": "use_stream_endpoint", "message": "流式请求请使用 /v1/llm/stream"})
    try:
        return await call_with_fallback(request)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/v1/llm/stream")
async def create_stream(request: LLMRequest) -> StreamingResponse:
    # 提供独立流式入口，明确禁止与 Structured Output 混用。
    if request.response_schema is not None:
        raise HTTPException(status_code=400, detail={"code": "unsupported_combination", "message": "流式输出不支持 response_schema"})
    try:
        validate_model(request.model, None)
        build_messages(request)
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc
    return StreamingResponse(stream_with_fallback(request), media_type="text/event-stream")
