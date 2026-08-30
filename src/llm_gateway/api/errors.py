"""OpenAI 风格错误体适配（M03 任务 4）。

错误响应的对外形态从 demo 的 `{"detail": {"code", "message"}}` 升级为 OpenAI
风格 `{"error": {"message", "type", "code"}}`（design.md §API 层：错误体统一
OpenAI 风格，openai SDK 端可正常解析）。code/message/status 三元组仍以
core/errors.py 注册表为唯一事实来源（M02 双冻结），本模块只做形态翻译、
不产生新的错误码，也不改写注册表的任何取值。

作用域（controller 裁决 + 旧端点行为不变约束的交点）：
- OpenAI 风格只作用于 OpenAI 兼容面（当前仅 /v1/chat/completions）。旧
  /v1/llm 端点在 M03 删除前保持既有行为：它的 422 请求校验形态被契约测试
  钉住（test_stream_with_response_schema_rejected_422），因此 RequestValidationError
  处理器按路径分流——旧路径原样复用 FastAPI 自带的默认处理器（响应逐字节
  等价于未注册本处理器时）。
- GatewayError 无需分流：旧端点在路由内 try/except 把它转成 HTTPException，
  从不逃逸到异常处理器；本处理器实际只会接到 OpenAI 兼容面的错误。
- 任务 B 删除旧端点后，路径分流随之收口为无条件 OpenAI 风格（届时唯一
  需要动的是 _ON_OPENAI_SURFACE 与旧路径回退分支）。
"""

from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from llm_gateway.core.errors import ERROR_REGISTRY, UNSUPPORTED_FIELD, GatewayError

# ---------------------------------------------------------------------------
# OpenAI 错误体的 type 取值（controller 裁决，design.md 未定，此处落锤）
# ---------------------------------------------------------------------------

# 4xx 类错误：请求本身不合法（白名单外字段、缺字段、未知模型等），调用方
# 改请求即可重试，openai SDK 会映射为 BadRequestError 等客户端侧异常。
INVALID_REQUEST_ERROR: Final[str] = "invalid_request_error"
# 5xx 类错误：Gateway 自身或上游的问题，调用方重试同请求是有意义的动作。
API_ERROR: Final[str] = "api_error"

# OpenAI 兼容面的路径前缀：RequestValidationError 处理器按它分流。新增
# OpenAI 面路由时在此追加；旧端点删除后此表即全量收敛。
_ON_OPENAI_SURFACE: Final[tuple[str, ...]] = ("/v1/chat/completions",)


def error_type_for_status(status_code: int) -> str:
    # controller 裁决：4xx -> invalid_request_error，5xx -> api_error。以 500
    # 为界实现（而非逐枚举 4xx），注册表未来增补状态码时无需同步本函数；
    # 非 4/5xx 的错误状态（当前注册表仅 499 request_cancelled）按 controller
    # 的 4xx 归类落入 invalid_request_error。
    return API_ERROR if status_code >= 500 else INVALID_REQUEST_ERROR


def openai_error_body(code: str | None, message: str, status_code: int) -> dict[str, Any]:
    # code=None 表示"无稳定注册码"的失败（如缺字段/类型错——注册表封闭且
    # 本里程碑不得扩员，见 M02 双冻结），保持三键齐全以固定响应形态；
    # openai SDK 对 null code 的解析行为与缺键一致（e.code 为 None）。
    return {
        "error": {
            "message": message,
            "type": error_type_for_status(status_code),
            "code": code,
        }
    }


def _loc_path(loc: tuple[int | str, ...]) -> str:
    # FastAPI 给 body 错误的 loc 前置 "body" 段，去掉后才是请求字段路径
    # （嵌套时形如 messages.0.role，比只报末段更有定位价值）。
    parts = loc[1:] if loc and loc[0] == "body" else loc
    return ".".join(str(part) for part in parts)


async def gateway_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # exc 参数声明为 Exception（而非 GatewayError）：Starlette 的
    # add_exception_handler 要求 Callable[[Request, Exception], ...]，其参数
    # 类型逆变不允许收窄声明；用 isinstance 收窄代替 cast——运行期 Starlette
    # 按注册键分发，窄化分支不可达，纯属满足类型系统。
    if not isinstance(exc, GatewayError):
        raise exc
    # 三元组全部取自注册表托管的 GatewayError，这里只做形态翻译——message
    # 即便被调用点动态覆盖过（如拼接缺失变量名），也只是注册表默认值之上
    # 的合法覆盖面，code/status 不可变。
    return JSONResponse(
        status_code=exc.status_code,
        content=openai_error_body(exc.code, exc.message, exc.status_code),
    )


async def request_validation_handler(request: Request, exc: Exception) -> Response:
    # 请求体校验失败（Pydantic ValidationError 经 FastAPI 包装）：OpenAI 面
    # 一律 400（controller 裁决，不用 FastAPI 默认 422），type 按状态码归为
    # invalid_request_error。isinstance 收窄的动机同 gateway_error_handler。
    if not isinstance(exc, RequestValidationError):
        raise exc
    if not any(request.scope["path"].startswith(prefix) for prefix in _ON_OPENAI_SURFACE):
        # 旧端点：复用 FastAPI 自带默认处理器，保证 422 形态与未注册本
        # 处理器时逐字节一致（等价迁移契约钉住的行为零改动）。
        return await request_validation_exception_handler(request, exc)

    errors = exc.errors()
    # 白名单优先：extra_forbidden（白名单外字段）与其他校验错（缺字段/类型
    # 错/越界）并存时报 unsupported_field——不变量 #4 的语义是"不接受白名单
    # 外的请求"，先告知调用方字段层面的问题。
    extra_errors = [error for error in errors if error["type"] == "extra_forbidden"]
    if extra_errors:
        # 动态拼接字段名是注册表默认 message 之上的合法覆盖面（与
        # missing_prompt_variable 拼变量名同一先例）；status 取注册表，不另写 400。
        spec = ERROR_REGISTRY[UNSUPPORTED_FIELD]
        fields = "、".join(dict.fromkeys(_loc_path(error["loc"]) for error in extra_errors))
        return JSONResponse(
            status_code=spec.status_code,
            content=openai_error_body(UNSUPPORTED_FIELD, f"{spec.message}: {fields}", spec.status_code),
        )
    # 其他校验失败（缺字段/类型错/取值越界）：注册表无对应码且本里程碑
    # 不得扩员，code 置 None；message 用字段路径 + Pydantic 原始描述，给
    # 调用方足够定位信息。
    detail = "; ".join(f"{_loc_path(error['loc'])}: {error['msg']}" for error in errors)
    return JSONResponse(
        status_code=400,
        content=openai_error_body(None, f"请求校验失败: {detail}", 400),
    )


def register_error_handlers(app: FastAPI) -> None:
    # 应用组装入口（main.py 调用）：异常处理器集中在 api/errors.py 定义、
    # main.py 一行注册，避免处理器散落在路由文件里与旧端点的 detail 形态
    # 意外耦合。
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(GatewayError, gateway_error_handler)
