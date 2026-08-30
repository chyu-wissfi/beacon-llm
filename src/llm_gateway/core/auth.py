"""调用方认证（M05 任务 1）。

`Authorization: Bearer <key>` 对调用方表（core/config 加载的 callers）：
- 常数时间比较：`hmac.compare_digest` 逐键比对。遍历完全部键才判定，
  命中不提前返回——提前短路会让"表中第几个键命中"泄漏进响应时延，
  唯一允许的短路发生在比较全部完成之后；
- 失败一律 401 `unauthorized`（注册表三元组），不区分"无头/格式错/键不存在"
  ——错误形态对探测者无信息增量；
- 认证结果（CallerConfig）由 api 层注入请求上下文（request.state.caller），
  本模块只回答"这个头是谁"，不做准入决策（那是 ratelimit/breaker 的面）。

依赖边界：core 层，只依赖 stdlib + core/errors + core/config 的形态。
"""

import hmac

from llm_gateway.core.config import CallerConfig
from llm_gateway.core.errors import UNAUTHORIZED, GatewayError


def parse_bearer(authorization: str | None) -> str | None:
    # 解析 `Authorization` 头：只承认 Bearer 方案（大小写不敏感），
    # 其余形态（Basic、裸 token、空值）与缺头同罪——返回 None 交给
    # authenticate 统一拒绝，不各自造错误文案。
    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def authenticate(authorization: str | None, callers: dict[str, CallerConfig]) -> CallerConfig:
    # 认证入口：合法返回调用方信息，否则抛 UNAUTHORIZED（401）。
    # compare_digest 对 str 只接受纯 ASCII，先编码成 bytes 再比（任意
    # 取值都安全，不同长度的比较由 compare_digest 内部常数时间处理）。
    token = parse_bearer(authorization)
    matched: CallerConfig | None = None
    if token is not None:
        token_bytes = token.encode()
        for key, caller in callers.items():
            if hmac.compare_digest(key.encode(), token_bytes):
                # 不 return：遍历完全部键，命中位置不进时延侧信道。
                matched = caller
    if matched is None:
        raise GatewayError(UNAUTHORIZED)
    return matched
