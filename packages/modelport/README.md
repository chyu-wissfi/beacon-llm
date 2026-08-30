# ModelPort

LLM Gateway 的 Agent 侧客户端包：**Agent 的唯一依赖**。封装网关接入细节、
映射网关错误码、透传 request_id——Agent 不导入任何供应商 SDK（内部使用
openai SDK 是实现细节，对 Agent 是黑盒）。

## 安装（本地可编辑，不发 PyPI）

```bash
uv pip install -e packages/modelport
```

## 用法

```python
import modelport

port = modelport.ModelPort(base_url="http://localhost:8000", api_key="sk-...")

# 非流式
result = await port.complete(
    model="general-primary",
    messages=[{"role": "user", "content": "你好"}],
)
result.content        # 模型输出
result.request_id     # 与 /v1/traces 的 request_id 同源，可对账
result.usage          # prompt_tokens / completion_tokens / total_tokens

# 流式
async for chunk in port.stream(model="general-primary", messages=[...]):
    print(chunk.delta, end="")

# 结构化输出：response_format 原样透传，输出已过网关双重校验
result = await port.complete(
    model="general-primary",
    messages=[...],
    response_format={"type": "json_object"},
    validation=modelport.validation_ref("order_decision", "v1"),
)
payload = result.json()

# 模板引用：渲染在网关侧完成，Agent 只提交坐标与变量
result = await port.complete(
    model="general-primary",
    messages=[{"role": "user", "content": "..."}],
    prompt=modelport.prompt_ref("knowledge_decision", "v1", {"product_name": "Beacon"}),
)
```

## 错误处理

一切失败都是 `ModelPortError` 层级（`.code` 是网关稳定错误码，`.retry_after`
在 429 类携带建议间隔）：

| 异常 | 语义 | 该怎么办 |
|---|---|---|
| `AuthenticationError` | 401 | 换 key |
| `RequestRejectedError` | 400（未知模型/模板/档案、白名单外字段） | 改请求，重试无意义 |
| `RateLimitedError` | 429 | 按 `retry_after` 退避 |
| `UpstreamFailedError` | 502 / 流内终态错误 | 网关已用尽预算，自行决定降级 |
| `GatewayUnavailableError` | 503（缺凭据 / 熔断） | 稍后重试 |
| `CancelledRequestError` | 499 | 请求被取消 |
| `TransportError` | 连不上网关 / 超时 | 检查网关是否在运行 |

供应商 SDK 异常不外泄：异常类型与异常链中都只有本包类型。

## request_id 与 Trace 对账

非流式 `result.request_id`、流式 `chunk.request_id` 均与网关 Trace 的
`request_id` 同源，可直接用于 `GET /v1/traces` 排查（路由理由、尝试数、
成本、价格版本都在那里）。
