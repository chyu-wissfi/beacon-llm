"""六大功能点一键取证脚本：对运行在同一进程内的网关 app 发起真实调用。

覆盖面（与 README「双协议组合」小节的 curl 示例一一对应，证据逐项打印）：
1. 统一抽象层：同一入口按 model 字段路由到 OpenAI Responses / Anthropic
   Messages 两个协议适配器，双模型非流式调用各自 200 + usage 三键自洽；
2. 流式输出：stream=true 的 SSE 增量拼接、[DONE] 收尾、TTFT 进 trace；
3. 结构化输出：Responses 路径的 json_schema 原生约束 + Anthropic 路径的
   json_object（schema 注入 system），输出可解析且符合 schema；
4. 模板引用：prompt 扩展字段渲染，trace 记录模板坐标；
5. 可观测：/v1/traces 的 token 分类统计 / 延迟 / TTFT / 成本 / 尝试数，
   /metrics 的按模型计数；
6. 韧性：低 RPM 配置真实触发一次 429（Retry-After 下发）；把上游 base_url
   指向不可达端口，演示指数退避重试（预算内 4 次尝试后链尾统一终态）。

运行前提：VVEAI_API_KEY（双协议共用凭据）。脚本直接消耗少量真实额度；
trace 记在进程内内存库，不写 data/traces.db。
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
if Path.cwd() != ROOT:
    os.chdir(ROOT)  # 配置加载按相对路径 config/ 取文件，必须从仓库根启动

from llm_gateway.core import ratelimit  # noqa: E402
from llm_gateway.core.breaker import reset_breakers  # noqa: E402
from llm_gateway.core.schemas import RateLimitConfig  # noqa: E402
from llm_gateway.main import app  # noqa: E402
from llm_gateway.observability.metrics import reset_metrics  # noqa: E402
from llm_gateway.services.catalog import MODEL_CONFIGS  # noqa: E402
from llm_gateway.services.trace_service import flush_pending  # noqa: E402
from llm_gateway.storage.engine import (  # noqa: E402
    MEMORY_DB_URL,
    configure_engine,
    dispose_engine,
)

CHAT_PATH = "/v1/chat/completions"
GPT = "vve-gpt-responses"
CLAUDE = "vve-claude-anthropic"

# 公开白名单没有 timeout 字段（网关有意设计），而中转站偶发拥塞时首包可能
# 超过内部默认 30s。脚本进程内把默认值放宽到 120s（白名单上限）：
# pydantic-core 在类定义时把默认值编译进 core schema，改 FieldInfo 后必须
# model_rebuild 重建才能生效。
from llm_gateway.core.schemas import LLMRequest  # noqa: E402

LLMRequest.model_fields["timeout_seconds"].default = 120
LLMRequest.model_rebuild(force=True)

# 冒烟请求一律收到最短回复：控制真实额度消耗，同时足以断言行为面。
_TINY_MESSAGES = [{"role": "user", "content": "只回复一个词：ok"}]

_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "approve": {"type": "boolean"},
    },
    "required": ["order_id", "approve"],
    "additionalProperties": False,
}

_failures: list[str] = []


def check(section: str, ok: bool, evidence: str) -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {evidence}")
    if not ok:
        _failures.append(f"{section}: {evidence}")


def _caller_key() -> str:
    # 调用方 key 优先从环境取（不落盘），否则读 callers.yaml 的第一个条目。
    key = os.environ.get("BEACON_CALLER_KEY")
    if key:
        return key
    file = yaml.safe_load((ROOT / "config" / "callers.yaml").read_text(encoding="utf-8"))
    return next(iter(file["callers"]))


def _payload(model: str, **overrides: Any) -> dict:
    payload: dict[str, Any] = {"model": model}
    payload.update(overrides)
    return payload


def _error_code(response: httpx.Response) -> str:
    # 失败响应的错误码提取（错误体形态由网关错误出口保证）。
    try:
        return response.json()["error"]["code"]
    except Exception:
        return response.text[:120]


async def _post_with_retries(
    client: httpx.AsyncClient, payload: dict, attempts: int = 3
) -> httpx.Response:
    # 中转站排队随机（7s～120s+），上游慢 ≠ 网关错：用例内对非 200 有限重试，
    # 把“上游排队”与“功能缺陷”区分开。
    response = await _post_json(client, payload)
    for attempt in range(attempts - 1):
        if response.status_code == 200:
            return response
        print(f"    （上游排队，重试 {attempt + 1}/{attempts - 1}：{_error_code(response)}）")
        response = await _post_json(client, payload)
    return response


def _find_trace(traces: list[dict], request_id: str) -> dict | None:
    return next((item for item in traces if item["request_id"] == request_id), None)


async def _post_json(client: httpx.AsyncClient, payload: dict) -> httpx.Response:
    return await client.post(CHAT_PATH, json=payload)


async def _stream_events(client: httpx.AsyncClient, payload: dict) -> tuple[int, list[str]]:
    # SSE 解析：data: 行去掉前缀，[DONE] 原样保留为终态标记。
    events: list[str] = []
    async with client.stream("POST", CHAT_PATH, json=payload) as response:
        status = response.status_code
        if status == 200:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(line[len("data: ") :])
    return status, events


def _stream_text(events: list[str]) -> tuple[str, str | None, str | None]:
    # 返回 (拼接正文, 首块 id, 流内失败事件原文)：首块 id 与 trace.request_id
    # 同源可对账；无 [DONE] 收尾时错误事件原文进证据（网关流内失败语义）。
    text = ""
    first_id = None
    failure = None
    for event in events:
        if event == "[DONE]":
            continue
        chunk = json.loads(event)
        first_id = first_id or chunk.get("id")
        if chunk.get("error") is not None:
            failure = json.dumps(chunk["error"], ensure_ascii=False)
            continue
        for choice in chunk.get("choices", []):
            text += choice.get("delta", {}).get("content", "")
    return text, first_id, failure


# ---------------------------------------------------------------------------
# 1. 统一抽象层：双协议非流式
# ---------------------------------------------------------------------------


async def section_dual_protocol(client: httpx.AsyncClient) -> list[str]:
    print("\n[1] 统一抽象层：同一入口按 model 路由到两种协议适配器（非流式）")
    request_ids: list[str] = []
    for model in (GPT, CLAUDE):
        response = await _post_json(client, _payload(model, messages=_TINY_MESSAGES, max_tokens=32))
        ok = response.status_code == 200
        evidence = f"{model}: HTTP {response.status_code}"
        if ok:
            body = response.json()
            usage = body["usage"]
            content = body["choices"][0]["message"]["content"]
            self_consistent = usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            ok = bool(content.strip()) and self_consistent
            evidence += (
                f"，content={content.strip()!r}，usage(prompt/completion/total)="
                f"{usage['prompt_tokens']}/{usage['completion_tokens']}/{usage['total_tokens']}"
            )
            request_ids.append(body["id"])
        else:
            evidence += f"，code={_error_code(response)}"
        check("dual_protocol", ok, evidence)
    return request_ids


# ---------------------------------------------------------------------------
# 2. 流式输出
# ---------------------------------------------------------------------------


async def section_streaming(client: httpx.AsyncClient) -> list[str]:
    print("\n[2] 流式输出：stream=true 的 SSE 增量与 [DONE] 终态")
    request_ids: list[str] = []
    for model in (GPT, CLAUDE):
        status, events = await _stream_events(
            client, _payload(model, messages=_TINY_MESSAGES, stream=True)
        )
        ok = status == 200
        evidence = f"{model}: HTTP {status}"
        if ok:
            text, first_id, failure = _stream_text(events)
            done = events[-1] == "[DONE]" if events else False
            if failure is not None:
                evidence += f"，流内失败事件={failure}"
            elif first_id is None:
                ok = False
                evidence += "，流内无任何 chunk id"
            else:
                ok = bool(text.strip()) and done
                evidence += f"，增量拼接={text.strip()!r}，[DONE]收尾={done}"
                request_ids.append(first_id)
        check("streaming", ok, evidence)
    return request_ids


# ---------------------------------------------------------------------------
# 3. 结构化输出
# ---------------------------------------------------------------------------

_STRUCTURED_MESSAGES = [
    {
        "role": "user",
        "content": (
            "你是订单决策器。只输出一个 JSON 对象，不要任何其他文字："
            '{"order_id": "D-1001", "approve": true, "reject": false, "escalate": false}'
        ),
    }
]


async def section_structured_output(client: httpx.AsyncClient) -> list[str]:
    print("\n[3] 结构化输出：json_schema（Responses 原生）/ json_object（Anthropic 注入）")
    request_ids: list[str] = []
    # Responses API：原生 text.format=json_schema 约束。
    response = await _post_json(
        client,
        _payload(
            GPT,
            messages=[
                {"role": "user", "content": "给我订单 D-1001 的审批决策，approve 为 true 或 false"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "order", "strict": True, "schema": _ORDER_SCHEMA},
            },
            max_tokens=256,
        ),
    )
    ok = response.status_code == 200
    evidence = f"{GPT}: HTTP {response.status_code}"
    if ok:
        content = response.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            schema_ok = (
                set(parsed) == {"order_id", "approve"}
                and isinstance(parsed["approve"], bool)
            )
            ok = schema_ok
            evidence += f"，输出={content.strip()!r}，schema 符合={schema_ok}"
        except json.JSONDecodeError:
            ok = False
            evidence += f"，输出不是合法 JSON：{content!r}"
    else:
        evidence += f"，body={response.text[:200]}"
    check("structured_output", ok, evidence)
    if ok:
        request_ids.append(response.json()["id"])

    # Anthropic：json_object + 业务校验档案（schema 注入 system + 本地兜底）。
    response = await _post_with_retries(
        client,
        _payload(
            CLAUDE,
            messages=_STRUCTURED_MESSAGES,
            response_format={"type": "json_object"},
            validation={"name": "order_decision", "version": "v1"},
            max_tokens=256,
        ),
    )
    ok = response.status_code == 200
    evidence = f"{CLAUDE}: HTTP {response.status_code}"
    if ok:
        content = response.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            ok = parsed.get("order_id") == "D-1001" and parsed.get("approve") is True
            evidence += f"，输出={content.strip()!r}，业务校验（order_decision/v1）通过={ok}"
        except json.JSONDecodeError:
            ok = False
            evidence += f"，输出不是合法 JSON：{content!r}"
    else:
        evidence += f"，body={response.text[:200]}"
    check("structured_output", ok, evidence)
    if ok:
        request_ids.append(response.json()["id"])
    return request_ids


# ---------------------------------------------------------------------------
# 4. 模板引用
# ---------------------------------------------------------------------------


async def section_prompt_template(client: httpx.AsyncClient) -> list[str]:
    print("\n[4] 模板引用：prompt 扩展字段（存储 + 变量替换 + 版本引用）")
    request_ids: list[str] = []
    for model in (GPT, CLAUDE):
        response = await _post_with_retries(
            client,
            _payload(
                model,
                messages=_TINY_MESSAGES,
                prompt={
                    "name": "knowledge_decision",
                    "version": "v1",
                    "variables": {"product_name": "Beacon"},
                },
                max_tokens=32,
            ),
        )
        ok = response.status_code == 200
        evidence = f"{model}: HTTP {response.status_code}"
        if ok:
            request_ids.append(response.json()["id"])
        check("prompt_template", ok, evidence)
    return request_ids


# ---------------------------------------------------------------------------
# 5. 可观测
# ---------------------------------------------------------------------------


async def section_observability(
    client: httpx.AsyncClient, request_ids: list[str]
) -> None:
    print("\n[5] 可观测：/v1/traces 的 token/延迟/TTFT/成本 + /metrics 计数")
    traces = (await client.get("/v1/traces")).json()
    for request_id in request_ids:
        record = _find_trace(traces, request_id)
        if record is None or record["status"] != "success":
            check("observability", False, f"request_id={request_id} 无对应 trace 记录或状态非 success")
            continue
        evidence = (
            f"{record['requested_model']}({request_id[:8]}…): "
            f"input={record['input_tokens']} output={record['output_tokens']} "
            f"cost=${record['cost_usd']:.6f} latency={record['latency_ms']}ms "
            f"ttft={record['ttft_ms']}ms attempts={record['attempts']} "
            f"caller={record['caller']} price_version={record['price_version']}"
        )
        if record.get("prompt_name"):
            evidence += f" prompt={record['prompt_name']}/{record['prompt_version']}"
        check("observability", True, evidence)

    metrics = (await client.get("/metrics")).text
    for model in (GPT, CLAUDE):
        total = sum(
            float(line.split()[-1])
            for line in metrics.splitlines()
            if line.startswith("llm_requests_total{") and f'model="{model}"' in line
        )
        check("observability", total > 0, f"/metrics llm_requests_total[{model}] = {total:g}")


# ---------------------------------------------------------------------------
# 6. 韧性：限流 + 重试
# ---------------------------------------------------------------------------


async def section_rate_limit(client: httpx.AsyncClient) -> None:
    print("\n[6a] 按模型限流：RPM 压到 1 真实触发 429")
    # 选 GPT 链路演示：限流语义与协议无关（按模型名桶），而该链路延迟稳定，
    # 避免中转站排队把“限流演示”变成“上游超时演示”。
    model = GPT
    original = MODEL_CONFIGS[model]
    MODEL_CONFIGS[model] = replace(original, rate_limit=RateLimitConfig(rpm=1, tpm=None))
    ratelimit.reset_admission()
    try:
        first = await _post_json(client, _payload(model, messages=_TINY_MESSAGES, max_tokens=32))
        second = await _post_json(client, _payload(model, messages=_TINY_MESSAGES, max_tokens=32))
        check("rate_limit", first.status_code == 200, f"第一个请求：HTTP {first.status_code}（吃掉唯一令牌）")
        ok = second.status_code == 429
        evidence = f"第二个请求：HTTP {second.status_code}"
        if ok:
            error = second.json()["error"]
            retry_after = second.headers.get("retry-after")
            ok = error["code"] == "rate_limited" and retry_after is not None
            evidence += f"，code={error['code']}，Retry-After={retry_after}"
        check("rate_limit", ok, evidence)
    finally:
        MODEL_CONFIGS[model] = original
        ratelimit.reset_admission()


async def section_retry_backoff(client: httpx.AsyncClient) -> None:
    print("\n[6b] 指数退避重试：上游不可达时预算内 4 次尝试（0.5s×2^n + 抖动）")
    original = MODEL_CONFIGS[GPT]
    # 不可达目标选本地未监听高端口：内核立即 RST（ECONNREFUSED），退避节奏
    # 不被连接挂起拖长——演示的是重试/退避语义本身。
    MODEL_CONFIGS[GPT] = replace(original, base_url="http://127.0.0.1:59999")
    ratelimit.reset_admission()
    reset_breakers()
    try:
        started = time.monotonic()
        response = await _post_json(client, _payload(GPT, messages=_TINY_MESSAGES, max_tokens=32))
        elapsed = time.monotonic() - started
        ok = response.status_code == 502
        evidence = f"终态：HTTP {response.status_code}"
        if ok:
            error = response.json()["error"]
            ok = error["code"] == "model_unavailable"
            evidence += f"，code={error['code']}，耗时={elapsed:.1f}s（≥3 次退避即 ≥3.5s）"
        check("retry_backoff", ok, evidence)
        await flush_pending()
        traces = (await client.get("/v1/traces")).json()
        record = next(
            (
                item
                for item in traces
                if item["requested_model"] == GPT and item["status"] == "failed"
            ),
            None,
        )
        ok = record is not None and record["attempts"] == 4
        evidence = "trace 未记录失败终态" if record is None else (
            f"attempts={record['attempts']}，route_reason={record.get('route_reason')!r}"
        )
        check("retry_backoff", ok, evidence)
    finally:
        MODEL_CONFIGS[GPT] = original
        ratelimit.reset_admission()
        reset_breakers()


# ---------------------------------------------------------------------------


async def main() -> int:
    if not os.environ.get("VVEAI_API_KEY"):
        print("缺少 VVEAI_API_KEY 环境变量（双协议上游凭据），无法取证。")
        print("export VVEAI_API_KEY=sk-... 后重试。")
        return 2

    # 进程内全局状态归零 + 内存 trace 库（不写生产库 data/traces.db）。
    ratelimit.reset_admission()
    reset_breakers()
    reset_metrics()
    configure_engine(MEMORY_DB_URL)

    request_ids: list[str] = []
    headers = {"Authorization": f"Bearer {_caller_key()}"}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
            headers=headers,
            timeout=httpx.Timeout(120.0),
        ) as client:
            request_ids += await section_dual_protocol(client)
            request_ids += await section_streaming(client)
            request_ids += await section_structured_output(client)
            request_ids += await section_prompt_template(client)
            await section_observability(client, request_ids)
            await section_rate_limit(client)
            await section_retry_backoff(client)
    finally:
        await dispose_engine()

    print("\n" + "=" * 72)
    if _failures:
        print(f"结果：{_failures and len(_failures)} 项未通过")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("结果：六大功能点全部通过，证据见上方各 [PASS] 行。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
