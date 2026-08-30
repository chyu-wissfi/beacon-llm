"""ModelPort 契约测试（M12 任务 2 + 3）：打本地 ASGI app + Fake Adapter。

断言面（与 spec 验收逐条对应）：
- 错误映射完整：网关稳定码 -> ModelPortError 层级，映射表与错误码注册表
  全集对齐；SDK 异常不外泄（类型与异常链都只有本包类型）；
- request_id 可取：非流式（响应 id）与流式（chunk id）均与 Trace 的
  request_id 同源，可与 /v1/traces 对账（网关侧透传见 api/chat.py）；
- 流式可用：增量拼接完整、终态块恰好一个、流内失败映射为终态异常且
  已交付增量不丢；
- extra_body 透传：模板（prompt）与校验档案（validation）选择项经
  ModelPort 直达网关扩展字段；
- 隔离证明（任务 3）：公开面无供应商 SDK 符号再导出、源码无星号再导出、
  示例 Agent 仅 import modelport（+ 标准库）。

夹具分工：网关行为面的卫生夹具（鉴权环境变量 / trace 内存库 / 准入复位 /
openai legacy httpx shim）沿用 tests/contract/conftest.py 的 autouse 夹具；
本文件只加 ModelPort 专属的 http 传输与 fake 模型注册。
"""

import ast
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import modelport
import pytest
import pytest_asyncio
from modelport.client import _ERROR_CLASS_BY_CODE

from llm_gateway.core.errors import ERROR_REGISTRY
from llm_gateway.core.schemas import ModelConfig, RateLimitConfig
from llm_gateway.main import app
from llm_gateway.providers import PROVIDER_REGISTRY
from llm_gateway.providers.fake import (
    FakeAdapter,
    RateLimited,
    StreamInterrupt,
    Success,
    Timeout,
)
from llm_gateway.services.catalog import MODEL_CONFIGS, PRICE_PER_MILLION
from llm_gateway.services.trace_service import CALL_TRACES
from tests.contract.helpers import VALID_CALLER_KEY

pytestmark = pytest.mark.asyncio

# 测试专用平台模型：provider 指向 Fake Adapter，剧本随用例注册。
FAKE_MODEL = "fake-model"

# dict 字面量即合法消息载荷（同 test_openai_sdk_contract.py 的口径）。
_MESSAGES: Any = [{"role": "user", "content": "你好"}]

# 仓库根：隔离证明测试要读 packages/ 下的源码与示例。
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _register_fake(
    monkeypatch: pytest.MonkeyPatch, adapter: FakeAdapter
) -> FakeAdapter:
    # 注册可控剧本的 fake 模型（同 test_admission.py 的接线形态）：
    # PROVIDER_REGISTRY 与 MODEL_CONFIGS 都是 monkeypatch 的 setitem，
    # 用例间零残留；attempts 计数是"上游请求数"的可观测面。
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake", adapter)
    monkeypatch.setitem(
        MODEL_CONFIGS,
        FAKE_MODEL,
        ModelConfig(
            provider_model="fake",
            base_url="http://fake.test",
            api_key_env="FAKE_API_KEY",
            supports_structured_output=True,
            provider="fake",
        ),
    )
    # trace 记账按模型查价：补一条零价条目，避免成功路径 KeyError。
    monkeypatch.setitem(PRICE_PER_MILLION, FAKE_MODEL, {"input": 0.0, "output": 0.0})
    return adapter


@pytest_asyncio.fixture
async def http():
    # httpx ASGI 传输直打 app（不起端口）：ModelPort 的 http_client 注入口。
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
        yield c


@pytest_asyncio.fixture
async def port(http):
    # 合法调用方身份的 ModelPort 实例（api_key 取自 callers.yaml 加载产物）。
    return modelport.ModelPort(
        base_url="http://gateway.test", api_key=VALID_CALLER_KEY, http_client=http
    )


def _port_with_key(http: httpx.AsyncClient, api_key: str) -> modelport.ModelPort:
    return modelport.ModelPort(
        base_url="http://gateway.test", api_key=api_key, http_client=http
    )


# ---------------------------------------------------------------------------
# 非流式 / 流式：成功路径 + request_id 与 Trace 对账（任务 1+2）
# ---------------------------------------------------------------------------


async def test_complete_success_and_request_id_matches_trace(port, monkeypatch):
    # 成功三元组原样搬运；响应 id = Trace 的 request_id（对账钥匙）。
    adapter = _register_fake(monkeypatch, FakeAdapter(scenario=Success(content="ok")))
    result = await port.complete(FAKE_MODEL, _MESSAGES)
    assert result.content == "ok"
    assert result.model == FAKE_MODEL
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 2
    assert result.usage.total_tokens == 5
    assert adapter.attempts == 1
    assert len(CALL_TRACES) == 1
    assert result.request_id == CALL_TRACES[-1].request_id
    assert CALL_TRACES[-1].status == "success"


async def test_stream_yields_chunks_and_request_id_matches_trace(port, monkeypatch):
    # 增量拼接完整、终态块恰好一个；所有 chunk 的 request_id 与 Trace 同源。
    _register_fake(monkeypatch, FakeAdapter(scenario=Success(content="hello stream")))
    chunks = [chunk async for chunk in port.stream(FAKE_MODEL, _MESSAGES)]
    assert "".join(chunk.delta for chunk in chunks) == "hello stream"
    terminal = [chunk for chunk in chunks if chunk.finish_reason is not None]
    assert len(terminal) == 1
    assert terminal[0].finish_reason == "stop"
    assert {chunk.request_id for chunk in chunks} == {CALL_TRACES[-1].request_id}
    assert CALL_TRACES[-1].status == "success"


async def test_stream_inflight_failure_maps_upstream_failed(port, monkeypatch):
    # 流内终态错误（HTTP 200 上的错误事件）映射为 UpstreamFailedError：
    # 已交付增量完好保留，异常只描述"流在这里断了"（无 HTTP 状态可携带）。
    _register_fake(
        monkeypatch, FakeAdapter(scenario=StreamInterrupt(chunks_before_failure=2))
    )
    received: list[modelport.StreamChunk] = []
    with pytest.raises(modelport.UpstreamFailedError) as exc_info:
        async for chunk in port.stream(FAKE_MODEL, _MESSAGES):
            received.append(chunk)
    assert [chunk.delta for chunk in received] == ["chunk-0", "chunk-1"]
    assert exc_info.value.code == "upstream_stream_failed"
    assert exc_info.value.status_code is None
    assert CALL_TRACES[-1].status == "failed"
    assert CALL_TRACES[-1].error_code == "upstream_stream_failed"


# ---------------------------------------------------------------------------
# 错误映射（任务 1）：网关错误码 -> ModelPortError 层级
# ---------------------------------------------------------------------------


async def test_error_mapping_covers_full_gateway_registry():
    # 映射表与网关错误码封闭集合全集对齐：注册表新增码而映射表漏登记，
    # 这条断言先于任何行为漂移失败。
    assert set(_ERROR_CLASS_BY_CODE) == set(ERROR_REGISTRY)


async def test_unauthorized_maps_authentication_error(http):
    bad = _port_with_key(http, api_key="sk-bogus")
    with pytest.raises(modelport.AuthenticationError) as exc_info:
        await bad.complete(FAKE_MODEL, _MESSAGES)
    assert exc_info.value.code == "unauthorized"
    assert exc_info.value.status_code == 401
    # SDK 异常不外泄：异常链上也只有 ModelPort 族类型。
    assert exc_info.value.__cause__ is None


async def test_unknown_model_maps_request_rejected(port, monkeypatch):
    adapter = _register_fake(monkeypatch, FakeAdapter())
    with pytest.raises(modelport.RequestRejectedError) as exc_info:
        await port.complete("no-such-model", _MESSAGES)
    assert exc_info.value.code == "unknown_model"
    assert exc_info.value.status_code == 400
    assert adapter.attempts == 0  # 请求级拒绝在任何上游调用之前


async def test_unsupported_field_maps_request_rejected(port, monkeypatch):
    adapter = _register_fake(monkeypatch, FakeAdapter())
    with pytest.raises(modelport.RequestRejectedError) as exc_info:
        await port.complete(
            FAKE_MODEL, _MESSAGES, response_format={"type": "no-such-format"}
        )
    assert exc_info.value.code == "unsupported_field"
    assert adapter.attempts == 0


async def test_rate_limited_maps_rate_limited_error_with_retry_after(port, monkeypatch):
    # RPM 令牌桶压到 1：首个请求吃掉唯一令牌，第二个准入期即拒——
    # 429 + Retry-After 头原样进 RateLimitedError。
    adapter = _register_fake(monkeypatch, FakeAdapter())
    monkeypatch.setitem(
        MODEL_CONFIGS,
        FAKE_MODEL,
        replace(MODEL_CONFIGS[FAKE_MODEL], rate_limit=RateLimitConfig(rpm=1, tpm=None)),
    )
    first = await port.complete(FAKE_MODEL, _MESSAGES)
    assert first.content == "ok"
    with pytest.raises(modelport.RateLimitedError) as exc_info:
        await port.complete(FAKE_MODEL, _MESSAGES)
    assert exc_info.value.code == "rate_limited"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after is not None
    assert adapter.attempts == 1  # 准入拒绝零上游波及


async def test_provider_rate_limit_exhausts_budget_as_model_unavailable(port, monkeypatch):
    # 上游 429（剧本）是可重试故障：网关按 Run 预算退避重试，用尽后按链尾
    # 统一终态 model_unavailable 对外（invocation 语义）——客户端看到的是"预算
    # 用尽"而不是单次 429；准入层的 429（rate_limited 用例）才是直达的限流拒绝。
    # route_reason 里保留 provider_overloaded 逐次记录（降级决策可审计）。
    adapter = _register_fake(monkeypatch, FakeAdapter(scenario=RateLimited()))
    with pytest.raises(modelport.UpstreamFailedError) as exc_info:
        await port.complete(FAKE_MODEL, _MESSAGES)
    assert exc_info.value.code == "model_unavailable"
    assert exc_info.value.status_code == 502
    assert adapter.attempts == 4  # Run 预算上限 4，无隐藏放大
    assert "provider_overloaded" in (CALL_TRACES[-1].route_reason or "")


async def test_timeout_maps_upstream_failed(port, monkeypatch):
    # 传输故障（剧本）映射为 MODEL_UNAVAILABLE -> UpstreamFailedError；
    # 可重试码，预算内 4 次尝试后终态（与上一条同款节奏，证据互补）。
    adapter = _register_fake(monkeypatch, FakeAdapter(scenario=Timeout()))
    with pytest.raises(modelport.UpstreamFailedError) as exc_info:
        await port.complete(FAKE_MODEL, _MESSAGES)
    assert exc_info.value.code == "model_unavailable"
    assert exc_info.value.status_code == 502
    assert adapter.attempts == 4  # Run 预算上限 4，无隐藏放大


async def test_missing_credentials_maps_gateway_unavailable(port, monkeypatch):
    # 缺上游凭据 -> 503 gateway_misconfigured（provider 层，非重试码即终态）：
    # 复用 general-primary（api_key_env=DEEPSEEK_API_KEY），delenv 制造缺失。
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(modelport.GatewayUnavailableError) as exc_info:
        await port.complete("general-primary", _MESSAGES)
    assert exc_info.value.code == "gateway_misconfigured"
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# extra_body 透传（任务 1）：模板与校验档案选择项
# ---------------------------------------------------------------------------


async def test_prompt_ref_passthrough_renders_template(port, monkeypatch):
    # 模板选择项经 extra_body 直达：渲染在网关侧完成（fake 不关心消息体），
    # Trace 记录模板坐标（不变量 #12 的 ModelPort 视角）。
    _register_fake(monkeypatch, FakeAdapter())
    result = await port.complete(
        FAKE_MODEL,
        _MESSAGES,
        prompt=modelport.prompt_ref("knowledge_decision", "v1", {"product_name": "Beacon"}),
    )
    assert result.content == "ok"
    trace = CALL_TRACES[-1]
    assert trace.prompt_name == "knowledge_decision"
    assert trace.prompt_version == "v1"


async def test_missing_prompt_variable_rejected_before_upstream(port, monkeypatch):
    adapter = _register_fake(monkeypatch, FakeAdapter())
    with pytest.raises(modelport.RequestRejectedError) as exc_info:
        await port.complete(
            FAKE_MODEL,
            _MESSAGES,
            prompt=modelport.prompt_ref("knowledge_decision", "v1", {}),
        )
    assert exc_info.value.code == "missing_prompt_variable"
    assert adapter.attempts == 0  # 在调用模型之前失败（不变量 #11）


async def test_unknown_validation_profile_rejected_before_upstream(port, monkeypatch):
    adapter = _register_fake(monkeypatch, FakeAdapter())
    with pytest.raises(modelport.RequestRejectedError) as exc_info:
        await port.complete(
            FAKE_MODEL, _MESSAGES, validation=modelport.validation_ref("nope", "v9")
        )
    assert exc_info.value.code == "unknown_validation_profile"
    assert adapter.attempts == 0


async def test_structured_output_passthrough_with_schema_validation(port, monkeypatch):
    # response_format（json_schema）原样透传：输出过网关双重校验后交付，
    # result.json() 直接可用。
    _register_fake(monkeypatch, FakeAdapter(scenario=Success(content='{"answer": "42"}')))
    result = await port.complete(
        FAKE_MODEL,
        _MESSAGES,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        },
    )
    assert result.json() == {"answer": "42"}


async def test_business_validation_failure_maps_upstream_failed(port, monkeypatch):
    # 结构合法但业务非法（approve 与 reject 同真）：修复调用一次后仍违规，
    # business_validation_failed 终态——"业务非法不进 Agent Loop"的客户端视角。
    adapter = _register_fake(
        monkeypatch,
        FakeAdapter(
            scenario=Success(content='{"order_id": "D-1", "approve": true, "reject": true}')
        ),
    )
    with pytest.raises(modelport.UpstreamFailedError) as exc_info:
        await port.complete(
            FAKE_MODEL,
            _MESSAGES,
            response_format={"type": "json_object"},
            validation=modelport.validation_ref("order_decision", "v1"),
        )
    assert exc_info.value.code == "business_validation_failed"
    assert adapter.attempts == 2  # 首次 + 修复调用恰好一次
    trace = CALL_TRACES[-1]
    assert trace.status == "failed"
    assert trace.validation_profile == "order_decision/v1"


async def test_business_validation_success_records_profile_in_trace(port, monkeypatch):
    _register_fake(
        monkeypatch,
        FakeAdapter(scenario=Success(content='{"order_id": "D-1", "approve": true}')),
    )
    result = await port.complete(
        FAKE_MODEL,
        _MESSAGES,
        response_format={"type": "json_object"},
        validation=modelport.validation_ref("order_decision", "v1"),
    )
    assert result.json() == {"order_id": "D-1", "approve": True}
    assert CALL_TRACES[-1].validation_profile == "order_decision/v1"


# ---------------------------------------------------------------------------
# SDK 隔离证明（任务 3）："Agent 不依赖供应商 SDK"是可验收行为
# ---------------------------------------------------------------------------

_VENDOR_PREFIXES = ("openai", "anthropic")


async def test_public_surface_exposes_no_vendor_sdk_symbols():
    # 公开面（__all__）只有本包自有类型：任何成员的来源模块都不是供应商 SDK。
    for name in modelport.__all__:
        obj = getattr(modelport, name)
        module = getattr(obj, "__module__", "")
        assert not module.startswith(_VENDOR_PREFIXES), f"{name} 来自 {module}"
        assert not name.startswith(_VENDOR_PREFIXES)


async def test_source_has_no_star_reexport_of_vendor_sdks():
    # 源码不出现对供应商 SDK 的星号再导出（`import openai` 是实现细节，
    # 具名内部使用不构成再导出——再出口的可验收面是 __all__ 与星号导入）。
    source_root = _REPO_ROOT / "packages" / "modelport" / "src"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(_VENDOR_PREFIXES) and any(
                    alias.name == "*" for alias in node.names
                ):
                    offenders.append(str(path))
    assert offenders == []


async def test_example_agent_imports_only_modelport():
    # 示例 Agent 的 import 面 = modelport + 标准库：供应商 SDK 零出现
    # （不变量 #2 的字面证明）。
    example = _REPO_ROOT / "packages" / "modelport" / "examples" / "agent_example.py"
    tree = ast.parse(example.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".")[0])
    third_party = imports - {"modelport"} - set(sys.stdlib_module_names)
    assert third_party == set()
