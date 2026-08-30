"""治理类只读端点：/v1/models（OpenAI 面）与 /v1/traces（网关审计域）。

design.md §3.1：/v1/models 返回平台模型列表，供调用方发现可用模型；/v1/traces
保留内部 CallTrace 形态——审计是网关自己的域，不属于 OpenAI 协议，故不套
OpenAI 包装。

M09：/v1/traces 数据源从内存 list 迁移到 SQLite（终态恰好一次落库在
trace_service）。支持四过滤（caller / model / prompt_version / status）与
聚合（group_by）：
- 无 group_by：返回过滤后的 list[CallTrace]——响应面与迁移前逐字一致；
- 有 group_by：返回 {summary, groups}（总量/总成本/平均延迟/平均 TTFT，
  按调用方/模型/Prompt 版本分组，spec 任务 5）。
model 维度口径 = COALESCE(actual_model, requested_model)：服务过取实际模型，
未服务到（失败/取消早退）归因到请求模型；过滤与分组同一口径（controller 裁决）。
"""

from collections.abc import Sequence
from datetime import timezone
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_gateway.core.schemas import CallTrace
from llm_gateway.services.catalog import MODEL_CONFIGS
from llm_gateway.services.trace_service import flush_pending
from llm_gateway.storage.engine import get_engine
from llm_gateway.storage.models import TraceRow

router = APIRouter()

# 平台自营模型的 owned_by 标识：模型表内的全部条目都由本网关运营，M03 无
# 多运营主体的区分，统一标注（OpenAI Model 对象要求该键，SDK 模型必填）。
_OWNED_BY: Literal["beacon-llm"] = "beacon-llm"

# 聚合分组维度的合法词表：非法值由 Literal 查询参数自动 422（零新增错误码）。
GroupBy = Literal["caller", "model", "prompt_version"]


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


class TraceStats(BaseModel):
    # 聚合口径（spec 任务 5）：总量、总成本、平均延迟/TTFT。
    # 空范围：count=0、合计为 0、均值为 None（无样本不算 0）。
    model_config = ConfigDict(extra="forbid")

    count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    avg_latency_ms: float | None
    avg_ttft_ms: float | None


class TraceGroup(TraceStats):
    # 分组项：key 是分组维度的取值（caller 未记 / prompt 未用时为 None）。
    key: str | None


class TraceAggregation(BaseModel):
    # group_by 响应体：整体汇总 + 逐组明细（同一过滤范围）。
    model_config = ConfigDict(extra="forbid")

    summary: TraceStats
    groups: list[TraceGroup]


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


def _model_dimension() -> Any:
    # model 维度派生列：服务过取 actual_model，未服务到归因 requested_model。
    return func.coalesce(TraceRow.actual_model, TraceRow.requested_model)


def _filters(
    caller: str | None,
    model: str | None,
    prompt_version: str | None,
    status: str | None,
) -> list[Any]:
    # 四过滤全部字面等值；任一参数缺省即不过滤该维度。
    conditions: list[Any] = []
    if caller is not None:
        conditions.append(TraceRow.caller == caller)
    if model is not None:
        conditions.append(_model_dimension() == model)
    if prompt_version is not None:
        conditions.append(TraceRow.prompt_version == prompt_version)
    if status is not None:
        conditions.append(TraceRow.status == status)
    return conditions


def _stats_columns() -> list[Any]:
    # 聚合列（summary 与 groups 共用）：count / token 合计 / 成本合计 /
    # 平均延迟 / 平均 TTFT。SUM 空集为 NULL，coalesce 归 0；AVG 空集为
    # NULL 即 None（ttft 的 NULL 样本被 AVG 天然忽略——只统计观测到首块的）。
    return [
        func.count(TraceRow.id),
        func.coalesce(func.sum(TraceRow.input_tokens), 0),
        func.coalesce(func.sum(TraceRow.output_tokens), 0),
        func.coalesce(func.sum(TraceRow.cost_usd), 0.0),
        func.avg(TraceRow.latency_ms),
        func.avg(TraceRow.ttft_ms),
    ]


def _to_stats(row: Sequence[Any]) -> dict[str, Any]:
    return {
        "count": row[0],
        "input_tokens": row[1],
        "output_tokens": row[2],
        "cost_usd": row[3],
        "avg_latency_ms": row[4],
        "avg_ttft_ms": row[5],
    }


def _to_call_trace(row: TraceRow) -> CallTrace:
    # 行 -> 协议对象：SQLite 回读的 timestamp 可能丢时区（方言行为），
    # 统一 coerce 为 UTC aware——写入面本来就是 UTC（record_trace）。
    timestamp = row.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return CallTrace(
        request_id=row.request_id,
        timestamp=timestamp,
        requested_model=row.requested_model,
        actual_model=row.actual_model,
        prompt_name=row.prompt_name,
        prompt_version=row.prompt_version,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cost_usd=row.cost_usd,
        latency_ms=row.latency_ms,
        attempts=row.attempts,
        status=row.status,  # type: ignore[arg-type]  # 写入面只有三终态字面量
        error_code=row.error_code,
        caller=row.caller,
        route_reason=row.route_reason,
        ttft_ms=row.ttft_ms,
        final_endpoint=row.final_endpoint,
        validation_profile=row.validation_profile,
        price_version=row.price_version,
    )


@router.get("/v1/traces")
async def list_traces(
    caller: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    status: str | None = None,
    group_by: GroupBy | None = None,
) -> list[CallTrace] | TraceAggregation:
    # 暴露调用审计记录，供成本分析与故障排查使用（M09：读库）。
    # 读前对账：等掉已调度未完成的落库任务——同进程 read-your-writes。
    await flush_pending()
    engine = await get_engine()
    conditions = _filters(caller, model, prompt_version, status)
    # ORM 实体水合只在 Session 语境发生（Connection 是 Core 语义，只回列值），
    # 读路径因此走 AsyncSession；聚合查询同享一个会话语境。
    async with AsyncSession(engine) as session:
        if group_by is None:
            # 无聚合：返回过滤后的全字段列表（响应面与迁移前逐字一致）。
            result = await session.execute(
                select(TraceRow).where(*conditions).order_by(TraceRow.id)
            )
            return [_to_call_trace(row) for row in result.scalars().all()]
        group_column = {
            "caller": TraceRow.caller,
            "model": _model_dimension(),
            "prompt_version": TraceRow.prompt_version,
        }[group_by]
        summary_row = (
            await session.execute(
                select(*_stats_columns()).select_from(TraceRow).where(*conditions)
            )
        ).one()
        group_rows = (
            await session.execute(
                select(group_column, *_stats_columns())
                .where(*conditions)
                .group_by(group_column)
                .order_by(group_column)
            )
        ).all()
    return TraceAggregation(
        summary=TraceStats(**_to_stats(summary_row)),
        groups=[TraceGroup(key=row[0], **_to_stats(row[1:])) for row in group_rows],
    )
