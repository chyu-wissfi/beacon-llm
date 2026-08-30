"""traces 表的 ORM 映射（M09 任务 1/2）。

字段面 = spec 任务 2 的 19 字段一个不漏（demo 全量 + M06/M09 新增），与
core/schemas.py 的 CallTrace 一一对应——CallTrace 是协议面（Pydantic），
TraceRow 是存储面（SQLAlchemy），两者互转在 trace_service（写）与
governance 读路径（读），不在本模块掺业务语义。

id 自增主键是存储层自己的行坐标（按写入序全表扫描/分页用），不属于
trace 语义面，不出现在 CallTrace 上；request_id 唯一约束是"恰好一次落库"
的底层防线（上层主防线是 TraceDraft.finalized 旗标，见 trace_service 模块注）。
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    # 声明式基类：本仓库只此一个 metadata（create_all 的唯一入口在
    # storage/engine.py），新增表在本包内登记即可被建表覆盖。
    pass


class TraceRow(Base):
    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 与响应 id 对账的调用坐标（唯一约束 = 恰好一次落库的底层防线）。
    request_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # 过滤/聚合维度列（spec 任务 5）：caller / 模型 / prompt_version / status
    # 带索引；其余列只随行存取，不建索引（单实例低流量，spec 边界无轮转）。
    caller: Mapped[str | None] = mapped_column(String, index=True)
    requested_model: Mapped[str] = mapped_column(String)
    actual_model: Mapped[str | None] = mapped_column(String)
    # 实际服务模型的上游地址（ModelConfig.base_url）；未服务到任何模型的
    # 终态记 None（语义注见 core/schemas.py CallTrace.final_endpoint）。
    final_endpoint: Mapped[str | None] = mapped_column(String)
    route_reason: Mapped[str | None] = mapped_column(String)
    prompt_name: Mapped[str | None] = mapped_column(String)
    prompt_version: Mapped[str | None] = mapped_column(String, index=True)
    # Validation Profile 注册表坐标 "{name}/{version}"（未指定为 None）。
    validation_profile: Mapped[str | None] = mapped_column(String)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    # 成本计算所用的价格表版本（config/prices.yaml 的 version 快照）。
    price_version: Mapped[str | None] = mapped_column(String)
    latency_ms: Mapped[int] = mapped_column(Integer)
    ttft_ms: Mapped[int | None] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, index=True)
    error_code: Mapped[str | None] = mapped_column(String)
