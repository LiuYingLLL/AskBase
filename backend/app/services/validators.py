"""跨模块共用的业务校验

此处集中处理"名称唯一"这类被多个模块各写一遍的校验逻辑，
避免同一套判断在 agents / data_source / mcp / skill 四处分叉演进。
"""

from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def ensure_unique_name(
    db: AsyncSession,
    model,
    name: str,
    *,
    label: str,
    exclude_id: Optional[int] = None,
) -> None:
    """名称唯一性校验，冲突时抛 409

    Args:
        model: ORM 模型类，需有 name 与 id 字段
        name: 待校验名称（首尾空白会被忽略）
        label: 业务名称，用于拼接错误文案，如"智能体""数据源"
        exclude_id: 更新场景传自身 ID，避免和自己冲突
    """
    query = select(model).where(model.name == name.strip())
    if exclude_id is not None:
        query = query.where(model.id != exclude_id)

    existing = (await db.execute(query)).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409, detail=f"{label}名称「{name}」已存在，请使用其他名称"
        )
