"""父块仓储 — 父块的写入、按 ID 回查与清理

父块只服务于"命中子块后补全上下文"，不做语义检索，因此没有向量字段。
"""

import json
from typing import Dict, Iterable, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.parent_chunk import ParentChunk
from app.rag.parent_child import ParentChunkDraft


def _dumps_path(path: List[str]) -> str:
    return json.dumps(path, ensure_ascii=False)


def loads_path(raw: Optional[str]) -> List[str]:
    """解析章节路径，脏数据一律当空路径处理，不抛异常"""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


class ParentChunkStore:
    """基于传入 session 的父块操作，事务由调用方控制"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def replace_for_document(
        self,
        kb_doc_id: int,
        drafts: Iterable[ParentChunkDraft],
        filename: str,
        kb_id: Optional[int] = None,
    ) -> int:
        """覆盖式写入：先删该文档的旧父块，再插新的，保证重复入库幂等"""
        await self.delete_for_document(kb_doc_id)

        rows = [
            ParentChunk(
                id=draft.id,
                kb_doc_id=kb_doc_id,
                kb_id=kb_id,
                filename=filename,
                section_path=_dumps_path(draft.section_path),
                content=draft.content,
                child_count=draft.child_count,
            )
            for draft in drafts
        ]
        if not rows:
            return 0

        self.session.add_all(rows)
        await self.session.flush()
        return len(rows)

    async def delete_for_document(self, kb_doc_id: int) -> None:
        await self.session.execute(
            delete(ParentChunk).where(ParentChunk.kb_doc_id == kb_doc_id)
        )
        await self.session.flush()

    async def load_many(self, ids: Iterable[str]) -> Dict[str, ParentChunk]:
        """按 ID 批量取父块，返回 id -> 行 的映射

        注意 SQL 的 IN 查询不保证返回顺序，调用方需要自己按命中顺序重排。
        """
        id_list = [str(i) for i in ids if i]
        if not id_list:
            return {}

        result = await self.session.execute(
            select(ParentChunk).where(ParentChunk.id.in_(id_list))
        )
        return {row.id: row for row in result.scalars().all()}


async def load_parents_isolated(ids: Iterable[str]) -> Dict[str, ParentChunk]:
    """在独立会话里回查父块 —— 供检索链路使用（不持有请求级 session）"""
    from app.database import async_session_factory

    id_list = [str(i) for i in ids if i]
    if not id_list:
        return {}

    async with async_session_factory() as session:
        store = ParentChunkStore(session)
        return await store.load_many(id_list)
