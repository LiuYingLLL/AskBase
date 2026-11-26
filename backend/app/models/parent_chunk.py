"""父块模型 — 父子块切分里的完整章节

父块承载"完整语境"，子块承载"被检索命中"。父块内容不进向量库，
只在检索命中子块后按 ID 回查，用于给模型提供完整条件与例外。
"""

from typing import Optional

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models._utils import now_iso
from app.database import Base


class ParentChunk(Base):
    __tablename__ = "kb_parent_chunks"

    # 切分阶段生成的 UUID，子块当场记录，不依赖自增 ID 回填
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kb_doc_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("knowledge_documents.id"),
        nullable=False,
        index=True,
    )
    kb_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # 章节路径，JSON 数组字符串（SQLite 无原生数组类型）
    section_path: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    child_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(String(30), nullable=False, default=now_iso)

    def __repr__(self) -> str:
        return f"<ParentChunk(id={self.id[:8]}, filename='{self.filename}', children={self.child_count})>"
