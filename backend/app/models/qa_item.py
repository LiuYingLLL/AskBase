"""问答集模型 — A 类（文档型）知识库的人工录入 Q&A，用于提升会话效果"""

from sqlalchemy import String, Integer, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.models._utils import now_iso
from app.database import Base


class QAItem(Base):
    __tablename__ = "qa_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, comment="所属知识库")
    question: Mapped[str] = mapped_column(String(500), nullable=False, comment="问题")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="答案")
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[str] = mapped_column(String(30), nullable=False, default=now_iso)
    updated_at: Mapped[str] = mapped_column(String(30), nullable=False, default=now_iso, onupdate=now_iso)

    def __repr__(self) -> str:
        return f"<QAItem(id={self.id}, kb_id={self.kb_id}, question='{self.question[:30]}')>"
