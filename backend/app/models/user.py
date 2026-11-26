"""用户模型"""

from sqlalchemy import String, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models._utils import now_iso
from app.database import Base
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.knowledge_document import KnowledgeDocument
    from app.models.agent import Agent


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False, default="user")  # "admin" | "user"
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[str] = mapped_column(
        String(30), nullable=False, default=now_iso
    )
    updated_at: Mapped[str] = mapped_column(
        String(30), nullable=False, default=now_iso, onupdate=now_iso
    )

    # 关系
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    knowledge_documents: Mapped[List["KnowledgeDocument"]] = relationship(
        "KnowledgeDocument", back_populates="uploader", cascade="all, delete-orphan"
    )
    agents: Mapped[List["Agent"]] = relationship(
        "Agent", back_populates="creator"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"
