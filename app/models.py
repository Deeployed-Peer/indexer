"""SQLAlchemy ORM models for the repository indexing service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .db import Base


class RepositoryRecord(Base):
    """Tracked git repository and the most recent indexing metadata."""

    __tablename__ = "repositories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_url: Mapped[str] = mapped_column(String(512), nullable=False)
    branch: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    commit_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_indexed_commit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    index_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class CodeFileRecord(Base):
    """A source file discovered during indexing."""

    __tablename__ = "code_files"
    __table_args__ = (
        UniqueConstraint("repository_id", "path", name="uq_code_files_repo_path"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    language: Mapped[str] = mapped_column(String(64), nullable=False)
    digest: Mapped[str] = mapped_column(String(128), nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    keyword_index: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class CodeElementRecord(Base):
    """Individual code element (function, class, variable, module)."""

    __tablename__ = "code_elements"
    __table_args__ = (
        UniqueConstraint("repository_id", "qualified_name", name="uq_elements_repo_qualified_name"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("code_files.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(256), nullable=False)
    qualified_name: Mapped[str] = mapped_column(String(512), nullable=False)
    element_type: Mapped[str] = mapped_column(String(32), nullable=False)
    signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    span_start: Mapped[int] = mapped_column(Integer, nullable=False)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False)
    ast_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    tokens: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    is_external: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class RelationshipRecord(Base):
    """Directed relationship between code elements."""

    __tablename__ = "relationships"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    source_element_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("code_elements.id", ondelete="CASCADE"), nullable=False
    )
    source_symbol: Mapped[str] = mapped_column(String(512), nullable=False)
    target_element_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("code_elements.id", ondelete="CASCADE"), nullable=True
    )
    target_symbol: Mapped[str] = mapped_column(String(512), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(32), nullable=False)
    attributes_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class EmbeddingRecord(Base):
    """Vector embedding for a code element."""

    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    element_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("code_elements.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[List[float]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class LegacyGraphRecord(Base):
    """Retained for backward compatibility with the previous implementation."""

    __tablename__ = "graphs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_url: Mapped[str] = mapped_column(String(512), nullable=False)
    branch: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    commit_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    graph_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    tags_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


__all__ = [
    "RepositoryRecord",
    "CodeFileRecord",
    "CodeElementRecord",
    "RelationshipRecord",
    "EmbeddingRecord",
    "LegacyGraphRecord",
]
