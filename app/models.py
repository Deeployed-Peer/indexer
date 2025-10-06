"""SQLAlchemy ORM models for the repo graph service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .db import Base


class GraphRecord(Base):
    __tablename__ = "graphs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repository_url: Mapped[str] = mapped_column(String(512), nullable=False)
    branch: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    commit_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    graph_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    tags_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


__all__ = ["GraphRecord"]
