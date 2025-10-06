"""Database setup using SQLAlchemy."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


if settings.database_url.startswith("sqlite:///"):
    db_path = Path(settings.database_url.replace("sqlite:///", ""))
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(settings.database_url, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


__all__ = ["Base", "engine", "SessionLocal", "session_scope", "init_db"]
