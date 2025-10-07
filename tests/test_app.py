from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from app.db import Base, engine, session_scope  # noqa: E402
from app.main import create_app  # noqa: E402
import app.services.repository_indexer as repository_indexer  # noqa: E402
import app.config as app_config  # noqa: E402
from app.models import EmbeddingRecord  # noqa: E402


class DummyRepo:
    class _Head:
        def __init__(self) -> None:
            self.commit = type("Commit", (), {"hexsha": "abc123"})()
            self.is_detached = False

    def __init__(self, path: str) -> None:  # noqa: D401
        self.path = path
        self.head = self._Head()
        self.active_branch = type("Branch", (), {"name": "main"})()


@pytest.fixture(autouse=True)
def clean_test_db() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


def _patch_git(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setattr(repository_indexer, "clone_repository", lambda *args, **kwargs: repo_root)
    monkeypatch.setattr(repository_indexer, "cleanup_workspace", lambda path: None)
    monkeypatch.setattr(repository_indexer, "Repo", DummyRepo)


def test_ingest_repository_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = create_app()
    client = TestClient(app)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "def foo():\n    return 1\n",
        encoding="utf-8",
    )

    _patch_git(monkeypatch, repo_root)

    response = client.post(
        "/repositories",
        json={"repository_url": "https://example.com/repo.git"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["file_count"] == 1
    repository_id = payload["repository_id"]

    status = client.get(f"/repositories/{repository_id}")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["node_count"] >= 2

    search = client.post(
        f"/repositories/{repository_id}/search",
        json={"query": "foo"},
    )
    assert search.status_code == 200
    search_payload = search.json()
    assert search_payload["results"]
    top = search_payload["results"][0]
    assert "foo" in top["qualified_name"]
    assert top["related"]

    context = client.post(
        f"/repositories/{repository_id}/context",
        json={"symbol": "foo", "depth": 1},
    )
    assert context.status_code == 200
    context_payload = context.json()
    assert context_payload["matched_symbols"]
    assert context_payload["contexts"]
    assert context_payload["contexts"][0]["visited"]

    snippet_context = client.post(
        f"/repositories/{repository_id}/context",
        json={"snippet": "def foo():\n    return 1", "depth": 1},
    )
    assert snippet_context.status_code == 200
    snippet_payload = snippet_context.json()
    assert snippet_payload["matched_symbols"]
    assert snippet_payload["contexts"]


def test_ingest_repository_incremental(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = create_app()
    client = TestClient(app)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "def foo():\n    return 1\n",
        encoding="utf-8",
    )

    _patch_git(monkeypatch, repo_root)

    first = client.post(
        "/repositories",
        json={"repository_url": "https://example.com/repo.git"},
    )
    assert first.status_code == 200

    second = client.post(
        "/repositories",
        json={"repository_url": "https://example.com/repo.git"},
    )
    assert second.status_code == 200
    payload = second.json()
    assert payload["stats"]["file_count"] == 0
    assert payload["stats"]["was_incremental"] is True
    assert payload["stats"]["skipped_files"] >= 1
    assert payload["skipped_files"]


def test_ingest_repository_without_embeddings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_config.settings, "embeddings_enabled", False, raising=False)

    app = create_app()
    client = TestClient(app)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "def foo():\n    return 1\n",
        encoding="utf-8",
    )

    _patch_git(monkeypatch, repo_root)

    response = client.post(
        "/repositories",
        json={"repository_url": "https://example.com/repo.git"},
    )
    assert response.status_code == 200
    payload = response.json()
    repository_id = payload["repository_id"]

    with session_scope() as session:
        count = session.scalar(
            select(func.count(EmbeddingRecord.id)).where(EmbeddingRecord.repository_id == repository_id)
        )
    assert count == 0

    search = client.post(
        f"/repositories/{repository_id}/search",
        json={"query": "foo"},
    )
    assert search.status_code == 200
    assert search.json()["results"]

    context = client.post(
        f"/repositories/{repository_id}/context",
        json={"symbol": "foo", "depth": 1},
    )
    assert context.status_code == 200
    assert context.json()["contexts"]


def test_get_unknown_repository_returns_404() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/repositories/missing")
    assert response.status_code == 404


def test_search_unknown_repository_returns_404() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/repositories/missing/search",
        json={"query": "foo"},
    )
    assert response.status_code == 404
