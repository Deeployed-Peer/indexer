from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from app.db import Base, engine  # noqa: E402
from app.services.git_service import GitCloneError  # noqa: E402
import app.services.graph_service as graph_service  # noqa: E402
from app.main import create_app  # noqa: E402


class DummyCodeGraph:
    def __init__(self, map_tokens: int, root: str) -> None:  # noqa: D401
        self.root = Path(root)

    def find_files(self, directories: List[str]) -> List[str]:
        return [str(self.root / "sample.py")]

    def get_code_graph(self, files: List[str]) -> Tuple[List[dict], nx.MultiDiGraph]:
        graph = nx.MultiDiGraph()
        graph.add_node("foo", category="function", info="sample", fname=str(self.root / "sample.py"))
        graph.add_edge("foo", "foo")
        tags = [
            {
                "rel_fname": "sample.py",
                "fname": str(self.root / "sample.py"),
                "line": [1, 1],
                "name": "foo",
                "kind": "def",
                "category": "function",
                "info": "def foo(): ...",
            }
        ]
        return tags, graph


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
def clean_test_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


def test_build_graph_success(monkeypatch, tmp_path):
    app = create_app()
    client = TestClient(app)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "sample.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(graph_service, "CodeGraph", DummyCodeGraph)
    monkeypatch.setattr(graph_service, "clone_repository", lambda *args, **kwargs: repo_root)
    monkeypatch.setattr(graph_service, "cleanup_workspace", lambda path: None)
    monkeypatch.setattr(graph_service, "Repo", DummyRepo)

    response = client.post(
        "/graphs",
        json={"repository_url": "https://example.com/repo.git"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["node_count"] == 1
    assert payload["nodes"][0]["id"] == "foo"
    assert payload["commit_sha"] == "abc123"

    graph_id = payload["graph_id"]
    context = client.post(
        "/graphs/context",
        json={"graph_id": graph_id, "symbol": "foo", "depth": 1},
    )
    assert context.status_code == 200
    context_payload = context.json()
    assert context_payload["symbol"] == "foo"
    assert context_payload["neighbors"] in (["foo"], [])


def test_build_graph_clone_failure(monkeypatch):
    app = create_app()
    client = TestClient(app)

    def _raise(*args, **kwargs):
        raise GitCloneError("cannot clone")

    monkeypatch.setattr(graph_service, "clone_repository", _raise)

    response = client.post(
        "/graphs",
        json={"repository_url": "https://example.com/repo.git"},
    )
    assert response.status_code == 400


def test_context_unknown_graph_id():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/graphs/context",
        json={"graph_id": "missing", "symbol": "foo"},
    )
    assert response.status_code == 404
