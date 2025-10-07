# Repository Indexer Service

FastAPI microservice that clones Git repositories, builds an AST-backed code graph, stores embeddings, and exposes hybrid semantic/graph search capabilities.

## Capabilities

- **Repository scanning & ingestion** – clone or fetch repositories on demand, honouring ignore patterns and supporting branch/commit targets.
- **Structural parsing** – walk source files, build AST representations, and capture functions, classes, variables, and modules with natural-language summaries.
- **Relationship mapping** – persist call, import, inheritance, and data-flow edges between code elements, including external dependency references.
- **Embedding generation** – create deterministic (or provider-backed) embeddings per element and maintain lexical tokens for hybrid retrieval.
- **Hybrid search** – start from embedding similarity then traverse graph edges to surface related symbols across modules.
- **Graph storage** – persist repositories, files, elements, relationships, and embeddings in a relational store for later queries.
- **Incremental updates** – detect unchanged files, re-index only what changed, and support forced reloads when desired.

## Requirements

- **Python 3.11+**
- **Git 2.40+**
- **SQLite** (bundled) or any SQLAlchemy-compatible database via `DATABASE_URL`

Optional: `uvicorn` for local serving, `pytest` for testing, `httpx` for smoke checks.

## Configuration

Environment variables are read via `pydantic-settings` (see `app/config.py`). Common options:

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./data/repo_graph.db` | SQLAlchemy connection string |
| `WORKSPACE_ROOT` | `./workspaces` | Clone workspace root |
| `GIT_CLONE_DEPTH` | `1` | Depth for shallow clones |
| `INDEX_IGNORE` | `.git,.hg,.svn,.venv,node_modules,__pycache__,dist,build` | Ignore fragments during discovery |
| `EMBEDDING_PROVIDER` | `hash` | `hash` (deterministic) or `openai` |
| `EMBEDDING_MODEL` | `hash://sha256` | Provider-specific model identifier |
| `EMBEDDING_DIMENSIONS` | `256` | Target embedding width |
| `EMBEDDINGS_ENABLED` | `true` | Disable to skip embedding generation and rely on lexical + graph traversal |

If `EMBEDDING_PROVIDER=openai`, set `OPENAI_API_KEY` accordingly.

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The service listens on `localhost:8000` by default.

## API

### 1. Ingest / Re-index a Repository

```http
POST /repositories
Content-Type: application/json

{
  "repository_url": "https://github.com/example/project.git",
  "branch": "main",
  "reload": false
}
```

Response:

```json
{
  "repository_id": "7b7f0c81771d4b1a8f6d5bb7f2f96aa8",
  "repository_url": "https://github.com/example/project.git",
  "branch": "main",
  "commit_sha": "abc123",
  "indexed_files": ["src/module.py"],
  "skipped_files": [],
  "removed_files": [],
  "stats": {
    "file_count": 1,
    "element_count": 4,
    "relationship_count": 3,
    "embedding_count": 4,
    "skipped_files": 0,
    "removed_files": 0,
    "was_incremental": false
  }
}
```

Set `reload=true` to force a full re-index even if nothing changed.

### 2. Repository Status

```http
GET /repositories/{repository_id}
```

Returns the current commit, indexing status, counts, and most recent file lists.

### 3. Hybrid Search

```http
POST /repositories/{repository_id}/search
Content-Type: application/json

{
  "query": "where is the auth configured",
  "limit": 5,
  "depth": 2
}
```

Response includes ranked elements with semantic scores and relationship traces (imports, calls, inheritance, etc.) discovered via graph traversal.

### 4. Code Context (Symbol or Snippet)

```http
POST /repositories/{repository_id}/context
Content-Type: application/json

{
  "symbol": "module.foo",
  "depth": 2
}
```

Alternatively, send a snippet and the service infers identifiers:

```http
POST /repositories/{repository_id}/context

{
  "snippet": "def foo():\n    return bar()",
  "depth": 2
}
```

The response lists matched symbols, per-symbol traversals (visited nodes with distances), and one-hop neighbors, enabling quick inspection of related code without relying on embeddings.

## Testing

The test suite targets the FastAPI layer, incremental indexing, and the embeddings toggle:

```bash
python -m pytest
```

Ensure `pytest` is installed in your active environment. Tests rely on the in-memory SQLite database configured via `DATABASE_URL`.
