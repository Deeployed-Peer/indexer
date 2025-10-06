# Repo Graph Service

Python FastAPI microservice that clones git repositories, builds call graphs with the original `CodeGraph` utilities, persists the results, and exposes HTTP endpoints for retrieving graph context.

## Requirements

- **Python 3.11+**
- **Git 2.40+** (used for cloning)
- **SQLite** (bundled) or any database supported by SQLAlchemy via `DATABASE_URL`

Optional: `uvicorn`/`gunicorn` for production serving; development scripts rely on `pytest` and `httpx`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
cp .env.example .env
```

The default `.env` uses a local SQLite database at `./data/repo_graph.db` and clones repositories into `./workspaces`.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_PORT` | `8000` | Port passed to uvicorn when running locally |
| `DATABASE_URL` | `sqlite:///./data/repo_graph.db` | SQLAlchemy connection string |
| `WORKSPACE_ROOT` | `./workspaces` | Directory where repositories are cloned |
| `GIT_CLONE_DEPTH` | `1` | Depth passed to git clone (0 / unset for full clone) |

## Running the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-8000}
```

On startup the service will create the configured workspace directory and ensure the `graphs` table exists.

## HTTP API

### `POST /graphs`

Builds a new graph by cloning the provided repository URL, running the analyzer, storing the result, and returning the persisted graph metadata.

Request body:

```json
{
  "repository_url": "https://github.com/owner/project.git",
  "branch": "main",
  "commit_sha": "optional commit",
  "max_map_tokens": 2048
}
```

Response body:

```json
{
  "graph_id": "uuid",
  "repository_url": "https://github.com/owner/project.git",
  "branch": "main",
  "commit_sha": "abc123",
  "nodes": [ ... ],
  "edges": [ ... ],
  "tags": [ ... ],
  "stats": { "node_count": 42, "edge_count": 99 }
}
```

### `POST /graphs/context`

Fetches a stored graph and performs a BFS around the requested symbol.

Request:

```json
{
  "graph_id": "uuid returned from /graphs",
  "symbol": "function_name",
  "depth": 2
}
```

Response:

```json
{
  "symbol": "function_name",
  "depth": 2,
  "visited": [ { "id": "function_name", ... } ],
  "neighbors": ["helper"],
  "path": ["function_name", "helper"]
}
```

If the graph is unknown or the symbol cannot be found a `404` is returned with an explanatory message.

## Development workflow

- Run tests: `pytest`
- Format / lint is intentionally left to tooling of your choice
- Python dependencies are tracked in `pyproject.toml`; pinned requirement files live under `requirements/`

Graphs are stored in the `graphs` table with their serialized nodes/edges for repeatable retrieval without re-cloning.

## Project layout

```
app/
├── __init__.py
├── config.py          # Settings sourced from .env / environment
├── db.py              # SQLAlchemy engine/session helpers
├── main.py            # FastAPI wiring
├── models.py          # ORM models
├── schemas.py         # Pydantic contracts
└── services/
    ├── git_service.py # Git clone helpers
    └── graph_service.py
original/              # Legacy graph builders reused by the service
requirements/          # Runtime and dev dependency lists
tests/                 # pytest suite
```

## Persistence details

Graphs are persisted as JSON blobs (`nodes`, `edges`, `tags`) keyed by a UUID alongside repository metadata. The default SQLite configuration is production-ready for small deployments; switch `DATABASE_URL` to Postgres/MySQL/etc. for higher concurrency.

## Cleanup

The service removes temporary workspaces after indexing. If you want to keep cloned repositories (e.g., for debugging), comment out the `cleanup_workspace` call in `graph_service.py`.
