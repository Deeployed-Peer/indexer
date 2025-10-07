"""FastAPI application exposing repository ingestion and search endpoints."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from .schemas import (
    RepositoryIngestRequest,
    RepositoryIngestResponse,
    RepositoryContextRequest,
    RepositoryContextResponse,
    RepositorySearchRequest,
    RepositorySearchResponse,
    RepositoryStatusResponse,
)
from .services.git_service import GitCloneError
from .services.graph_service import RepositoryNotFoundError, RepositoryService
from .services.search_service import EmptyIndexError, RepositoryNotIndexedError


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(title="Repository Indexer", version="1.0.0")
    service = RepositoryService()

    @app.post("/repositories", response_model=RepositoryIngestResponse)
    async def ingest_repository(payload: RepositoryIngestRequest) -> RepositoryIngestResponse:
        try:
            return service.ingest_repository(payload)
        except GitCloneError as exc:
            raise HTTPException(status_code=400, detail=f"Failed to clone repository: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/repositories/{repository_id}", response_model=RepositoryStatusResponse)
    async def get_repository(repository_id: str) -> RepositoryStatusResponse:
        try:
            return service.get_repository(repository_id)
        except RepositoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown repository id: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        "/repositories/{repository_id}/search",
        response_model=RepositorySearchResponse,
    )
    async def search_repository(
        repository_id: str,
        payload: RepositorySearchRequest,
    ) -> RepositorySearchResponse:
        try:
            return service.search_repository(repository_id, payload)
        except RepositoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown repository id: {exc}") from exc
        except RepositoryNotIndexedError as exc:
            raise HTTPException(status_code=409, detail=f"Repository not indexed: {exc}") from exc
        except EmptyIndexError as exc:
            raise HTTPException(status_code=409, detail=f"Repository index empty: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        "/repositories/{repository_id}/context",
        response_model=RepositoryContextResponse,
    )
    async def context_repository(
        repository_id: str,
        payload: RepositoryContextRequest,
    ) -> RepositoryContextResponse:
        try:
            return service.context_repository(repository_id, payload)
        except RepositoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown repository id: {exc}") from exc
        except RepositoryNotIndexedError as exc:
            raise HTTPException(status_code=409, detail=f"Repository not indexed: {exc}") from exc
        except EmptyIndexError as exc:
            raise HTTPException(status_code=409, detail=f"Repository index empty: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
