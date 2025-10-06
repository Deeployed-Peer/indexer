"""FastAPI application exposing repository graph operations."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from .schemas import BuildGraphRequest, BuildGraphResponse, ContextRequest, ContextResponse
from .services.git_service import GitCloneError
from .services.graph_service import GraphNotFoundError, GraphService, SymbolNotFoundError


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(title="Repo Graph Service", version="0.2.0")
    graph_service = GraphService()

    @app.post("/graphs", response_model=BuildGraphResponse)
    async def build_graph(payload: BuildGraphRequest) -> BuildGraphResponse:
        try:
            return graph_service.build_graph(payload)
        except GitCloneError as exc:
            raise HTTPException(status_code=400, detail=f"Failed to clone repository: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/graphs/context", response_model=ContextResponse)
    async def get_context(payload: ContextRequest) -> ContextResponse:
        try:
            return graph_service.get_context(payload)
        except GraphNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown graph id: {exc}") from exc
        except SymbolNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown symbol: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
