"""Pydantic schemas for repository ingestion and search APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator, validator


class RepositoryIngestRequest(BaseModel):
    """Request to trigger repository indexing."""

    repository_url: str = Field(..., description="Clone URL for the git repository")
    branch: Optional[str] = Field(default=None, description="Branch to checkout")
    commit_sha: Optional[str] = Field(default=None, description="Specific commit to checkout")
    reload: bool = Field(default=False, description="Force a full reindex even if unchanged")

    @validator("repository_url")
    def _validate_repository_url(cls, value: str) -> str:  # noqa: N805
        value = value.strip()
        if not value:
            raise ValueError("repository_url must not be empty")
        return value


class IndexingStatsSchema(BaseModel):
    file_count: int
    element_count: int
    relationship_count: int
    embedding_count: int
    skipped_files: int
    removed_files: int
    was_incremental: bool


class RepositoryIngestResponse(BaseModel):
    repository_id: str
    repository_url: str
    branch: Optional[str]
    commit_sha: str
    indexed_files: List[str]
    skipped_files: List[str]
    removed_files: List[str]
    stats: IndexingStatsSchema


class RepositoryStatusResponse(BaseModel):
    repository_id: str
    repository_url: str
    branch: Optional[str]
    commit_sha: Optional[str]
    last_indexed_commit: Optional[str]
    index_status: str
    node_count: int
    indexed_files: List[str]
    skipped_files: List[str]
    removed_files: List[str]
    created_at: datetime
    updated_at: datetime


class RepositorySearchRequest(BaseModel):
    query: str = Field(..., description="Free-form search query")
    limit: int = Field(default=5, ge=1, le=50)
    depth: int = Field(default=2, ge=0, le=6)

    @validator("query")
    def _validate_query(cls, value: str) -> str:  # noqa: N805
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class RelatedNodeSchema(BaseModel):
    element_id: Optional[str]
    qualified_name: str
    summary: Optional[str]
    relationship: str
    direction: str
    distance: int
    attributes: Dict[str, Any]
    is_external: bool
    file_path: Optional[str]
    span: Tuple[int, int]


class SearchResultSchema(BaseModel):
    element_id: str
    qualified_name: str
    summary: Optional[str]
    docstring: Optional[str]
    file_path: Optional[str]
    span: Tuple[int, int]
    element_type: str
    score: float
    vector_score: float
    lexical_score: float
    related: List[RelatedNodeSchema]


class RepositorySearchResponse(BaseModel):
    repository_id: str
    query: str
    limit: int
    depth: int
    results: List[SearchResultSchema]


class RepositoryContextRequest(BaseModel):
    symbol: Optional[str] = Field(default=None, description="Exact symbol or qualified name")
    snippet: Optional[str] = Field(default=None, description="Snippet used to infer symbols")
    depth: int = Field(default=2, ge=0, le=6)

    @validator("symbol")
    def _validate_symbol(cls, value: Optional[str]) -> Optional[str]:  # noqa: N805
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("symbol must not be empty")
        return value

    @validator("snippet")
    def _validate_snippet(cls, value: Optional[str]) -> Optional[str]:  # noqa: N805
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("snippet must not be empty")
        return value

    @model_validator(mode="after")
    def _require_symbol_or_snippet(cls, values: "RepositoryContextRequest") -> "RepositoryContextRequest":  # noqa: N805
        symbol = values.symbol
        snippet = values.snippet
        if bool(symbol) == bool(snippet):
            raise ValueError("Provide exactly one of symbol or snippet")
        return values


class ContextResultNodeSchema(BaseModel):
    element_id: Optional[str]
    qualified_name: str
    summary: Optional[str]
    file_path: Optional[str]
    span: Tuple[int, int]
    element_type: str
    distance: int
    is_external: bool


class SymbolContextSchema(BaseModel):
    requested_symbol: str
    resolved_symbol: str
    visited: List[ContextResultNodeSchema]
    neighbors: List[str]
    path: List[str]


class RepositoryContextResponse(BaseModel):
    repository_id: str
    depth: int
    snippet: Optional[str]
    requested_symbols: List[str]
    matched_symbols: List[str]
    unmatched_symbols: List[str]
    contexts: List[SymbolContextSchema]


__all__ = [
    "RepositoryIngestRequest",
    "RepositoryIngestResponse",
    "RepositoryStatusResponse",
    "RepositorySearchRequest",
    "RepositorySearchResponse",
    "SearchResultSchema",
    "RelatedNodeSchema",
    "IndexingStatsSchema",
    "RepositoryContextRequest",
    "RepositoryContextResponse",
    "SymbolContextSchema",
    "ContextResultNodeSchema",
]
