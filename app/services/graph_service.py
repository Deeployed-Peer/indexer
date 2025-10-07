"""High-level service orchestrating repository ingestion, search, and context."""

from __future__ import annotations

import builtins
import keyword
import re
from dataclasses import asdict
from typing import Iterable, List

from sqlalchemy import func, select

from ..db import session_scope
from ..models import CodeElementRecord, RepositoryRecord
from ..schemas import (
    IndexingStatsSchema,
    ContextResultNodeSchema,
    RelatedNodeSchema,
    RepositoryIngestRequest,
    RepositoryIngestResponse,
    RepositoryContextRequest,
    RepositoryContextResponse,
    RepositorySearchRequest,
    RepositorySearchResponse,
    RepositoryStatusResponse,
    SearchResultSchema,
    SymbolContextSchema,
)
from .repository_indexer import IndexingRequest, RepositoryIndexer
from .search_service import EmptyIndexError, RepositoryNotIndexedError, SearchService


class RepositoryNotFoundError(RuntimeError):
    """Raised when a repository id cannot be located."""


class RepositoryService:
    """Facade providing repository ingestion, status lookup, and search."""

    def __init__(self) -> None:
        self.indexer = RepositoryIndexer()
        self.search_service = SearchService()

    def ingest_repository(self, request: RepositoryIngestRequest) -> RepositoryIngestResponse:
        result = self.indexer.index(
            IndexingRequest(
                repository_url=request.repository_url,
                branch=request.branch,
                commit_sha=request.commit_sha,
                reload=request.reload,
            )
        )

        stats = IndexingStatsSchema(**asdict(result.stats))

        return RepositoryIngestResponse(
            repository_id=result.repository_id,
            repository_url=request.repository_url,
            branch=result.branch,
            commit_sha=result.commit_sha,
            indexed_files=result.indexed_files,
            skipped_files=result.skipped_files,
            removed_files=result.removed_files,
            stats=stats,
        )

    def get_repository(self, repository_id: str) -> RepositoryStatusResponse:
        with session_scope() as session:
            record = session.get(RepositoryRecord, repository_id)
            if record is None:
                raise RepositoryNotFoundError(repository_id)

            node_count = session.scalar(
                select(func.count(CodeElementRecord.id)).where(
                    CodeElementRecord.repository_id == repository_id
                )
            ) or 0

            metadata = record.metadata_json or {}

            return RepositoryStatusResponse(
                repository_id=record.id,
                repository_url=record.repository_url,
                branch=record.branch,
                commit_sha=record.commit_sha,
                last_indexed_commit=record.last_indexed_commit,
                index_status=record.index_status,
                node_count=node_count,
                indexed_files=metadata.get("last_indexed_files", []),
                skipped_files=metadata.get("skipped_files", []),
                removed_files=metadata.get("removed_files", []),
                created_at=record.created_at,
                updated_at=record.updated_at,
            )

    def search_repository(
        self,
        repository_id: str,
        request: RepositorySearchRequest,
    ) -> RepositorySearchResponse:
        results = self.search_service.search(
            repository_id,
            request.query,
            limit=request.limit,
            traversal_depth=request.depth,
        )

        payload: List[SearchResultSchema] = []
        for match in results:
            related = [
                RelatedNodeSchema(
                    element_id=node.element_id,
                    qualified_name=node.qualified_name,
                    summary=node.summary,
                    relationship=node.relationship,
                    direction=node.direction,
                    distance=node.distance,
                    attributes=node.attributes,
                    is_external=node.is_external,
                    file_path=node.file_path,
                    span=node.span,
                )
                for node in match.related
            ]
            payload.append(
                SearchResultSchema(
                    element_id=match.element_id,
                    qualified_name=match.qualified_name,
                    summary=match.summary,
                    docstring=match.docstring,
                    file_path=match.file_path,
                    span=match.span,
                    element_type=match.element_type,
                    score=match.score,
                    vector_score=match.vector_score,
                    lexical_score=match.lexical_score,
                    related=related,
                )
            )

        return RepositorySearchResponse(
            repository_id=repository_id,
            query=request.query,
            limit=request.limit,
            depth=request.depth,
            results=payload,
        )

    def context_repository(
        self,
        repository_id: str,
        request: RepositoryContextRequest,
    ) -> RepositoryContextResponse:
        if request.symbol:
            requested_symbols = [request.symbol]
        else:
            requested_symbols = self._extract_symbols_from_snippet(request.snippet or "")
            if not requested_symbols:
                raise ValueError("No symbols could be inferred from snippet")

        contexts, matched, unmatched = self.search_service.context(
            repository_id,
            requested_symbols,
            depth=request.depth,
        )

        context_payload: List[SymbolContextSchema] = []
        for item in contexts:
            visited = [
                ContextResultNodeSchema(
                    element_id=node.element_id,
                    qualified_name=node.qualified_name,
                    summary=node.summary,
                    file_path=node.file_path,
                    span=node.span,
                    element_type=node.element_type,
                    distance=node.distance,
                    is_external=node.is_external,
                )
                for node in item.visited
            ]
            context_payload.append(
                SymbolContextSchema(
                    requested_symbol=item.requested_symbol,
                    resolved_symbol=item.resolved_symbol,
                    visited=visited,
                    neighbors=item.neighbors,
                    path=item.path,
                )
            )

        return RepositoryContextResponse(
            repository_id=repository_id,
            depth=request.depth,
            snippet=request.snippet,
            requested_symbols=requested_symbols,
            matched_symbols=matched,
            unmatched_symbols=unmatched,
            contexts=context_payload,
        )

    def _extract_symbols_from_snippet(self, snippet: str) -> List[str]:
        pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
        raw_tokens = pattern.findall(snippet)

        ignored = {name.lower() for name in keyword.kwlist}
        ignored.update({"self", "cls"})
        ignored.update(name.lower() for name in dir(builtins))

        symbols: List[str] = []
        seen: set[str] = set()

        for token in raw_tokens:
            for candidate in self._expand_symbol_candidate(token):
                cleaned = candidate.strip("._")
                if not cleaned:
                    continue
                lowered = cleaned.lower()
                if lowered in ignored:
                    continue
                if lowered in seen:
                    continue
                seen.add(lowered)
                symbols.append(cleaned)

        return symbols

    def _expand_symbol_candidate(self, token: str) -> Iterable[str]:
        yield token
        for part in re.split(r"[.:]", token):
            if part and part != token:
                yield part


__all__ = ["RepositoryService", "RepositoryNotFoundError"]
