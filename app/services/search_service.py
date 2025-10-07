"""Hybrid embedding + graph traversal search service."""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
from sqlalchemy import select

from ..db import session_scope
from ..models import CodeElementRecord, CodeFileRecord, EmbeddingRecord, RelationshipRecord, RepositoryRecord
from .embedding_service import EmbeddingService


@dataclass(frozen=True)
class RelatedNode:
    element_id: Optional[str]
    qualified_name: str
    summary: str
    relationship: str
    direction: str
    distance: int
    attributes: Dict[str, object]
    is_external: bool
    file_path: Optional[str]
    span: Tuple[int, int]


@dataclass(frozen=True)
class SearchResult:
    element_id: str
    qualified_name: str
    summary: str
    docstring: Optional[str]
    file_path: Optional[str]
    span: Tuple[int, int]
    element_type: str
    score: float
    vector_score: float
    lexical_score: float
    related: List[RelatedNode]


class RepositoryNotIndexedError(RuntimeError):
    """Raised when a repository lookup fails."""


class EmptyIndexError(RuntimeError):
    """Raised when a repository has not been indexed yet."""


@dataclass(frozen=True)
class ContextNode:
    element_id: Optional[str]
    qualified_name: str
    summary: Optional[str]
    file_path: Optional[str]
    span: Tuple[int, int]
    element_type: str
    distance: int
    is_external: bool


@dataclass(frozen=True)
class SymbolContextResult:
    requested_symbol: str
    resolved_symbol: str
    visited: List[ContextNode]
    neighbors: List[str]
    path: List[str]


class SearchService:
    """Performs hybrid similarity + graph traversal queries."""

    def __init__(self) -> None:
        self.embeddings = EmbeddingService()

    def _load_repository_graph(
        self,
        repository_id: str,
    ) -> Tuple[nx.MultiDiGraph, Dict[str, "_ElementData"]]:
        with session_scope() as session:
            repository = session.get(RepositoryRecord, repository_id)
            if repository is None:
                raise RepositoryNotIndexedError(repository_id)

            element_rows = session.execute(
                select(
                    CodeElementRecord.id,
                    CodeElementRecord.symbol,
                    CodeElementRecord.qualified_name,
                    CodeElementRecord.summary,
                    CodeElementRecord.docstring,
                    CodeElementRecord.tokens,
                    CodeElementRecord.element_type,
                    CodeElementRecord.span_start,
                    CodeElementRecord.span_end,
                    CodeFileRecord.path,
                    EmbeddingRecord.vector,
                )
                .join(CodeFileRecord, CodeElementRecord.file_id == CodeFileRecord.id)
                .join(EmbeddingRecord, EmbeddingRecord.element_id == CodeElementRecord.id, isouter=True)
                .where(CodeElementRecord.repository_id == repository_id)
            ).all()

            if not element_rows:
                raise EmptyIndexError(repository_id)

            relationship_rows = session.execute(
                select(
                    RelationshipRecord.source_element_id,
                    RelationshipRecord.target_element_id,
                    RelationshipRecord.target_symbol,
                    RelationshipRecord.relationship_type,
                    RelationshipRecord.attributes_json,
                ).where(RelationshipRecord.repository_id == repository_id)
            ).all()

        graph = nx.MultiDiGraph()
        elements: Dict[str, _ElementData] = {}

        for row in element_rows:
            element_id = row.id
            summary = row.summary or row.docstring or row.qualified_name
            span = (row.span_start or 0, row.span_end or 0)
            tokens_normalized = list(dict.fromkeys(token.lower() for token in row.tokens or []))
            elements[element_id] = _ElementData(
                id=element_id,
                symbol=row.symbol,
                qualified_name=row.qualified_name,
                summary=summary,
                docstring=row.docstring,
                tokens=tokens_normalized,
                vector=row.vector,
                file_path=row.path,
                span=span,
                element_type=row.element_type,
            )
            graph.add_node(
                element_id,
                id=element_id,
                symbol=row.symbol,
                qualified_name=row.qualified_name,
                summary=summary,
                docstring=row.docstring,
                tokens=tokens_normalized,
                file_path=row.path,
                span=span,
                element_type=row.element_type,
                is_external=False,
            )

        for row in relationship_rows:
            source_id = row.source_element_id
            target_id = row.target_element_id or f"external::{row.target_symbol}"
            if target_id not in graph:
                graph.add_node(
                    target_id,
                    id=None,
                    symbol=row.target_symbol,
                    qualified_name=row.target_symbol,
                    summary=row.target_symbol,
                    docstring=None,
                    tokens=[],
                    file_path=None,
                    span=(0, 0),
                    element_type="external",
                    is_external=True,
                )
            graph.add_edge(
                source_id,
                target_id,
                relationship_type=row.relationship_type,
                **(row.attributes_json or {}),
            )

        return graph, elements

    def search(
        self,
        repository_id: str,
        query: str,
        *,
        limit: int = 10,
        traversal_depth: int = 2,
    ) -> List[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")

        tokens = _tokenize(query)
        token_set = set(tokens)
        query_vector = self.embeddings.embed(query)

        graph, elements = self._load_repository_graph(repository_id)

        vector_weight = 0.7 if self.embeddings.enabled else 0.0
        lexical_weight = 0.3 if self.embeddings.enabled else 1.0

        scored = [
            self._score(
                element=element,
                query_vector=query_vector,
                query_tokens=token_set,
                vector_weight=vector_weight,
                lexical_weight=lexical_weight,
            )
            for element in elements.values()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        top_results = scored[:limit]

        results: List[SearchResult] = []
        for candidate in top_results:
            context = self._traverse(
                graph,
                node_id=candidate.element.id,
                depth=traversal_depth,
            )
            results.append(
                SearchResult(
                    element_id=candidate.element.id,
                    qualified_name=candidate.element.qualified_name,
                    summary=candidate.element.summary,
                    docstring=candidate.element.docstring,
                    file_path=candidate.element.file_path,
                    span=candidate.element.span,
                    element_type=candidate.element.element_type,
                    score=candidate.score,
                    vector_score=candidate.vector_score,
                    lexical_score=candidate.lexical_score,
                    related=context,
                )
            )

        return results

    def context(
        self,
        repository_id: str,
        requested_symbols: Sequence[str],
        *,
        depth: int,
    ) -> Tuple[List[SymbolContextResult], List[str], List[str]]:
        graph, elements = self._load_repository_graph(repository_id)
        if not requested_symbols:
            return [], [], []

        qualified_map = {data.qualified_name: data.id for data in elements.values()}
        symbol_map: Dict[str, List[str]] = {}
        for element_id, data in elements.items():
            symbol_map.setdefault(data.symbol, []).append(element_id)

        lower_qualified = {name.lower(): element_id for name, element_id in qualified_map.items()}
        lower_symbol_map = {
            symbol.lower(): [element_id for element_id in ids]
            for symbol, ids in symbol_map.items()
        }

        contexts: List[SymbolContextResult] = []
        unmatched: List[str] = []

        for requested in requested_symbols:
            node_id, resolved = self._resolve_symbol(
                requested,
                elements,
                qualified_map,
                symbol_map,
                lower_qualified,
                lower_symbol_map,
            )
            if node_id is None or resolved is None:
                unmatched.append(requested)
                continue

            contexts.append(
                self._build_symbol_context(
                    graph,
                    elements,
                    start_id=node_id,
                    requested_symbol=requested,
                    resolved_symbol=resolved,
                    depth=depth,
                )
            )

        matched = [item.resolved_symbol for item in contexts]
        return contexts, matched, unmatched

    def _score(
        self,
        *,
        element: "_ElementData",
        query_vector: List[float],
        query_tokens: Set[str],
        vector_weight: float,
        lexical_weight: float,
    ) -> "_ScoredElement":
        vector_score = 0.0
        if element.vector and query_vector:
            vector_score = _cosine_similarity(query_vector, element.vector)

        lexical_score = 0.0
        if query_tokens:
            tokens = set(element.tokens)
            overlap = tokens.intersection(query_tokens)
            if overlap:
                lexical_score = len(overlap) / len(query_tokens)

        combined = (vector_weight * vector_score) + (lexical_weight * lexical_score)
        return _ScoredElement(element=element, score=combined, vector_score=vector_score, lexical_score=lexical_score)

    def _resolve_symbol(
        self,
        requested: str,
        elements: Dict[str, "_ElementData"],
        qualified_map: Dict[str, str],
        symbol_map: Dict[str, List[str]],
        lower_qualified: Dict[str, str],
        lower_symbol_map: Dict[str, List[str]],
    ) -> Tuple[Optional[str], Optional[str]]:
        exact = qualified_map.get(requested)
        if exact:
            return exact, elements[exact].qualified_name

        requested_lower = requested.lower()
        lowered = lower_qualified.get(requested_lower)
        if lowered:
            return lowered, elements[lowered].qualified_name

        tail_candidates = self._candidate_ids_for_symbol(requested, symbol_map)
        if tail_candidates:
            best = self._select_best_candidate(tail_candidates, requested, elements)
            if best:
                return best, elements[best].qualified_name

        lower_tail_candidates = self._candidate_ids_for_symbol(requested_lower, lower_symbol_map, lowercase=True)
        if lower_tail_candidates:
            best = self._select_best_candidate(lower_tail_candidates, requested, elements)
            if best:
                return best, elements[best].qualified_name

        for element_id, data in elements.items():
            qualified_lower = data.qualified_name.lower()
            if data.qualified_name.endswith(requested) or qualified_lower.endswith(requested_lower):
                return element_id, data.qualified_name

        return None, None

    def _candidate_ids_for_symbol(
        self,
        requested: str,
        symbol_map: Dict[str, List[str]],
        *,
        lowercase: bool = False,
    ) -> List[str]:
        if lowercase:
            key = requested
        else:
            key = requested

        if key in symbol_map:
            return symbol_map[key]

        parts = [part for part in re.split(r"[.:/]+", requested) if part]
        if not parts:
            return []
        tail = parts[-1]
        if lowercase:
            tail = tail.lower()
        return symbol_map.get(tail, [])

    def _select_best_candidate(
        self,
        candidates: Sequence[str],
        requested: str,
        elements: Dict[str, "_ElementData"],
    ) -> Optional[str]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        requested_tail = re.split(r"[.:/]+", requested)[-1]

        for candidate in candidates:
            data = elements[candidate]
            if data.qualified_name == requested:
                return candidate
        for candidate in candidates:
            data = elements[candidate]
            if data.qualified_name.endswith(requested):
                return candidate
        for candidate in candidates:
            data = elements[candidate]
            if data.symbol == requested or data.symbol == requested_tail:
                return candidate

        return min(candidates, key=lambda cid: len(elements[cid].qualified_name))

    def _build_symbol_context(
        self,
        graph: nx.MultiDiGraph,
        elements: Dict[str, "_ElementData"],
        *,
        start_id: str,
        requested_symbol: str,
        resolved_symbol: str,
        depth: int,
    ) -> SymbolContextResult:
        queue: deque[Tuple[str, int]] = deque([(start_id, 0)])
        visited_order: List[Tuple[str, int]] = []
        seen: Dict[str, int] = {start_id: 0}

        while queue:
            node_id, current_depth = queue.popleft()
            visited_order.append((node_id, current_depth))
            if current_depth >= depth:
                continue
            for neighbor in self._iter_neighbors(graph, node_id):
                if neighbor not in seen:
                    seen[neighbor] = current_depth + 1
                    queue.append((neighbor, current_depth + 1))

        visited_nodes = [self._make_context_node(graph, node_id, distance) for node_id, distance in visited_order]
        neighbors = self._neighbor_names(graph, start_id)
        path = [node.qualified_name for node in visited_nodes]

        return SymbolContextResult(
            requested_symbol=requested_symbol,
            resolved_symbol=resolved_symbol,
            visited=visited_nodes,
            neighbors=neighbors,
            path=path,
        )

    def _iter_neighbors(self, graph: nx.MultiDiGraph, node_id: str) -> Set[str]:
        successors = set(graph.successors(node_id)) if graph.has_node(node_id) else set()
        predecessors = set(graph.predecessors(node_id)) if graph.has_node(node_id) else set()
        return successors | predecessors

    def _make_context_node(
        self,
        graph: nx.MultiDiGraph,
        node_id: str,
        distance: int,
    ) -> ContextNode:
        data = graph.nodes.get(node_id, {})
        span_raw = data.get("span") or (0, 0)
        if isinstance(span_raw, (list, tuple)):
            if len(span_raw) >= 2:
                span = (int(span_raw[0] or 0), int(span_raw[1] or 0))
            elif len(span_raw) == 1:
                start = int(span_raw[0] or 0)
                span = (start, start)
            else:
                span = (0, 0)
        else:
            span = (0, 0)
        return ContextNode(
            element_id=data.get("id"),
            qualified_name=data.get("qualified_name", node_id),
            summary=data.get("summary"),
            file_path=data.get("file_path"),
            span=span,
            element_type=data.get("element_type", "unknown"),
            distance=distance,
            is_external=bool(data.get("is_external")),
        )

    def _neighbor_names(self, graph: nx.MultiDiGraph, node_id: str) -> List[str]:
        if not graph.has_node(node_id):
            return []
        neighbors = {
            graph.nodes[neighbor].get("qualified_name", neighbor)
            for neighbor in self._iter_neighbors(graph, node_id)
        }
        return sorted(neighbors)

    def _traverse(self, graph: nx.MultiDiGraph, *, node_id: str, depth: int) -> List[RelatedNode]:
        if node_id not in graph:
            return []
        if depth <= 0:
            return []

        queue: deque[Tuple[str, int]] = deque([(node_id, 0)])
        visited: Dict[str, int] = {node_id: 0}
        related: List[RelatedNode] = []
        seen_edges: set[Tuple[str, str, str]] = set()

        while queue:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            next_depth = current_depth + 1

            for direction, neighbors in (
                ("out", graph.successors(current)),
                ("in", graph.predecessors(current)),
            ):
                for neighbor in neighbors:
                    edge_records = _edge_records(graph, current, neighbor, direction)
                    if not edge_records:
                        continue

                    if neighbor not in visited or visited[neighbor] > next_depth:
                        visited[neighbor] = next_depth
                        queue.append((neighbor, next_depth))

                    for edge_attrs in edge_records:
                        key = (neighbor, edge_attrs["relationship_type"], direction)
                        if key in seen_edges:
                            continue
                        seen_edges.add(key)
                        node_data = graph.nodes[neighbor]
                        related.append(
                            RelatedNode(
                                element_id=node_data.get("id"),
                                qualified_name=node_data.get("qualified_name", neighbor),
                                summary=node_data.get("summary", ""),
                                relationship=edge_attrs["relationship_type"],
                                direction=direction,
                                distance=next_depth,
                                attributes=edge_attrs["attributes"],
                                is_external=bool(node_data.get("is_external")),
                                file_path=node_data.get("file_path"),
                                span=node_data.get("span", (0, 0)),
                            )
                        )

        related.sort(key=lambda item: (item.distance, item.relationship, item.qualified_name))
        return related


@dataclass(frozen=True)
class _ElementData:
    id: str
    symbol: str
    qualified_name: str
    summary: str
    docstring: Optional[str]
    tokens: List[str]
    vector: Optional[List[float]]
    file_path: Optional[str]
    span: Tuple[int, int]
    element_type: str


@dataclass(frozen=True)
class _ScoredElement:
    element: _ElementData
    score: float
    vector_score: float
    lexical_score: float


_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _tokenize(text: str) -> List[str]:
    return list(dict.fromkeys(token.lower() for token in _TOKEN_PATTERN.findall(text)))


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (norm_a * norm_b)


def _edge_records(graph: nx.MultiDiGraph, source: str, target: str, direction: str) -> List[Dict[str, object]]:
    if direction == "out":
        data = graph.get_edge_data(source, target) or {}
    else:
        data = graph.get_edge_data(target, source) or {}

    edges: List[Dict[str, object]] = []
    for attrs in data.values():
        edges.append(
            {
                "relationship_type": attrs.get("relationship_type", "related"),
                "attributes": {
                    key: value for key, value in attrs.items() if key != "relationship_type"
                },
            }
        )
    return edges


__all__ = [
    "SearchService",
    "SearchResult",
    "RelatedNode",
    "RepositoryNotIndexedError",
    "EmptyIndexError",
]
