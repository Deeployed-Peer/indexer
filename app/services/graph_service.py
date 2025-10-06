"""High-level orchestration for graph construction, persistence, and context lookup."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

import networkx as nx
from git import Repo

from original.construct_graph import CodeGraph
from original.graph_searcher import RepoSearcher

from ..db import init_db, session_scope
from ..models import GraphRecord
from ..schemas import (
    BuildGraphRequest,
    BuildGraphResponse,
    ContextRequest,
    ContextResponse,
    ContextResultNode,
    GraphEdgeSchema,
    GraphNodeSchema,
    GraphStatsSchema,
    TagSchema,
)
from .git_service import cleanup_workspace, clone_repository


logger = logging.getLogger(__name__)

init_db()


class GraphNotFoundError(Exception):
    """Raised when a requested graph id is unknown to the service."""


class SymbolNotFoundError(Exception):
    """Raised when a requested symbol is absent from the stored graph."""


class GraphService:
    """Facade around CodeGraph construction and RepoSearcher context queries."""

    def build_graph(self, payload: BuildGraphRequest) -> BuildGraphResponse:
        logger.info(
            "Graph build starting",
            extra={
                "repository_url": payload.repository_url,
                "branch": payload.branch,
                "commit_sha": payload.commit_sha,
            },
        )
        checkout_path = clone_repository(
            payload.repository_url,
            branch=payload.branch,
            commit_sha=payload.commit_sha,
        )
        repo = Repo(str(checkout_path))
        commit_sha = payload.commit_sha or repo.head.commit.hexsha
        branch = payload.branch
        if branch is None and not repo.head.is_detached:
            try:
                branch = repo.active_branch.name
            except TypeError:
                branch = None

        map_tokens = payload.max_map_tokens if payload.max_map_tokens else 1024
        code_graph = CodeGraph(map_tokens=map_tokens, root=str(checkout_path))

        try:
            files = code_graph.find_files([str(checkout_path)])
            if not files:
                raise ValueError("No Python source files discovered in repository")

            logger.info(
                "Code graph generation starting",
                extra={
                    "repository_url": payload.repository_url,
                    "file_count": len(files),
                    "map_tokens": map_tokens,
                },
            )
            tags_raw, graph = code_graph.get_code_graph(files)
            graph_id = str(uuid4())

            logger.info("Converting nodes...")
            node_schemas = list(self._convert_nodes(graph))
            
            logger.info("Converting edges...")
            edge_schemas = list(self._convert_edges(graph))
            
            logger.info("Converting tags...")
            tag_schemas = [self._convert_tag(tag) for tag in tags_raw]

            logger.info(
                "Persisting graph to database",
                extra={
                    "graph_id": graph_id,
                    "node_count": len(node_schemas),
                    "edge_count": len(edge_schemas),
                },
            )

            with session_scope() as session:
                record = GraphRecord(
                    id=graph_id,
                    repository_url=payload.repository_url,
                    branch=branch,
                    commit_sha=commit_sha,
                    graph_json={
                        "nodes": [node.model_dump() for node in node_schemas],
                        "edges": [edge.model_dump() for edge in edge_schemas],
                    },
                    tags_json=[tag.model_dump() for tag in tag_schemas],
                )
                session.add(record)

        finally:
            cleanup_workspace(checkout_path)

        logger.info(
            "Graph build finished",
            extra={
                "graph_id": graph_id,
                "repository_url": payload.repository_url,
                "branch": branch,
                "commit_sha": commit_sha,
            },
        )

        return BuildGraphResponse(
            graph_id=graph_id,
            repository_url=payload.repository_url,
            branch=branch,
            commit_sha=commit_sha,
            nodes=node_schemas,
            edges=edge_schemas,
            tags=tag_schemas,
            stats=GraphStatsSchema(node_count=len(node_schemas), edge_count=len(edge_schemas)),
        )

    def get_context(self, payload: ContextRequest) -> ContextResponse:
        logger.info(
            "Context lookup starting",
            extra={"graph_id": payload.graph_id, "symbol": payload.symbol, "depth": payload.depth},
        )
        record = self._load_graph_record(payload.graph_id)
        graph = self._graph_from_json(record.graph_json)

        if payload.symbol not in graph:
            raise SymbolNotFoundError(payload.symbol)

        searcher = RepoSearcher(graph)
        visited = searcher.bfs(payload.symbol, payload.depth)
        neighbors = searcher.one_hop_neighbors(payload.symbol)

        nodes = [self._to_context_node(graph, node_id) for node_id in visited]

        logger.info(
            "Context lookup finished",
            extra={
                "graph_id": payload.graph_id,
                "symbol": payload.symbol,
                "visited_count": len(nodes),
                "neighbor_count": len(neighbors),
            },
        )

        return ContextResponse(
            symbol=payload.symbol,
            depth=payload.depth,
            visited=nodes,
            neighbors=neighbors,
            path=visited,
        )

    def _load_graph_record(self, graph_id: str) -> GraphRecord:
        with session_scope() as session:
            record = session.get(GraphRecord, graph_id)
            if record is None:
                raise GraphNotFoundError(graph_id)
            # Explicitly detach to avoid lazy loading issues outside session scope.
            session.expunge(record)
            return record

    def _graph_from_json(self, data: Dict[str, List[Dict]]) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for node in data.get("nodes", []):
            node_data = {k: v for k, v in node.items() if k != "id"}
            graph.add_node(node["id"], **node_data)
        for edge in data.get("edges", []):
            metadata = {k: v for k, v in edge.items() if k not in {"source", "target", "kind"}}
            graph.add_edge(edge["source"], edge["target"], kind=edge.get("kind"), **metadata)
        return graph

    def _convert_nodes(self, graph: nx.MultiDiGraph) -> Iterable[GraphNodeSchema]:
        for node_id, data in graph.nodes(data=True):
            metadata = {
                key: value
                for key, value in data.items()
                if key not in {"category", "info", "fname", "line", "kind"}
            }
            yield GraphNodeSchema(
                id=str(node_id),
                category=data.get("category"),
                info=data.get("info"),
                fname=data.get("fname"),
                line=data.get("line"),
                kind=data.get("kind"),
                metadata=metadata,
            )

    def _convert_edges(self, graph: nx.MultiDiGraph) -> Iterable[GraphEdgeSchema]:
        edges_iter: Iterable[Tuple[str, str, Dict[str, object]]]
        edges_iter = graph.edges(data=True)
        for source, target, data in edges_iter:
            metadata = {key: value for key, value in data.items() if key not in {"kind"}}
            yield GraphEdgeSchema(
                source=str(source),
                target=str(target),
                kind=data.get("kind"),
                metadata=metadata,
            )

    def _convert_tag(self, tag) -> TagSchema:  # type: ignore[override]
        if hasattr(tag, "_asdict"):
            raw = tag._asdict()
        elif isinstance(tag, dict):
            raw = tag
        else:
            fields = [
                "rel_fname",
                "fname",
                "line",
                "name",
                "kind",
                "category",
                "info",
            ]
            raw = {field: getattr(tag, field, None) for field in fields}
        return TagSchema(**raw)

    def _to_context_node(self, graph: nx.MultiDiGraph, node_id: str) -> ContextResultNode:
        data = graph.nodes.get(node_id, {})
        return ContextResultNode(
            id=str(node_id),
            category=data.get("category"),
            info=data.get("info"),
            fname=data.get("fname"),
            line=data.get("line"),
            kind=data.get("kind"),
        )


__all__ = [
    "GraphService",
    "GraphNotFoundError",
    "SymbolNotFoundError",
]
