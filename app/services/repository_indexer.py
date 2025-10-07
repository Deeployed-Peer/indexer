"""Repository ingestion and indexing pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

from git import Repo
from sqlalchemy import delete, or_, select

from ..config import settings
from ..db import session_scope
from ..models import (
    CodeElementRecord,
    CodeFileRecord,
    EmbeddingRecord,
    RelationshipRecord,
    RepositoryRecord,
)
from .embedding_service import EmbeddingService
from .git_service import cleanup_workspace, clone_repository
from .parser_service import ParsedElement, ParsedFile, ParserService


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingRequest:
    repository_url: str
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    reload: bool = False


@dataclass(frozen=True)
class IndexingStats:
    file_count: int
    element_count: int
    relationship_count: int
    embedding_count: int
    skipped_files: int
    removed_files: int
    was_incremental: bool


@dataclass(frozen=True)
class IndexingResult:
    repository_id: str
    branch: Optional[str]
    commit_sha: str
    indexed_files: List[str]
    skipped_files: List[str]
    removed_files: List[str]
    stats: IndexingStats


class RepositoryIndexer:
    """Coordinates cloning, parsing, relationship mapping, and persistence."""

    def __init__(self) -> None:
        self.parser = ParserService()
        self.embeddings = EmbeddingService()

    def index(self, request: IndexingRequest) -> IndexingResult:
        checkout_path = clone_repository(
            request.repository_url,
            branch=request.branch,
            commit_sha=request.commit_sha,
        )
        repo = Repo(str(checkout_path))
        commit_sha = request.commit_sha or repo.head.commit.hexsha
        branch = request.branch
        if branch is None and not repo.head.is_detached:
            try:
                branch = repo.active_branch.name
            except TypeError:
                branch = None

        repository_id, repo_metadata = self._prepare_repository_record(
            repository_url=request.repository_url,
            branch=branch,
            commit_sha=commit_sha,
        )

        try:
            discovered_files = self._discover_files(Path(checkout_path))
            file_index = {
                str(path.relative_to(checkout_path)): {
                    "path": path,
                    "digest": _file_digest(path),
                }
                for path in discovered_files
            }

            paths = set(file_index.keys())
            existing_files = repo_metadata.files
            removed_paths = sorted(existing_files.keys() - paths)
            reload = request.reload

            indexed_paths: List[str] = []
            skipped_paths: List[str] = []

            for rel_path, info in file_index.items():
                record = existing_files.get(rel_path)
                if reload:
                    indexed_paths.append(rel_path)
                    continue
                if record is None:
                    indexed_paths.append(rel_path)
                elif record.digest != info["digest"]:
                    indexed_paths.append(rel_path)
                else:
                    skipped_paths.append(rel_path)

            if not indexed_paths and not removed_paths and not reload and repo_metadata.last_indexed_commit == commit_sha:
                logger.info(
                    "Repository already indexed",
                    extra={"repository_id": repository_id, "commit_sha": commit_sha},
                )
                return IndexingResult(
                    repository_id=repository_id,
                    branch=branch,
                    commit_sha=commit_sha,
                    indexed_files=[],
                    skipped_files=sorted(skipped_paths),
                    removed_files=[],
                    stats=IndexingStats(
                        file_count=0,
                        element_count=0,
                        relationship_count=0,
                        embedding_count=0,
                        skipped_files=len(skipped_paths),
                        removed_files=0,
                        was_incremental=True,
                    ),
                )

            if reload:
                removed_paths = sorted(existing_files.keys() - paths)
                skipped_paths = []
                indexed_paths = sorted(paths)

            parsed_results: List[ParsedFile] = []
            if indexed_paths:
                to_parse = [file_index[path]["path"] for path in indexed_paths if path in file_index]
                parsed_results = self.parser.parse_files(Path(checkout_path), to_parse)

            stats = self._persist(
                repository_id=repository_id,
                commit_sha=commit_sha,
                branch=branch,
                parsed_files=parsed_results,
                removed_paths=removed_paths,
                indexed_paths=indexed_paths,
                skipped_paths=skipped_paths,
                file_index=file_index,
                repo_metadata=repo_metadata,
                reload=reload,
            )

            return IndexingResult(
                repository_id=repository_id,
                branch=branch,
                commit_sha=commit_sha,
                indexed_files=sorted(indexed_paths),
                skipped_files=sorted(skipped_paths),
                removed_files=removed_paths,
                stats=stats,
            )
        finally:
            cleanup_workspace(checkout_path)

    # -- Internals -----------------------------------------------------

    def _prepare_repository_record(
        self,
        *,
        repository_url: str,
        branch: Optional[str],
        commit_sha: str,
    ) -> Tuple[str, "_RepositoryMetadata"]:
        with session_scope() as session:
            query = select(RepositoryRecord).where(
                RepositoryRecord.repository_url == repository_url,
                RepositoryRecord.branch == branch,
            )
            record = session.scalar(query)
            if record is None:
                repository_id = uuid4().hex
                record = RepositoryRecord(
                    id=repository_id,
                    repository_url=repository_url,
                    branch=branch,
                    commit_sha=commit_sha,
                    index_status="processing",
                    metadata_json={},
                )
                session.add(record)
            else:
                repository_id = record.id
                record.commit_sha = commit_sha
                record.index_status = "processing"

            session.flush()

            file_rows = list(
                session.execute(
                    select(CodeFileRecord.id, CodeFileRecord.path, CodeFileRecord.digest)
                    .where(CodeFileRecord.repository_id == repository_id)
                )
            )
            files = {
                row.path: _ExistingFile(id=row.id, digest=row.digest)
                for row in file_rows
            }

            element_rows = list(
                session.execute(
                    select(
                        CodeElementRecord.id,
                        CodeElementRecord.qualified_name,
                        CodeFileRecord.path,
                    )
                    .join(CodeFileRecord, CodeElementRecord.file_id == CodeFileRecord.id)
                    .where(CodeElementRecord.repository_id == repository_id)
                )
            )
            qualified_index = {
                row.qualified_name: row.id for row in element_rows
            }
            file_to_elements: Dict[str, Set[str]] = {}
            for row in element_rows:
                file_to_elements.setdefault(row.path, set()).add(row.qualified_name)

            metadata = _RepositoryMetadata(
                id=repository_id,
                files=files,
                qualified_index=qualified_index,
                file_to_elements=file_to_elements,
                last_indexed_commit=record.last_indexed_commit,
                initial_file_count=len(files),
            )

        return repository_id, metadata

    def _discover_files(self, root: Path) -> List[Path]:
        results: List[Path] = []
        ignore = settings.ignore_patterns
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root)
            dirnames[:] = [
                name
                for name in dirnames
                if not self._is_ignored_path(rel_dir / name, ignore)
            ]
            for filename in filenames:
                rel_path = rel_dir / filename
                full = Path(dirpath) / filename
                if self._is_ignored_path(rel_path, ignore):
                    continue
                if _is_binary(full):
                    continue
                results.append(full)
        return results

    def _is_ignored_path(self, rel_path: Path, patterns: Sequence[str]) -> bool:
        rel_str = str(rel_path)
        parts = set(rel_path.parts)
        for pattern in patterns:
            if pattern in parts:
                return True
            if fnmatch(rel_str, pattern):
                return True
        return False

    def _persist(
        self,
        *,
        repository_id: str,
        commit_sha: str,
        branch: Optional[str],
        parsed_files: Sequence[ParsedFile],
        removed_paths: Sequence[str],
        indexed_paths: Sequence[str],
        skipped_paths: Sequence[str],
        file_index: Dict[str, Dict[str, object]],
        repo_metadata: "_RepositoryMetadata",
        reload: bool,
    ) -> IndexingStats:
        element_count = 0
        relationship_count = 0
        embedding_count = 0
        embeddings_disabled = not self.embeddings.enabled

        with session_scope() as session:
            repository = session.get(RepositoryRecord, repository_id)
            if repository is None:
                raise RuntimeError(f"Repository record {repository_id} disappeared")

            if embeddings_disabled:
                session.execute(
                    delete(EmbeddingRecord).where(EmbeddingRecord.repository_id == repository_id)
                )

            paths_to_clean = set(removed_paths) | {path for path in indexed_paths if path in repo_metadata.files}
            if paths_to_clean:
                self._purge_existing(session, repository_id, paths_to_clean, repo_metadata)

            code_file_records: List[CodeFileRecord] = []
            code_element_records: List[CodeElementRecord] = []
            embedding_targets: List[Tuple[str, str]] = []

            for parsed in parsed_files:
                file_id = _stable_id("file", repository_id, parsed.path)
                summary = parsed.summary
                keyword_index = parsed.keyword_index

                record = CodeFileRecord(
                    id=file_id,
                    repository_id=repository_id,
                    path=parsed.path,
                    language=parsed.language,
                    digest=parsed.digest,
                    summary=summary,
                    keyword_index=keyword_index,
                )
                code_file_records.append(record)
                repo_metadata.files[parsed.path] = _ExistingFile(id=file_id, digest=parsed.digest)

                qualified_for_file: Set[str] = set()

                for element in parsed.elements:
                    element_id = _stable_id("element", repository_id, element.qualified_name)
                    repo_metadata.qualified_index[element.qualified_name] = element_id
                    qualified_for_file.add(element.qualified_name)

                    code_element_records.append(
                        CodeElementRecord(
                            id=element_id,
                            repository_id=repository_id,
                            file_id=file_id,
                            symbol=element.symbol,
                            qualified_name=element.qualified_name,
                            element_type=element.element_type,
                            signature=element.signature,
                            docstring=element.docstring,
                            summary=element.summary,
                            span_start=element.span[0],
                            span_end=element.span[1],
                            ast_json=element.ast_repr,
                            tokens=element.tokens,
                            is_external=False,
                        )
                    )

                    if self.embeddings.enabled:
                        embedding_targets.append((element_id, self._embedding_text(element)))

                repo_metadata.file_to_elements[parsed.path] = qualified_for_file

            session.bulk_save_objects(code_file_records)
            session.bulk_save_objects(code_element_records)
            element_count += len(code_element_records)

            relationship_records = self._materialize_relationships(
                repository_id=repository_id,
                parsed_files=parsed_files,
                repo_metadata=repo_metadata,
            )
            if relationship_records:
                session.bulk_save_objects(relationship_records)
                relationship_count += len(relationship_records)

            if self.embeddings.enabled and embedding_targets:
                payloads = [text for _, text in embedding_targets]
                embeddings = self.embeddings.embed_many(payloads)
                embedding_records: List[EmbeddingRecord] = []
                for (element_id, _), vector in zip(embedding_targets, embeddings):
                    embedding_records.append(
                        EmbeddingRecord(
                            id=_stable_id("embedding", repository_id, element_id),
                            repository_id=repository_id,
                            element_id=element_id,
                            model=settings.embedding_model,
                            dimensions=settings.embedding_dimensions,
                            vector=vector,
                        )
                    )
                if embedding_records:
                    session.bulk_save_objects(embedding_records)
                    embedding_count += len(embedding_records)

            if embeddings_disabled:
                embedding_count = 0

            repository.commit_sha = commit_sha
            repository.last_indexed_commit = commit_sha
            repository.index_status = "ready"
            metadata = dict(repository.metadata_json or {})
            metadata.update(
                {
                    "last_indexed_files": sorted(indexed_paths),
                    "skipped_files": sorted(skipped_paths),
                    "removed_files": sorted(removed_paths),
                    "node_count": len(repo_metadata.qualified_index),
                }
            )
            repository.metadata_json = metadata

        total_files = len(file_index)
        was_incremental = (
            repo_metadata.initial_file_count > 0
            and not reload
            and total_files > 0
            and len(indexed_paths) < total_files
        )
        return IndexingStats(
            file_count=len(indexed_paths),
            element_count=element_count,
            relationship_count=relationship_count,
            embedding_count=embedding_count,
            skipped_files=len(skipped_paths),
            removed_files=len(removed_paths),
            was_incremental=was_incremental,
        )

    def _materialize_relationships(
        self,
        *,
        repository_id: str,
        parsed_files: Sequence[ParsedFile],
        repo_metadata: "_RepositoryMetadata",
    ) -> List[RelationshipRecord]:
        records: List[RelationshipRecord] = []
        seen: Set[str] = set()
        for parsed in parsed_files:
            for relationship in parsed.relationships:
                source_id = repo_metadata.qualified_index.get(relationship.source)
                if not source_id:
                    continue
                target_id = repo_metadata.qualified_index.get(relationship.target)
                attributes = dict(relationship.attributes)
                if target_id is None:
                    attributes.setdefault("external", True)
                attributes_key = json.dumps(attributes, sort_keys=True)
                fingerprint = _stable_id(
                    "rel",
                    repository_id,
                    ":".join(
                        [
                            relationship.relationship_type,
                            relationship.source,
                            relationship.target,
                            attributes_key,
                        ]
                    ),
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)

                records.append(
                    RelationshipRecord(
                        id=fingerprint,
                        repository_id=repository_id,
                        source_element_id=source_id,
                        source_symbol=relationship.source,
                        target_element_id=target_id,
                        target_symbol=relationship.target,
                        relationship_type=relationship.relationship_type,
                        attributes_json=attributes,
                    )
                )
        return records

    def _purge_existing(
        self,
        session,
        repository_id: str,
        paths: Set[str],
        repo_metadata: "_RepositoryMetadata",
    ) -> None:
        if not paths:
            return
        file_ids = [
            record.id
            for path, record in repo_metadata.files.items()
            if path in paths
        ]
        if not file_ids:
            return

        element_ids = list(
            session.execute(
                select(CodeElementRecord.id)
                .where(CodeElementRecord.file_id.in_(file_ids))
            )
        )
        element_id_values = [row.id for row in element_ids]

        if element_id_values:
            session.execute(
                delete(RelationshipRecord).where(
                    RelationshipRecord.repository_id == repository_id,
                    or_(
                        RelationshipRecord.source_element_id.in_(element_id_values),
                        RelationshipRecord.target_element_id.in_(element_id_values),
                    ),
                )
            )
            session.execute(
                delete(EmbeddingRecord).where(
                    EmbeddingRecord.repository_id == repository_id,
                    EmbeddingRecord.element_id.in_(element_id_values),
                )
            )
            session.execute(
                delete(CodeElementRecord).where(CodeElementRecord.id.in_(element_id_values))
            )

        session.execute(
            delete(CodeFileRecord).where(CodeFileRecord.id.in_(file_ids))
        )

        for path in paths:
            repo_metadata.files.pop(path, None)
            qualified_names = repo_metadata.file_to_elements.pop(path, set())
            for name in qualified_names:
                repo_metadata.qualified_index.pop(name, None)

    def _embedding_text(self, element: ParsedElement) -> str:
        parts = [element.summary or ""]
        if element.signature:
            parts.append(element.signature)
        if element.docstring:
            parts.append(element.docstring)
        parts.append(" ".join(element.tokens))
        return "\n".join(part for part in parts if part)


@dataclass
class _ExistingFile:
    id: str
    digest: str


@dataclass
class _RepositoryMetadata:
    id: str
    files: Dict[str, _ExistingFile]
    qualified_index: Dict[str, str]
    file_to_elements: Dict[str, Set[str]]
    last_indexed_commit: Optional[str]
    initial_file_count: int


def _file_digest(path: Path) -> str:
    sha1 = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def _is_binary(path: Path, chunk_size: int = 1024) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(chunk_size)
            if b"\0" in chunk:
                return True
    except OSError:
        return True
    return False


__all__ = [
    "RepositoryIndexer",
    "IndexingRequest",
    "IndexingResult",
    "IndexingStats",
]
