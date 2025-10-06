"""Helpers for cloning repositories into local workspaces."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from git import Repo

from ..config import settings


logger = logging.getLogger(__name__)


class GitCloneError(RuntimeError):
    """Raised when git clone operations fail."""


def clone_repository(
    repository_url: str,
    *,
    branch: Optional[str] = None,
    commit_sha: Optional[str] = None,
    depth: Optional[int] = None,
) -> Path:
    """Clone a repository into the configured workspace and return the checkout path."""

    depth = settings.git_clone_depth if depth is None else depth
    target = settings.workspace_root / _safe_workspace_name(repository_url)
    counter = 0
    while target.exists():
        counter += 1
        target = settings.workspace_root / f"{_safe_workspace_name(repository_url)}-{counter}"

    clone_kwargs = {"depth": depth} if depth else {}
    if branch:
        clone_kwargs["branch"] = branch

    logger.info(
        "Starting repository clone",
        extra={
            "repository_url": repository_url,
            "branch": branch,
            "commit_sha": commit_sha,
            "depth": clone_kwargs.get("depth"),
        },
    )

    try:
        repo = Repo.clone_from(repository_url, target, **clone_kwargs)
        if commit_sha:
            repo.git.checkout(commit_sha)
    except Exception as exc:  # noqa: BLE001
        cleanup_workspace(target)
        logger.exception("Repository clone failed", extra={"repository_url": repository_url})
        raise GitCloneError(str(exc)) from exc

    logger.info(
        "Repository clone finished",
        extra={
            "repository_url": repository_url,
            "branch": branch,
            "commit_sha": commit_sha or repo.head.commit.hexsha,
            "checkout_path": str(target),
        },
    )

    return target


def cleanup_workspace(path: Path) -> None:
    if path.exists():
        logger.info("Removing workspace", extra={"path": str(path)})
        shutil.rmtree(path, ignore_errors=True)


def _safe_workspace_name(repository_url: str) -> str:
    basename = repository_url.rstrip("/").split("/")[-1]
    if basename.endswith(".git"):
        basename = basename[:-4]
    return basename or "workspace"


__all__ = ["clone_repository", "cleanup_workspace", "GitCloneError"]
