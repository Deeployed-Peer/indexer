"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration values for the service."""

    app_port: int = Field(default=8000, alias="APP_PORT")
    database_url: str = Field(default="sqlite:///./data/repo_graph.db", alias="DATABASE_URL")
    workspace_root: Path = Field(default=Path("./workspaces"), alias="WORKSPACE_ROOT")
    git_clone_depth: int = Field(default=1, alias="GIT_CLONE_DEPTH")
    index_ignore: str = Field(
        default=".git,.hg,.svn,.venv,node_modules,__pycache__,dist,build",
        alias="INDEX_IGNORE",
        description="Comma separated list of glob fragments ignored during indexing",
    )
    embedding_provider: str = Field(
        default="hash",
        alias="EMBEDDING_PROVIDER",
        description="Embedding backend identifier (hash|openai)",
    )
    embedding_model: str = Field(
        default="hash://sha256",
        alias="EMBEDDING_MODEL",
        description="Embedding model name or identifier",
    )
    embedding_dimensions: int = Field(
        default=256,
        alias="EMBEDDING_DIMENSIONS",
        description="Target dimensionality for embeddings",
        ge=1,
    )
    embeddings_enabled: bool = Field(
        default=True,
        alias="EMBEDDINGS_ENABLED",
        description="Toggle embedding generation and vector storage",
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    incremental_window: int = Field(
        default=50,
        alias="INCREMENTAL_WINDOW",
        description="Maximum commit history depth considered for incremental updates",
        ge=0,
    )

    model_config = SettingsConfigDict(env_file=(".env",), case_sensitive=False)

    @validator("workspace_root", pre=True)
    def _ensure_path(cls, value):  # noqa: N805
        if isinstance(value, Path):
            return value
        return Path(str(value))

    @validator("git_clone_depth")
    def _validate_depth(cls, value: int):  # noqa: N805
        if value < 0:
            raise ValueError("GIT_CLONE_DEPTH must be non-negative")
        return value

    @property
    def ignore_patterns(self) -> tuple[str, ...]:
        return tuple(
            fragment.strip()
            for fragment in self.index_ignore.split(",")
            if fragment.strip()
        )


settings = Settings()

# Ensure workspace directory exists at startup.
settings.workspace_root.mkdir(parents=True, exist_ok=True)

__all__ = ["Settings", "settings"]
