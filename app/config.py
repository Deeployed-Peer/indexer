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


settings = Settings()

# Ensure workspace directory exists at startup.
settings.workspace_root.mkdir(parents=True, exist_ok=True)

__all__ = ["Settings", "settings"]
