"""Pydantic schemas used by the repository graph service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class BuildGraphRequest(BaseModel):
  """Request payload for building a repository graph."""

  repository_url: str = Field(..., description="Clone URL for the git repository")
  branch: Optional[str] = Field(default=None, description="Branch to checkout after cloning")
  commit_sha: Optional[str] = Field(default=None, description="Specific commit to checkout")
  max_map_tokens: Optional[int] = Field(
    default=None,
    description="Optional override for CodeGraph max_map_tokens parameter",
    ge=1,
  )

  @validator("repository_url")
  def _validate_repo_url(cls, value: str) -> str:  # noqa: N805
    value = value.strip()
    if not value:
      raise ValueError("repository_url must not be empty")
    return value


class TagSchema(BaseModel):
  rel_fname: Optional[str]
  fname: Optional[str]
  line: Optional[Union[int, List[int], str]]
  name: Optional[str]
  kind: Optional[str]
  category: Optional[str]
  info: Optional[str]


class GraphNodeSchema(BaseModel):
  id: str = Field(..., description="Unique node identifier (tag name)")
  category: Optional[str]
  info: Optional[str]
  fname: Optional[str]
  line: Optional[Union[int, List[int], str]]
  kind: Optional[str]
  metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdgeSchema(BaseModel):
  source: str
  target: str
  kind: Optional[str] = None
  metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphStatsSchema(BaseModel):
  node_count: int
  edge_count: int


class BuildGraphResponse(BaseModel):
  graph_id: str
  repository_url: str
  branch: Optional[str]
  commit_sha: Optional[str]
  nodes: List[GraphNodeSchema]
  edges: List[GraphEdgeSchema]
  tags: List[TagSchema]
  stats: GraphStatsSchema


class ContextRequest(BaseModel):
  graph_id: str
  symbol: str
  depth: int = Field(default=1, ge=0, le=6)

  @validator("symbol")
  def _validate_symbol(cls, value: str) -> str:  # noqa: N805
    value = value.strip()
    if not value:
      raise ValueError("symbol must not be empty")
    return value


class ContextResultNode(BaseModel):
  id: str
  category: Optional[str]
  info: Optional[str]
  fname: Optional[str]
  line: Optional[Union[int, List[int], str]]
  kind: Optional[str]


class ContextResponse(BaseModel):
  symbol: str
  depth: int
  visited: List[ContextResultNode]
  neighbors: List[str]
  path: List[str]
