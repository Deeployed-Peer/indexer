"""Embedding generation utilities."""

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass
from typing import List, Sequence

from ..config import settings


logger = logging.getLogger(__name__)


class BaseEmbeddingProvider:
    """Simple protocol for embedding providers."""

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class HashEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic embedding provider for offline usage."""

    dimensions: int

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
            rng = random.Random(seed)
            vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
            norm = math.sqrt(sum(component * component for component in vector)) or 1.0
            normalized = [component / norm for component in vector]
            vectors.append(normalized)
        return vectors


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by the OpenAI API."""

    def __init__(self, model: str, dimensions: int | None) -> None:
        try:
            from openai import OpenAI
        except ImportError as err:  # pragma: no cover - import guard
            raise RuntimeError("openai package is required for OpenAI embedding provider") from err

        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embedding provider")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        vectors: List[List[float]] = []
        for item in response.data:
            vector = list(item.embedding)
            if self.dimensions and len(vector) != self.dimensions:
                logger.debug(
                    "Resizing embedding vector",
                    extra={"original_dimensions": len(vector), "target_dimensions": self.dimensions},
                )
                vector = _resize(vector, self.dimensions)
            vectors.append(vector)
        return vectors


def _resize(vector: List[float], dimensions: int) -> List[float]:
    if len(vector) == dimensions:
        return vector
    if len(vector) > dimensions:
        return vector[:dimensions]
    padded = vector[:]
    generator = random.Random(0)
    while len(padded) < dimensions:
        padded.append(generator.uniform(-1.0, 1.0))
    return padded


class EmbeddingService:
    """Convenience layer over configured embedding provider with a hash fallback."""

    def __init__(self) -> None:
        provider_name = settings.embedding_provider.lower().strip()
        dim = settings.embedding_dimensions
        provider: BaseEmbeddingProvider
        self.enabled = settings.embeddings_enabled

        if not self.enabled:
            provider = HashEmbeddingProvider(dim)
            self.provider: BaseEmbeddingProvider | None = None
            self._fallback = provider
            return

        if provider_name == "openai":
            try:
                provider = OpenAIEmbeddingProvider(settings.embedding_model, dim or None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falling back to hash embeddings", exc_info=exc)
                provider = HashEmbeddingProvider(dim)
        else:
            provider = HashEmbeddingProvider(dim)

        self.provider = provider
        self._fallback = HashEmbeddingProvider(dim)

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        if not self.enabled:
            return [[] for _ in texts]

        try:
            return self.provider.embed_many(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding provider failed; using hash fallback", exc_info=exc)
            return self._fallback.embed_many(texts)

    def embed(self, text: str) -> List[float]:
        if not self.enabled:
            return []
        result = self.embed_many([text])
        return result[0] if result else []
