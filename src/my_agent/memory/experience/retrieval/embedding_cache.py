"""Revision-aware embedding cache with content-stable vector reuse."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class EmbeddingCacheKey:
    embedding_model_revision: str
    tokenizer_revision: str
    repository_revision: str
    memory_id: str
    content_hash: str
    embedding_prompt_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "embedding_model_revision",
            "tokenizer_revision",
            "repository_revision",
            "memory_id",
            "content_hash",
            "embedding_prompt_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"embedding cache {field_name} must not be empty")


class EmbeddingCache:
    def __init__(self) -> None:
        self._by_revision: dict[EmbeddingCacheKey, tuple[float, ...]] = {}
        self._by_content: dict[tuple[str, str, str, str, str], tuple[float, ...]] = {}
        self._lock = RLock()

    def get(self, key: EmbeddingCacheKey) -> tuple[float, ...] | None:
        with self._lock:
            direct = self._by_revision.get(key)
            if direct is not None:
                return direct
            reusable = self._by_content.get(_content_key(key))
            if reusable is not None:
                self._by_revision[key] = reusable
            return reusable

    def put(self, key: EmbeddingCacheKey, vector: tuple[float, ...]) -> None:
        normalized = _validate_vector(vector)
        with self._lock:
            self._by_revision[key] = normalized
            self._by_content[_content_key(key)] = normalized

    def invalidate_memory(self, memory_id: str) -> None:
        with self._lock:
            self._by_revision = {
                key: vector
                for key, vector in self._by_revision.items()
                if key.memory_id != memory_id
            }
            self._by_content = {
                key: vector
                for key, vector in self._by_content.items()
                if key[2] != memory_id
            }

    @property
    def revision_entry_count(self) -> int:
        with self._lock:
            return len(self._by_revision)

    @property
    def content_entry_count(self) -> int:
        with self._lock:
            return len(self._by_content)


def _content_key(key: EmbeddingCacheKey) -> tuple[str, str, str, str, str]:
    return (
        key.embedding_model_revision,
        key.tokenizer_revision,
        key.memory_id,
        key.content_hash,
        key.embedding_prompt_version,
    )


def _validate_vector(vector: tuple[float, ...]) -> tuple[float, ...]:
    if not vector:
        raise ValueError("embedding vector must not be empty")
    normalized = tuple(float(item) for item in vector)
    if any(item != item or item in {float("inf"), float("-inf")} for item in normalized):
        raise ValueError("embedding vector must contain finite values")
    return normalized


__all__ = ["EmbeddingCache", "EmbeddingCacheKey"]
