"""OpenAI-compatible embedding encoder used only by memory benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Callable

from my_agent.evaluation.memory_benchmark.api_config import ApiEndpoint


@dataclass(frozen=True)
class ApiEmbeddingMetrics:
    calls: int = 0
    encoded_texts: int = 0
    elapsed_sec: float = 0.0
    dimension: int | None = None


class MemoryBenchmarkApiEmbeddingEncoder:
    def __init__(
        self,
        endpoint: ApiEndpoint,
        *,
        client: Any | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.endpoint = endpoint
        self.model = endpoint.model
        self.model_revision = f"api:{endpoint.model}:{endpoint.endpoint_hash}"
        self.tokenizer_revision = self.model_revision
        self._client = client or _openai_client(endpoint)
        self._clock = clock
        self._calls = 0
        self._encoded_texts = 0
        self._elapsed_sec = 0.0
        self._dimension: int | None = None

    def encode_queries(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts)

    def encode_documents(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        return self._encode(texts)

    def metrics_snapshot(self) -> ApiEmbeddingMetrics:
        return ApiEmbeddingMetrics(
            calls=self._calls,
            encoded_texts=self._encoded_texts,
            elapsed_sec=self._elapsed_sec,
            dimension=self._dimension,
        )

    def metrics_since(self, snapshot: ApiEmbeddingMetrics) -> ApiEmbeddingMetrics:
        current = self.metrics_snapshot()
        return ApiEmbeddingMetrics(
            calls=current.calls - snapshot.calls,
            encoded_texts=current.encoded_texts - snapshot.encoded_texts,
            elapsed_sec=max(0.0, current.elapsed_sec - snapshot.elapsed_sec),
            dimension=current.dimension,
        )

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        inputs = tuple(str(text) for text in texts)
        if not inputs:
            return ()
        started = self._clock()
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=list(inputs),
            )
            vectors = _ordered_vectors(getattr(response, "data", None), len(inputs))
            dimension = len(vectors[0])
            if self._dimension is not None and dimension != self._dimension:
                raise ValueError(
                    "embedding API dimension changed from "
                    f"{self._dimension} to {dimension}"
                )
            self._dimension = dimension
            self._encoded_texts += len(inputs)
            return vectors
        finally:
            self._calls += 1
            self._elapsed_sec += max(0.0, self._clock() - started)


def _ordered_vectors(data: Any, expected_count: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ValueError("embedding API response data must be a sequence")
    if len(data) != expected_count:
        raise ValueError("embedding API returned the wrong vector count")
    ordered: list[tuple[float, ...] | None] = [None] * expected_count
    dimension: int | None = None
    for fallback_index, item in enumerate(data):
        index = _field(item, "index", fallback_index)
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("embedding response index must be an integer")
        if index < 0 or index >= expected_count or ordered[index] is not None:
            raise ValueError("embedding response indexes must uniquely cover the input")
        raw_vector = _field(item, "embedding", None)
        if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes)):
            raise ValueError("embedding vector must be a sequence")
        try:
            vector = tuple(float(value) for value in raw_vector)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding vector values must be numbers") from exc
        if not vector:
            raise ValueError("embedding vector must not be empty")
        if any(not isfinite(value) for value in vector):
            raise ValueError("embedding vector values must be finite")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("embedding vectors must have a consistent dimension")
        ordered[index] = vector
    if any(vector is None for vector in ordered):
        raise ValueError("embedding response indexes must cover the input")
    return tuple(vector for vector in ordered if vector is not None)


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _openai_client(endpoint: ApiEndpoint) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "memory benchmark API clients require the 'memory-benchmark' extra"
        ) from exc
    return OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url)


__all__ = ["ApiEmbeddingMetrics", "MemoryBenchmarkApiEmbeddingEncoder"]
