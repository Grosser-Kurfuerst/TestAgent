"""Shared contracts for Experience retrieval backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, runtime_checkable

from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.types import RetrievalHit

PerTierHits: TypeAlias = dict[
    ExperienceTier,
    tuple[RetrievalHit[ExperienceMemory], ...],
]


@runtime_checkable
class RetrievalMetrics(Protocol):
    repository_revision: str
    returned_count: int

    def to_trace_payload(self) -> dict[str, Any]: ...


@runtime_checkable
class ExperienceRetriever(Protocol):
    last_metrics: RetrievalMetrics

    def retrieve_per_tier(
        self,
        query: str,
        *,
        store: ExperienceStore,
        project_key: str,
        top_k_per_tier: int,
    ) -> PerTierHits: ...

    def retrieve_candidates(
        self,
        query: str,
        *,
        store: ExperienceStore,
        project_key: str,
        top_k_per_tier: int,
    ) -> tuple[RetrievalHit[ExperienceMemory], ...]: ...

    def fork(self) -> "ExperienceRetriever": ...


def flatten_per_tier_hits(
    per_tier: PerTierHits,
    *,
    sort_key: Callable[[RetrievalHit[ExperienceMemory]], Any] | None = None,
) -> tuple[RetrievalHit[ExperienceMemory], ...]:
    merged = [hit for tier in ExperienceTier for hit in per_tier[tier]]
    if sort_key is not None:
        merged.sort(key=sort_key)
    return tuple(merged)


__all__ = [
    "ExperienceRetriever",
    "PerTierHits",
    "RetrievalMetrics",
    "flatten_per_tier_hits",
]
