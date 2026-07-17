"""Immutable embedding index records coupled to one repository revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from my_agent.memory.evolver.types import ExperienceMemory, ExperienceTier


@dataclass(frozen=True)
class EmbeddingIndexEntry:
    memory: ExperienceMemory
    vector: tuple[float, ...]
    content_hash: str


@dataclass(frozen=True)
class EmbeddingIndexSnapshot:
    repository_revision: str
    entries_by_tier: Mapping[ExperienceTier, tuple[EmbeddingIndexEntry, ...]]

    def __post_init__(self) -> None:
        if not self.repository_revision:
            raise ValueError("embedding index requires repository_revision")
        if set(self.entries_by_tier) != set(ExperienceTier):
            raise ValueError("embedding index must include every experience tier")


__all__ = ["EmbeddingIndexEntry", "EmbeddingIndexSnapshot"]
