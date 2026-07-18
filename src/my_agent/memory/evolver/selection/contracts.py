"""Shared contracts for legacy and formal Experience selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.types import MemoryContext, RetrievalHit
from my_agent.training.decision_log import DecisionEventContext
from my_agent.training.role_views import CandidateSnapshotEntry


@dataclass(frozen=True)
class ExperienceCandidate:
    id: str
    hit: RetrievalHit[ExperienceMemory]
    tier: ExperienceTier
    retrieval_score: float
    selection_score: float
    matched_terms: tuple[str, ...] = ()
    token_count: int = 0
    value: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class SelectedExperience:
    candidate: ExperienceCandidate
    rank: int
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    candidates: tuple[ExperienceCandidate, ...]
    selected: tuple[SelectedExperience, ...]
    context: MemoryContext[ExperienceMemory]
    policy: str
    estimated_tokens: int
    omitted_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


SelectionPolicyCandidate: TypeAlias = ExperienceCandidate | CandidateSnapshotEntry


@runtime_checkable
class TaskSelectionPolicy(Protocol):
    def select(
        self,
        *,
        task: str,
        candidates: tuple[SelectionPolicyCandidate, ...],
        token_budget: int,
        max_items: int,
        context: DecisionEventContext | None,
    ) -> tuple[str, ...]: ...


__all__ = [
    "ExperienceCandidate",
    "SelectedExperience",
    "SelectionResult",
    "SelectionPolicyCandidate",
    "TaskSelectionPolicy",
]
