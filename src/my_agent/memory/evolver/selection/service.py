"""Shared deterministic candidate snapshot and selection limiting services."""

from __future__ import annotations

from dataclasses import dataclass

from my_agent.memory.evolver.selection.contracts import (
    SelectionPolicyCandidate,
    TaskSelectionPolicy,
)
from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.types import RetrievalHit
from my_agent.training.decision_log import DecisionEventContext
from my_agent.training.role_views import CandidateSnapshotEntry


@dataclass
class SelectionBudget:
    max_tokens: int
    max_items: int
    used_tokens: int = 0
    selected_items: int = 0

    def __post_init__(self) -> None:
        self.max_tokens = max(0, int(self.max_tokens))
        self.max_items = max(0, int(self.max_items))
        self.used_tokens = max(0, int(self.used_tokens))
        self.selected_items = max(0, int(self.selected_items))

    def accept(self, token_count: int) -> bool:
        tokens = max(0, int(token_count))
        if self.selected_items >= self.max_items:
            return False
        if self.used_tokens + tokens > self.max_tokens:
            return False
        self.used_tokens += tokens
        self.selected_items += 1
        return True


class SelectionService:
    def __init__(self, policy: TaskSelectionPolicy) -> None:
        self.policy = policy

    def select(
        self,
        *,
        task: str,
        candidates: tuple[SelectionPolicyCandidate, ...],
        token_budget: int,
        max_items: int,
        context: DecisionEventContext | None,
    ) -> tuple[str, ...]:
        selected_ids = self.policy.select(
            task=task,
            candidates=candidates,
            token_budget=token_budget,
            max_items=max_items,
            context=context,
        )
        if not isinstance(selected_ids, tuple) or any(
            not isinstance(item, str) for item in selected_ids
        ):
            raise ValueError("selector must return a tuple of memory IDs")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selector returned duplicate memory IDs")
        return selected_ids


def candidate_snapshot(
    hits: tuple[RetrievalHit[ExperienceMemory], ...],
) -> tuple[CandidateSnapshotEntry, ...]:
    ranks = {tier: 0 for tier in ExperienceTier}
    candidates: list[CandidateSnapshotEntry] = []
    for hit in hits:
        tier = hit.entry.tier
        ranks[tier] += 1
        candidates.append(CandidateSnapshotEntry(
            label=f"RETRIEVED_{tier.value.upper()}_{ranks[tier]:02d}",
            memory_id=hit.entry.id,
            tier=tier.value,
            content=hit.entry.content,
            retrieval_score=float(hit.score),
            rank=ranks[tier],
            token_count=hit.entry.token_count,
        ))
    return tuple(candidates)


def limit_selected_ids(
    selected_ids: tuple[str, ...],
    *,
    candidates: tuple[SelectionPolicyCandidate, ...],
    token_budget: int,
    max_items: int,
) -> tuple[str, ...]:
    if not isinstance(selected_ids, tuple) or any(
        not isinstance(item, str) for item in selected_ids
    ):
        raise ValueError("selector must return a tuple of memory IDs")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selector returned duplicate memory IDs")
    by_id = {_candidate_id(item): item for item in candidates}
    if any(memory_id not in by_id for memory_id in selected_ids):
        raise ValueError(
            "selector referenced memory outside the frozen candidate snapshot"
        )
    kept: list[str] = []
    budget = SelectionBudget(max_tokens=token_budget, max_items=max_items)
    for memory_id in selected_ids:
        if budget.selected_items >= budget.max_items:
            break
        candidate = by_id[memory_id]
        if not budget.accept(candidate.token_count):
            break
        kept.append(memory_id)
    return tuple(kept)


def _candidate_id(candidate: SelectionPolicyCandidate) -> str:
    if isinstance(candidate, CandidateSnapshotEntry):
        return candidate.memory_id
    return candidate.id


__all__ = [
    "SelectionBudget",
    "SelectionService",
    "candidate_snapshot",
    "limit_selected_ids",
]
