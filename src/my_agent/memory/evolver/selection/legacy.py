from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from my_agent.memory.evolver.selection.contracts import (
    ExperienceCandidate,
    SelectionPolicyCandidate,
    SelectedExperience,
    SelectionResult,
)
from my_agent.memory.evolver.selection.rendering import (
    LEGACY_SELECTED_CONTEXT_HEADER,
    render_legacy_selected_entry,
    render_selected_experiences,
    safe_task_ref,
)
from my_agent.memory.evolver.selection.service import (
    SelectionBudget,
    SelectionService,
)
from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import RetrievalHit
from my_agent.training.decision_log import DecisionEventContext


class LegacyWeightedSelectionPolicy:
    """Legacy tier-weighted policy behind the compatibility selector service."""

    def __init__(self, *, tier_caps: Mapping[str, int]) -> None:
        self.tier_caps = dict(tier_caps)

    def select(
        self,
        *,
        task: str,
        candidates: tuple[SelectionPolicyCandidate, ...],
        token_budget: int,
        max_items: int,
        context: DecisionEventContext | None,
    ) -> tuple[str, ...]:
        del task, context
        legacy_candidates = _legacy_candidates(candidates)
        if max_items <= 0 or token_budget <= 0:
            return ()
        header_tokens = estimate_tokens(LEGACY_SELECTED_CONTEXT_HEADER)
        if header_tokens > token_budget:
            return ()
        selected_ids: list[str] = []
        tier_counts: dict[str, int] = {}
        budget = SelectionBudget(
            max_tokens=token_budget,
            max_items=max_items,
            used_tokens=header_tokens,
        )
        for candidate in legacy_candidates:
            if budget.selected_items >= budget.max_items:
                continue
            tier_key = candidate.tier.value
            cap = self.tier_caps.get(tier_key, budget.max_items)
            if cap <= 0 or tier_counts.get(tier_key, 0) >= cap:
                continue
            item = SelectedExperience(
                candidate=candidate,
                rank=len(selected_ids) + 1,
                reason=f"selected by {candidate.reason}",
            )
            entry_tokens = estimate_tokens(render_legacy_selected_entry(item))
            separator_tokens = 1 if selected_ids else 0
            if not budget.accept(separator_tokens + entry_tokens):
                continue
            selected_ids.append(candidate.id)
            tier_counts[tier_key] = tier_counts.get(tier_key, 0) + 1
        return tuple(selected_ids)


class ExperienceSelector:
    """Pure tier-aware selector over typed four-tier experience hits."""

    def __init__(
        self,
        *,
        tier_weights: Mapping[str, float],
        tier_caps: Mapping[str, int],
        selected_max_items: int,
        min_score: float = 0.0,
    ) -> None:
        self.tier_weights = _normalize_tier_weights(tier_weights)
        self.tier_caps = _normalize_tier_caps(tier_caps, default=max(0, int(selected_max_items)))
        self.selected_max_items = max(0, int(selected_max_items))
        self.min_score = float(min_score)
        self.policy = LegacyWeightedSelectionPolicy(tier_caps=self.tier_caps)
        self.service = SelectionService(self.policy)

    def select(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit[ExperienceMemory]],
        max_tokens: int,
        max_items: int | None = None,
    ) -> SelectionResult:
        item_limit = self.selected_max_items if max_items is None else max(0, int(max_items))
        candidates = self._build_candidates(hits)
        selected_ids = self.service.select(
            task=query,
            candidates=tuple(candidates),
            token_budget=max_tokens,
            max_items=item_limit,
            context=None,
        )
        by_id = {candidate.id: candidate for candidate in candidates}
        selected = [
            SelectedExperience(
                candidate=by_id[memory_id],
                rank=rank,
                reason=f"selected by {by_id[memory_id].reason}",
            )
            for rank, memory_id in enumerate(selected_ids, 1)
        ]
        context = render_selected_experiences(selected, max_tokens=max_tokens)
        selected_ids = {item.candidate.id for item in selected}
        omitted_ids = tuple(candidate.id for candidate in candidates if candidate.id not in selected_ids)
        return SelectionResult(
            candidates=tuple(candidates),
            selected=tuple(selected),
            context=context,
            policy=_POLICY_NAME,
            estimated_tokens=context.estimated_tokens,
            omitted_ids=omitted_ids,
            metadata={
                "query_chars": len(query),
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "candidate_tier_counts": selection_tier_counts(candidates),
                "selected_tier_counts": selection_tier_counts(item.candidate for item in selected),
            },
        )

    def _build_candidates(
        self,
        hits: Sequence[RetrievalHit[ExperienceMemory]],
    ) -> list[ExperienceCandidate]:
        by_identity: dict[tuple[ExperienceTier, str], ExperienceCandidate] = {}
        for hit in hits:
            entry = hit.entry
            if not isinstance(entry, ExperienceMemory):
                raise TypeError("experience selector only accepts ExperienceMemory hits")
            if hit.score < self.min_score or not entry.content.strip():
                continue
            candidate = ExperienceCandidate(
                id=entry.id,
                hit=hit,
                tier=entry.tier,
                retrieval_score=float(hit.score),
                selection_score=selection_score(hit, tier_weights=self.tier_weights),
                matched_terms=tuple(hit.matched_terms),
                token_count=entry.token_count,
                value=entry.attribution_value,
                reason=_score_reason(hit, self.tier_weights),
            )
            identity = (entry.tier, entry.fingerprint or entry.id)
            previous = by_identity.get(identity)
            if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
                by_identity[identity] = candidate
        candidates = list(by_identity.values())
        candidates.sort(key=_candidate_sort_key)
        return candidates

def selection_score(
    hit: RetrievalHit[ExperienceMemory],
    *,
    tier_weights: Mapping[str, float] | None = None,
) -> float:
    entry = hit.entry
    if not isinstance(entry, ExperienceMemory):
        raise TypeError("selection_score requires an ExperienceMemory hit")
    confidence = _effective_confidence(entry)
    value_weight = clamp(1.0 + entry.attribution_value, 0.50, 1.50)
    confidence_weight = clamp(confidence, 0.50, 1.20)
    return (
        float(hit.score)
        * _tier_weight(tier_weights or {}, entry.tier)
        * value_weight
        * confidence_weight
    )


def selection_candidate_summary(candidate: ExperienceCandidate) -> dict[str, Any]:
    entry = candidate.hit.entry
    return {
        "id": candidate.id,
        "tier": candidate.tier.value,
        "score": candidate.selection_score,
        "tokens": candidate.token_count,
        "retrieval_score": candidate.retrieval_score,
        "selection_score": candidate.selection_score,
        "token_count": candidate.token_count,
        "value": candidate.value,
        "source_task": safe_task_ref(entry.source_task),
    }


def selection_tier_counts(candidates: Iterable[ExperienceCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        tier_key = candidate.tier.value
        counts[tier_key] = counts.get(tier_key, 0) + 1
    return counts


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_POLICY_NAME = "rule_tier_weighted_v1"
def _effective_confidence(entry: ExperienceMemory) -> float:
    if entry.candidate_count > 0:
        return entry.attribution_confidence
    return entry.writer_confidence


def _legacy_candidates(
    candidates: tuple[SelectionPolicyCandidate, ...],
) -> tuple[ExperienceCandidate, ...]:
    if any(not isinstance(candidate, ExperienceCandidate) for candidate in candidates):
        raise TypeError("legacy selector requires ExperienceCandidate candidates")
    return tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, ExperienceCandidate)
    )


def _score_reason(
    hit: RetrievalHit[ExperienceMemory],
    tier_weights: Mapping[str, float],
) -> str:
    entry = hit.entry
    return (
        f"retrieval={float(hit.score):.3f} "
        f"tier_weight={_tier_weight(tier_weights, entry.tier):.2f} "
        f"value={entry.attribution_value:.2f} "
        f"confidence={_effective_confidence(entry):.2f}"
    )


def _candidate_sort_key(candidate: ExperienceCandidate) -> tuple[float, float, int, str]:
    return (-candidate.selection_score, -candidate.retrieval_score, candidate.token_count, candidate.id)


def _normalize_tier_weights(mapping: Mapping[str, float]) -> dict[str, float]:
    return {tier.value: _mapping_float(mapping, tier, 1.0) for tier in ExperienceTier}


def _normalize_tier_caps(mapping: Mapping[str, int], *, default: int) -> dict[str, int]:
    caps: dict[str, int] = {}
    for tier in ExperienceTier:
        raw = _mapping_value(mapping, tier, default)
        try:
            caps[tier.value] = max(0, int(raw))
        except (TypeError, ValueError):
            caps[tier.value] = max(0, int(default))
    return caps


def _tier_weight(mapping: Mapping[str, float], tier: ExperienceTier) -> float:
    return _mapping_float(mapping, tier, 1.0)


def _mapping_float(mapping: Mapping[str, float], tier: ExperienceTier, default: float) -> float:
    raw = _mapping_value(mapping, tier, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _mapping_value(mapping: Mapping[str, Any], tier: ExperienceTier, default: Any) -> Any:
    if tier.value in mapping:
        return mapping[tier.value]
    if tier in mapping:
        return mapping[tier]  # type: ignore[index]
    return default


__all__ = [
    "ExperienceCandidate",
    "ExperienceSelector",
    "LegacyWeightedSelectionPolicy",
    "SelectedExperience",
    "SelectionResult",
    "clamp",
    "render_selected_experiences",
    "selection_candidate_summary",
    "selection_score",
    "selection_tier_counts",
]
