"""Shared read-only lookup and redundancy helpers for maintenance."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from my_agent.memory.evolver.maintenance.contracts import MaintenanceLookupHit
from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.experience.retrieval.text import tokenize_experience_text
from my_agent.memory.types import MemoryScope, normalize_content


def lookup_experiences(
    entries: Sequence[ExperienceMemory],
    query: str,
    *,
    project_key: str,
    tiers: Collection[str] | None = None,
    limit: int = 20,
) -> list[MaintenanceLookupHit]:
    if not project_key:
        raise ValueError("project_key must not be empty")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    requested_tiers: set[str] | None = None
    if tiers is not None:
        requested_tiers = set()
        for value in tiers:
            try:
                tier = ExperienceTier(str(value))
            except ValueError as exc:
                raise ValueError(f"invalid experience tier: {value!r}") from exc
            requested_tiers.add(tier.value)

    normalized_query = normalize_content(query)
    query_terms = set(tokenize_experience_text(query))
    if not normalized_query and not query_terms:
        return []

    hits: list[MaintenanceLookupHit] = []
    for entry in entries:
        if not _entry_visible_to_project(entry, project_key):
            continue
        if requested_tiers is not None and entry.tier.value not in requested_tiers:
            continue
        content_terms = set(tokenize_experience_text(entry.content))
        matched = tuple(sorted(query_terms.intersection(content_terms)))
        coverage = len(matched) / len(query_terms) if query_terms else 0.0
        substring_bonus = (
            0.25
            if normalized_query and normalized_query in normalize_content(entry.content)
            else 0.0
        )
        score = min(1.0, coverage + substring_bonus)
        if score <= 0:
            continue
        hits.append(MaintenanceLookupHit(
            memory=entry,
            tier=entry.tier.value,
            score=round(score, 6),
            matched_terms=matched,
        ))
    hits.sort(key=lambda item: (-item.score, item.tier, item.memory.id))
    return hits[:limit] if limit else []


def redundancy_score(left: ExperienceMemory, right: ExperienceMemory) -> float:
    if left.tier != right.tier:
        return 0.0
    if left.scope != right.scope:
        return 0.0
    if left.scope != MemoryScope.GLOBAL and left.project_key != right.project_key:
        return 0.0

    left_normalized = normalize_content(left.content)
    right_normalized = normalize_content(right.content)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    token_score = _jaccard(
        set(tokenize_experience_text(left.content)),
        set(tokenize_experience_text(right.content)),
    )
    trigram_score = _jaccard(
        _char_trigrams(left_normalized),
        _char_trigrams(right_normalized),
    )
    return round(min(1.0, max(0.0, token_score, trigram_score)), 6)


def _entry_visible_to_project(entry: ExperienceMemory, project_key: str) -> bool:
    return entry.scope == MemoryScope.GLOBAL or (
        entry.scope == MemoryScope.PROJECT and entry.project_key == project_key
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _char_trigrams(value: str) -> set[str]:
    compact = "".join(value.split())
    if not compact:
        return set()
    if len(compact) < 3:
        return {compact}
    return {
        compact[index:index + 3]
        for index in range(len(compact) - 2)
    }


__all__ = ["lookup_experiences", "redundancy_score"]
