"""Tier-local lexical retrieval over immutable Experience snapshots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.experience.repository_index import (
    ExperienceRepositoryIndexSnapshot,
    visible_ids_for_tier,
)
from my_agent.memory.experience.retrieval.contracts import (
    PerTierHits,
    flatten_per_tier_hits,
)
from my_agent.memory.experience.retrieval.text import (
    experience_index_terms,
    experience_searchable_text,
    tokenize_experience_text,
)
from my_agent.memory.types import RetrievalHit, normalize_content

_TIME_DECAY_FLOOR = 0.35
_TIME_DECAY_HORIZON_DAYS = 30.0
_LONG_TERM_SOURCE_WEIGHT = 1.2


@dataclass(frozen=True)
class ExperienceRetrievalMetrics:
    repository_revision: str = ""
    visible_count: int = 0
    indexed_count: int = 0
    posting_candidate_count: int = 0
    scored_count: int = 0
    matched_count: int = 0
    returned_count: int = 0
    per_tier: dict[str, dict[str, int]] = field(default_factory=dict)
    retrieval_fallback: str = ""
    retrieval_backend: str = "lexical"

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "repository_revision": self.repository_revision,
            "visible_count": self.visible_count,
            "indexed_count": self.indexed_count,
            "posting_candidate_count": self.posting_candidate_count,
            "scored_count": self.scored_count,
            "matched_count": self.matched_count,
            "returned_count": self.returned_count,
            "retrieval_per_tier": {
                tier: dict(counts) for tier, counts in self.per_tier.items()
            },
            "retrieval_fallback": self.retrieval_fallback,
            "retrieval_backend": self.retrieval_backend,
        }


@dataclass(frozen=True)
class LexicalIndexSnapshot:
    repository_revision: str
    postings_by_tier: Mapping[ExperienceTier, Mapping[str, frozenset[str]]]
    searchable_text_by_id: Mapping[str, str]


def build_lexical_index(
    repository: ExperienceRepositoryIndexSnapshot,
) -> LexicalIndexSnapshot:
    postings: dict[ExperienceTier, dict[str, set[str]]] = {
        tier: defaultdict(set) for tier in ExperienceTier
    }
    searchable_text: dict[str, str] = {}
    for memory in repository.by_id.values():
        text = experience_searchable_text(memory)
        searchable_text[memory.id] = text
        if memory.invalidated:
            continue
        for term in experience_index_terms(memory):
            postings[memory.tier][term].add(memory.id)

    frozen_postings: dict[ExperienceTier, Mapping[str, frozenset[str]]] = {}
    for tier in ExperienceTier:
        frozen_postings[tier] = MappingProxyType({
            term: frozenset(ids)
            for term, ids in sorted(postings[tier].items())
        })
    return LexicalIndexSnapshot(
        repository_revision=repository.revision,
        postings_by_tier=MappingProxyType(frozen_postings),
        searchable_text_by_id=MappingProxyType(dict(searchable_text)),
    )


class LexicalExperienceRetriever:
    """Tier-local lexical retrieval with a revision-coupled lexical index."""

    def __init__(self, *, now: datetime | None = None) -> None:
        self._now = now
        self._index: LexicalIndexSnapshot | None = None
        self.last_metrics = ExperienceRetrievalMetrics()

    @property
    def last_index(self) -> LexicalIndexSnapshot | None:
        return self._index

    def fork(self) -> "LexicalExperienceRetriever":
        return LexicalExperienceRetriever(now=self._now)

    def retrieve_per_tier(
        self,
        query: str,
        *,
        store: ExperienceStore,
        project_key: str,
        top_k_per_tier: int,
    ) -> PerTierHits:
        top_k = max(0, int(top_k_per_tier))
        normalized_query = normalize_content(query)
        query_terms = tokenize_experience_text(query)
        now = self._now or datetime.now(timezone.utc)

        try:
            repository = store.index_snapshot()
            index = self._lexical_index(repository)
        except Exception:  # noqa: BLE001 - explicit project/tier fallback is required
            return self._retrieve_with_bucket_fallback(
                normalized_query=normalized_query,
                query_terms=query_terms,
                store=store,
                project_key=project_key,
                top_k=top_k,
                now=now,
            )

        results: PerTierHits = {}
        per_tier: dict[str, dict[str, int]] = {}
        totals = {
            "visible": 0,
            "indexed": 0,
            "posting": 0,
            "scored": 0,
            "matched": 0,
            "returned": 0,
        }
        used_bucket_fallback = False
        for tier in ExperienceTier:
            visible_ids = visible_ids_for_tier(
                repository,
                project_key=project_key,
                tier=tier,
            )
            searchable_ids = tuple(
                memory_id
                for memory_id in visible_ids
                if not repository.by_id[memory_id].invalidated
            )
            candidate_ids, tier_fallback = _posting_candidates(
                index,
                tier=tier,
                visible_ids=searchable_ids,
                normalized_query=normalized_query,
                query_terms=query_terms,
            )
            used_bucket_fallback = used_bucket_fallback or tier_fallback
            hits = _score_candidates(
                candidate_ids,
                repository=repository,
                index=index,
                normalized_query=normalized_query,
                query_terms=query_terms,
                now=now,
            )
            returned = tuple(hits[:top_k]) if top_k > 0 else ()
            results[tier] = returned
            counts = {
                "visible_count": len(visible_ids),
                "indexed_count": len(searchable_ids),
                "posting_candidate_count": 0 if tier_fallback else len(candidate_ids),
                "scored_count": len(candidate_ids),
                "matched_count": len(hits),
                "returned_count": len(returned),
            }
            per_tier[tier.value] = counts
            totals["visible"] += counts["visible_count"]
            totals["indexed"] += counts["indexed_count"]
            totals["posting"] += counts["posting_candidate_count"]
            totals["scored"] += counts["scored_count"]
            totals["matched"] += counts["matched_count"]
            totals["returned"] += counts["returned_count"]

        self.last_metrics = ExperienceRetrievalMetrics(
            repository_revision=repository.revision,
            visible_count=totals["visible"],
            indexed_count=totals["indexed"],
            posting_candidate_count=totals["posting"],
            scored_count=totals["scored"],
            matched_count=totals["matched"],
            returned_count=totals["returned"],
            per_tier=per_tier,
            retrieval_fallback="tier_bucket_scan" if used_bucket_fallback else "",
        )
        return results

    def retrieve_candidates(
        self,
        query: str,
        *,
        store: ExperienceStore,
        project_key: str,
        top_k_per_tier: int,
    ) -> tuple[RetrievalHit[ExperienceMemory], ...]:
        return flatten_per_tier_hits(
            self.retrieve_per_tier(
                query,
                store=store,
                project_key=project_key,
                top_k_per_tier=top_k_per_tier,
            ),
            sort_key=_hit_sort_key,
        )

    def _lexical_index(
        self,
        repository: ExperienceRepositoryIndexSnapshot,
    ) -> LexicalIndexSnapshot:
        if self._index is None or self._index.repository_revision != repository.revision:
            self._index = build_lexical_index(repository)
        return self._index

    def _retrieve_with_bucket_fallback(
        self,
        *,
        normalized_query: str,
        query_terms: tuple[str, ...],
        store: ExperienceStore,
        project_key: str,
        top_k: int,
        now: datetime,
    ) -> PerTierHits:
        results: PerTierHits = {}
        per_tier: dict[str, dict[str, int]] = {}
        totals = {"visible": 0, "scored": 0, "matched": 0, "returned": 0}
        for tier in ExperienceTier:
            visible_bucket = tuple(
                store.all(project_key=project_key, tiers=frozenset({tier}))
            )
            bucket = tuple(memory for memory in visible_bucket if not memory.invalidated)
            hits = _score_memories(
                bucket,
                normalized_query=normalized_query,
                query_terms=query_terms,
                now=now,
            )
            returned = tuple(hits[:top_k]) if top_k > 0 else ()
            results[tier] = returned
            counts = {
                "visible_count": len(visible_bucket),
                "indexed_count": 0,
                "posting_candidate_count": 0,
                "scored_count": len(bucket),
                "matched_count": len(hits),
                "returned_count": len(returned),
            }
            per_tier[tier.value] = counts
            totals["visible"] += len(visible_bucket)
            totals["scored"] += len(bucket)
            totals["matched"] += len(hits)
            totals["returned"] += len(returned)
        try:
            revision = store.revision()
        except Exception:  # noqa: BLE001 - metrics must not hide fallback results
            revision = ""
        self.last_metrics = ExperienceRetrievalMetrics(
            repository_revision=revision,
            visible_count=totals["visible"],
            indexed_count=0,
            posting_candidate_count=0,
            scored_count=totals["scored"],
            matched_count=totals["matched"],
            returned_count=totals["returned"],
            per_tier=per_tier,
            retrieval_fallback="tier_bucket_scan",
        )
        return results


ExperienceRetriever = LexicalExperienceRetriever


def _posting_candidates(
    index: LexicalIndexSnapshot,
    *,
    tier: ExperienceTier,
    visible_ids: tuple[str, ...],
    normalized_query: str,
    query_terms: tuple[str, ...],
) -> tuple[tuple[str, ...], bool]:
    if not normalized_query and not query_terms:
        return (), False
    if not query_terms:
        return visible_ids, True

    postings = index.postings_by_tier.get(tier, {})
    visible = set(visible_ids)
    candidates: set[str] = set()
    for term in dict.fromkeys(query_terms):
        if len(term) < 2:
            return visible_ids, True
        width = 3 if len(term) >= 3 else 2
        grams = tuple(term[index:index + width] for index in range(len(term) - width + 1))
        if not grams:
            return visible_ids, True
        term_candidates: set[str] | None = None
        for gram in grams:
            ids = set(postings.get(gram, ()))
            term_candidates = ids if term_candidates is None else term_candidates & ids
            if not term_candidates:
                break
        if term_candidates:
            candidates.update(term_candidates & visible)
    return tuple(sorted(candidates)), False


def _score_candidates(
    candidate_ids: tuple[str, ...],
    *,
    repository: ExperienceRepositoryIndexSnapshot,
    index: LexicalIndexSnapshot,
    normalized_query: str,
    query_terms: tuple[str, ...],
    now: datetime,
) -> list[RetrievalHit[ExperienceMemory]]:
    hits: list[RetrievalHit[ExperienceMemory]] = []
    for memory_id in candidate_ids:
        memory = repository.by_id[memory_id]
        hit = _score_memory(
            memory,
            searchable_text=index.searchable_text_by_id[memory_id],
            normalized_query=normalized_query,
            query_terms=query_terms,
            now=now,
        )
        if hit is not None:
            hits.append(hit)
    hits.sort(key=_hit_sort_key)
    return hits


def _score_memories(
    memories: tuple[ExperienceMemory, ...],
    *,
    normalized_query: str,
    query_terms: tuple[str, ...],
    now: datetime,
) -> list[RetrievalHit[ExperienceMemory]]:
    hits: list[RetrievalHit[ExperienceMemory]] = []
    for memory in memories:
        hit = _score_memory(
            memory,
            searchable_text=experience_searchable_text(memory),
            normalized_query=normalized_query,
            query_terms=query_terms,
            now=now,
        )
        if hit is not None:
            hits.append(hit)
    hits.sort(key=_hit_sort_key)
    return hits


def _score_memory(
    memory: ExperienceMemory,
    *,
    searchable_text: str,
    normalized_query: str,
    query_terms: tuple[str, ...],
    now: datetime,
) -> RetrievalHit[ExperienceMemory] | None:
    if not normalized_query and not query_terms:
        return None
    matched_terms = tuple(term for term in query_terms if term and term in searchable_text)
    if normalized_query and normalized_query in searchable_text:
        base = 1.0
    elif not matched_terms:
        return None
    else:
        base = len(matched_terms) / max(1, len(query_terms))
    age_days = max(0.0, (now - memory.created_at).total_seconds() / 86400.0)
    time_decay = max(_TIME_DECAY_FLOOR, 1.0 - age_days / _TIME_DECAY_HORIZON_DAYS)
    return RetrievalHit(
        entry=memory,
        score=base * time_decay * _LONG_TERM_SOURCE_WEIGHT,
        matched_terms=matched_terms,
        source_weight=_LONG_TERM_SOURCE_WEIGHT,
        time_decay=time_decay,
    )


def _hit_sort_key(hit: RetrievalHit[ExperienceMemory]) -> tuple[float, int, str]:
    return (-float(hit.score), hit.entry.token_count, hit.entry.id)


__all__ = [
    "ExperienceRetrievalMetrics",
    "ExperienceRetriever",
    "LexicalExperienceRetriever",
    "LexicalIndexSnapshot",
    "build_lexical_index",
    "experience_index_terms",
    "experience_searchable_text",
    "tokenize_experience_text",
]
