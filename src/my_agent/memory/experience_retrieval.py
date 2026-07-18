from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from my_agent.memory.experience.models import (
    ExperienceMemory,
    ExperienceTier,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.types import normalize_content
from my_agent.memory.types import RetrievalHit

if TYPE_CHECKING:
    from my_agent.memory.experience.repository import ExperienceStore, ExperienceStoreIndexSnapshot


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[一-鿿]+")
_FIELD_CHAR_LIMIT = 2_000
_STEP_CHAR_LIMIT = 600
_TOTAL_CHAR_LIMIT = 12_000
_SCHEMA_MAX_DEPTH = 3
_SCHEMA_MAX_ITEMS = 64
_SCHEMA_SINGLE_CHILD_KEYS = (
    "additionalProperties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
)
_SCHEMA_SEQUENCE_CHILD_KEYS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_MAPPING_CHILD_KEYS = (
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
    "patternProperties",
)
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


class ExperienceRetriever:
    """Tier-local lexical retrieval over an immutable experience snapshot."""

    def __init__(self, *, now: datetime | None = None) -> None:
        self._now = now
        self.last_metrics = ExperienceRetrievalMetrics()

    def fork(self) -> "ExperienceRetriever":
        """Return a query-state-isolated retriever for a forked task manager."""
        return ExperienceRetriever(now=self._now)

    def retrieve_per_tier(
        self,
        query: str,
        *,
        store: "ExperienceStore",
        project_key: str,
        top_k_per_tier: int,
    ) -> dict[ExperienceTier, tuple[RetrievalHit[ExperienceMemory], ...]]:
        top_k = max(0, int(top_k_per_tier))
        normalized_query = normalize_content(query)
        query_terms = tokenize_experience_text(query)
        now = self._now or datetime.now(timezone.utc)

        try:
            index = store.index_snapshot()
        except Exception:  # noqa: BLE001 - explicit project/tier fallback is required
            return self._retrieve_with_bucket_fallback(
                normalized_query=normalized_query,
                query_terms=query_terms,
                store=store,
                project_key=project_key,
                top_k=top_k,
                now=now,
            )

        results: dict[ExperienceTier, tuple[RetrievalHit[ExperienceMemory], ...]] = {}
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
            visible_ids = _visible_ids(index, project_key=project_key, tier=tier)
            searchable_ids = tuple(
                memory_id
                for memory_id in visible_ids
                if not index.by_id[memory_id].invalidated
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
            repository_revision=index.revision,
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
        store: "ExperienceStore",
        project_key: str,
        top_k_per_tier: int,
    ) -> tuple[RetrievalHit[ExperienceMemory], ...]:
        per_tier = self.retrieve_per_tier(
            query,
            store=store,
            project_key=project_key,
            top_k_per_tier=top_k_per_tier,
        )
        merged = [hit for tier in ExperienceTier for hit in per_tier[tier]]
        merged.sort(key=_hit_sort_key)
        return tuple(merged)

    def _retrieve_with_bucket_fallback(
        self,
        *,
        normalized_query: str,
        query_terms: tuple[str, ...],
        store: "ExperienceStore",
        project_key: str,
        top_k: int,
        now: datetime,
    ) -> dict[ExperienceTier, tuple[RetrievalHit[ExperienceMemory], ...]]:
        results: dict[ExperienceTier, tuple[RetrievalHit[ExperienceMemory], ...]] = {}
        per_tier: dict[str, dict[str, int]] = {}
        totals = {"visible": 0, "scored": 0, "matched": 0, "returned": 0}
        for tier in ExperienceTier:
            visible_bucket = tuple(
                store.all(project_key=project_key, tiers=frozenset({tier}))
            )
            bucket = tuple(
                memory
                for memory in visible_bucket
                if not memory.invalidated
            )
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


def experience_searchable_text(memory: ExperienceMemory) -> str:
    """Build deterministic, bounded semantic text from a typed experience."""
    if not isinstance(memory, ExperienceMemory):
        raise TypeError("memory must be an ExperienceMemory")

    pieces: list[str] = [memory.content]
    payload = memory.payload
    if isinstance(payload, TrajectoryPayload):
        pieces.extend((payload.task_description, *payload.key_learnings, *payload.tags))
        for step in payload.steps:
            if step.reward is None or step.reward <= 0:
                continue
            pieces.append(
                " ".join(
                    part
                    for part in (step.observation, step.action, step.result)
                    if part
                )[:_STEP_CHAR_LIMIT]
            )
    elif isinstance(payload, TipPayload):
        pieces.extend((payload.category, payload.severity, payload.trigger))
    elif isinstance(payload, SkillPayload):
        pieces.extend((payload.category, payload.technique, *payload.preconditions, *payload.steps))
    elif isinstance(payload, ToolPayload):
        pieces.extend((
            payload.name,
            payload.language,
            payload.code,
            payload.command,
            payload.input_description,
            payload.output_description,
            *_schema_search_fragments(payload.args_schema),
            payload.repo_context,
        ))
    else:  # pragma: no cover - ExperienceMemory closes the payload union
        raise TypeError(f"unsupported experience payload: {type(payload).__name__}")

    normalized: list[str] = []
    seen: set[str] = set()
    used_chars = 0
    for piece in pieces:
        text = normalize_content(str(piece or ""))[:_FIELD_CHAR_LIMIT]
        if not text or text in seen:
            continue
        remaining = _TOTAL_CHAR_LIMIT - used_chars
        if remaining <= 0:
            break
        text = text[:remaining]
        if not text:
            break
        seen.add(text)
        normalized.append(text)
        used_chars += len(text) + 1
    return " ".join(normalized)


def experience_index_terms(memory: ExperienceMemory) -> frozenset[str]:
    """Return token and n-gram postings used for high-recall candidate lookup."""
    terms: set[str] = set()
    for token in tokenize_experience_text(experience_searchable_text(memory)):
        terms.add(token)
        if len(token) >= 2:
            terms.update(token[index:index + 2] for index in range(len(token) - 1))
        if len(token) >= 3:
            terms.update(token[index:index + 3] for index in range(len(token) - 2))
    return frozenset(terms)


def tokenize_experience_text(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(_TOKEN_RE.findall(text.casefold()))


def _visible_ids(
    index: "ExperienceStoreIndexSnapshot",
    *,
    project_key: str,
    tier: ExperienceTier,
) -> tuple[str, ...]:
    visible = set(index.global_ids_by_tier.get(tier, ()))
    if project_key:
        visible.update(index.project_ids_by_tier.get((project_key, tier), ()))
    return tuple(sorted(visible))


def _posting_candidates(
    index: "ExperienceStoreIndexSnapshot",
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

    postings = index.lexical_postings_by_tier.get(tier, {})
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
    index: "ExperienceStoreIndexSnapshot",
    normalized_query: str,
    query_terms: tuple[str, ...],
    now: datetime,
) -> list[RetrievalHit[ExperienceMemory]]:
    hits: list[RetrievalHit[ExperienceMemory]] = []
    for memory_id in candidate_ids:
        memory = index.by_id[memory_id]
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


def _schema_search_fragments(schema: dict[str, Any]) -> tuple[str, ...]:
    fragments: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str) or len(fragments) >= _SCHEMA_MAX_ITEMS:
            return
        text = value[:_FIELD_CHAR_LIMIT]
        if not text or text in seen:
            return
        seen.add(text)
        fragments.append(text)

    def visit_schema(value: Any, *, depth: int) -> None:
        if (
            depth > _SCHEMA_MAX_DEPTH
            or len(fragments) >= _SCHEMA_MAX_ITEMS
            or not isinstance(value, dict)
        ):
            return

        add(value.get("description"))

        properties = value.get("properties")
        if isinstance(properties, dict):
            for field_name in sorted(properties, key=str):
                if len(fragments) >= _SCHEMA_MAX_ITEMS:
                    return
                add(str(field_name))
                visit_schema(properties[field_name], depth=depth + 1)

        for key in _SCHEMA_SINGLE_CHILD_KEYS:
            child = value.get(key)
            if isinstance(child, dict):
                visit_schema(child, depth=depth + 1)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    visit_schema(item, depth=depth + 1)

        for key in _SCHEMA_SEQUENCE_CHILD_KEYS:
            children = value.get(key)
            if isinstance(children, (list, tuple)):
                for child in children:
                    visit_schema(child, depth=depth + 1)

        for key in _SCHEMA_MAPPING_CHILD_KEYS:
            children = value.get(key)
            if not isinstance(children, dict):
                continue
            for child_name in sorted(children, key=str):
                visit_schema(children[child_name], depth=depth + 1)

    visit_schema(schema, depth=0)
    # Ensure arbitrary objects never leak through a future relaxed ToolPayload.
    json.dumps(fragments, ensure_ascii=False, allow_nan=False)
    return tuple(fragments)


__all__ = [
    "ExperienceRetrievalMetrics",
    "ExperienceRetriever",
    "experience_index_terms",
    "experience_searchable_text",
    "tokenize_experience_text",
]
