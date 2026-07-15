from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from my_agent.memory.evolver.types import ExperienceMemory, ExperienceTier
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryContext, RetrievalHit


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
        selected = self._select_with_budget(candidates, max_tokens=max_tokens, item_limit=item_limit)
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

    def _select_with_budget(
        self,
        candidates: Sequence[ExperienceCandidate],
        *,
        max_tokens: int,
        item_limit: int,
    ) -> list[SelectedExperience]:
        if item_limit <= 0 or max_tokens <= 0:
            return []
        header_tokens = estimate_tokens(_RENDER_HEADER)
        if header_tokens > max_tokens:
            return []
        selected: list[SelectedExperience] = []
        tier_counts: dict[str, int] = {}
        used_tokens = header_tokens
        for candidate in candidates:
            if len(selected) >= item_limit:
                continue
            tier_key = candidate.tier.value
            cap = self.tier_caps.get(tier_key, item_limit)
            if cap <= 0 or tier_counts.get(tier_key, 0) >= cap:
                continue
            item = SelectedExperience(
                candidate=candidate,
                rank=len(selected) + 1,
                reason=f"selected by {candidate.reason}",
            )
            entry_tokens = estimate_tokens(_render_selected_entry(item))
            separator_tokens = 1 if selected else 0
            if used_tokens + separator_tokens + entry_tokens > max_tokens:
                continue
            selected.append(item)
            tier_counts[tier_key] = tier_counts.get(tier_key, 0) + 1
            used_tokens += separator_tokens + entry_tokens
        return selected


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


def render_selected_experiences(
    selected: Sequence[SelectedExperience],
    *,
    max_tokens: int,
) -> MemoryContext[ExperienceMemory]:
    if not selected or max_tokens <= 0:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
    header_tokens = estimate_tokens(_RENDER_HEADER)
    if header_tokens > max_tokens:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
    lines = [_RENDER_HEADER]
    kept: list[SelectedExperience] = []
    used_tokens = header_tokens
    for item in selected:
        entry_text = _render_selected_entry(item)
        entry_tokens = estimate_tokens(entry_text)
        separator_tokens = 1 if kept else 0
        if used_tokens + separator_tokens + entry_tokens > max_tokens:
            continue
        lines.append(entry_text)
        kept.append(item)
        used_tokens += separator_tokens + entry_tokens
    if not kept:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
    injected_text = "\n".join(lines)
    return MemoryContext(
        injected_text=injected_text,
        hits=[item.candidate.hit for item in kept],
        estimated_tokens=estimate_tokens(injected_text),
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
        "source_task": _safe_task_ref(entry.source_task),
    }


def selection_tier_counts(candidates: Iterable[ExperienceCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        tier_key = candidate.tier.value
        counts[tier_key] = counts.get(tier_key, 0) + 1
    return counts


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_RENDER_HEADER = "Relevant selected experience:"
_POLICY_NAME = "rule_tier_weighted_v1"
_TASK_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,80}")
_KNOWN_TASK_REF_RE = re.compile(
    r"(?:"
    r"(?:task|run|stream|case|eval|bench|benchmark|humaneval|mbpp|problem|sample)[A-Za-z0-9_.:-]{0,72}"
    r"|[A-Za-z0-9_.:-]{0,48}(?:task|case|eval|bench|humaneval|mbpp)[A-Za-z0-9_.:-]{0,48}"
    r"|\d{1,12}"
    r")"
)
_SECRET_PREFIX_RE = re.compile(
    r"(?i)(?:ghp_|github_pat_|glpat-|xox[baprs]-|AKIA|ASIA|AIza|ya29\.|eyJ[A-Za-z0-9_-]{8,})"
)
_SECRET_MARKERS = (
    "secret",
    "token",
    "password",
    "api_key",
    "apikey",
    "credential",
    "bearer",
    "private_key",
    "access_key",
)


def _render_selected_entry(item: SelectedExperience) -> str:
    candidate = item.candidate
    entry = candidate.hit.entry
    header = (
        f"- [memory_id={entry.id} tier={candidate.tier.value} "
        f"score={candidate.selection_score:.2f} source_task={_safe_task_ref(entry.source_task)}]"
    )
    return f"{header}\n{_indent_content(entry.content.strip())}"


def _indent_content(content: str) -> str:
    return "\n".join(f"  {line}" for line in (content.splitlines() or [""]))


def _safe_task_ref(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if _looks_secret_like(text):
        return "[redacted]"
    if _TASK_ID_RE.fullmatch(text) and _KNOWN_TASK_REF_RE.fullmatch(text):
        return text
    return f"task_ref_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def _looks_secret_like(text: str) -> bool:
    lower = text.casefold()
    if _SECRET_PREFIX_RE.search(text):
        return True
    if any(marker in lower for marker in _SECRET_MARKERS):
        return True
    if lower.startswith("sk-") or " sk-" in lower or "=sk-" in lower or ":sk-" in lower:
        return True
    if "key=" in lower or "key:" in lower or "key-" in lower or "key_" in lower:
        return True
    if "?" in text or "&" in text or "=" in text or "://" in lower:
        return True
    return False


def _effective_confidence(entry: ExperienceMemory) -> float:
    if entry.candidate_count > 0:
        return entry.attribution_confidence
    return entry.writer_confidence


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
    "SelectedExperience",
    "SelectionResult",
    "clamp",
    "render_selected_experiences",
    "selection_candidate_summary",
    "selection_score",
    "selection_tier_counts",
]
