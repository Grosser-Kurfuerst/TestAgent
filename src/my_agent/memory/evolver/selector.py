from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from my_agent.memory.evolver.types import ExperienceTier, experience_tier, is_experience_entry
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryContext, MemoryEntry, RetrievalHit


@dataclass(frozen=True)
class ExperienceCandidate:
    id: str
    hit: RetrievalHit
    tier: ExperienceTier | None
    retrieval_score: float
    selection_score: float
    matched_terms: tuple[str, ...] = ()
    token_count: int = 0
    value: float | None = None
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
    context: MemoryContext
    policy: str
    estimated_tokens: int
    omitted_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperienceSelector:
    """Pure rule-based selector for OPD-Evolver experience memory."""

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
        hits: list[RetrievalHit],
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

    def _build_candidates(self, hits: Sequence[RetrievalHit]) -> list[ExperienceCandidate]:
        by_fingerprint: dict[str, ExperienceCandidate] = {}
        for hit in hits:
            entry = hit.entry
            if not is_experience_entry(entry):
                continue
            if hit.score < self.min_score:
                continue
            if not entry.content.strip():
                continue

            tier = candidate_tier(entry)
            if tier is None:
                continue
            score = selection_score(hit, tier, entry.metadata, tier_weights=self.tier_weights)
            candidate = ExperienceCandidate(
                id=entry.id,
                hit=hit,
                tier=tier,
                retrieval_score=hit.score,
                selection_score=score,
                matched_terms=tuple(hit.matched_terms),
                token_count=estimate_tokens(entry.content),
                value=candidate_value(entry),
                reason=_score_reason(hit, tier, entry.metadata, self.tier_weights),
            )
            fingerprint = entry.fingerprint or entry.id
            previous = by_fingerprint.get(fingerprint)
            if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
                by_fingerprint[fingerprint] = candidate

        candidates = list(by_fingerprint.values())
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
            tier_key = candidate.tier.value if candidate.tier is not None else ""
            cap = self.tier_caps.get(tier_key, item_limit)
            if cap <= 0 or tier_counts.get(tier_key, 0) >= cap:
                continue

            selected_item = SelectedExperience(
                candidate=candidate,
                rank=len(selected) + 1,
                reason=f"selected by {candidate.reason}",
            )
            entry_text = _render_selected_entry(selected_item)
            separator_tokens = 1 if selected else 0
            entry_tokens = estimate_tokens(entry_text)
            if used_tokens + separator_tokens + entry_tokens > max_tokens:
                continue

            selected.append(selected_item)
            tier_counts[tier_key] = tier_counts.get(tier_key, 0) + 1
            used_tokens += separator_tokens + entry_tokens
        return selected


def candidate_tier(entry: MemoryEntry) -> ExperienceTier | None:
    return experience_tier(entry)


def candidate_value(entry: MemoryEntry) -> float | None:
    return _metadata_float(entry.metadata, "evolver_value", default=None)


def selection_score(
    hit: RetrievalHit,
    tier: ExperienceTier | None,
    metadata: Mapping[str, Any],
    *,
    tier_weights: Mapping[str, float] | None = None,
) -> float:
    if tier is None:
        return 0.0
    tier_weight = _tier_weight(tier_weights or {}, tier)
    value = _metadata_float(metadata, "evolver_value", default=0.0) or 0.0
    confidence = _metadata_float(metadata, "confidence", default=1.0) or 1.0
    value_weight = clamp(1.0 + value, 0.50, 1.50)
    confidence_weight = clamp(confidence, 0.50, 1.20)
    return float(hit.score) * tier_weight * value_weight * confidence_weight


def render_selected_experiences(
    selected: Sequence[SelectedExperience],
    *,
    max_tokens: int,
) -> MemoryContext:
    if not selected or max_tokens <= 0:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

    header_tokens = estimate_tokens(_RENDER_HEADER)
    if header_tokens > max_tokens:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

    lines: list[str] = [_RENDER_HEADER]
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
        "tier": candidate.tier.value if candidate.tier is not None else "",
        "score": candidate.selection_score,
        "tokens": candidate.token_count,
        "retrieval_score": candidate.retrieval_score,
        "selection_score": candidate.selection_score,
        "token_count": candidate.token_count,
        "value": candidate.value,
        "source_task": _safe_task_ref(entry.metadata.get("source_task") or entry.metadata.get("task_id") or ""),
    }


def selection_tier_counts(candidates: Sequence[ExperienceCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate.tier is None:
            continue
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
    r"(?i)(?:"
    r"ghp_|github_pat_|glpat-|xox[baprs]-|AKIA|ASIA|AIza|ya29\\.|"
    r"eyJ[A-Za-z0-9_-]{8,}"
    r")"
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
    tier = candidate.tier.value if candidate.tier is not None else "unknown"
    source_task = _safe_task_ref(entry.metadata.get("source_task") or entry.metadata.get("task_id") or "")
    header = (
        f"- [memory_id={entry.id} tier={tier} "
        f"score={candidate.selection_score:.2f} source_task={source_task}]"
    )
    content = _indent_content(entry.content.strip())
    return f"{header}\n{content}"


def _indent_content(content: str) -> str:
    lines = content.splitlines() or [""]
    return "\n".join(f"  {line}" for line in lines)


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
    if "?" in text or "&" in text or "=" in text:
        return True
    if "://" in lower:
        return True
    return False


def _score_reason(
    hit: RetrievalHit,
    tier: ExperienceTier,
    metadata: Mapping[str, Any],
    tier_weights: Mapping[str, float],
) -> str:
    return (
        f"retrieval={float(hit.score):.3f} "
        f"tier_weight={_tier_weight(tier_weights, tier):.2f} "
        f"value={_metadata_float(metadata, 'evolver_value', default=0.0) or 0.0:.2f} "
        f"confidence={_metadata_float(metadata, 'confidence', default=1.0) or 1.0:.2f}"
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


def _metadata_float(metadata: Mapping[str, Any], key: str, *, default: float | None) -> float | None:
    raw = metadata.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


__all__ = [
    "ExperienceCandidate",
    "ExperienceSelector",
    "SelectedExperience",
    "SelectionResult",
    "candidate_tier",
    "candidate_value",
    "clamp",
    "render_selected_experiences",
    "selection_candidate_summary",
    "selection_score",
    "selection_tier_counts",
]
