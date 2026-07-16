"""Outcome-calibrated attribution for evolver experience memories.

Given a usage log (candidate / selected / outcome per task) this module scores
each experience memory by how much selecting it moves the average reward away
from the pool baseline, within a task-type group. See
``docs/opd-evolver-reproduction/phase5-outcome-calibrated-attribution-implementation-plan.md``
-> "Attribution Formula".

The scorer is pure w.r.t. time and ordering: for the same inputs it produces
byte-stable JSONL, with output sorted by ``memory_id``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from my_agent.memory.evolver.types import ExperienceMemory, ExperienceTier
from my_agent.memory.evolver.usage_log import UsageLogEntry
from my_agent.memory.types import MemoryScope
from my_agent.text_safety import sanitize_json_value

if TYPE_CHECKING:
    from my_agent.memory.experience_store import ExperienceStore


DEFAULT_TIER_WEIGHTS: dict[str, float] = {
    "trajectory": 0.90,
    "tip": 0.80,
    "skill": 1.00,
    "tool": 1.20,
}


@dataclass(frozen=True)
class AttributionConfig:
    """Knobs for the attribution formula. Defaults match the plan."""

    tier_weights: Mapping[str, float] = field(default_factory=lambda: DEFAULT_TIER_WEIGHTS)
    min_candidate_count: int = 2
    min_selected_count: int = 1
    min_not_selected_count: int = 1
    min_selected_count_for_full_confidence: int = 8
    value_clip: float = 0.5


@dataclass(frozen=True)
class MemoryAttributionRecord:
    """One memory's attribution result for a single project key."""

    memory_id: str
    tier: str
    memory_project_key: str = ""
    candidate_count: int = 0
    selected_count: int = 0
    not_selected_count: int = 0
    success_when_selected: float | None = None
    success_when_candidate_not_selected: float | None = None
    reward_when_selected: float | None = None
    reward_when_candidate_not_selected: float | None = None
    value: float = 0.0
    confidence: float = 0.0
    groups: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    stream_ids: tuple[str, ...] = ()
    selected_task_ids: tuple[str, ...] = ()
    not_selected_task_ids: tuple[str, ...] = ()
    last_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "memory_id": str(self.memory_id),
            "tier": str(self.tier),
            "memory_project_key": str(self.memory_project_key or ""),
            "candidate_count": int(self.candidate_count),
            "selected_count": int(self.selected_count),
            "not_selected_count": int(self.not_selected_count),
            "success_when_selected": _maybe_round(self.success_when_selected),
            "success_when_candidate_not_selected": _maybe_round(self.success_when_candidate_not_selected),
            "reward_when_selected": _maybe_round(self.reward_when_selected),
            "reward_when_candidate_not_selected": _maybe_round(self.reward_when_candidate_not_selected),
            "value": round(float(self.value), 6),
            "confidence": round(float(self.confidence), 6),
            "groups": list(self.groups),
            "task_types": list(self.task_types),
            "stream_ids": list(self.stream_ids),
            "selected_task_ids": list(self.selected_task_ids),
            "not_selected_task_ids": list(self.not_selected_task_ids),
            "last_used": str(self.last_used or ""),
        }
        return sanitize_json_value(payload)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryAttributionRecord":
        return cls(
            memory_id=str(data.get("memory_id", "") or ""),
            tier=str(data.get("tier", "") or ""),
            memory_project_key=str(data.get("memory_project_key", "") or ""),
            candidate_count=int(data.get("candidate_count", 0) or 0),
            selected_count=int(data.get("selected_count", 0) or 0),
            not_selected_count=int(data.get("not_selected_count", 0) or 0),
            success_when_selected=_maybe_float(data.get("success_when_selected")),
            success_when_candidate_not_selected=_maybe_float(data.get("success_when_candidate_not_selected")),
            reward_when_selected=_maybe_float(data.get("reward_when_selected")),
            reward_when_candidate_not_selected=_maybe_float(data.get("reward_when_candidate_not_selected")),
            value=float(data.get("value", 0.0) or 0.0),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            groups=tuple(str(g) for g in (data.get("groups") or [])),
            task_types=tuple(str(t) for t in (data.get("task_types") or [])),
            stream_ids=tuple(str(s) for s in (data.get("stream_ids") or [])),
            selected_task_ids=tuple(str(t) for t in (data.get("selected_task_ids") or [])),
            not_selected_task_ids=tuple(str(t) for t in (data.get("not_selected_task_ids") or [])),
            last_used=str(data.get("last_used", "") or ""),
        )


@dataclass(frozen=True)
class AttributionWriteBackSummary:
    """Counters from writing attribution values back into long-term memory."""

    attempted: int = 0
    updated: int = 0
    skipped_missing: int = 0
    skipped_by_project_key: int = 0
    skipped_tier_mismatch: int = 0
    skipped_low_evidence: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "attempted": int(self.attempted),
            "updated": int(self.updated),
            "skipped_missing": int(self.skipped_missing),
            "skipped_by_project_key": int(self.skipped_by_project_key),
            "skipped_tier_mismatch": int(self.skipped_tier_mismatch),
            "skipped_low_evidence": int(self.skipped_low_evidence),
        }


def _maybe_round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _tier_weight(config: AttributionConfig, tier: str) -> float:
    return float(config.tier_weights.get(tier, 1.0))


def _group_logs_by_task_type(logs: Sequence[UsageLogEntry]) -> dict[str, list[UsageLogEntry]]:
    groups: dict[str, list[UsageLogEntry]] = {}
    for log in logs:
        groups.setdefault(log.task_type or "unknown", []).append(log)
    return groups


def score_memory(
    *,
    memory_id: str,
    tier: str,
    usage_logs: Sequence[UsageLogEntry],
    config: AttributionConfig | None = None,
    project_key: str = "",
) -> MemoryAttributionRecord:
    """Score one memory against a pool of complete usage logs.

    ``usage_logs`` should already be filtered to the target project scope by
    the caller (see :func:`score_all_memories`). The optional ``project_key``
    is only carried onto the record for traceability; it does not re-filter
    when non-empty, so the function stays a pure formula over the supplied pool.
    """
    cfg = config or AttributionConfig()
    tier_weight = _tier_weight(cfg, tier)

    # Pool: complete logs where this memory was a candidate. The caller is
    # responsible for cross-stream project filtering (see score_all_memories).
    pool_all: list[UsageLogEntry] = []
    for log in usage_logs:
        if not log.is_complete:
            continue
        if memory_id in log.all_candidate_ids():
            pool_all.append(log)

    selected_logs = [log for log in pool_all if memory_id in log.all_selected_ids()]
    not_selected_logs = [log for log in pool_all if memory_id not in log.all_selected_ids()]

    candidate_count = len(pool_all)
    selected_count = len(selected_logs)
    not_selected_count = len(not_selected_logs)

    success_when_selected = _rate([bool(log.success) for log in selected_logs]) if selected_logs else None
    success_when_not = _rate([bool(log.success) for log in not_selected_logs]) if not_selected_logs else None
    reward_when_selected = mean([float(log.env_reward) for log in selected_logs]) if selected_logs else None
    reward_when_not = mean([float(log.env_reward) for log in not_selected_logs]) if not_selected_logs else None

    confidence = _confidence(cfg, selected_count)

    contributions: list[float] = []
    used_task_types: list[str] = []
    if candidate_count >= cfg.min_candidate_count:
        # Compute per-task-type contributions only when both selected and
        # not-selected evidence exist in that task_type and meet min counts.
        by_task_type = _group_logs_by_task_type(pool_all)
        for task_type, logs in sorted(by_task_type.items()):
            sel = [log for log in logs if memory_id in log.all_selected_ids()]
            nsel = [log for log in logs if memory_id not in log.all_selected_ids()]
            if len(sel) < cfg.min_selected_count or len(nsel) < cfg.min_not_selected_count:
                continue
            if len(logs) == 0:
                continue
            pool_reward = mean([float(log.env_reward) for log in logs])
            sel_reward = mean([float(log.env_reward) for log in sel])
            delta = sel_reward - pool_reward
            weight = len(sel) / len(logs)
            contributions.append(delta * weight)
            used_task_types.append(task_type)

    raw_value = mean(contributions) if contributions else 0.0
    value = tier_weight * raw_value * confidence
    value = _clamp(value, -cfg.value_clip, cfg.value_clip)

    task_types = tuple(sorted({log.task_type or "unknown" for log in pool_all}))
    stream_ids = tuple(sorted({log.stream_id for log in pool_all if log.stream_id}))
    groups = stream_ids  # group identity aligns with stream in AgentCli
    selected_task_ids = tuple(sorted({log.task_id for log in selected_logs if log.task_id}))
    not_selected_task_ids = tuple(sorted({log.task_id for log in not_selected_logs if log.task_id}))
    last_used = max(
        (str(log.timestamp) for log in selected_logs if str(log.timestamp or "")),
        default="",
    )

    return MemoryAttributionRecord(
        memory_id=memory_id,
        tier=tier,
        memory_project_key=project_key,
        candidate_count=candidate_count,
        selected_count=selected_count,
        not_selected_count=not_selected_count,
        success_when_selected=success_when_selected,
        success_when_candidate_not_selected=success_when_not,
        reward_when_selected=reward_when_selected,
        reward_when_candidate_not_selected=reward_when_not,
        value=value,
        confidence=confidence,
        groups=groups,
        task_types=task_types,
        stream_ids=stream_ids,
        selected_task_ids=selected_task_ids,
        not_selected_task_ids=not_selected_task_ids,
        last_used=last_used,
    )


def _confidence(config: AttributionConfig, selected_count: int) -> float:
    if selected_count <= 0:
        return 0.0
    denom = sqrt(config.min_selected_count_for_full_confidence)
    if denom <= 0:
        return 1.0
    return min(1.0, sqrt(selected_count) / denom)


def _rate(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for v in values if v) / len(values)


def score_all_memories(
    *,
    entries: Sequence[ExperienceMemory],
    usage_logs: Sequence[UsageLogEntry],
    project_key: str = "",
    config: AttributionConfig | None = None,
) -> list[MemoryAttributionRecord]:
    """Score every experience memory that appears in the candidate/selected pool.

    Filtering rules (see plan -> "Attribution Formula"):
    - Every entry must be a typed four-tier experience.
    - A non-empty ``project_key`` requires an exact
      ``log.memory_project_key == project_key`` match and restricts candidate
      entries to that project. An explicitly empty API key is the
      per-task/global fallback and scores all complete logs.
    """
    cfg = config or AttributionConfig()
    # Pre-filter usage logs to the target project scope:
    # - empty project_key: per-task / global fallback -> all complete logs.
    # - non-empty project_key (shared stream): only logs whose memory_project_key
    #   equals it, so other streams (and per-task rows) never pollute this score.
    scoped_logs: list[UsageLogEntry] = []
    candidate_ids: set[str] = set()
    for log in usage_logs:
        if not log.is_complete:
            continue
        if project_key and log.memory_project_key != project_key:
            continue
        scoped_logs.append(log)
        candidate_ids.update(log.all_candidate_ids())

    # Project-visible typed experience entries. The store is a closed domain;
    # silently filtering legacy generic values would hide a cutover bug.
    visible_entries: list[ExperienceMemory] = []
    for entry in entries:
        if not isinstance(entry, ExperienceMemory):
            raise TypeError("score_all_memories requires ExperienceMemory entries")
        if (
            project_key
            and entry.scope != MemoryScope.GLOBAL
            and entry.project_key != project_key
        ):
            continue
        if entry.id in candidate_ids:
            visible_entries.append(entry)

    records: list[MemoryAttributionRecord] = []
    for entry in visible_entries:
        record = score_memory(
            memory_id=entry.id,
            tier=entry.tier.value,
            usage_logs=scoped_logs,
            config=cfg,
            project_key=project_key,
        )
        records.append(record)

    records.sort(key=lambda r: r.memory_id)
    return records


def write_attribution_jsonl(records: Sequence[MemoryAttributionRecord], output: str | Path) -> None:
    """Write records sorted by memory_id, byte-stable for the same inputs."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.memory_id)
    with path.open("w", encoding="utf-8") as fh:
        for record in ordered:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_attribution_jsonl(path: str | Path) -> dict[str, MemoryAttributionRecord]:
    """Load attribution records keyed by memory_id, skipping bad lines."""
    p = Path(path)
    records: dict[str, MemoryAttributionRecord] = {}
    if not p.exists():
        return records
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"attribution: skipping corrupt line {lineno} in {p}: {exc}"
                )
                continue
            if not isinstance(data, dict):
                continue
            record = MemoryAttributionRecord.from_dict(data)
            if record.memory_id:
                records[record.memory_id] = record
    return records


def write_back_attribution(
    *,
    store: ExperienceStore,
    records: Sequence[MemoryAttributionRecord],
    project_key: str | None = None,
    all_projects: bool = False,
    min_abs_value_to_write: float = 0.01,
    min_candidate_count: int = 2,
    updated_at: datetime | str | None = None,
) -> AttributionWriteBackSummary:
    """Write attribution fields back through ``ExperienceStore``.

    The update is intentionally narrow: only flat attribution fields change,
    and the store enforces atomic persistence. Unless ``all_projects`` is true,
    each record is restricted to the requested ``project_key`` or the record's
    own ``memory_project_key``.
    """
    attempted = 0
    updated = 0
    skipped_missing = 0
    skipped_by_project_key = 0
    skipped_tier_mismatch = 0
    skipped_low_evidence = 0

    all_entries = {entry.id: entry for entry in store.all()}
    for record in records:
        attempted += 1

        entry = all_entries.get(record.memory_id)
        if entry is None:
            skipped_missing += 1
            continue

        target_project_key = _record_project_key(project_key, record)
        if (
            not all_projects
            and target_project_key is not None
            and not _memory_visible_to_project(entry, target_project_key)
        ):
            skipped_by_project_key += 1
            continue

        try:
            expected_tier = ExperienceTier(record.tier)
        except ValueError:
            skipped_tier_mismatch += 1
            continue
        if entry.tier != expected_tier:
            skipped_tier_mismatch += 1
            continue

        value_to_write = float(record.value)
        if record.candidate_count < int(min_candidate_count) or abs(value_to_write) < float(min_abs_value_to_write):
            value_to_write = 0.0
            skipped_low_evidence += 1

        stored_record = replace(record, value=value_to_write)
        changed = store.update_attribution(
            stored_record,
            project_key=target_project_key,
            expected_tier=expected_tier,
            all_projects=all_projects,
            updated_at=updated_at,
        )
        if changed:
            updated += 1
        else:
            skipped_missing += 1

    return AttributionWriteBackSummary(
        attempted=attempted,
        updated=updated,
        skipped_missing=skipped_missing,
        skipped_by_project_key=skipped_by_project_key,
        skipped_tier_mismatch=skipped_tier_mismatch,
        skipped_low_evidence=skipped_low_evidence,
    )


def attribution_summary(
    records: Sequence[MemoryAttributionRecord],
    *,
    output: str | Path = "",
    top_n: int = 5,
    write_back: AttributionWriteBackSummary | None = None,
    config: AttributionConfig | None = None,
) -> dict[str, Any]:
    """Build the CLI / summary.json payload for attribution scoring."""
    ordered = sorted(records, key=lambda r: r.memory_id)
    positives = [r for r in ordered if r.value > 0]
    negatives = [r for r in ordered if r.value < 0]
    zero = [r for r in ordered if r.value == 0]
    top = sorted(ordered, key=lambda r: (-r.value, r.memory_id))[: max(0, int(top_n))]
    bottom = sorted(ordered, key=lambda r: (r.value, r.memory_id))[: max(0, int(top_n))]
    cfg = config or AttributionConfig()
    low_evidence = sum(
        1
        for record in ordered
        if record.candidate_count < cfg.min_candidate_count
        or record.selected_count < cfg.min_selected_count
        or record.not_selected_count < cfg.min_not_selected_count
    )
    payload: dict[str, Any] = {
        "output": str(output or ""),
        "records": len(ordered),
        "positive": len(positives),
        "negative": len(negatives),
        "zero": len(zero),
        "skipped_by_project_key": 0,
        "skipped_low_evidence": low_evidence,
        "top": [_summary_item(r) for r in top],
        "bottom": [_summary_item(r) for r in bottom],
    }
    if write_back is not None:
        payload["write_back_updated"] = write_back.updated
        payload["write_back"] = write_back.to_dict()
        payload["skipped_by_project_key"] = write_back.skipped_by_project_key
        payload["skipped_low_evidence"] = write_back.skipped_low_evidence
    else:
        payload["write_back_updated"] = 0
    return sanitize_json_value(payload)  # type: ignore[return-value]


def render_attribution_summary(summary: Mapping[str, Any]) -> str:
    return (
        f"Attribution records: {int(summary.get('records') or 0)} "
        f"(positive={int(summary.get('positive') or 0)}, "
        f"negative={int(summary.get('negative') or 0)}, "
        f"zero={int(summary.get('zero') or 0)})\n"
        f"Output: {summary.get('output') or ''}\n"
        f"Write-back updated: {int(summary.get('write_back_updated') or 0)}\n"
        f"Skipped by project key: {int(summary.get('skipped_by_project_key') or 0)}\n"
        f"Skipped low evidence: {int(summary.get('skipped_low_evidence') or 0)}"
    )


def _record_project_key(project_key: str | None, record: MemoryAttributionRecord) -> str | None:
    if project_key is not None:
        return project_key
    if record.memory_project_key:
        return record.memory_project_key
    return None


def _memory_visible_to_project(memory: ExperienceMemory, project_key: str) -> bool:
    if memory.scope == MemoryScope.GLOBAL:
        return True
    return bool(project_key) and memory.project_key == project_key


def _summary_item(record: MemoryAttributionRecord) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "tier": record.tier,
        "value": round(float(record.value), 6),
        "confidence": round(float(record.confidence), 6),
        "candidate_count": int(record.candidate_count),
        "selected_count": int(record.selected_count),
    }


__all__ = [
    "AttributionConfig",
    "AttributionWriteBackSummary",
    "DEFAULT_TIER_WEIGHTS",
    "MemoryAttributionRecord",
    "attribution_summary",
    "load_attribution_jsonl",
    "render_attribution_summary",
    "score_all_memories",
    "score_memory",
    "write_back_attribution",
    "write_attribution_jsonl",
]
