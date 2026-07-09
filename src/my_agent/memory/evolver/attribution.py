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

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence
import json
import warnings

from my_agent.memory.evolver.types import experience_tier
from my_agent.memory.evolver.usage_log import UsageLogEntry
from my_agent.memory.types import MemoryEntry
from my_agent.text_safety import sanitize_json_value


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

    # Compute per-task-type contributions only when both selected and
    # not-selected evidence exist in that task_type and meet min counts.
    by_task_type = _group_logs_by_task_type(pool_all)
    contributions: list[float] = []
    used_task_types: list[str] = []
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
        last_used="",
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
    entries: Sequence[MemoryEntry],
    usage_logs: Sequence[UsageLogEntry],
    project_key: str = "",
    config: AttributionConfig | None = None,
) -> list[MemoryAttributionRecord]:
    """Score every experience memory that appears in the candidate/selected pool.

    Filtering rules (see plan -> "Attribution Formula"):
    - Only entries with an experience tier are considered.
    - ``project_key`` filters usage logs by ``log.memory_project_key == project_key``
      (empty log keys are per-task/global and always included). When non-empty it
      also restricts candidate entries to those visible to ``project_key``.
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
        candidate_ids.update(log.all_selected_ids())

    # Project-visible experience entries.
    visible_entries: list[MemoryEntry] = []
    for entry in entries:
        tier = experience_tier(entry)
        if tier is None:
            continue
        if project_key and entry.project_key and entry.project_key != project_key:
            continue
        if entry.id in candidate_ids:
            visible_entries.append(entry)

    records: list[MemoryAttributionRecord] = []
    for entry in visible_entries:
        record = score_memory(
            memory_id=entry.id,
            tier=experience_tier(entry).value,  # type: ignore[union-attr]
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


__all__ = [
    "AttributionConfig",
    "DEFAULT_TIER_WEIGHTS",
    "MemoryAttributionRecord",
    "load_attribution_jsonl",
    "score_all_memories",
    "score_memory",
    "write_attribution_jsonl",
]