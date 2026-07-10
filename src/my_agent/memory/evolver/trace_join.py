"""Join manifest ``results.jsonl`` rows and agent traces into usage-log entries.

This is the offline bridge that turns Phase 3 selection traces +
Phase 4 outcome records into :class:`UsageLogEntry` rows for Phase 5
attribution. It is deliberately pure w.r.t. time: every field is read from
already-persisted input, never from the wall clock, so reruns are byte-stable.

See ``phase5-outcome-calibrated-attribution-implementation-plan.md`` ->
"Trace And Result Join".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from my_agent.memory.evolver.usage_log import (
    UsageLogEntry,
    group_ids_by_tier,
)


_TRACE_SELECTION_CANDIDATES = "memory.evolver_candidates"
_TRACE_SELECTION_SELECTED = "memory.evolver_selected"
_TRACE_BENCHMARK_RESULT = "benchmark_result"


@dataclass(frozen=True)
class SelectionSnapshot:
    """The last observed selector state for one run."""

    retrieved_candidates: dict[str, list[str]] = field(default_factory=dict)
    selected_memory_ids: dict[str, list[str]] = field(default_factory=dict)
    candidate_count: int = 0
    selected_count: int = 0
    selection_policy: str = ""
    timestamp: str = ""
    run_id: str = ""
    selection_events_seen: int = 0
    selection_events_used: int = 0

    @property
    def is_empty(self) -> bool:
        return self.candidate_count == 0 and self.selected_count == 0


@dataclass(frozen=True)
class BenchmarkOutcome:
    """Post-hoc outcome parsed from a manifest row or trace benchmark event."""

    task_id: str = ""
    resolved: bool = False
    status: str = ""
    failure_type: str = ""
    source: str = ""
    mode: str = ""
    tags: tuple[str, ...] = ()
    stream_id: str = ""
    memory_mode: str = ""
    memory_dir: str = ""
    memory_project_key: str = ""
    timestamp: str = ""
    run_id: str = ""
    has_resolved: bool = False


def read_trace_events(path: str | Path) -> list[dict[str, Any]]:
    """Read a trace JSONL file, skipping blank/corrupt lines."""
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, dict):
        return payload
    return {}


def selection_from_trace(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str = "",
) -> SelectionSnapshot:
    """Return the last non-empty selection state observed in the trace.

    A run may prepare context multiple times (multi-turn ReAct). The plan says the
    "last non-empty selected" event is the most-recent selection actually applied
    before execution. If the last selection is empty but candidates exist, that is
    still recorded (valuable for the not-selected baseline).
    """
    candidate_event: Mapping[str, Any] | None = None
    last_snapshot: SelectionSnapshot | None = None
    candidate_events_seen = 0
    selection_events_seen = 0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if not _event_matches_run(event, run_id):
            continue
        name = event.get("event")
        if name == _TRACE_SELECTION_CANDIDATES:
            candidate_events_seen += 1
            candidate_event = event
        elif name == _TRACE_SELECTION_SELECTED:
            selection_events_seen += 1
            snapshot = _selection_snapshot_from_pair(candidate_event, event)
            if not snapshot.is_empty:
                last_snapshot = snapshot
            elif last_snapshot is None:
                last_snapshot = snapshot
    if last_snapshot is None and candidate_event is not None:
        last_snapshot = _selection_snapshot_from_pair(candidate_event, None)
    snapshot = last_snapshot or SelectionSnapshot(run_id=run_id)
    return replace(
        snapshot,
        selection_events_seen=max(candidate_events_seen, selection_events_seen),
        selection_events_used=0 if snapshot.is_empty else 1,
    )


def _selection_snapshot_from_pair(
    candidate_event: Mapping[str, Any] | None,
    selected_event: Mapping[str, Any] | None,
) -> SelectionSnapshot:
    candidate_payload = _payload(candidate_event) if candidate_event else {}
    selected_payload = _payload(selected_event) if selected_event else {}

    candidate_tier_by_id: dict[str, str] = {}
    for summary in candidate_payload.get("candidate_summaries") or []:
        if isinstance(summary, Mapping):
            memory_id = summary.get("id")
            if memory_id:
                candidate_tier_by_id[str(memory_id)] = str(summary.get("tier") or "unknown")

    retrieved_candidates = group_ids_by_tier(candidate_payload.get("candidate_summaries") or [])
    # Fall back to flat candidate_ids (no tier) if summaries are absent.
    if not retrieved_candidates:
        flat_ids = [str(i) for i in (candidate_payload.get("candidate_ids") or []) if i]
        if flat_ids:
            retrieved_candidates = {"unknown": flat_ids}

    selected_ids = [str(i) for i in (selected_payload.get("selected_ids") or []) if i]
    selected_memory_ids: dict[str, list[str]] = {}
    for memory_id in selected_ids:
        tier = candidate_tier_by_id.get(memory_id, "")
        if not selected_payload.get("tiers"):
            tier = tier or "unknown"
        selected_memory_ids.setdefault(tier or "unknown", []).append(memory_id)

    timestamp = str(
        selected_payload.get("timestamp")
        or (selected_event.get("time") if selected_event else "")
        or candidate_payload.get("timestamp")
        or (candidate_event.get("time") if candidate_event else "")
        or ""
    )
    run_id = str(
        (selected_event.get("run_id") if selected_event else "")
        or (candidate_event.get("run_id") if candidate_event else "")
        or ""
    )
    return SelectionSnapshot(
        retrieved_candidates=retrieved_candidates,
        selected_memory_ids=selected_memory_ids,
        candidate_count=int(candidate_payload.get("candidate_count") or len(_flatten(retrieved_candidates))),
        selected_count=int(selected_payload.get("selected_count") or len(selected_ids)),
        selection_policy=str(selected_payload.get("selection_policy") or candidate_payload.get("selection_policy") or ""),
        timestamp=timestamp,
        run_id=run_id,
    )


def _flatten(ids_by_tier: Mapping[str, list[str]]) -> list[str]:
    flat: list[str] = []
    for ids in ids_by_tier.values():
        flat.extend(ids)
    return flat


def benchmark_outcome_from_trace(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str = "",
) -> BenchmarkOutcome | None:
    """Return the last benchmark_result outcome observed in the trace, if any."""
    last: Mapping[str, Any] | None = None
    for event in events:
        if (
            isinstance(event, Mapping)
            and event.get("event") == _TRACE_BENCHMARK_RESULT
            and _event_matches_run(event, run_id)
        ):
            last = event
    if last is None:
        return None
    payload = _payload(last)
    status = str(payload.get("status") or "")
    has_resolved = "resolved" in payload and payload.get("resolved") is not None
    return BenchmarkOutcome(
        task_id=str(payload.get("task_id") or ""),
        resolved=bool(payload.get("resolved")) if has_resolved else _status_is_success(status),
        status=status,
        failure_type=str(payload.get("failure_type") or ""),
        source=str(payload.get("source") or ""),
        mode=str(payload.get("mode") or ""),
        tags=tuple(str(t) for t in (payload.get("tags") or [])),
        stream_id=str(payload.get("stream_id") or ""),
        memory_mode=str(payload.get("memory_mode") or ""),
        memory_dir=str(payload.get("memory_dir") or ""),
        memory_project_key=str(payload.get("memory_project_key") or ""),
        timestamp=str(payload.get("timestamp") or last.get("time") or ""),
        run_id=str(last.get("run_id") or payload.get("run_id") or ""),
        has_resolved=has_resolved,
    )


def _event_matches_run(event: Mapping[str, Any], run_id: str) -> bool:
    if not run_id:
        return True
    event_run_id = str(event.get("run_id") or _payload(event).get("run_id") or "")
    return not event_run_id or event_run_id == run_id


def _status_is_success(status: str) -> bool:
    return status.strip().lower() in {"passed", "resolved", "success", "succeeded"}


def _resolve_task_type(row: Mapping[str, Any], outcome: BenchmarkOutcome) -> str:
    """task_type prefers manifest source, then mode, then 'unknown'."""
    for value in (row.get("source"), outcome.source, row.get("mode"), outcome.mode):
        if value:
            return str(value)
    return "unknown"


def usage_entry_from_result_row(
    row: Mapping[str, Any],
    *,
    trace_events: Sequence[Mapping[str, Any]] | None = None,
) -> UsageLogEntry | None:
    """Build a usage log entry from a manifest results.jsonl row.

    ``trace_events`` is the already-read event list for ``row["trace_path"]``;
    if omitted, the trace is read lazily from ``trace_path``.
    Returns ``None`` when the row has no usable selection or outcome.
    """
    task_id = str(row.get("task_id") or "")
    if not task_id:
        return None

    events: Sequence[Mapping[str, Any]]
    if trace_events is not None:
        events = trace_events
    else:
        trace_path = str(row.get("trace_path") or "")
        events = read_trace_events(trace_path) if trace_path else []

    row_run_id = str(row.get("run_id") or "")
    outcome_trace = benchmark_outcome_from_trace(events, run_id=row_run_id)
    target_run_id = row_run_id or (outcome_trace.run_id if outcome_trace else "")
    selection = selection_from_trace(events, run_id=target_run_id)

    # outcome from manifest row is authoritative; trace benchmark_result is fallback.
    row_has_resolved = "resolved" in row and row.get("resolved") is not None
    status = str(row.get("status") or "")
    if row_has_resolved:
        resolved = bool(row.get("resolved"))
    elif status:
        resolved = _status_is_success(status)
    elif outcome_trace is not None:
        resolved = bool(outcome_trace.resolved)
    else:
        resolved = False
    if not status:
        status = outcome_trace.status if outcome_trace is not None else ""

    resolved_bool = bool(resolved)
    # A complete usage log requires a post-hoc outcome signal. finish_called /
    # other runtime stop reasons must NOT be treated as success; only manifest
    # ``resolved`` / trace ``benchmark_result.resolved/status`` count.
    has_outcome = bool(
        row_has_resolved
        or status
        or (
            outcome_trace is not None
            and (outcome_trace.has_resolved or outcome_trace.status)
        )
    )
    if not has_outcome:
        return None

    memory_project_key = str(row.get("memory_project_key") or (outcome_trace.memory_project_key if outcome_trace else "") or "")
    stream_id = str(row.get("stream_id") or (outcome_trace.stream_id if outcome_trace else "") or "")
    memory_mode = str(row.get("memory_mode") or (outcome_trace.memory_mode if outcome_trace else "") or "")
    source_task = str(row.get("source_task") or row.get("task_id") or "")
    tags = tuple(str(t) for t in (row.get("tags") or (outcome_trace.tags if outcome_trace else []) or []) if t)
    task_type = _resolve_task_type(row, outcome_trace or BenchmarkOutcome())

    timestamp = str(
        selection.timestamp
        or (outcome_trace.timestamp if outcome_trace else "")
        or row.get("timestamp")
        or ""
    )

    failure_type = str(row.get("failure_type") or (outcome_trace.failure_type if outcome_trace else "") or "")

    return UsageLogEntry(
        task_id=task_id,
        task_type=task_type,
        timestamp=timestamp,
        run_id=target_run_id or selection.run_id,
        trace_path=str(row.get("trace_path") or ""),
        source_task=source_task,
        stream_id=stream_id,
        memory_project_key=memory_project_key,
        memory_mode=memory_mode,
        retrieved_candidates=dict(selection.retrieved_candidates),
        selected_memory_ids=dict(selection.selected_memory_ids),
        env_reward=1.0 if resolved_bool else 0.0,
        success=resolved_bool,
        status="complete",
        failure_type=failure_type,
        tags=tags,
    )


def usage_entry_from_manifest_result(result: Mapping[str, Any]) -> UsageLogEntry | None:
    """Build a usage log entry from a single manifest result dict (with trace_path)."""
    return usage_entry_from_result_row(result)


def usage_entry_from_trace(
    trace_path: str | Path,
    *,
    result: Mapping[str, Any] | None = None,
) -> UsageLogEntry | None:
    """Build a usage log entry directly from a trace file (+ optional result row)."""
    events = read_trace_events(trace_path)
    if result is not None:
        return usage_entry_from_result_row(result, trace_events=events)
    # No manifest row: synthesize a minimal row carrying trace-derived outcome only.
    outcome = benchmark_outcome_from_trace(events)
    selection = selection_from_trace(events, run_id=outcome.run_id if outcome else "")
    if selection.is_empty and outcome is None:
        return None
    if outcome is None:
        return None
    row: dict[str, Any] = {
        "task_id": outcome.task_id,
        "trace_path": str(trace_path),
        "resolved": outcome.resolved,
        "status": outcome.status,
        "failure_type": outcome.failure_type,
        "source": outcome.source,
        "mode": outcome.mode,
        "tags": list(outcome.tags),
        "stream_id": outcome.stream_id,
        "memory_mode": outcome.memory_mode,
        "memory_project_key": outcome.memory_project_key,
        "run_id": outcome.run_id,
    }
    return usage_entry_from_result_row(row, trace_events=events)


def collect_usage_from_manifest_results(results_path: str | Path) -> list[UsageLogEntry]:
    """Read a manifest ``results.jsonl`` and return one usage entry per row."""
    entries: list[UsageLogEntry] = []
    path = Path(results_path)
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            entry = usage_entry_from_result_row(row)
            if entry is not None:
                entries.append(entry)
    return entries


__all__ = [
    "BenchmarkOutcome",
    "SelectionSnapshot",
    "benchmark_outcome_from_trace",
    "collect_usage_from_manifest_results",
    "read_trace_events",
    "selection_from_trace",
    "usage_entry_from_manifest_result",
    "usage_entry_from_result_row",
    "usage_entry_from_trace",
]
