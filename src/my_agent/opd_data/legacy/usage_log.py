"""Usage log data model and JSONL logger for OPD-Evolver Phase 5 attribution.

The usage log is the fact table that connects retrieved candidate memories,
selected memories, and a post-hoc benchmark outcome for a single task. It is
the input to :mod:`my_agent.opd_data.legacy.attribution`.

Design notes (see
``docs/opd-evolver-reproduction/phase5-outcome-calibrated-attribution-implementation-plan.md``):

- ``timestamp`` must be reproducible offline. It is constructed only from stable
  fields already present in the inputs (trace selection event / benchmark result
  / manifest result). It never calls ``datetime.now()`` so that the same inputs
  produce byte-level stable usage logs.
- outcome (``success`` / ``env_reward``) only ever comes from manifest
  ``resolved`` or the trace ``benchmark_result``; it never comes from a runtime
  ``stop_reason=finish_called`` signal.
- started/complete merge key is ``(memory_project_key, task_id)`` (or
  ``(run_id, task_id)`` when ``run_id`` is available) so that the same
  ``task_id`` in different streams/projects never overwrites each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import warnings

from my_agent.text_safety import sanitize_json_value


@dataclass(frozen=True)
class UsageLogEntry:
    """One task's candidate/selected/outcome triplet for attribution."""

    task_id: str
    task_type: str
    timestamp: str = ""
    run_id: str = ""
    trace_path: str = ""
    source_task: str = ""
    stream_id: str = ""
    memory_project_key: str = ""
    memory_mode: str = ""
    retrieved_candidates: dict[str, list[str]] = field(default_factory=dict)
    selected_memory_ids: dict[str, list[str]] = field(default_factory=dict)
    env_reward: float = 0.0
    success: bool = False
    status: str | None = None
    failure_type: str = ""
    tags: tuple[str, ...] = ()

    def all_candidate_ids(self) -> list[str]:
        """Return candidate ids in tier order, de-duplicated, preserving order."""
        seen: set[str] = set()
        ids: list[str] = []
        for tier_ids in self.retrieved_candidates.values():
            for memory_id in tier_ids:
                if memory_id and memory_id not in seen:
                    seen.add(memory_id)
                    ids.append(memory_id)
        return ids

    def all_selected_ids(self) -> list[str]:
        """Return selected ids in tier order, de-duplicated, preserving order."""
        seen: set[str] = set()
        ids: list[str] = []
        for tier_ids in self.selected_memory_ids.values():
            for memory_id in tier_ids:
                if memory_id and memory_id not in seen:
                    seen.add(memory_id)
                    ids.append(memory_id)
        return ids

    @property
    def is_complete(self) -> bool:
        return self.status in (None, "complete")

    def merge_outcome(self, *, env_reward: float, success: bool, status: str = "complete") -> "UsageLogEntry":
        """Return a copy with outcome fields filled (started -> complete)."""
        return UsageLogEntry(
            task_id=self.task_id,
            task_type=self.task_type,
            timestamp=self.timestamp,
            run_id=self.run_id,
            trace_path=self.trace_path,
            source_task=self.source_task,
            stream_id=self.stream_id,
            memory_project_key=self.memory_project_key,
            memory_mode=self.memory_mode,
            retrieved_candidates=self.retrieved_candidates,
            selected_memory_ids=self.selected_memory_ids,
            env_reward=float(env_reward),
            success=bool(success),
            status=status,
            failure_type=self.failure_type,
            tags=self.tags,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": str(self.task_id),
            "task_type": str(self.task_type),
            "timestamp": str(self.timestamp or ""),
            "run_id": str(self.run_id or ""),
            "trace_path": str(self.trace_path or ""),
            "source_task": str(self.source_task or ""),
            "stream_id": str(self.stream_id or ""),
            "memory_project_key": str(self.memory_project_key or ""),
            "memory_mode": str(self.memory_mode or ""),
            "retrieved_candidates": {str(k): list(v) for k, v in self.retrieved_candidates.items()},
            "selected_memory_ids": {str(k): list(v) for k, v in self.selected_memory_ids.items()},
            "env_reward": float(self.env_reward),
            "success": bool(self.success),
            "status": self.status,
            "failure_type": str(self.failure_type or ""),
            "tags": list(self.tags),
        }
        return sanitize_json_value(payload)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UsageLogEntry":
        return cls(
            task_id=str(data.get("task_id", "") or ""),
            task_type=str(data.get("task_type", "") or ""),
            timestamp=str(data.get("timestamp", "") or ""),
            run_id=str(data.get("run_id", "") or ""),
            trace_path=str(data.get("trace_path", "") or ""),
            source_task=str(data.get("source_task", "") or ""),
            stream_id=str(data.get("stream_id", "") or ""),
            memory_project_key=str(data.get("memory_project_key", "") or ""),
            memory_mode=str(data.get("memory_mode", "") or ""),
            retrieved_candidates={
                str(k): [str(i) for i in v]
                for k, v in dict(data.get("retrieved_candidates") or {}).items()
            },
            selected_memory_ids={
                str(k): [str(i) for i in v]
                for k, v in dict(data.get("selected_memory_ids") or {}).items()
            },
            env_reward=float(data.get("env_reward", 0.0) or 0.0),
            success=bool(data.get("success", False)),
            status=data.get("status"),
            failure_type=str(data.get("failure_type", "") or ""),
            tags=tuple(str(t) for t in (data.get("tags") or [])),
        )


def group_ids_by_tier(items: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Group id lists by ``tier`` for a sequence of summary/record mappings.

    Each item may carry ``tier`` and ``id`` (candidate summaries) or ``tier`` and
    ``ids``. Missing tiers fall back to ``"unknown"``.
    """
    grouped: dict[str, list[str]] = {}
    for item in items:
        tier = str(item.get("tier") or item.get("evolver_tier") or "unknown") or "unknown"
        bucket = grouped.setdefault(tier, [])
        memory_id = item.get("id")
        if memory_id:
            bucket.append(str(memory_id))
        for mid in item.get("ids") or []:
            bucket.append(str(mid))
    return grouped


def flatten_tier_ids(ids_by_tier: Mapping[str, Iterable[str]]) -> list[str]:
    """Flatten a tier -> ids map into an ordered, de-duplicated id list."""
    seen: set[str] = set()
    flat: list[str] = []
    for tier_ids in ids_by_tier.values():
        for memory_id in tier_ids:
            memory_id = str(memory_id)
            if memory_id and memory_id not in seen:
                seen.add(memory_id)
                flat.append(memory_id)
    return flat


def _merge_key(entry: Mapping[str, Any]) -> tuple[str, str]:
    """Stable merge key so same task_id across streams/projects never collide."""
    run_id = str(entry.get("run_id") or "")
    project_key = str(entry.get("memory_project_key") or "")
    task_id = str(entry.get("task_id") or "")
    if run_id:
        return (run_id, task_id)
    return (project_key, task_id)


class UsageLogger:
    """Append-only JSONL usage log with dedup-aware loading."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, entry: UsageLogEntry) -> None:
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def append_many(self, entries: Iterable[UsageLogEntry]) -> int:
        count = 0
        with self.path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        return count

    def overwrite(self, entries: Iterable[UsageLogEntry]) -> int:
        """Replace the file contents (default CLI mode for reproducible output)."""
        count = 0
        with self.path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        return count

    def load_all(self) -> list[UsageLogEntry]:
        """Load and merge started/complete records, skipping corrupt lines."""
        if not self.path.exists():
            return []
        started: dict[tuple[str, str], dict[str, Any]] = {}
        complete: dict[tuple[str, str], dict[str, Any]] = {}
        legacy: dict[tuple[str, str], dict[str, Any]] = {}
        order: list[tuple[str, str]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    warnings.warn(
                        f"UsageLogger: skipping corrupt line {lineno} in "
                        f"{self.path}: {exc}"
                    )
                    continue
                if not isinstance(data, dict):
                    continue
                key = _merge_key(data)
                if not key[-1]:
                    continue
                status = data.get("status")
                if status == "complete":
                    if key not in complete and key not in started and key not in legacy:
                        order.append(key)
                    complete[key] = data
                elif status == "started":
                    if key not in started and key not in complete and key not in legacy:
                        order.append(key)
                    started[key] = data
                else:
                    if key not in legacy and key not in started and key not in complete:
                        order.append(key)
                    legacy[key] = data
        entries: list[UsageLogEntry] = []
        for key in order:
            data: Mapping[str, Any] | None
            if key in started:
                base = dict(started[key])
                if key in complete:
                    outcome = complete[key]
                    if outcome.get("retrieved_candidates") or outcome.get("selected_memory_ids"):
                        base = dict(outcome)
                        base["status"] = "complete"
                    else:
                        base["env_reward"] = outcome.get("env_reward", base.get("env_reward", 0.0))
                        base["success"] = outcome.get("success", base.get("success", False))
                        base["failure_type"] = outcome.get(
                            "failure_type", base.get("failure_type", "")
                        )
                        base["status"] = "complete"
                data = base
            elif key in complete:
                data = complete[key]
            else:
                data = legacy.get(key)
            if data is None:
                continue
            try:
                entries.append(UsageLogEntry.from_dict(data))
            except (KeyError, TypeError, ValueError) as exc:
                warnings.warn(
                    f"UsageLogger: skipping malformed usage record for "
                    f"task_id={key[-1]!r}: {exc}"
                )
        return entries

    def load_for_memory(self, memory_id: str) -> list[UsageLogEntry]:
        return [
            entry
            for entry in self.load_all()
            if memory_id in entry.all_candidate_ids() or memory_id in entry.all_selected_ids()
        ]

    def count(self) -> int:
        return len(self.load_all())

    def count_lines(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())


__all__ = [
    "UsageLogEntry",
    "UsageLogger",
    "flatten_tier_ids",
    "group_ids_by_tier",
]
