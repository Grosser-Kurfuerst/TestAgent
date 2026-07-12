from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EDIT_TOOLS = {"replace_in_file", "write_file"}


@dataclass(frozen=True)
class TraceStats:
    trace_files: int = 0
    runs: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    blocked_tool_calls: int = 0
    test_runs: int = 0
    passed_test_runs: int = 0
    edit_count: int = 0
    tool_distribution: dict[str, int] = field(default_factory=dict)
    evolver_candidate_events: int = 0
    evolver_selected_events: int = 0
    evolver_candidates_total: int = 0
    evolver_selected_total: int = 0
    evolver_selected_by_tier: dict[str, int] = field(default_factory=dict)
    evolver_selection_policies: dict[str, int] = field(default_factory=dict)
    evolver_writer_started_events: int = 0
    evolver_writer_saved_events: int = 0
    evolver_writer_saved_total: int = 0
    evolver_writer_saved_by_tier: dict[str, int] = field(default_factory=dict)
    evolver_writer_failed_events: int = 0
    maintenance_runs: int = 0
    maintenance_applied_runs: int = 0
    maintenance_keep: int = 0
    maintenance_delete: int = 0
    maintenance_merge: int = 0
    maintenance_promote: int = 0
    maintenance_removed_entries: int = 0
    maintenance_added_entries: int = 0
    maintenance_failures: int = 0
    maintenance_committed_with_audit_error: int = 0

    @property
    def tool_success_rate(self) -> float:
        return _rate(self.successful_tool_calls, self.tool_calls)

    @property
    def test_pass_rate(self) -> float:
        return _rate(self.passed_test_runs, self.test_runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_files": self.trace_files,
            "runs": self.runs,
            "tool_calls": self.tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "tool_success_rate": self.tool_success_rate,
            "blocked_tool_calls": self.blocked_tool_calls,
            "test_runs": self.test_runs,
            "passed_test_runs": self.passed_test_runs,
            "test_pass_rate": self.test_pass_rate,
            "edit_count": self.edit_count,
            "tool_distribution": self.tool_distribution,
            "evolver_candidate_events": self.evolver_candidate_events,
            "evolver_selected_events": self.evolver_selected_events,
            "evolver_candidates_total": self.evolver_candidates_total,
            "evolver_selected_total": self.evolver_selected_total,
            "evolver_selected_by_tier": self.evolver_selected_by_tier,
            "evolver_selection_policies": self.evolver_selection_policies,
            "evolver_writer_started_events": self.evolver_writer_started_events,
            "evolver_writer_saved_events": self.evolver_writer_saved_events,
            "evolver_writer_saved_total": self.evolver_writer_saved_total,
            "evolver_writer_saved_by_tier": self.evolver_writer_saved_by_tier,
            "evolver_writer_failed_events": self.evolver_writer_failed_events,
            "maintenance_runs": self.maintenance_runs,
            "maintenance_applied_runs": self.maintenance_applied_runs,
            "maintenance_keep": self.maintenance_keep,
            "maintenance_delete": self.maintenance_delete,
            "maintenance_merge": self.maintenance_merge,
            "maintenance_promote": self.maintenance_promote,
            "maintenance_removed_entries": self.maintenance_removed_entries,
            "maintenance_added_entries": self.maintenance_added_entries,
            "maintenance_failures": self.maintenance_failures,
            "maintenance_committed_with_audit_error": (
                self.maintenance_committed_with_audit_error
            ),
        }


def collect_trace_stats(path: str | Path) -> TraceStats:
    trace_files = _trace_files(Path(path))
    run_ids: set[str] = set()
    distribution: Counter[str] = Counter()
    tool_calls = 0
    successful_tool_calls = 0
    blocked_tool_calls = 0
    test_runs = 0
    passed_test_runs = 0
    edit_count = 0
    evolver_candidate_events = 0
    evolver_selected_events = 0
    evolver_candidates_total = 0
    evolver_selected_total = 0
    evolver_selected_by_tier: Counter[str] = Counter()
    evolver_selection_policies: Counter[str] = Counter()
    evolver_writer_started_events = 0
    evolver_writer_saved_events = 0
    evolver_writer_saved_total = 0
    evolver_writer_saved_by_tier: Counter[str] = Counter()
    evolver_writer_failed_events = 0
    maintenance_runs = 0
    maintenance_applied_runs = 0
    maintenance_keep = 0
    maintenance_delete = 0
    maintenance_merge = 0
    maintenance_promote = 0
    maintenance_removed_entries = 0
    maintenance_added_entries = 0
    maintenance_failures = 0
    maintenance_committed_with_audit_error = 0

    for trace_file in trace_files:
        for event in _read_trace_file(trace_file):
            run_id = event.get("run_id")
            if isinstance(run_id, str) and run_id:
                run_ids.add(run_id)
            event_name = event.get("event")
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue

            if event_name == "memory.evolver_candidates":
                evolver_candidate_events += 1
                evolver_candidates_total += _payload_count(
                    payload,
                    count_key="candidate_count",
                    fallback_keys=("candidate_ids", "candidate_summaries"),
                )
                continue
            if event_name == "memory.evolver_selected":
                evolver_selected_events += 1
                evolver_selected_total += _payload_count(
                    payload,
                    count_key="selected_count",
                    fallback_keys=("selected_ids",),
                )
                evolver_selected_by_tier.update(_tier_counts(payload.get("tiers")))
                policy = payload.get("selection_policy")
                if isinstance(policy, str) and policy:
                    evolver_selection_policies[policy] += 1
                continue

            if event_name == "memory.evolver_writer_started":
                evolver_writer_started_events += 1
                continue
            if event_name == "memory.evolver_writer_saved":
                evolver_writer_saved_events += 1
                evolver_writer_saved_total += _payload_count(
                    payload,
                    count_key="saved_count",
                    fallback_keys=("saved_ids", "saved_records"),
                )
                evolver_writer_saved_by_tier.update(_tier_counts(payload.get("tiers")))
                continue
            if event_name == "memory.evolver_writer_failed":
                evolver_writer_failed_events += 1
                continue

            if event_name == "memory.maintenance_started":
                maintenance_runs += 1
                continue
            if event_name == "memory.maintenance_proposed":
                maintenance_keep += _nonnegative_int(payload.get("keep"))
                maintenance_delete += _nonnegative_int(payload.get("delete"))
                maintenance_merge += _nonnegative_int(payload.get("merge"))
                maintenance_promote += _nonnegative_int(payload.get("promote"))
                maintenance_removed_entries += _nonnegative_int(
                    payload.get("source_entries_removed")
                )
                maintenance_added_entries += _nonnegative_int(
                    payload.get("entries_added")
                )
                continue
            if event_name == "memory.maintenance_completed":
                maintenance_applied_runs += 1
                if payload.get("status") == "committed_with_audit_error":
                    maintenance_committed_with_audit_error += 1
                continue
            if event_name == "memory.maintenance_failed":
                maintenance_failures += 1
                continue

            if event_name != "tool.completed":
                continue

            tool = payload.get("name")
            if not isinstance(tool, str) or not tool:
                continue

            tool_calls += 1
            distribution[tool] += 1
            ok = bool(payload.get("ok"))
            blocked = bool(payload.get("blocked"))
            if ok:
                successful_tool_calls += 1
            if blocked:
                blocked_tool_calls += 1
            if tool in EDIT_TOOLS and ok:
                edit_count += 1
            if tool == "run_tests":
                test_runs += 1
                if ok:
                    passed_test_runs += 1

    return TraceStats(
        trace_files=len(trace_files),
        runs=len(run_ids),
        tool_calls=tool_calls,
        successful_tool_calls=successful_tool_calls,
        blocked_tool_calls=blocked_tool_calls,
        test_runs=test_runs,
        passed_test_runs=passed_test_runs,
        edit_count=edit_count,
        tool_distribution=dict(sorted(distribution.items())),
        evolver_candidate_events=evolver_candidate_events,
        evolver_selected_events=evolver_selected_events,
        evolver_candidates_total=evolver_candidates_total,
        evolver_selected_total=evolver_selected_total,
        evolver_selected_by_tier=dict(sorted(evolver_selected_by_tier.items())),
        evolver_selection_policies=dict(sorted(evolver_selection_policies.items())),
        evolver_writer_started_events=evolver_writer_started_events,
        evolver_writer_saved_events=evolver_writer_saved_events,
        evolver_writer_saved_total=evolver_writer_saved_total,
        evolver_writer_saved_by_tier=dict(sorted(evolver_writer_saved_by_tier.items())),
        evolver_writer_failed_events=evolver_writer_failed_events,
        maintenance_runs=maintenance_runs,
        maintenance_applied_runs=maintenance_applied_runs,
        maintenance_keep=maintenance_keep,
        maintenance_delete=maintenance_delete,
        maintenance_merge=maintenance_merge,
        maintenance_promote=maintenance_promote,
        maintenance_removed_entries=maintenance_removed_entries,
        maintenance_added_entries=maintenance_added_entries,
        maintenance_failures=maintenance_failures,
        maintenance_committed_with_audit_error=maintenance_committed_with_audit_error,
    )


def format_trace_stats(stats: TraceStats) -> str:
    lines = [
        f"Trace files: {stats.trace_files}",
        f"Runs: {stats.runs}",
        f"Tool calls: {stats.tool_calls}",
        (
            "Tool success rate: "
            f"{stats.successful_tool_calls}/{stats.tool_calls} ({stats.tool_success_rate:.1%})"
        ),
        f"Blocked tool calls: {stats.blocked_tool_calls}",
        f"Test pass rate: {stats.passed_test_runs}/{stats.test_runs} ({stats.test_pass_rate:.1%})",
        f"Edit count: {stats.edit_count}",
        (
            "Evolver selection: "
            f"candidate_events={stats.evolver_candidate_events}, "
            f"candidates={stats.evolver_candidates_total}, "
            f"selected_events={stats.evolver_selected_events}, "
            f"selected={stats.evolver_selected_total}"
        ),
        (
            "Evolver writer: "
            f"started_events={stats.evolver_writer_started_events}, "
            f"saved_events={stats.evolver_writer_saved_events}, "
            f"saved={stats.evolver_writer_saved_total}, "
            f"failed_events={stats.evolver_writer_failed_events}"
        ),
        (
            "Memory maintenance: "
            f"runs={stats.maintenance_runs}, "
            f"applied={stats.maintenance_applied_runs}, "
            f"failures={stats.maintenance_failures}, "
            "committed_with_audit_error="
            f"{stats.maintenance_committed_with_audit_error}"
        ),
        (
            "Maintenance actions: "
            f"keep={stats.maintenance_keep}, "
            f"delete={stats.maintenance_delete}, "
            f"merge={stats.maintenance_merge}, "
            f"promote={stats.maintenance_promote}, "
            f"removed={stats.maintenance_removed_entries}, "
            f"added={stats.maintenance_added_entries}"
        ),
        "Tool distribution:",
    ]
    if stats.tool_distribution:
        lines.extend(f"- {tool}: {count}" for tool, count in stats.tool_distribution.items())
    else:
        lines.append("- none: 0")
    lines.append("Evolver selected by tier:")
    if stats.evolver_selected_by_tier:
        lines.extend(f"- {tier}: {count}" for tier, count in stats.evolver_selected_by_tier.items())
    else:
        lines.append("- none: 0")
    lines.append("Evolver selection policies:")
    if stats.evolver_selection_policies:
        lines.extend(f"- {policy}: {count}" for policy, count in stats.evolver_selection_policies.items())
    else:
        lines.append("- none: 0")
    lines.append("Evolver writer saved by tier:")
    if stats.evolver_writer_saved_by_tier:
        lines.extend(f"- {tier}: {count}" for tier, count in stats.evolver_writer_saved_by_tier.items())
    else:
        lines.append("- none: 0")
    return "\n".join(lines)


def _trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(file for file in path.glob("*.jsonl") if file.is_file())
    raise FileNotFoundError(f"Trace path not found: {path}")


def _read_trace_file(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL trace at {path}:{line_number}: {exc}") from exc
        if isinstance(event, dict):
            events.append(event)
    return events


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _payload_count(payload: dict[str, Any], *, count_key: str, fallback_keys: tuple[str, ...]) -> int:
    count = payload.get(count_key)
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    for key in fallback_keys:
        values = payload.get(key)
        if isinstance(values, list):
            return len(values)
    return 0


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _tier_counts(value: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counts
    for tier, count in value.items():
        if not isinstance(tier, str) or not tier:
            continue
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            counts[tier] += count
    return counts
