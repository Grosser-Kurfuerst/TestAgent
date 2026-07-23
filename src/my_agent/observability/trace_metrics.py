from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from my_agent.observability.maintenance_events import MaintenanceEventCounters


EDIT_TOOLS = {"replace_in_file", "write_file"}
SUCCESS_STOP_REASONS = {
    "finish_called",
    "assistant_final",
    "plan_completed",
    "team_completed",
}


@dataclass(frozen=True)
class TraceMetrics:
    trace_files: int = 0
    runs: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    blocked_tool_calls: int = 0
    test_runs: int = 0
    passed_test_runs: int = 0
    edit_count: int = 0
    tool_distribution: dict[str, int] = field(default_factory=dict)
    llm_iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    actor_usage_available: bool = False
    memory_prompt_tokens: int | None = 0
    memory_completion_tokens: int | None = 0
    memory_total_tokens: int | None = 0
    memory_tokens_by_role: dict[str, dict[str, int | None]] = field(default_factory=dict)
    memory_usage_available: bool = True
    memory_usage_unavailable_reason: str = ""
    system_total_tokens: int | None = None
    tokens_by_phase: dict[str, dict[str, int]] = field(default_factory=dict)
    stop_reason_distribution: dict[str, int] = field(default_factory=dict)
    no_test_finish: int = 0
    budget_stop: int = 0
    evolver_candidate_events: int = 0
    evolver_selected_events: int = 0
    evolver_candidates_total: int = 0
    evolver_selected_total: int = 0
    evolver_selected_by_tier: dict[str, int] = field(default_factory=dict)
    evolver_selected_ids: tuple[str, ...] = ()
    evolver_selection_policies: dict[str, int] = field(default_factory=dict)
    evolver_writer_started_events: int = 0
    evolver_writer_saved_events: int = 0
    evolver_writer_saved_total: int = 0
    evolver_writer_saved_by_tier: dict[str, int] = field(default_factory=dict)
    evolver_writer_failed_events: int = 0
    evolver_writer_statuses: dict[str, int] = field(default_factory=dict)
    evolver_written_ids: tuple[str, ...] = ()
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
            "llm_iterations": self.llm_iterations,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "actor_usage_available": self.actor_usage_available,
            "memory_prompt_tokens": self.memory_prompt_tokens,
            "memory_completion_tokens": self.memory_completion_tokens,
            "memory_total_tokens": self.memory_total_tokens,
            "memory_tokens_by_role": self.memory_tokens_by_role,
            "memory_usage_available": self.memory_usage_available,
            "memory_usage_unavailable_reason": self.memory_usage_unavailable_reason,
            "system_total_tokens": self.system_total_tokens,
            "tokens_by_phase": self.tokens_by_phase,
            "stop_reason_distribution": self.stop_reason_distribution,
            "no_test_finish": self.no_test_finish,
            "budget_stop": self.budget_stop,
            "evolver_candidate_events": self.evolver_candidate_events,
            "evolver_selected_events": self.evolver_selected_events,
            "evolver_candidates_total": self.evolver_candidates_total,
            "evolver_selected_total": self.evolver_selected_total,
            "evolver_selected_by_tier": self.evolver_selected_by_tier,
            "evolver_selected_ids": list(self.evolver_selected_ids),
            "evolver_selection_policies": self.evolver_selection_policies,
            "evolver_writer_started_events": self.evolver_writer_started_events,
            "evolver_writer_saved_events": self.evolver_writer_saved_events,
            "evolver_writer_saved_total": self.evolver_writer_saved_total,
            "evolver_writer_saved_by_tier": self.evolver_writer_saved_by_tier,
            "evolver_writer_failed_events": self.evolver_writer_failed_events,
            "evolver_writer_statuses": self.evolver_writer_statuses,
            "evolver_written_ids": list(self.evolver_written_ids),
            "maintenance_runs": self.maintenance_runs,
            "maintenance_applied_runs": self.maintenance_applied_runs,
            "maintenance_keep": self.maintenance_keep,
            "maintenance_delete": self.maintenance_delete,
            "maintenance_merge": self.maintenance_merge,
            "maintenance_promote": self.maintenance_promote,
            "maintenance_removed_entries": self.maintenance_removed_entries,
            "maintenance_added_entries": self.maintenance_added_entries,
            "maintenance_failures": self.maintenance_failures,
            "maintenance_committed_with_audit_error": (self.maintenance_committed_with_audit_error),
        }


@dataclass
class _EvolverEventMetrics:
    candidate_events: int = 0
    selected_events: int = 0
    candidates_total: int = 0
    selected_total: int = 0
    selected_by_tier: Counter[str] = field(default_factory=Counter)
    selected_ids: list[str] = field(default_factory=list)
    selection_policies: Counter[str] = field(default_factory=Counter)
    writer_started_events: int = 0
    writer_saved_events: int = 0
    writer_saved_total: int = 0
    writer_saved_by_tier: Counter[str] = field(default_factory=Counter)
    writer_failed_events: int = 0
    writer_statuses: Counter[str] = field(default_factory=Counter)
    written_ids: list[str] = field(default_factory=list)


def collect_trace_metrics(path: str | Path, *, recursive: bool = True) -> TraceMetrics:
    trace_files = _trace_files(Path(path), recursive=recursive)
    run_ids: set[str] = set()
    distribution: Counter[str] = Counter()
    stop_reasons: dict[str, str] = {}
    test_runs_by_run: Counter[str] = Counter()
    trace_runs: dict[Path, set[str]] = defaultdict(set)
    child_paths_by_run: dict[str, list[Path]] = defaultdict(list)
    budget_runs: set[str] = set()
    child_paths: list[Path] = []
    tool_calls = 0
    successful_tool_calls = 0
    blocked_tool_calls = 0
    test_runs = 0
    passed_test_runs = 0
    edit_count = 0
    llm_iterations = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    actor_usage_incomplete = False
    phase_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    evolver_events: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    memory_role_prompt_tokens: Counter[str] = Counter()
    memory_role_completion_tokens: Counter[str] = Counter()
    memory_role_decisions: Counter[str] = Counter()
    memory_role_unknown_events: Counter[str] = Counter()
    maintenance = MaintenanceEventCounters()

    scanned: set[Path] = set()
    pending = list(trace_files)
    while pending:
        trace_file = pending.pop(0).resolve()
        if trace_file in scanned:
            continue
        scanned.add(trace_file)
        for event in _read_trace_file(trace_file):
            run_id = event.get("run_id")
            if isinstance(run_id, str) and run_id:
                run_ids.add(run_id)
                trace_runs[trace_file].add(run_id)
            event_name = event.get("event")
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            scope = run_id if isinstance(run_id, str) and run_id else str(trace_file)
            if event_name in {
                "memory.evolver_session_started",
                "memory.evolver_task_finalized",
                "memory.evolver_candidates",
                "memory.evolver_selected",
                "memory.evolver_writer_started",
                "memory.evolver_writer_saved",
                "memory.evolver_writer_failed",
            }:
                evolver_events[scope].append((str(event_name), payload))

            if event_name == "tool.completed":
                tool = payload.get("name")
                if isinstance(tool, str) and tool:
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
                        if isinstance(run_id, str) and run_id:
                            test_runs_by_run[run_id] += 1
                        if ok:
                            passed_test_runs += 1

            if event_name == "llm.completed":
                llm_iterations += 1
                phase = str(payload.get("phase") or "unknown")
                prompt, completion, total, usage_available = _usage_values(payload.get("usage"))
                if not usage_available:
                    actor_usage_incomplete = True
                prompt_tokens += prompt
                completion_tokens += completion
                total_tokens += total
                phase_tokens[phase]["prompt_tokens"] += prompt
                phase_tokens[phase]["completion_tokens"] += completion
                phase_tokens[phase]["total_tokens"] += total
                phase_tokens[phase]["llm_iterations"] += 1

            if event_name == "opd.decision":
                role = str(payload.get("role") or "")
                if role in {"selection", "writing", "maintenance"}:
                    memory_role_decisions[role] += 1
                    prompt_count = _token_id_count(payload.get("prompt_token_ids"))
                    completion_count = _token_id_count(payload.get("completion_token_ids"))
                    if prompt_count is None or completion_count is None or prompt_count + completion_count == 0:
                        memory_role_unknown_events[role] += 1
                    else:
                        memory_role_prompt_tokens[role] += prompt_count
                        memory_role_completion_tokens[role] += completion_count

            if event_name == "budget.exceeded" and isinstance(run_id, str) and run_id:
                budget_runs.add(run_id)

            maintenance.observe(event_name, payload)

            if event_name == "agent.completed":
                reason = str(payload.get("stop_reason") or "")
                if isinstance(run_id, str) and run_id:
                    stop_reasons[run_id] = reason
                if recursive:
                    for child in payload.get("child_trace_paths") or []:
                        child_path = _resolve_child_path(trace_file, child)
                        if child_path is not None:
                            child_paths.append(child_path)
                            if isinstance(run_id, str) and run_id:
                                child_paths_by_run[run_id].append(child_path.resolve())
            elif event_name == "run.completed" and isinstance(run_id, str) and run_id and run_id not in stop_reasons:
                stop_reasons[run_id] = str(payload.get("stop_reason") or "")

        if recursive:
            pending.extend(path for path in child_paths if path.exists())
            child_paths.clear()

    evolver = _collect_evolver_event_metrics(evolver_events)
    memory_tokens_by_role: dict[str, dict[str, int | None]] = {}
    for role in sorted(memory_role_decisions):
        unknown_events = memory_role_unknown_events[role]
        prompt = memory_role_prompt_tokens[role]
        completion = memory_role_completion_tokens[role]
        memory_tokens_by_role[role] = {
            "prompt_tokens": None if unknown_events else prompt,
            "completion_tokens": None if unknown_events else completion,
            "total_tokens": None if unknown_events else prompt + completion,
            "decisions": memory_role_decisions[role],
            "unknown_events": unknown_events,
        }
    unknown_roles = sorted(role for role, count in memory_role_unknown_events.items() if count)
    memory_usage_available = not unknown_roles
    memory_usage_unavailable_reason = (
        "" if memory_usage_available else "formal memory decision token IDs unavailable for roles: " + ", ".join(unknown_roles)
    )
    memory_prompt_tokens = sum(memory_role_prompt_tokens.values()) if memory_usage_available else None
    memory_completion_tokens = sum(memory_role_completion_tokens.values()) if memory_usage_available else None
    memory_total_tokens = memory_prompt_tokens + memory_completion_tokens if memory_usage_available else None
    actor_usage_available = llm_iterations > 0 and not actor_usage_incomplete
    system_total_tokens = total_tokens + memory_total_tokens if actor_usage_available and memory_usage_available else None

    stop_distribution = Counter(reason or "unknown" for reason in stop_reasons.values())
    no_test_finish = sum(
        1
        for run_id, reason in stop_reasons.items()
        if reason in SUCCESS_STOP_REASONS
        and _tests_for_run_tree(
            run_id,
            test_runs_by_run=test_runs_by_run,
            child_paths_by_run=child_paths_by_run,
            trace_runs=trace_runs,
            recursive=recursive,
        )
        == 0
    )
    budget_stop = len(
        budget_runs.union(
            run_id
            for run_id, reason in stop_reasons.items()
            if any(marker in reason.lower() for marker in ("budget", "max_", "timeout", "timed_out"))
        )
    )

    return TraceMetrics(
        trace_files=len(scanned),
        runs=len(run_ids),
        tool_calls=tool_calls,
        successful_tool_calls=successful_tool_calls,
        blocked_tool_calls=blocked_tool_calls,
        test_runs=test_runs,
        passed_test_runs=passed_test_runs,
        edit_count=edit_count,
        tool_distribution=dict(sorted(distribution.items())),
        llm_iterations=llm_iterations,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        actor_usage_available=actor_usage_available,
        memory_prompt_tokens=memory_prompt_tokens,
        memory_completion_tokens=memory_completion_tokens,
        memory_total_tokens=memory_total_tokens,
        memory_tokens_by_role=memory_tokens_by_role,
        memory_usage_available=memory_usage_available,
        memory_usage_unavailable_reason=memory_usage_unavailable_reason,
        system_total_tokens=system_total_tokens,
        tokens_by_phase={phase: dict(values) for phase, values in sorted(phase_tokens.items())},
        stop_reason_distribution=dict(sorted(stop_distribution.items())),
        no_test_finish=no_test_finish,
        budget_stop=budget_stop,
        evolver_candidate_events=evolver.candidate_events,
        evolver_selected_events=evolver.selected_events,
        evolver_candidates_total=evolver.candidates_total,
        evolver_selected_total=evolver.selected_total,
        evolver_selected_by_tier=dict(sorted(evolver.selected_by_tier.items())),
        evolver_selected_ids=tuple(evolver.selected_ids),
        evolver_selection_policies=dict(sorted(evolver.selection_policies.items())),
        evolver_writer_started_events=evolver.writer_started_events,
        evolver_writer_saved_events=evolver.writer_saved_events,
        evolver_writer_saved_total=evolver.writer_saved_total,
        evolver_writer_saved_by_tier=dict(sorted(evolver.writer_saved_by_tier.items())),
        evolver_writer_failed_events=evolver.writer_failed_events,
        evolver_writer_statuses=dict(sorted(evolver.writer_statuses.items())),
        evolver_written_ids=tuple(evolver.written_ids),
        maintenance_runs=maintenance.runs,
        maintenance_applied_runs=maintenance.applied_runs,
        maintenance_keep=maintenance.keep,
        maintenance_delete=maintenance.delete,
        maintenance_merge=maintenance.merge,
        maintenance_promote=maintenance.promote,
        maintenance_removed_entries=maintenance.removed_entries,
        maintenance_added_entries=maintenance.added_entries,
        maintenance_failures=maintenance.failures,
        maintenance_committed_with_audit_error=maintenance.committed_with_audit_error,
    )


def format_trace_metrics(metrics: TraceMetrics) -> str:
    lines = [
        f"Trace files: {metrics.trace_files}",
        f"Runs: {metrics.runs}",
        f"Tool calls: {metrics.tool_calls}",
        (f"Tool success rate: {metrics.successful_tool_calls}/{metrics.tool_calls} ({metrics.tool_success_rate:.1%})"),
        f"Blocked tool calls: {metrics.blocked_tool_calls}",
        f"Test pass rate: {metrics.passed_test_runs}/{metrics.test_runs} ({metrics.test_pass_rate:.1%})",
        f"Edit count: {metrics.edit_count}",
        f"LLM iterations: {metrics.llm_iterations}",
        f"Tokens: prompt={metrics.prompt_tokens}, completion={metrics.completion_tokens}, total={metrics.total_tokens}",
        (
            "Memory tokens: "
            f"prompt={metrics.memory_prompt_tokens}, "
            f"completion={metrics.memory_completion_tokens}, "
            f"total={metrics.memory_total_tokens}, "
            f"available={metrics.memory_usage_available}"
        ),
        f"System total tokens: {metrics.system_total_tokens}",
        f"No-test finishes: {metrics.no_test_finish}",
        f"Budget stops: {metrics.budget_stop}",
        (
            "Evolver selection: "
            f"candidate_events={metrics.evolver_candidate_events}, "
            f"candidates={metrics.evolver_candidates_total}, "
            f"selected_events={metrics.evolver_selected_events}, "
            f"selected={metrics.evolver_selected_total}"
        ),
        (
            "Evolver writer: "
            f"started_events={metrics.evolver_writer_started_events}, "
            f"saved_events={metrics.evolver_writer_saved_events}, "
            f"saved={metrics.evolver_writer_saved_total}, "
            f"failed_events={metrics.evolver_writer_failed_events}"
        ),
        (
            "Memory maintenance: "
            f"runs={metrics.maintenance_runs}, "
            f"applied={metrics.maintenance_applied_runs}, "
            f"failures={metrics.maintenance_failures}, "
            "committed_with_audit_error="
            f"{metrics.maintenance_committed_with_audit_error}"
        ),
        (
            "Maintenance actions: "
            f"keep={metrics.maintenance_keep}, "
            f"delete={metrics.maintenance_delete}, "
            f"merge={metrics.maintenance_merge}, "
            f"promote={metrics.maintenance_promote}, "
            f"removed={metrics.maintenance_removed_entries}, "
            f"added={metrics.maintenance_added_entries}"
        ),
        "Tool distribution:",
    ]
    if metrics.tool_distribution:
        lines.extend(f"- {tool}: {count}" for tool, count in metrics.tool_distribution.items())
    else:
        lines.append("- none: 0")
    lines.append("Stop reasons:")
    if metrics.stop_reason_distribution:
        lines.extend(f"- {reason}: {count}" for reason, count in metrics.stop_reason_distribution.items())
    else:
        lines.append("- none: 0")
    lines.append("Tokens by phase:")
    if metrics.tokens_by_phase:
        for phase, values in metrics.tokens_by_phase.items():
            lines.append(
                f"- {phase}: prompt={values.get('prompt_tokens', 0)}, "
                f"completion={values.get('completion_tokens', 0)}, "
                f"total={values.get('total_tokens', 0)}, "
                f"iterations={values.get('llm_iterations', 0)}"
            )
    else:
        lines.append("- none: 0")
    lines.append("Evolver selected by tier:")
    if metrics.evolver_selected_by_tier:
        lines.extend(f"- {tier}: {count}" for tier, count in metrics.evolver_selected_by_tier.items())
    else:
        lines.append("- none: 0")
    lines.append("Evolver selection policies:")
    if metrics.evolver_selection_policies:
        lines.extend(f"- {policy}: {count}" for policy, count in metrics.evolver_selection_policies.items())
    else:
        lines.append("- none: 0")
    lines.append("Evolver writer saved by tier:")
    if metrics.evolver_writer_saved_by_tier:
        lines.extend(f"- {tier}: {count}" for tier, count in metrics.evolver_writer_saved_by_tier.items())
    else:
        lines.append("- none: 0")
    lines.append("Evolver writer statuses:")
    if metrics.evolver_writer_statuses:
        lines.extend(f"- {status}: {count}" for status, count in metrics.evolver_writer_statuses.items())
    else:
        lines.append("- none: 0")
    lines.append("Memory tokens by role:")
    if metrics.memory_tokens_by_role:
        for role, values in metrics.memory_tokens_by_role.items():
            lines.append(
                f"- {role}: prompt={values.get('prompt_tokens')}, "
                f"completion={values.get('completion_tokens')}, "
                f"total={values.get('total_tokens')}, "
                f"decisions={values.get('decisions', 0)}, "
                f"unknown={values.get('unknown_events', 0)}"
            )
    else:
        lines.append("- none: 0")
    return "\n".join(lines)


def _trace_files(path: Path, *, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        pattern = "**/*.jsonl" if recursive else "*.jsonl"
        return sorted(file for file in path.glob(pattern) if file.is_file())
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


def _resolve_child_path(parent_trace: Path, child: object) -> Path | None:
    if not isinstance(child, str) or not child.strip():
        return None
    path = Path(child)
    return path if path.is_absolute() else parent_trace.parent / path


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _usage_values(value: object) -> tuple[int, int, int, bool]:
    if not isinstance(value, dict):
        return 0, 0, 0, False
    prompt = _optional_nonnegative_int(value.get("prompt_tokens"))
    completion = _optional_nonnegative_int(value.get("completion_tokens"))
    total = _optional_nonnegative_int(value.get("total_tokens"))
    resolved_prompt = prompt or 0
    resolved_completion = completion or 0
    resolved_total = (
        total if total is not None and total > 0 else resolved_prompt + resolved_completion
    )
    # ChatUsage normalizes missing provider usage to three zero values. Treat
    # that shape as unknown rather than manufacturing a zero-cost Actor call.
    available = resolved_total > 0
    return resolved_prompt, resolved_completion, resolved_total, available


def _token_id_count(value: object) -> int | None:
    if not isinstance(value, list):
        return None
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in value):
        return None
    return len(value)


def _collect_evolver_event_metrics(
    event_groups: dict[str, list[tuple[str, dict[str, Any]]]],
) -> _EvolverEventMetrics:
    metrics = _EvolverEventMetrics()
    for events in event_groups.values():
        formal_selection = [payload for event_name, payload in events if event_name == "memory.evolver_session_started"]
        if formal_selection:
            for payload in formal_selection:
                metrics.candidate_events += 1
                metrics.selected_events += 1
                metrics.candidates_total += _payload_count(
                    payload,
                    count_key="candidate_count",
                    fallback_keys=("candidates",),
                )
                selected_ids = _string_list(payload.get("selected_memory_ids") or payload.get("selected_ids"))
                metrics.selected_total += _payload_count(
                    payload,
                    count_key="selected_count",
                    fallback_keys=("selected_memory_ids", "selected_ids"),
                )
                metrics.selected_ids.extend(selected_ids)
                metrics.selected_by_tier.update(_formal_selected_tiers(payload, selected_ids=selected_ids))
        else:
            for event_name, payload in events:
                if event_name == "memory.evolver_candidates":
                    metrics.candidate_events += 1
                    metrics.candidates_total += _payload_count(
                        payload,
                        count_key="candidate_count",
                        fallback_keys=("candidate_ids", "candidate_summaries"),
                    )
                elif event_name == "memory.evolver_selected":
                    metrics.selected_events += 1
                    metrics.selected_total += _payload_count(
                        payload,
                        count_key="selected_count",
                        fallback_keys=("selected_ids",),
                    )
                    metrics.selected_ids.extend(_string_list(payload.get("selected_ids")))
                    metrics.selected_by_tier.update(_tier_counts(payload.get("tiers")))
                    policy = payload.get("selection_policy")
                    if isinstance(policy, str) and policy:
                        metrics.selection_policies[policy] += 1

        formal_writer = [payload for event_name, payload in events if event_name == "memory.evolver_task_finalized"]
        if formal_writer:
            for payload in formal_writer:
                metrics.writer_started_events += 1
                status = payload.get("writer_status")
                if isinstance(status, str) and status:
                    metrics.writer_statuses[status] += 1
                written_ids = _string_list(payload.get("written_memory_ids"))
                metrics.written_ids.extend(written_ids)
                if status == "committed":
                    metrics.writer_saved_events += 1
                    metrics.writer_saved_total += len(written_ids)
                elif status == "failed_no_write":
                    metrics.writer_failed_events += 1
        else:
            for event_name, payload in events:
                if event_name == "memory.evolver_writer_started":
                    metrics.writer_started_events += 1
                elif event_name == "memory.evolver_writer_saved":
                    metrics.writer_saved_events += 1
                    metrics.writer_saved_total += _payload_count(
                        payload,
                        count_key="saved_count",
                        fallback_keys=("saved_ids", "saved_records"),
                    )
                    metrics.written_ids.extend(_string_list(payload.get("saved_ids")))
                    metrics.writer_saved_by_tier.update(_tier_counts(payload.get("tiers")))
                elif event_name == "memory.evolver_writer_failed":
                    metrics.writer_failed_events += 1
    return metrics


def _formal_selected_tiers(
    payload: dict[str, Any],
    *,
    selected_ids: list[str],
) -> Counter[str]:
    tiers_by_id: dict[str, str] = {}
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            memory_id = candidate.get("memory_id")
            tier = candidate.get("tier")
            if isinstance(memory_id, str) and memory_id and isinstance(tier, str) and tier:
                tiers_by_id[memory_id] = tier
    return Counter(tiers_by_id[memory_id] for memory_id in selected_ids if memory_id in tiers_by_id)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _payload_count(payload: dict[str, Any], *, count_key: str, fallback_keys: tuple[str, ...]) -> int:
    count = payload.get(count_key)
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    for key in fallback_keys:
        values = payload.get(key)
        if isinstance(values, list):
            return len(values)
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


def _tests_for_run_tree(
    run_id: str,
    *,
    test_runs_by_run: Counter[str],
    child_paths_by_run: dict[str, list[Path]],
    trace_runs: dict[Path, set[str]],
    recursive: bool,
    seen: set[str] | None = None,
) -> int:
    total = test_runs_by_run[run_id]
    if not recursive:
        return total
    visited = seen or set()
    if run_id in visited:
        return total
    visited.add(run_id)
    for child_path in child_paths_by_run.get(run_id, []):
        for child_run_id in trace_runs.get(child_path.resolve(), set()):
            total += _tests_for_run_tree(
                child_run_id,
                test_runs_by_run=test_runs_by_run,
                child_paths_by_run=child_paths_by_run,
                trace_runs=trace_runs,
                recursive=True,
                seen=visited,
            )
    return total


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
