from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EDIT_TOOLS = {"replace_in_file", "write_file"}
SUCCESS_STOP_REASONS = {"finish_called", "assistant_final", "plan_completed", "team_completed"}


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
    tokens_by_phase: dict[str, dict[str, int]] = field(default_factory=dict)
    stop_reason_distribution: dict[str, int] = field(default_factory=dict)
    no_test_finish: int = 0
    budget_stop: int = 0
    evolver_candidate_events: int = 0
    evolver_selected_events: int = 0
    evolver_candidates_total: int = 0
    evolver_selected_total: int = 0
    evolver_selected_by_tier: dict[str, int] = field(default_factory=dict)
    evolver_selection_policies: dict[str, int] = field(default_factory=dict)

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
            "tokens_by_phase": self.tokens_by_phase,
            "stop_reason_distribution": self.stop_reason_distribution,
            "no_test_finish": self.no_test_finish,
            "budget_stop": self.budget_stop,
            "evolver_candidate_events": self.evolver_candidate_events,
            "evolver_selected_events": self.evolver_selected_events,
            "evolver_candidates_total": self.evolver_candidates_total,
            "evolver_selected_total": self.evolver_selected_total,
            "evolver_selected_by_tier": self.evolver_selected_by_tier,
            "evolver_selection_policies": self.evolver_selection_policies,
        }


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
    phase_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    evolver_candidate_events = 0
    evolver_selected_events = 0
    evolver_candidates_total = 0
    evolver_selected_total = 0
    evolver_selected_by_tier: Counter[str] = Counter()
    evolver_selection_policies: Counter[str] = Counter()

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
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    prompt = _nonnegative_int(usage.get("prompt_tokens"))
                    completion = _nonnegative_int(usage.get("completion_tokens"))
                    total = _nonnegative_int(usage.get("total_tokens")) or prompt + completion
                    prompt_tokens += prompt
                    completion_tokens += completion
                    total_tokens += total
                    phase_tokens[phase]["prompt_tokens"] += prompt
                    phase_tokens[phase]["completion_tokens"] += completion
                    phase_tokens[phase]["total_tokens"] += total
                phase_tokens[phase]["llm_iterations"] += 1

            if event_name == "budget.exceeded" and isinstance(run_id, str) and run_id:
                budget_runs.add(run_id)

            if event_name == "memory.evolver_candidates":
                evolver_candidate_events += 1
                evolver_candidates_total += _payload_count(
                    payload,
                    count_key="candidate_count",
                    fallback_keys=("candidate_ids", "candidate_summaries"),
                )
            elif event_name == "memory.evolver_selected":
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
        tokens_by_phase={
            phase: dict(values)
            for phase, values in sorted(phase_tokens.items())
        },
        stop_reason_distribution=dict(sorted(stop_distribution.items())),
        no_test_finish=no_test_finish,
        budget_stop=budget_stop,
        evolver_candidate_events=evolver_candidate_events,
        evolver_selected_events=evolver_selected_events,
        evolver_candidates_total=evolver_candidates_total,
        evolver_selected_total=evolver_selected_total,
        evolver_selected_by_tier=dict(sorted(evolver_selected_by_tier.items())),
        evolver_selection_policies=dict(sorted(evolver_selection_policies.items())),
    )


def format_trace_metrics(metrics: TraceMetrics) -> str:
    lines = [
        f"Trace files: {metrics.trace_files}",
        f"Runs: {metrics.runs}",
        f"Tool calls: {metrics.tool_calls}",
        (
            "Tool success rate: "
            f"{metrics.successful_tool_calls}/{metrics.tool_calls} ({metrics.tool_success_rate:.1%})"
        ),
        f"Blocked tool calls: {metrics.blocked_tool_calls}",
        f"Test pass rate: {metrics.passed_test_runs}/{metrics.test_runs} ({metrics.test_pass_rate:.1%})",
        f"Edit count: {metrics.edit_count}",
        f"LLM iterations: {metrics.llm_iterations}",
        f"Tokens: prompt={metrics.prompt_tokens}, completion={metrics.completion_tokens}, total={metrics.total_tokens}",
        f"No-test finishes: {metrics.no_test_finish}",
        f"Budget stops: {metrics.budget_stop}",
        (
            "Evolver selection: "
            f"candidate_events={metrics.evolver_candidate_events}, "
            f"candidates={metrics.evolver_candidates_total}, "
            f"selected_events={metrics.evolver_selected_events}, "
            f"selected={metrics.evolver_selected_total}"
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


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


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
