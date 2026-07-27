"""Deterministic aggregation and paired comparison for memory benchmark runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
import json
import os

from my_agent.evaluation.memory_benchmark.contracts import MemoryBenchmarkTaskResult
from my_agent.evaluation.memory_benchmark.protocol import (
    MemoryBenchmarkProtocol,
    backend_config_hash,
)
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256


COMPARISON_SCHEMA_VERSION = "memory-benchmark-comparison-v1"
_ARM_ORDER = ("no_memory", "agentcli_four_tier", "mem0")
_INTERVALS = ((1, 10), (11, 20), (21, 30), (31, 40))


@dataclass(frozen=True)
class _RunArtifacts:
    run_dir: Path
    run_id: str
    protocol: MemoryBenchmarkProtocol
    preflight: Mapping[str, Any]
    suite: Mapping[str, Any]
    backend_configs: Mapping[str, Mapping[str, Any]]
    results: Mapping[tuple[str, int, str], tuple[MemoryBenchmarkTaskResult, ...]]


def generate_memory_benchmark_report(run_dir: str | Path) -> dict[str, Any]:
    """Validate one immutable run and atomically write JSON/Markdown reports."""

    artifacts = _load_run_artifacts(Path(run_dir).expanduser().resolve())
    report = _build_report(artifacts)
    _write_bytes_atomic(
        artifacts.run_dir / "comparison.json",
        canonical_json_bytes(report) + b"\n",
    )
    _write_bytes_atomic(
        artifacts.run_dir / "comparison.md",
        _render_markdown(report).encode("utf-8"),
    )
    return report


def _load_run_artifacts(run_dir: Path) -> _RunArtifacts:
    protocol = MemoryBenchmarkProtocol.from_dict(_load_mapping(run_dir / "protocol.json"))
    recorded_protocol_hash = (run_dir / "protocol_hash.txt").read_text(
        encoding="utf-8"
    ).strip()
    if recorded_protocol_hash != protocol.protocol_hash:
        raise ValueError("protocol hash file does not match protocol.json")
    preflight = _load_mapping(run_dir / "preflight.json")
    if preflight.get("schema_version") != "memory-benchmark-preflight-v2":
        raise ValueError("unsupported memory benchmark preflight schema")
    if preflight.get("status") != "passed":
        raise ValueError("memory benchmark preflight has not passed")
    if preflight.get("protocol_hash") != protocol.protocol_hash:
        raise ValueError("preflight protocol hash does not match protocol.json")
    if bool(preflight.get("pilot")) != protocol.pilot:
        raise ValueError("preflight pilot declaration does not match protocol")
    run_id = _required_string(preflight.get("run_id"), "preflight run_id")

    source_lock = _load_mapping(run_dir / "source-lock.json")
    if canonical_sha256(source_lock) != protocol.source_lock_hash:
        raise ValueError("source lock snapshot does not match protocol")
    suite = _load_mapping(run_dir / "suite_manifest.json")
    suite_metadata = _validate_suite_manifest(suite, protocol)

    backend_configs: dict[str, Mapping[str, Any]] = {}
    for arm, expected_hash in protocol.backend_config_hashes.items():
        config_path = run_dir / "arms" / arm / "backend_config.json"
        config = _load_mapping(config_path)
        actual_hash = backend_config_hash(config)
        recorded_hash = (config_path.parent / "backend_config_hash.txt").read_text(
            encoding="utf-8"
        ).strip()
        if actual_hash != recorded_hash or actual_hash != expected_hash:
            raise ValueError(f"backend config hash mismatch for arm {arm}")
        backend_configs[arm] = config

    missing_arms = sorted(set(_ARM_ORDER) - set(backend_configs))
    if missing_arms:
        raise ValueError(
            "comparison report requires all three memory benchmark arms: "
            + ", ".join(missing_arms)
        )

    expected_arms = tuple(arm for arm in _ARM_ORDER if arm in backend_configs)
    results: dict[
        tuple[str, int, str], tuple[MemoryBenchmarkTaskResult, ...]
    ] = {}
    for arm in expected_arms:
        for seed in protocol.repetition_ids:
            for benchmark in protocol.ordered_task_ids_by_benchmark:
                path = (
                    run_dir
                    / "arms"
                    / arm
                    / f"seed_{seed}"
                    / benchmark
                    / "results.jsonl"
                )
                rows = _load_results(path)
                _validate_result_stream(
                    rows,
                    arm=arm,
                    seed=seed,
                    benchmark=benchmark,
                    run_id=run_id,
                    protocol=protocol,
                    task_metadata=suite_metadata[benchmark],
                )
                results[(arm, seed, benchmark)] = rows
    return _RunArtifacts(
        run_dir=run_dir,
        run_id=run_id,
        protocol=protocol,
        preflight=preflight,
        suite=suite,
        backend_configs=backend_configs,
        results=results,
    )


def _validate_suite_manifest(
    suite: Mapping[str, Any],
    protocol: MemoryBenchmarkProtocol,
) -> dict[str, dict[str, Any]]:
    if suite.get("schema_version") != "memory-benchmark-prepared-suite-v1":
        raise ValueError("unsupported memory benchmark suite schema")
    if bool(suite.get("pilot")) != protocol.pilot:
        raise ValueError("suite pilot declaration does not match protocol")
    benchmarks = _mapping(suite.get("benchmarks"), "suite benchmarks")
    if set(benchmarks) != set(protocol.ordered_task_ids_by_benchmark):
        raise ValueError("suite benchmark set does not match protocol")
    metadata: dict[str, dict[str, Any]] = {}
    for benchmark, expected_ids in protocol.ordered_task_ids_by_benchmark.items():
        payload = _mapping(benchmarks.get(benchmark), f"suite benchmark {benchmark}")
        _required_string(payload.get("subset"), f"{benchmark} subset")
        task_ids = _string_sequence(payload.get("task_ids"), f"{benchmark} task_ids")
        if task_ids != expected_ids:
            raise ValueError(f"suite task order does not match protocol for {benchmark}")
        if payload.get("task_manifest_hash") != protocol.task_manifest_hashes[benchmark]:
            raise ValueError(f"suite task manifest hash mismatch for {benchmark}")
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
            raise ValueError(f"suite tasks must be an array for {benchmark}")
        tasks = tuple(_mapping(item, f"suite task {benchmark}") for item in raw_tasks)
        if tuple(str(task.get("task_id")) for task in tasks) != expected_ids:
            raise ValueError(f"suite task payload order mismatch for {benchmark}")
        if canonical_sha256([dict(task) for task in tasks]) != protocol.task_manifest_hashes[
            benchmark
        ]:
            raise ValueError(f"suite task payload hash mismatch for {benchmark}")
        metadata[benchmark] = {
            "task_hashes": {
                str(task["task_id"]): _required_string(
                    task.get("content_hash"),
                    f"{benchmark} task content_hash",
                )
                for task in tasks
            },
            "task_subsets": {
                str(task["task_id"]): _required_string(
                    task.get("subset"),
                    f"{benchmark} task subset",
                )
                for task in tasks
            },
        }
    return metadata


def _load_results(path: Path) -> tuple[MemoryBenchmarkTaskResult, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"memory benchmark result stream is missing: {path}")
    rows: list[MemoryBenchmarkTaskResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid result JSONL at {path}:{line_number}") from exc
        rows.append(
            MemoryBenchmarkTaskResult.from_dict(
                _mapping(payload, f"result row {path}:{line_number}")
            )
        )
    if not rows:
        raise ValueError(f"memory benchmark result stream is empty: {path}")
    return tuple(rows)


def _validate_result_stream(
    rows: Sequence[MemoryBenchmarkTaskResult],
    *,
    arm: str,
    seed: int,
    benchmark: str,
    run_id: str,
    protocol: MemoryBenchmarkProtocol,
    task_metadata: Mapping[str, Any],
) -> None:
    expected_ids = protocol.ordered_task_ids_by_benchmark[benchmark]
    if tuple(row.task_id for row in rows) != expected_ids:
        raise ValueError(f"result task order does not match protocol for {arm}/{seed}/{benchmark}")
    if tuple(row.order_index for row in rows) != tuple(range(1, len(rows) + 1)):
        raise ValueError(f"result order_index is invalid for {arm}/{seed}/{benchmark}")
    task_hashes = _mapping(task_metadata.get("task_hashes"), "task hashes")
    task_subsets = _mapping(task_metadata.get("task_subsets"), "task subsets")
    expected_effective_seed = seed if protocol.actor_sampling_seed_supported else None
    for row in rows:
        checks = {
            "run_id": (row.run_id, run_id),
            "arm": (row.arm, arm),
            "seed": (row.seed, seed),
            "benchmark": (row.benchmark, benchmark),
            "subset": (row.subset, task_subsets[row.task_id]),
            "protocol_hash": (row.protocol_hash, protocol.protocol_hash),
            "backend_config_hash": (
                row.backend_config_hash,
                protocol.backend_config_hashes[arm],
            ),
            "actor_identity_hash": (
                row.actor_identity_hash,
                protocol.actor_identity_hash,
            ),
            "tools_hash": (row.tools_hash, protocol.tools_hash),
            "evaluator_hash": (
                row.evaluator_hash,
                protocol.evaluator_hashes[benchmark],
            ),
            "task_content_hash": (
                row.task_content_hash,
                task_hashes[row.task_id],
            ),
            "actor_sampling_seed_supported": (
                row.actor_sampling_seed_supported,
                protocol.actor_sampling_seed_supported,
            ),
            "actor_sampling_seed_effective": (
                row.actor_sampling_seed_effective,
                expected_effective_seed,
            ),
        }
        mismatched = [name for name, values in checks.items() if values[0] != values[1]]
        if mismatched:
            raise ValueError(
                f"result identity mismatch for {arm}/{seed}/{benchmark}/{row.task_id}: "
                + ", ".join(mismatched)
            )
        if not row.outcome_finalized:
            raise ValueError(f"unfinalized task result cannot be reported: {row.task_id}")
        _validate_memory_payload(row)


def _validate_memory_payload(row: MemoryBenchmarkTaskResult) -> None:
    required_ints = (
        "candidate_count",
        "selected_count",
        "selected_content_tokens",
        "injected_tokens",
        "written_count",
        "entries_before",
        "entries_after",
        "repository_bytes_after",
        "maintenance_runs",
        "maintenance_applied_runs",
        "maintenance_failures",
    )
    missing = [field_name for field_name in required_ints if field_name not in row.memory]
    required_objects = (
        "tier_counts_after",
        "maintenance_actions",
        "backend_finalize_status",
        "maintenance_status",
        "repository_revision_before",
        "repository_revision_after",
    )
    missing.extend(
        field_name for field_name in required_objects if field_name not in row.memory
    )
    if missing:
        raise ValueError(
            f"task result memory payload is incomplete for {row.task_id}: "
            + ", ".join(sorted(missing))
        )
    for field_name in required_ints:
        _non_negative_int(row.memory[field_name], f"memory.{field_name}")
    if _memory_int(row, "selected_count") > _memory_int(row, "candidate_count"):
        raise ValueError(f"selected_count exceeds candidate_count for {row.task_id}")
    tier_counts = _mapping(row.memory["tier_counts_after"], "tier_counts_after")
    normalized_tiers = {
        str(tier): _non_negative_int(count, f"tier_counts_after.{tier}")
        for tier, count in tier_counts.items()
    }
    if sum(normalized_tiers.values()) != _memory_int(row, "entries_after"):
        raise ValueError(f"tier_counts_after does not sum to entries_after for {row.task_id}")
    actions = _mapping(row.memory["maintenance_actions"], "maintenance_actions")
    expected_actions = {
        "keep",
        "delete",
        "merge",
        "promote",
        "removed_entries",
        "added_entries",
    }
    if set(actions) != expected_actions:
        raise ValueError(f"maintenance_actions fields are invalid for {row.task_id}")
    for name, value in actions.items():
        _non_negative_int(value, f"maintenance_actions.{name}")
    for field_name in (
        "backend_finalize_status",
        "maintenance_status",
        "repository_revision_before",
        "repository_revision_after",
    ):
        _required_string(row.memory[field_name], f"memory.{field_name}")


def _build_report(artifacts: _RunArtifacts) -> dict[str, Any]:
    protocol = artifacts.protocol
    arm_metrics = [
        _aggregate_stream(rows)
        for (_arm, _seed, _benchmark), rows in sorted(
            artifacts.results.items(),
            key=lambda item: (
                item[0][2],
                _ARM_ORDER.index(item[0][0]),
                item[0][1],
            ),
        )
    ]
    paired = {
        benchmark: _benchmark_paired_comparisons(artifacts, benchmark)
        for benchmark in protocol.ordered_task_ids_by_benchmark
    }
    fairness = _fairness_violations(artifacts)
    combined = _combined_arm_summaries(artifacts, paired)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "completed",
        "run_id": artifacts.run_id,
        "pilot": protocol.pilot,
        "protocol_hash": protocol.protocol_hash,
        "repetition_ids": list(protocol.repetition_ids),
        "actor_sampling_seed_supported": protocol.actor_sampling_seed_supported,
        "sampling_control": (
            "effective" if protocol.actor_sampling_seed_supported else "unavailable"
        ),
        "benchmarks": list(protocol.ordered_task_ids_by_benchmark),
        "arms": [arm for arm in _ARM_ORDER if arm in protocol.backend_config_hashes],
        "backend_config_hashes": dict(protocol.backend_config_hashes),
        "backend_configs": {
            arm: dict(config)
            for arm, config in sorted(artifacts.backend_configs.items())
        },
        "arm_metrics": arm_metrics,
        "benchmark_arm_summary": combined,
        "paired_comparisons": paired,
        "fairness_violations": fairness,
        "infrastructure_failures": [],
        "conclusions": {
            benchmark: {
                name: comparison["aggregate"]["verdict"]
                for name, comparison in paired[benchmark].items()
            }
            for benchmark in paired
        },
    }


def _aggregate_stream(
    rows: Sequence[MemoryBenchmarkTaskResult],
) -> dict[str, Any]:
    total = len(rows)
    resolved = sum(row.resolved for row in rows)
    actor_available = all(row.actor_usage_available for row in rows)
    memory_available = all(row.memory_usage_available for row in rows)
    system_available = all(row.system_total_tokens is not None for row in rows)
    actor_prompt = _sum_optional(rows, "actor_prompt_tokens", actor_available)
    actor_completion = _sum_optional(rows, "actor_completion_tokens", actor_available)
    actor_total = _sum_optional(rows, "actor_total_tokens", actor_available)
    memory_prompt = _sum_optional(rows, "memory_prompt_tokens", memory_available)
    memory_completion = _sum_optional(rows, "memory_completion_tokens", memory_available)
    memory_total = _sum_optional(rows, "memory_total_tokens", memory_available)
    system_total = _sum_optional(rows, "system_total_tokens", system_available)
    final_memory = rows[-1].memory
    roles = sorted({role for row in rows for role in row.memory_tokens_by_role})
    usage_by_role: dict[str, dict[str, Any]] = {}
    for role in roles:
        usages = [row.memory_tokens_by_role.get(role) for row in rows]
        available = all(usage is None or usage.available for usage in usages)
        usage_by_role[role] = {
            "available": available,
            "prompt_tokens": (
                sum(usage.prompt_tokens or 0 for usage in usages if usage is not None)
                if available
                else None
            ),
            "completion_tokens": (
                sum(usage.completion_tokens or 0 for usage in usages if usage is not None)
                if available
                else None
            ),
            "total_tokens": (
                sum(usage.resolved_total_tokens or 0 for usage in usages if usage is not None)
                if available
                else None
            ),
        }
    maintenance_actions: Counter[str] = Counter()
    for row in rows:
        raw_actions = row.memory.get("maintenance_actions", {})
        if isinstance(raw_actions, Mapping):
            for name, value in raw_actions.items():
                maintenance_actions[str(name)] += _non_negative_int(
                    value,
                    f"maintenance action {name}",
                )
    unavailable_reasons = sorted(
        {
            row.memory_usage_unavailable_reason
            for row in rows
            if not row.memory_usage_available and row.memory_usage_unavailable_reason
        }
    )
    actor_unavailable_count = sum(not row.actor_usage_available for row in rows)
    solved_per_million: float | None = None
    solved_per_million_reason = ""
    if system_total is not None and system_total > 0:
        solved_per_million = resolved * 1_000_000 / system_total
    elif not actor_available:
        solved_per_million_reason = (
            f"actor usage unavailable for {actor_unavailable_count} task result(s)"
        )
    elif not memory_available:
        solved_per_million_reason = "; ".join(unavailable_reasons) or (
            "memory usage unavailable for one or more task results"
        )
    else:
        solved_per_million_reason = "system token total is zero"
    return {
        "benchmark": rows[0].benchmark,
        "arm": rows[0].arm,
        "seed": rows[0].seed,
        "total": total,
        "scored": total,
        "resolved": resolved,
        "success_rate": resolved / total,
        "actor_prompt_tokens": actor_prompt,
        "actor_completion_tokens": actor_completion,
        "actor_total_tokens": actor_total,
        "memory_prompt_tokens": memory_prompt,
        "memory_completion_tokens": memory_completion,
        "memory_total_tokens": memory_total,
        "memory_tokens_by_role": usage_by_role,
        "system_total_tokens": system_total,
        "actor_usage_available": actor_available,
        "actor_usage_unavailable_reason": (
            ""
            if actor_available
            else f"actor usage unavailable for {actor_unavailable_count} task result(s)"
        ),
        "memory_usage_available": memory_available,
        "memory_usage_unavailable_reasons": unavailable_reasons,
        "solved_per_million_system_tokens": solved_per_million,
        "solved_per_million_unavailable_reason": solved_per_million_reason,
        "average_steps": sum(row.agent_steps for row in rows) / total,
        "average_tool_calls": sum(row.tool_calls for row in rows) / total,
        "average_elapsed_sec": sum(row.elapsed_sec for row in rows) / total,
        "embedding_calls": sum(row.embedding_calls for row in rows),
        "embedding_elapsed_sec": sum(row.embedding_elapsed_sec for row in rows),
        "memory_candidate_total": sum(
            _memory_int(row, "candidate_count") for row in rows
        ),
        "memory_selected_total": sum(
            _memory_int(row, "selected_count") for row in rows
        ),
        "memory_written_total": sum(
            _memory_int(row, "written_count") for row in rows
        ),
        "repository_final_entries": _memory_int(rows[-1], "entries_after"),
        "repository_final_bytes": _memory_int(rows[-1], "repository_bytes_after"),
        "tier_counts_final": dict(
            _mapping(final_memory.get("tier_counts_after", {}), "tier_counts_after")
        ),
        "maintenance_runs": sum(
            _memory_int(row, "maintenance_runs") for row in rows
        ),
        "maintenance_failures": sum(
            _memory_int(row, "maintenance_failures") for row in rows
        ),
        "maintenance_actions": dict(sorted(maintenance_actions.items())),
        "mem0_search_calls": total if rows[0].arm == "mem0" else 0,
        "mem0_add_calls": total if rows[0].arm == "mem0" else 0,
        "intervals": _interval_metrics(rows),
        "post_maintenance": _range_metrics(rows, 31, 40),
        "protocol_hash": rows[0].protocol_hash,
        "backend_config_hash": rows[0].backend_config_hash,
        "actor_sampling_seed_supported": rows[0].actor_sampling_seed_supported,
        "actor_sampling_seed_effective": rows[0].actor_sampling_seed_effective,
    }


def _benchmark_paired_comparisons(
    artifacts: _RunArtifacts,
    benchmark: str,
) -> dict[str, dict[str, Any]]:
    return {
        "agentcli_vs_no_memory": _paired_comparison(
            artifacts,
            benchmark=benchmark,
            candidate_arm="agentcli_four_tier",
            baseline_arm="no_memory",
            win_label="helped",
            loss_label="hurt",
        ),
        "mem0_vs_no_memory": _paired_comparison(
            artifacts,
            benchmark=benchmark,
            candidate_arm="mem0",
            baseline_arm="no_memory",
            win_label="helped",
            loss_label="hurt",
        ),
        "agentcli_vs_mem0": _paired_comparison(
            artifacts,
            benchmark=benchmark,
            candidate_arm="agentcli_four_tier",
            baseline_arm="mem0",
            win_label="agentcli_wins",
            loss_label="mem0_wins",
        ),
    }


def _paired_comparison(
    artifacts: _RunArtifacts,
    *,
    benchmark: str,
    candidate_arm: str,
    baseline_arm: str,
    win_label: str,
    loss_label: str,
) -> dict[str, Any]:
    if candidate_arm not in artifacts.protocol.backend_config_hashes:
        raise ValueError(f"comparison arm is unavailable: {candidate_arm}")
    if baseline_arm not in artifacts.protocol.backend_config_hashes:
        raise ValueError(f"comparison arm is unavailable: {baseline_arm}")
    per_seed: list[dict[str, Any]] = []
    task_details: list[dict[str, Any]] = []
    post_counts: Counter[str] = Counter()
    post_candidate_resolved = 0
    post_baseline_resolved = 0
    post_total = 0
    aggregate_counts: Counter[str] = Counter()
    deltas: list[int] = []
    for seed in artifacts.protocol.repetition_ids:
        candidate_rows = artifacts.results[(candidate_arm, seed, benchmark)]
        baseline_rows = artifacts.results[(baseline_arm, seed, benchmark)]
        counts: Counter[str] = Counter()
        for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
            if candidate.task_id != baseline.task_id:
                raise ValueError("paired comparison task IDs do not align")
            if candidate.resolved and not baseline.resolved:
                outcome = win_label
            elif baseline.resolved and not candidate.resolved:
                outcome = loss_label
            elif candidate.resolved:
                outcome = "both_pass"
            else:
                outcome = "both_fail"
            counts[outcome] += 1
            aggregate_counts[outcome] += 1
            if 31 <= candidate.order_index <= 40:
                post_counts[outcome] += 1
                post_total += 1
                post_candidate_resolved += int(candidate.resolved)
                post_baseline_resolved += int(baseline.resolved)
            task_details.append(
                {
                    "seed": seed,
                    "task_id": candidate.task_id,
                    "order_index": candidate.order_index,
                    "candidate_resolved": candidate.resolved,
                    "baseline_resolved": baseline.resolved,
                    "outcome": outcome,
                }
            )
        candidate_resolved = sum(row.resolved for row in candidate_rows)
        baseline_resolved = sum(row.resolved for row in baseline_rows)
        delta = candidate_resolved - baseline_resolved
        deltas.append(delta)
        per_seed.append(
            {
                "seed": seed,
                "candidate_resolved": candidate_resolved,
                "baseline_resolved": baseline_resolved,
                "delta_resolved": delta,
                "success_rate_delta": delta / len(candidate_rows),
                "classification": _stream_classification(delta),
                **dict(sorted(counts.items())),
            }
        )
    task_total = sum(len(artifacts.results[(candidate_arm, seed, benchmark)]) for seed in artifacts.protocol.repetition_ids)
    aggregate = {
        "candidate_arm": candidate_arm,
        "baseline_arm": baseline_arm,
        **dict(sorted(aggregate_counts.items())),
        "success_rate_delta": sum(deltas) / task_total,
        "delta_resolved_mean": mean(deltas),
        "delta_resolved_min": min(deltas),
        "delta_resolved_max": max(deltas),
        "verdict": _aggregate_verdict(deltas, pilot=artifacts.protocol.pilot),
    }
    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
        "post_maintenance": {
            "total": post_total,
            "candidate_resolved": post_candidate_resolved,
            "baseline_resolved": post_baseline_resolved,
            "delta_resolved": post_candidate_resolved - post_baseline_resolved,
            **dict(sorted(post_counts.items())),
        },
        "task_details": task_details,
    }


def _combined_arm_summaries(
    artifacts: _RunArtifacts,
    paired: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for benchmark in artifacts.protocol.ordered_task_ids_by_benchmark:
        for arm in _ARM_ORDER:
            if arm not in artifacts.protocol.backend_config_hashes:
                continue
            rows = tuple(
                row
                for seed in artifacts.protocol.repetition_ids
                for row in artifacts.results[(arm, seed, benchmark)]
            )
            actor_available = all(row.actor_usage_available for row in rows)
            memory_available = all(row.memory_usage_available for row in rows)
            system_available = all(row.system_total_tokens is not None for row in rows)
            helped: int | None = None
            hurt: int | None = None
            if arm == "agentcli_four_tier":
                aggregate = paired[benchmark]["agentcli_vs_no_memory"]["aggregate"]
                helped = int(aggregate.get("helped", 0))
                hurt = int(aggregate.get("hurt", 0))
            elif arm == "mem0":
                aggregate = paired[benchmark]["mem0_vs_no_memory"]["aggregate"]
                helped = int(aggregate.get("helped", 0))
                hurt = int(aggregate.get("hurt", 0))
            summaries.append(
                {
                    "benchmark": benchmark,
                    "arm": arm,
                    "total": len(rows),
                    "resolved": sum(row.resolved for row in rows),
                    "success_rate": sum(row.resolved for row in rows) / len(rows),
                    "helped": helped,
                    "hurt": hurt,
                    "actor_tokens_per_task": _per_task_optional(
                        rows,
                        "actor_total_tokens",
                        actor_available,
                    ),
                    "memory_tokens_per_task": _per_task_optional(
                        rows,
                        "memory_total_tokens",
                        memory_available,
                    ),
                    "system_tokens_per_task": _per_task_optional(
                        rows,
                        "system_total_tokens",
                        system_available,
                    ),
                    "steps_per_task": sum(row.agent_steps for row in rows) / len(rows),
                    "tool_calls_per_task": sum(row.tool_calls for row in rows) / len(rows),
                    "elapsed_sec_per_task": sum(row.elapsed_sec for row in rows) / len(rows),
                    "average_final_memory_entries": mean(
                        _memory_int(
                            artifacts.results[(arm, seed, benchmark)][-1],
                            "entries_after",
                        )
                        for seed in artifacts.protocol.repetition_ids
                    ),
                    "usage_complete": actor_available and memory_available,
                }
            )
    return summaries


def _fairness_violations(artifacts: _RunArtifacts) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    context = _mapping(
        _mapping(artifacts.preflight.get("checks"), "preflight checks").get(
            "context_budget"
        ),
        "preflight context budget",
    )
    fixed_tokens = _non_negative_int(
        context.get("fixed_with_memory_tokens"),
        "fixed_with_memory_tokens",
    )
    trigger_tokens = _non_negative_int(
        context.get("compression_trigger_tokens"),
        "compression_trigger_tokens",
    )
    synthetic_memory_tokens = _non_negative_int(
        context.get("synthetic_memory_tokens"),
        "synthetic_memory_tokens",
    )
    fixed_base_tokens = fixed_tokens - synthetic_memory_tokens
    if fixed_base_tokens < 0:
        raise ValueError("preflight synthetic memory tokens exceed fixed context tokens")
    if fixed_tokens >= trigger_tokens:
        violations.append(
            {
                "type": "context_budget_invalid",
                "detail": "preflight fixed context is not below the compression trigger",
            }
        )
    for (arm, seed, benchmark), rows in artifacts.results.items():
        for row in rows:
            selected = _memory_int(row, "selected_count")
            selected_tokens = _memory_int(row, "selected_content_tokens")
            injected_tokens = _memory_int(row, "injected_tokens")
            identity = {
                "arm": arm,
                "seed": seed,
                "benchmark": benchmark,
                "task_id": row.task_id,
            }
            if selected > artifacts.protocol.selected_max_items:
                violations.append(
                    {
                        "type": "selected_item_budget_exceeded",
                        **identity,
                        "actual": selected,
                        "limit": artifacts.protocol.selected_max_items,
                    }
                )
            if selected_tokens > artifacts.protocol.selected_content_max_tokens:
                violations.append(
                    {
                        "type": "selected_content_budget_exceeded",
                        **identity,
                        "actual": selected_tokens,
                        "limit": artifacts.protocol.selected_content_max_tokens,
                    }
                )
            actual_fixed_tokens = fixed_base_tokens + injected_tokens
            if actual_fixed_tokens >= trigger_tokens:
                violations.append(
                    {
                        "type": "rendered_context_budget_exceeded",
                        **identity,
                        "actual": actual_fixed_tokens,
                        "limit": trigger_tokens - 1,
                    }
                )
            if arm == "no_memory" and any(
                (
                    _memory_int(row, "candidate_count"),
                    selected,
                    _memory_int(row, "written_count"),
                    _memory_int(row, "entries_before"),
                    _memory_int(row, "entries_after"),
                    _memory_int(row, "repository_bytes_after"),
                )
            ):
                violations.append({"type": "no_memory_repository_growth", **identity})
    return violations


def _interval_metrics(
    rows: Sequence[MemoryBenchmarkTaskResult],
) -> list[dict[str, Any]]:
    return [_range_metrics(rows, start, end) for start, end in _INTERVALS]


def _range_metrics(
    rows: Sequence[MemoryBenchmarkTaskResult],
    start: int,
    end: int,
) -> dict[str, Any]:
    selected = [row for row in rows if start <= row.order_index <= end]
    return {
        "range": f"{start}-{end}",
        "total": len(selected),
        "resolved": sum(row.resolved for row in selected),
        "success_rate": (
            sum(row.resolved for row in selected) / len(selected) if selected else None
        ),
    }


def _stream_classification(delta: int) -> str:
    if abs(delta) <= 1:
        return "practical_tie"
    return "improvement" if delta > 0 else "regression"


def _aggregate_verdict(deltas: Sequence[int], *, pilot: bool) -> str:
    if pilot:
        return "pilot_only"
    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    average = mean(deltas)
    majority = len(deltas) // 2 + 1
    if positive >= majority and average >= 2:
        return "practical_improvement"
    if negative >= majority and average <= -2:
        return "practical_regression"
    if all(abs(delta) <= 1 for delta in deltas):
        return "practical_tie"
    return "inconclusive"


def _sum_optional(
    rows: Sequence[MemoryBenchmarkTaskResult],
    field_name: str,
    available: bool,
) -> int | None:
    if not available:
        return None
    values = [getattr(row, field_name) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values if value is not None)


def _per_task_optional(
    rows: Sequence[MemoryBenchmarkTaskResult],
    field_name: str,
    available: bool,
) -> float | None:
    total = _sum_optional(rows, field_name, available)
    return total / len(rows) if total is not None else None


def _memory_int(row: MemoryBenchmarkTaskResult, field_name: str) -> int:
    if field_name not in row.memory:
        raise ValueError(f"memory.{field_name} is required")
    return _non_negative_int(row.memory[field_name], f"memory.{field_name}")


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# AgentCli Memory Benchmark Comparison",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Protocol: `{report['protocol_hash']}`",
        f"- Pilot: `{str(report['pilot']).lower()}`",
        f"- Sampling control: `{report['sampling_control']}`",
        "",
        "## Summary",
        "",
        "| Benchmark | Arm | Success | Helped | Hurt | Actor tokens/task | Memory tokens/task | System tokens/task | Steps/task | Memory entries |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["benchmark_arm_summary"]:
        lines.append(
            "| {benchmark} | {arm} | {resolved}/{total} ({rate}) | {helped} | {hurt} | "
            "{actor} | {memory} | {system} | {steps} | {entries} |".format(
                benchmark=row["benchmark"],
                arm=row["arm"],
                resolved=row["resolved"],
                total=row["total"],
                rate=_format_rate(row["success_rate"]),
                helped=_format_optional(row["helped"]),
                hurt=_format_optional(row["hurt"]),
                actor=_format_optional(row["actor_tokens_per_task"], digits=1),
                memory=_format_optional(row["memory_tokens_per_task"], digits=1),
                system=_format_optional(row["system_tokens_per_task"], digits=1),
                steps=_format_optional(row["steps_per_task"], digits=2),
                entries=_format_optional(row["average_final_memory_entries"], digits=1),
            )
        )
    lines.extend(["", "## Paired comparisons", ""])
    for benchmark, comparisons in report["paired_comparisons"].items():
        lines.append(f"### {benchmark}")
        lines.append("")
        lines.append("| Comparison | Mean delta | Min | Max | Verdict |")
        lines.append("|---|---:|---:|---:|---|")
        for name, comparison in comparisons.items():
            aggregate = comparison["aggregate"]
            lines.append(
                f"| {name} | {aggregate['delta_resolved_mean']:.2f} | "
                f"{aggregate['delta_resolved_min']} | {aggregate['delta_resolved_max']} | "
                f"{aggregate['verdict']} |"
            )
        lines.append("")
    lines.extend(
        [
            "Full seed/task paired details are available in `comparison.json`.",
            "",
            "## Learning curves",
            "",
            "| Benchmark | Arm | Seed | 1-10 | 11-20 | 21-30 | 31-40 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metrics in report["arm_metrics"]:
        intervals = metrics["intervals"]
        lines.append(
            f"| {metrics['benchmark']} | {metrics['arm']} | {metrics['seed']} | "
            + " | ".join(_format_rate(item["success_rate"]) for item in intervals)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Post-maintenance tasks 31-40",
            "",
            "| Benchmark | Arm | Seed | Resolved | Success rate | Maintenance runs | Failures |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metrics in report["arm_metrics"]:
        post = metrics["post_maintenance"]
        lines.append(
            f"| {metrics['benchmark']} | {metrics['arm']} | {metrics['seed']} | "
            f"{post['resolved']}/{post['total']} | {_format_rate(post['success_rate'])} | "
            f"{metrics['maintenance_runs']} | {metrics['maintenance_failures']} |"
        )
    lines.extend(["", "## Fairness", ""])
    violations = report["fairness_violations"]
    if violations:
        for violation in violations:
            lines.append(f"- `{violation['type']}`: `{json.dumps(violation, sort_keys=True)}`")
    else:
        lines.append("No fairness violations detected.")
    lines.append("")
    if report["sampling_control"] == "unavailable":
        lines.append(
            "Sampling control is unavailable; Helped/Hurt values are observational "
            "system-level differences across independent repetitions."
        )
        lines.append("")
    return "\n".join(lines)


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _format_optional(value: Any, *, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    return _mapping(payload, str(path))


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


def _string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array")
    result = tuple(_required_string(item, field_name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} contains duplicates")
    return result


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["generate_memory_benchmark_report"]
