from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
import json

import pytest

from my_agent.evaluation.memory_benchmark.contracts import (
    BenchmarkTask,
    MemoryBenchmarkTaskResult,
    ProviderUsage,
)
from my_agent.evaluation.memory_benchmark.protocol import (
    MemoryBenchmarkProtocol,
    backend_config_hash,
)
from my_agent.evaluation.memory_benchmark.reporting import (
    generate_memory_benchmark_report,
)
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256


_ARMS = ("no_memory", "agentcli_four_tier", "mem0")
_SEEDS = (42, 43, 44)
_BENCHMARKS = {"lifelong_os": "os", "intercode_bash": "bash"}


def _write_fake_run(root: Path) -> Path:
    run_dir = root / "run"
    source_lock = {"schema_version": "fixture-source-lock-v1", "sources": {}}
    tasks_by_benchmark = {
        benchmark: tuple(
            _task(benchmark, subset, index)
            for index in range(1, 41)
        )
        for benchmark, subset in _BENCHMARKS.items()
    }
    backend_configs = {
        arm: {
            "schema_version": "memory-benchmark-backend-config-v1",
            "arm": arm,
            "mode": arm,
        }
        for arm in _ARMS
    }
    backend_hashes = {
        arm: backend_config_hash(config)
        for arm, config in backend_configs.items()
    }
    protocol = MemoryBenchmarkProtocol(
        ordered_task_ids_by_benchmark={
            benchmark: tuple(task.task_id for task in tasks)
            for benchmark, tasks in tasks_by_benchmark.items()
        },
        task_manifest_hashes={
            benchmark: canonical_sha256([task.to_dict() for task in tasks])
            for benchmark, tasks in tasks_by_benchmark.items()
        },
        source_lock_hash=canonical_sha256(source_lock),
        actor_identity_hash=canonical_sha256("actor"),
        tools_hash=canonical_sha256("tools"),
        evaluator_hashes={
            benchmark: _evaluator_hash(benchmark)
            for benchmark in tasks_by_benchmark
        },
        docker_image_digests={
            benchmark: canonical_sha256(f"image:{benchmark}")
            for benchmark in tasks_by_benchmark
        },
        backend_config_hashes=backend_hashes,
        agentcli_commit="a" * 40,
        uv_lock_hash=canonical_sha256("uv.lock"),
        python_version="3.12.0",
        runtime_environment_hash=canonical_sha256("runtime"),
        repetition_ids=_SEEDS,
        agent_mode="react",
        context_window=32768,
        response_reserve_tokens=4096,
        compression_buffer_tokens=2048,
        repo_context_budget_tokens=6000,
        tool_schema_budget_tokens=8000,
        memory_short_term_tokens=16000,
        memory_context_tokens=1800,
        memory_tool_result_chars=4000,
        max_steps=40,
        command_timeout=120,
        actor_temperature=1.0,
        memory_generation_temperature=1.0,
        memory_generation_top_p=0.95,
        selected_max_items=20,
        selected_content_max_tokens=1800,
        maintenance_interval_tasks=30,
        actor_sampling_seed_supported=False,
    )
    _write_json(run_dir / "protocol.json", protocol.to_dict())
    (run_dir / "protocol_hash.txt").write_text(
        protocol.protocol_hash + "\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "source-lock.json", source_lock)
    _write_json(
        run_dir / "suite_manifest.json",
        {
            "schema_version": "memory-benchmark-prepared-suite-v1",
            "pilot": False,
            "benchmarks": {
                benchmark: {
                    "subset": _BENCHMARKS[benchmark],
                    "task_ids": [task.task_id for task in tasks],
                    "task_manifest_hash": protocol.task_manifest_hashes[benchmark],
                    "tasks": [task.to_dict() for task in tasks],
                }
                for benchmark, tasks in tasks_by_benchmark.items()
            },
        },
    )
    _write_json(
        run_dir / "preflight.json",
        {
            "schema_version": "memory-benchmark-preflight-v1",
            "status": "passed",
            "run_id": "fixture-run",
            "protocol_hash": protocol.protocol_hash,
            "pilot": False,
            "checks": {
                "context_budget": {
                    "fixed_with_memory_tokens": 12_000,
                    "synthetic_memory_tokens": 2_000,
                    "compression_trigger_tokens": 30_000,
                }
            },
        },
    )
    for arm, config in backend_configs.items():
        arm_dir = run_dir / "arms" / arm
        _write_json(arm_dir / "backend_config.json", config)
        (arm_dir / "backend_config_hash.txt").write_text(
            backend_hashes[arm] + "\n",
            encoding="utf-8",
        )
        for seed in _SEEDS:
            for benchmark, tasks in tasks_by_benchmark.items():
                rows = [
                    _result(
                        task,
                        arm=arm,
                        seed=seed,
                        protocol=protocol,
                        backend_hash=backend_hashes[arm],
                    ).to_dict()
                    for task in tasks
                ]
                _write_jsonl(
                    arm_dir / f"seed_{seed}" / benchmark / "results.jsonl",
                    rows,
                )
    return run_dir


def _task(benchmark: str, subset: str, index: int) -> BenchmarkTask:
    task_id = f"task-{index:02d}"
    return BenchmarkTask(
        benchmark=benchmark,
        subset=subset,
        task_id=task_id,
        order_index=index,
        task_group=f"{benchmark}:{subset}",
        instruction=f"Solve {benchmark} task {index}",
        split="test",
        source_revision="b" * 40,
        content_hash=canonical_sha256(
            {"benchmark": benchmark, "task_id": task_id, "index": index}
        ),
        environment_spec={"image": benchmark},
        evaluator_spec={
            "name": benchmark,
            "version": "fixture",
            "hash": _evaluator_hash(benchmark),
        },
    )


def _evaluator_hash(benchmark: str) -> str:
    return canonical_sha256(f"evaluator:{benchmark}")


def _result(
    task: BenchmarkTask,
    *,
    arm: str,
    seed: int,
    protocol: MemoryBenchmarkProtocol,
    backend_hash: str,
) -> MemoryBenchmarkTaskResult:
    resolved = _resolved(arm, task.order_index)
    memory_total = {"no_memory": 0, "agentcli_four_tier": 10, "mem0": 5}[arm]
    role_usage = {
        "no_memory": {},
        "agentcli_four_tier": {
            "selection": ProviderUsage(prompt_tokens=3, completion_tokens=2),
            "writing": ProviderUsage(prompt_tokens=3, completion_tokens=2),
        },
        "mem0": {
            "search": ProviderUsage(prompt_tokens=1, completion_tokens=1),
            "add": ProviderUsage(prompt_tokens=2, completion_tokens=1),
        },
    }[arm]
    entries_before = 0 if arm == "no_memory" else task.order_index - 1
    entries_after = 0 if arm == "no_memory" else task.order_index
    maintenance_due = arm == "agentcli_four_tier" and task.order_index == 30
    return MemoryBenchmarkTaskResult(
        run_id="fixture-run",
        seed=seed,
        actor_sampling_seed_supported=False,
        actor_sampling_seed_effective=None,
        arm=arm,
        benchmark=task.benchmark,
        subset=task.subset,
        task_id=task.task_id,
        order_index=task.order_index,
        task_content_hash=task.content_hash,
        actor_identity_hash=protocol.actor_identity_hash,
        tools_hash=protocol.tools_hash,
        evaluator_hash=protocol.evaluator_hashes[task.benchmark],
        resolved=resolved,
        reward=1.0 if resolved else 0.0,
        outcome_finalized=True,
        failure_type="" if resolved else "task_failed",
        agent_steps=10,
        tool_calls=6,
        actor_prompt_tokens=80,
        actor_completion_tokens=20,
        actor_total_tokens=100,
        memory_prompt_tokens=memory_total,
        memory_completion_tokens=0,
        memory_total_tokens=memory_total,
        memory_tokens_by_role=role_usage,
        actor_usage_available=True,
        memory_usage_available=True,
        memory_usage_unavailable_reason="",
        system_total_tokens=100 + memory_total,
        embedding_calls=0 if arm == "no_memory" else 1,
        embedding_elapsed_sec=0.0 if arm == "no_memory" else 0.1,
        elapsed_sec=4.0,
        memory={
            "candidate_count": 0 if arm == "no_memory" else 10,
            "selected_count": 0 if arm == "no_memory" else 2,
            "selected_content_tokens": 0 if arm == "no_memory" else 100,
            "injected_tokens": 0 if arm == "no_memory" else 120,
            "written_count": 0 if arm == "no_memory" else 1,
            "entries_before": entries_before,
            "entries_after": entries_after,
            "repository_bytes_after": 0 if arm == "no_memory" else entries_after * 100,
            "tier_counts_after": (
                {} if arm == "no_memory" else {"external": entries_after}
            ),
            "repository_revision_before": canonical_sha256(
                {"arm": arm, "entries": entries_before}
            ),
            "repository_revision_after": canonical_sha256(
                {"arm": arm, "entries": entries_after}
            ),
            "backend_finalize_status": "committed" if arm != "no_memory" else "no_write",
            "maintenance_status": "committed" if maintenance_due else "not_due",
            "maintenance_runs": 1 if maintenance_due else 0,
            "maintenance_applied_runs": 1 if maintenance_due else 0,
            "maintenance_failures": 0,
            "maintenance_actions": {
                "keep": 1 if maintenance_due else 0,
                "delete": 0,
                "merge": 0,
                "promote": 0,
                "removed_entries": 0,
                "added_entries": 0,
            },
        },
        trace_path=f"traces/{arm}/{seed}/{task.task_id}.jsonl",
        action_log_path=f"actions/{arm}/{seed}/{task.task_id}.jsonl",
        official_result_path=f"official/{arm}/{seed}/{task.task_id}.json",
        public_episode_path=f"episodes/{arm}/{seed}/{task.task_id}.json",
        protocol_hash=protocol.protocol_hash,
        backend_config_hash=backend_hash,
    )


def _resolved(arm: str, order_index: int) -> bool:
    if arm == "no_memory":
        return order_index <= 20
    if arm == "agentcli_four_tier":
        return order_index <= 19 or order_index in {21, 22, 23}
    return order_index <= 19 or order_index == 21


def _mutate_result(
    run_dir: Path,
    *,
    arm: str,
    seed: int = 42,
    benchmark: str = "lifelong_os",
    row_index: int = 0,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    path = run_dir / "arms" / arm / f"seed_{seed}" / benchmark / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows[row_index])
    _write_jsonl(path, rows)


def test_report_is_deterministic_and_computes_paired_results(tmp_path: Path) -> None:
    run_dir = _write_fake_run(tmp_path)

    first = generate_memory_benchmark_report(run_dir)
    first_json = (run_dir / "comparison.json").read_bytes()
    first_markdown = (run_dir / "comparison.md").read_bytes()
    second = generate_memory_benchmark_report(run_dir)

    assert first == second
    assert first_json == (run_dir / "comparison.json").read_bytes()
    assert first_markdown == (run_dir / "comparison.md").read_bytes()
    comparison = first["paired_comparisons"]["lifelong_os"][
        "agentcli_vs_no_memory"
    ]
    assert comparison["aggregate"]["helped"] == 9
    assert comparison["aggregate"]["hurt"] == 3
    assert comparison["aggregate"]["delta_resolved_mean"] == 2
    assert comparison["aggregate"]["delta_resolved_min"] == 2
    assert comparison["aggregate"]["delta_resolved_max"] == 2
    assert comparison["aggregate"]["verdict"] == "practical_improvement"
    assert comparison["post_maintenance"]["total"] == 30
    assert comparison["post_maintenance"]["delta_resolved"] == 0
    mem0_comparison = first["paired_comparisons"]["lifelong_os"][
        "mem0_vs_no_memory"
    ]
    assert mem0_comparison["aggregate"]["delta_resolved_mean"] == 0
    assert mem0_comparison["aggregate"]["verdict"] == "practical_tie"
    assert first["sampling_control"] == "unavailable"
    assert first["fairness_violations"] == []
    agent_metrics = next(
        row
        for row in first["arm_metrics"]
        if row["benchmark"] == "lifelong_os"
        and row["arm"] == "agentcli_four_tier"
        and row["seed"] == 42
    )
    assert agent_metrics["maintenance_runs"] == 1
    assert agent_metrics["intervals"][0]["success_rate"] == 1.0
    assert agent_metrics["intervals"][1]["success_rate"] == 0.9
    assert agent_metrics["intervals"][2]["success_rate"] == 0.3
    assert agent_metrics["post_maintenance"]["success_rate"] == 0.0


def test_report_rejects_identity_and_task_order_mismatch(tmp_path: Path) -> None:
    run_dir = _write_fake_run(tmp_path)
    _mutate_result(
        run_dir,
        arm="agentcli_four_tier",
        mutate=lambda row: row.__setitem__(
            "actor_identity_hash", canonical_sha256("other-actor")
        ),
    )
    with pytest.raises(ValueError, match="result identity mismatch"):
        generate_memory_benchmark_report(run_dir)

    run_dir = _write_fake_run(tmp_path / "second")
    path = (
        run_dir
        / "arms"
        / "no_memory"
        / "seed_42"
        / "lifelong_os"
        / "results.jsonl"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task order"):
        generate_memory_benchmark_report(run_dir)


def test_report_records_fairness_violations(tmp_path: Path) -> None:
    run_dir = _write_fake_run(tmp_path)

    def grow_no_memory(row: dict[str, Any]) -> None:
        row["memory"]["entries_after"] = 1
        row["memory"]["repository_bytes_after"] = 100
        row["memory"]["tier_counts_after"]["trajectory"] = 1

    def exceed_budget(row: dict[str, Any]) -> None:
        row["memory"]["candidate_count"] = 21
        row["memory"]["selected_count"] = 21
        row["memory"]["selected_content_tokens"] = 1801
        row["memory"]["injected_tokens"] = 30_000

    _mutate_result(run_dir, arm="no_memory", mutate=grow_no_memory)
    _mutate_result(run_dir, arm="agentcli_four_tier", mutate=exceed_budget)

    report = generate_memory_benchmark_report(run_dir)
    violation_types = {item["type"] for item in report["fairness_violations"]}
    assert "no_memory_repository_growth" in violation_types
    assert "selected_item_budget_exceeded" in violation_types
    assert "selected_content_budget_exceeded" in violation_types
    assert "rendered_context_budget_exceeded" in violation_types


def test_report_preserves_unknown_memory_usage(tmp_path: Path) -> None:
    run_dir = _write_fake_run(tmp_path)

    def remove_usage(row: dict[str, Any]) -> None:
        row["memory_prompt_tokens"] = None
        row["memory_completion_tokens"] = None
        row["memory_total_tokens"] = None
        row["memory_usage_available"] = False
        row["memory_usage_unavailable_reason"] = "provider usage unavailable"
        row["system_total_tokens"] = None
        row["memory_tokens_by_role"]["search"] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    _mutate_result(run_dir, arm="mem0", mutate=remove_usage)
    report = generate_memory_benchmark_report(run_dir)
    metrics = next(
        row
        for row in report["arm_metrics"]
        if row["benchmark"] == "lifelong_os"
        and row["arm"] == "mem0"
        and row["seed"] == 42
    )
    assert metrics["memory_usage_available"] is False
    assert metrics["memory_total_tokens"] is None
    assert metrics["system_total_tokens"] is None
    assert metrics["solved_per_million_system_tokens"] is None
    assert metrics["solved_per_million_unavailable_reason"] == (
        "provider usage unavailable"
    )
    assert metrics["memory_usage_unavailable_reasons"] == [
        "provider usage unavailable"
    ]


def test_report_rejects_incomplete_memory_payload(tmp_path: Path) -> None:
    run_dir = _write_fake_run(tmp_path)

    def remove_required_fields(row: dict[str, Any]) -> None:
        del row["memory"]["selected_content_tokens"]
        del row["memory"]["maintenance_actions"]

    _mutate_result(
        run_dir,
        arm="agentcli_four_tier",
        mutate=remove_required_fields,
    )

    with pytest.raises(ValueError, match="memory payload is incomplete"):
        generate_memory_benchmark_report(run_dir)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(payload)) + b"\n")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    )
