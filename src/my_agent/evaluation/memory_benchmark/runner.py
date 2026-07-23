"""Ordered task runner shared by memory benchmark arms."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import os
import re

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import (
    ManifestBenchmarkResult,
    ManifestEvalResult,
    ManifestInfrastructureError,
    run_manifest_benchmark,
)
from my_agent.evaluation.memory_benchmark.adapters.base import BenchmarkAdapter
from my_agent.evaluation.memory_benchmark.backends import (
    MemoryBenchmarkBackend,
    memory_stream_project_key,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    BackendFinalizeResult,
    BenchmarkTask,
    MemoryBenchmarkTaskResult,
    MemoryContextSelection,
    MemoryRepositorySnapshot,
    PreparedBenchmarkTask,
    PublicEpisode,
)
from my_agent.policy.identity import canonical_json_bytes, require_sha256


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_INFRASTRUCTURE_FAILURE_TYPES = frozenset(
    {
        "environment_setup_failed",
        "agent_infrastructure_failed",
        "evaluator_error",
        "evolver_finalize_failed",
    }
)

ManifestRunnerFn = Callable[..., ManifestBenchmarkResult]


@dataclass(frozen=True)
class MemoryBenchmarkTaskExecution:
    task: BenchmarkTask
    prepared: PreparedBenchmarkTask
    context: MemoryContextSelection
    manifest_result: ManifestEvalResult
    backend_finalize: BackendFinalizeResult
    memory_before: MemoryRepositorySnapshot
    memory_after: MemoryRepositorySnapshot
    public_episode_path: Path
    task_result: MemoryBenchmarkTaskResult


@dataclass(frozen=True)
class MemoryBenchmarkStreamResult:
    arm: str
    seed: int
    benchmark: str
    output_dir: Path
    results_path: Path
    executions: tuple[MemoryBenchmarkTaskExecution, ...]


class MemoryBenchmarkInfrastructureError(RuntimeError):
    """Raised after preserving every available task diagnostic artifact."""

    def __init__(
        self,
        task: BenchmarkTask,
        failures: Sequence[tuple[str, BaseException]],
        *,
        manifest_result: ManifestEvalResult | None = None,
    ) -> None:
        self.task = task
        self.failures = tuple(failures)
        self.manifest_result = manifest_result
        details = "; ".join(
            f"{stage}={type(error).__name__}: {error}" for stage, error in self.failures
        )
        super().__init__(f"memory benchmark task {task.task_id!r} aborted: {details}")


def run_memory_benchmark_stream(
    *,
    tasks: Sequence[BenchmarkTask],
    adapter: BenchmarkAdapter,
    backend: MemoryBenchmarkBackend,
    base_config: AgentConfig,
    output_dir: str | Path,
    run_id: str,
    seed: int,
    stream_memory_dir: str | Path,
    stream_project_key: str,
    protocol_hash: str,
    actor_identity_hash: str,
    tools_hash: str,
    backend_config_hash: str,
    actor_sampling_seed_supported: bool = False,
    actor_sampling_seed_effective: int | None = None,
    max_steps: int | None = None,
    command_timeout: int | None = None,
    manifest_runner: ManifestRunnerFn = run_manifest_benchmark,
) -> MemoryBenchmarkStreamResult:
    ordered_tasks = _validate_ordered_tasks(tasks)
    if not ordered_tasks:
        raise ValueError("memory benchmark stream requires at least one task")
    if not str(run_id).strip():
        raise ValueError("run_id must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    require_sha256(protocol_hash, field_name="protocol_hash")
    require_sha256(actor_identity_hash, field_name="actor_identity_hash")
    require_sha256(tools_hash, field_name="tools_hash")
    require_sha256(backend_config_hash, field_name="backend_config_hash")
    if not isinstance(actor_sampling_seed_supported, bool):
        raise ValueError("actor_sampling_seed_supported must be a bool")
    if actor_sampling_seed_effective is not None and (
        isinstance(actor_sampling_seed_effective, bool)
        or not isinstance(actor_sampling_seed_effective, int)
        or actor_sampling_seed_effective < 0
    ):
        raise ValueError("actor_sampling_seed_effective must be a non-negative integer")
    if not actor_sampling_seed_supported and actor_sampling_seed_effective is not None:
        raise ValueError("unsupported actor sampling seed cannot be effective")
    stream_dir = Path(output_dir).expanduser().resolve()
    if stream_dir.exists():
        raise FileExistsError(f"memory benchmark stream output already exists: {stream_dir}")
    memory_dir = Path(stream_memory_dir).expanduser().resolve()
    if memory_dir != stream_dir / "memory":
        raise ValueError("stream_memory_dir must be the stream output memory directory")
    if memory_dir.exists() and any(memory_dir.iterdir()):
        raise FileExistsError(f"stream memory directory is not empty: {memory_dir}")
    expected_project_key = memory_stream_project_key(
        run_id=run_id,
        seed=seed,
        benchmark=ordered_tasks[0].benchmark,
        arm=backend.name,
    )
    if stream_project_key != expected_project_key:
        raise ValueError("stream_project_key does not match run/seed/benchmark/arm isolation")
    stream_dir.mkdir(parents=True)
    task_root = stream_dir / "tasks"
    task_root.mkdir()
    results_path = stream_dir / "results.jsonl"
    results_path.touch(exist_ok=False)
    executions: list[MemoryBenchmarkTaskExecution] = []
    stream_failure: BaseException | None = None
    try:
        for task in ordered_tasks:
            task_dir = task_root / f"{task.order_index:04d}_{_safe_id(task.task_id)}"
            task_dir.mkdir()
            try:
                prepared = adapter.prepare_task(task, task_dir=task_dir, seed=seed)
            except Exception as exc:  # noqa: BLE001 - adapter owns partial-resource cleanup.
                raise MemoryBenchmarkInfrastructureError(
                    task,
                    (("adapter_prepare", exc),),
                ) from exc
            execution = _execute_prepared_task(
                expected_task=task,
                prepared=prepared,
                adapter=adapter,
                backend=backend,
                base_config=base_config,
                task_dir=task_dir,
                stream_memory_dir=memory_dir,
                stream_project_key=stream_project_key,
                protocol_hash=protocol_hash,
                run_id=run_id,
                seed=seed,
                actor_identity_hash=actor_identity_hash,
                tools_hash=tools_hash,
                backend_config_hash=backend_config_hash,
                actor_sampling_seed_supported=actor_sampling_seed_supported,
                actor_sampling_seed_effective=actor_sampling_seed_effective,
                max_steps=max_steps,
                command_timeout=command_timeout,
                manifest_runner=manifest_runner,
            )
            try:
                _append_task_result(results_path, execution.task_result)
            except Exception as exc:  # noqa: BLE001 - result persistence is infrastructure.
                raise MemoryBenchmarkInfrastructureError(
                    task,
                    (("result_persist", exc),),
                    manifest_result=execution.manifest_result,
                ) from exc
            executions.append(execution)
    except BaseException as exc:  # preserve the primary cause while still closing the backend.
        stream_failure = exc
    finally:
        try:
            backend.close()
        except Exception as close_exc:  # noqa: BLE001 - close failure is infrastructure failure.
            if stream_failure is not None:
                stream_failure.add_note(f"backend close also failed: {close_exc}")
            else:
                stream_failure = close_exc
    if stream_failure is not None:
        raise stream_failure
    return MemoryBenchmarkStreamResult(
        arm=backend.name,
        seed=seed,
        benchmark=ordered_tasks[0].benchmark,
        output_dir=stream_dir,
        results_path=results_path,
        executions=tuple(executions),
    )


def _execute_prepared_task(
    *,
    expected_task: BenchmarkTask,
    prepared: PreparedBenchmarkTask,
    adapter: BenchmarkAdapter,
    backend: MemoryBenchmarkBackend,
    base_config: AgentConfig,
    task_dir: Path,
    stream_memory_dir: Path,
    stream_project_key: str,
    protocol_hash: str,
    run_id: str,
    seed: int,
    actor_identity_hash: str,
    tools_hash: str,
    backend_config_hash: str,
    actor_sampling_seed_supported: bool,
    actor_sampling_seed_effective: int | None,
    max_steps: int | None,
    command_timeout: int | None,
    manifest_runner: ManifestRunnerFn,
) -> MemoryBenchmarkTaskExecution:
    task = expected_task
    failures: list[tuple[str, BaseException]] = []
    manifest_result: ManifestEvalResult | None = None
    execution: MemoryBenchmarkTaskExecution | None = None
    artifacts_finalized = False
    try:
        if prepared.task != task:
            raise ValueError("adapter returned a PreparedBenchmarkTask for the wrong task")
        _validate_prepared_paths(prepared, task_dir=task_dir)
        if prepared.agent_test_command is not None:
            raise ValueError("external-state benchmark tasks must not expose agent_test_command")
        memory_before = backend.snapshot()
        context = backend.prepare_context(task)
        task_config = backend.configure_task(
            base_config,
            stream_memory_dir=stream_memory_dir,
            stream_project_key=stream_project_key,
            context=context,
        )
        _validate_benchmark_task_config(task_config, backend_name=backend.name)
        derived_manifest_path = task_dir / "derived_manifest.json"
        _write_json_atomic(
            derived_manifest_path,
            _derived_manifest_payload(
                prepared,
                stream_memory_dir=stream_memory_dir,
                stream_project_key=stream_project_key,
                protocol_hash=protocol_hash,
            ),
        )
        try:
            manifest = manifest_runner(
                tasks_path=derived_manifest_path,
                output_dir=task_dir / "eval",
                config=task_config,
                mode="react",
                max_steps=max_steps,
                command_timeout=command_timeout,
                agent_runner=backend.build_agent_runner(context=context),
            )
        except ManifestInfrastructureError as exc:
            manifest_result = exc.result
            raise RuntimeError(str(exc)) from exc
        if len(manifest.results) != 1:
            raise RuntimeError("single-task derived manifest must return exactly one result")
        manifest_result = manifest.results[0]
        _validate_manifest_result(task, manifest_result)
        adapter.finalize_task_artifacts(prepared)
        artifacts_finalized = True
        episode = _build_public_episode(prepared, manifest_result)
        public_episode_path = task_dir / "public_episode.json"
        _write_json_atomic(public_episode_path, episode.to_dict())
        backend_finalize = backend.finalize_task(episode, manifest_result)
        memory_after = backend.snapshot()
        if backend.name == "no_memory" and (
            memory_after.entry_count != 0
            or memory_after.revision != memory_before.revision
            or memory_after.repository_bytes != memory_before.repository_bytes
            or backend_finalize.written_ids
            or _no_memory_trace_activity(manifest_result.metrics)
        ):
            raise RuntimeError("No Memory backend produced persistent experience growth")
        task_result = _build_task_result(
            task=task,
            prepared=prepared,
            manifest_result=manifest_result,
            context=context,
            backend_finalize=backend_finalize,
            memory_before=memory_before,
            memory_after=memory_after,
            public_episode_path=public_episode_path,
            run_id=run_id,
            seed=seed,
            arm=backend.name,
            protocol_hash=protocol_hash,
            actor_identity_hash=actor_identity_hash,
            tools_hash=tools_hash,
            backend_config_hash=backend_config_hash,
            actor_sampling_seed_supported=actor_sampling_seed_supported,
            actor_sampling_seed_effective=actor_sampling_seed_effective,
        )
        execution = MemoryBenchmarkTaskExecution(
            task=task,
            prepared=prepared,
            context=context,
            manifest_result=manifest_result,
            backend_finalize=backend_finalize,
            memory_before=memory_before,
            memory_after=memory_after,
            public_episode_path=public_episode_path,
            task_result=task_result,
        )
    except Exception as exc:  # noqa: BLE001 - cleanup and artifacts must still run.
        failures.append(("task_execution", exc))
    finally:
        if not artifacts_finalized:
            try:
                adapter.finalize_task_artifacts(prepared)
            except Exception as exc:  # noqa: BLE001 - retain action logs on every path.
                failures.append(("artifact_finalize", exc))
        try:
            adapter.cleanup_task(prepared)
        except Exception as exc:  # noqa: BLE001 - cleanup failure aborts the stream.
            failures.append(("adapter_cleanup", exc))
    if failures:
        raise MemoryBenchmarkInfrastructureError(
            task,
            failures,
            manifest_result=manifest_result,
        ) from failures[0][1]
    if execution is None:  # pragma: no cover - defensive invariant.
        raise RuntimeError("task execution completed without a result")
    return execution


def _derived_manifest_payload(
    prepared: PreparedBenchmarkTask,
    *,
    stream_memory_dir: Path,
    stream_project_key: str,
    protocol_hash: str,
) -> dict[str, Any]:
    task = prepared.task
    evaluator_name = _required_spec_string(task.evaluator_spec, "name")
    evaluator_version = _required_spec_string(task.evaluator_spec, "version")
    evaluator_hash = _required_spec_string(task.evaluator_spec, "hash")
    require_sha256(evaluator_hash, field_name="evaluator_hash")
    env_overrides = {str(key): str(value) for key, value in prepared.env_overrides.items()}
    env_overrides.update(
        {
            "AGENTCLI_MEMORY_DIR": str(stream_memory_dir),
            "AGENTCLI_MEMORY_PROJECT_KEY": stream_project_key,
        }
    )
    return {
        "memory_mode": "shared_stream",
        "stream_id": f"{task.benchmark}:{task.subset}",
        "tasks": [
            {
                "id": task.task_id,
                "task_group": task.task_group,
                "source": task.benchmark,
                "repo": str(prepared.repo_path.resolve()),
                "task": prepared.public_prompt,
                "evaluation_kind": "external_state",
                "initial_environment_command": list(prepared.initial_environment_command),
                "agent_test_command": None,
                "hidden_test_command": list(prepared.hidden_evaluator_command),
                "official_result_path": str(prepared.official_result_path.resolve()),
                "evaluator_name": evaluator_name,
                "evaluator_version": evaluator_version,
                "evaluator_hash": evaluator_hash,
                "env_overrides": env_overrides,
                "tags": list(task.tags),
                "split": task.split,
                "protocol_hash": protocol_hash,
            }
        ],
    }


def _validate_manifest_result(task: BenchmarkTask, result: ManifestEvalResult) -> None:
    if result.task_id != task.task_id:
        raise RuntimeError("manifest result task_id does not match ordered benchmark task")
    if result.evaluation_kind != "external_state":
        raise RuntimeError("memory benchmark requires external_state manifest results")
    expected_hash = _required_spec_string(task.evaluator_spec, "hash")
    if result.evaluator_hash != expected_hash:
        raise RuntimeError("manifest result evaluator_hash mismatch")
    if not result.outcome_finalized or result.failure_type in _INFRASTRUCTURE_FAILURE_TYPES:
        detail = result.failure_type or "outcome_not_finalized"
        raise RuntimeError(f"manifest infrastructure failure: {detail}: {result.error}")


def _build_public_episode(
    prepared: PreparedBenchmarkTask,
    result: ManifestEvalResult,
) -> PublicEpisode:
    actions: list[Mapping[str, Any]] = []
    if not prepared.action_log_path.exists():
        raise FileNotFoundError(f"final action log is missing: {prepared.action_log_path}")
    for line_number, line in enumerate(
        prepared.action_log_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"action log line {line_number} must be a JSON object")
        actions.append(dict(payload))
    return PublicEpisode(
        task_id=prepared.task.task_id,
        instruction=prepared.public_prompt,
        actions=tuple(actions),
        final_response=result.agent_final_answer,
        resolved=result.resolved,
        reward=result.reward,
        failure_type=result.failure_type,
    )


def _validate_benchmark_task_config(config: AgentConfig, *, backend_name: str) -> None:
    if not config.memory_enabled:
        raise RuntimeError("memory benchmark task must keep MemoryManager enabled")
    if not config.enable_project_tools or config.tool_config_paths:
        raise RuntimeError("memory benchmark task must use only project benchmark tools")
    if config.enable_project_plugins or config.mcp_enabled or config.hitl_enabled:
        raise RuntimeError("plugins, MCP, and HITL must be disabled for memory benchmark")
    expected_mode = "formal" if backend_name == "agentcli_four_tier" else "off"
    if config.memory_evolver_mode != expected_mode:
        raise RuntimeError(
            f"backend {backend_name!r} requires memory_evolver_mode={expected_mode!r}"
        )


def _validate_prepared_paths(prepared: PreparedBenchmarkTask, *, task_dir: Path) -> None:
    root = task_dir.resolve()
    repo = prepared.repo_path.resolve()
    if not repo.is_dir() or not repo.is_relative_to(root):
        raise ValueError("prepared benchmark repo must be a directory inside task output")
    for field_name in (
        "action_log_path",
        "adapter_state_path",
        "official_result_path",
    ):
        path = getattr(prepared, field_name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{field_name} must remain inside task output")
    if prepared.runtime_action_log_path.resolve().is_relative_to(repo):
        raise ValueError("runtime_action_log_path must remain outside the prepared repo")
    expected_public_state = repo / ".agentcli" / "benchmark_state.json"
    if prepared.public_tool_state_path.resolve() != expected_public_state:
        raise ValueError("public_tool_state_path must use .agentcli/benchmark_state.json")


def _no_memory_trace_activity(metrics: Mapping[str, Any]) -> bool:
    keys = (
        "evolver_candidate_events",
        "evolver_selected_events",
        "evolver_candidates_total",
        "evolver_selected_total",
        "evolver_writer_started_events",
        "evolver_writer_saved_events",
        "evolver_writer_saved_total",
    )
    return any(int(metrics.get(key, 0) or 0) != 0 for key in keys)


def _build_task_result(
    *,
    task: BenchmarkTask,
    prepared: PreparedBenchmarkTask,
    manifest_result: ManifestEvalResult,
    context: MemoryContextSelection,
    backend_finalize: BackendFinalizeResult,
    memory_before: MemoryRepositorySnapshot,
    memory_after: MemoryRepositorySnapshot,
    public_episode_path: Path,
    run_id: str,
    seed: int,
    arm: str,
    protocol_hash: str,
    actor_identity_hash: str,
    tools_hash: str,
    backend_config_hash: str,
    actor_sampling_seed_supported: bool,
    actor_sampling_seed_effective: int | None,
) -> MemoryBenchmarkTaskResult:
    metrics = manifest_result.metrics
    raw_actor_prompt = int(metrics.get("prompt_tokens", 0) or 0)
    raw_actor_completion = int(metrics.get("completion_tokens", 0) or 0)
    raw_actor_total = int(metrics.get("total_tokens", 0) or 0)
    actor_usage_available = (
        int(metrics.get("llm_iterations", 0) or 0) > 0
        and raw_actor_total > 0
    )
    actor_prompt_tokens = (
        raw_actor_prompt if actor_usage_available else None
    )
    actor_completion_tokens = (
        raw_actor_completion if actor_usage_available else None
    )
    actor_total_tokens = (
        raw_actor_total if actor_usage_available else None
    )
    if arm == "no_memory":
        memory_prompt_tokens = 0
        memory_completion_tokens = 0
        memory_total_tokens = 0
        memory_usage_available = True
        memory_usage_unavailable_reason = ""
    else:
        memory_prompt_tokens = None
        memory_completion_tokens = None
        memory_total_tokens = None
        memory_usage_available = False
        memory_usage_unavailable_reason = "formal memory decision usage is not available"
    system_total_tokens = (
        actor_total_tokens + memory_total_tokens
        if actor_usage_available and memory_usage_available
        else None
    )
    return MemoryBenchmarkTaskResult(
        run_id=run_id,
        seed=seed,
        actor_sampling_seed_supported=actor_sampling_seed_supported,
        actor_sampling_seed_effective=actor_sampling_seed_effective,
        arm=arm,
        benchmark=task.benchmark,
        subset=task.subset,
        task_id=task.task_id,
        order_index=task.order_index,
        task_content_hash=task.content_hash,
        actor_identity_hash=actor_identity_hash,
        tools_hash=tools_hash,
        evaluator_hash=manifest_result.evaluator_hash,
        resolved=manifest_result.resolved,
        reward=manifest_result.reward,
        outcome_finalized=manifest_result.outcome_finalized,
        failure_type=manifest_result.failure_type,
        agent_steps=manifest_result.agent_steps,
        tool_calls=int(metrics.get("tool_calls", 0) or 0),
        actor_prompt_tokens=actor_prompt_tokens,
        actor_completion_tokens=actor_completion_tokens,
        actor_total_tokens=actor_total_tokens,
        memory_prompt_tokens=memory_prompt_tokens,
        memory_completion_tokens=memory_completion_tokens,
        memory_total_tokens=memory_total_tokens,
        memory_tokens_by_role={},
        actor_usage_available=actor_usage_available,
        memory_usage_available=memory_usage_available,
        memory_usage_unavailable_reason=memory_usage_unavailable_reason,
        system_total_tokens=system_total_tokens,
        embedding_calls=backend_finalize.embedding_calls,
        embedding_elapsed_sec=0.0,
        elapsed_sec=manifest_result.elapsed_sec,
        memory={
            "candidate_count": int(metrics.get("evolver_candidates_total", 0) or 0),
            "selected_count": int(metrics.get("evolver_selected_total", 0) or 0),
            "selected_content_tokens": context.selected_content_tokens,
            "injected_tokens": context.estimated_tokens,
            "written_count": len(backend_finalize.written_ids),
            "entries_before": memory_before.entry_count,
            "entries_after": memory_after.entry_count,
            "repository_bytes_after": memory_after.repository_bytes,
            "tier_counts_after": dict(memory_after.tier_counts),
            "repository_revision_before": memory_before.revision,
            "repository_revision_after": memory_after.revision,
            "backend_finalize_status": backend_finalize.status,
            "maintenance_status": manifest_result.evolver_maintenance_status or "not_due",
        },
        trace_path=manifest_result.trace_path,
        action_log_path=str(prepared.action_log_path),
        official_result_path=str(prepared.official_result_path),
        public_episode_path=str(public_episode_path),
        protocol_hash=protocol_hash,
        backend_config_hash=backend_config_hash,
    )


def _validate_ordered_tasks(tasks: Sequence[BenchmarkTask]) -> tuple[BenchmarkTask, ...]:
    ordered = tuple(tasks)
    if any(not isinstance(task, BenchmarkTask) for task in ordered):
        raise ValueError("tasks must contain only BenchmarkTask values")
    if not ordered:
        return ordered
    benchmark = ordered[0].benchmark
    subset = ordered[0].subset
    task_ids = [task.task_id for task in ordered]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("benchmark stream task IDs must be unique")
    if any(task.benchmark != benchmark or task.subset != subset for task in ordered):
        raise ValueError("one memory stream cannot mix benchmark/subset identities")
    if [task.order_index for task in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("benchmark tasks must be passed in consecutive order_index order")
    return ordered


def _required_spec_string(spec: Mapping[str, Any], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"evaluator_spec requires non-empty {key}")
    return value.strip()


def _safe_id(value: str) -> str:
    normalized = _SAFE_ID_RE.sub("_", str(value)).strip("_.-")
    return normalized[:120] or "task"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(payload)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _append_task_result(path: Path, result: MemoryBenchmarkTaskResult) -> None:
    payload = canonical_json_bytes(result.to_dict()) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short append to {path}: wrote {written} of {len(payload)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "MemoryBenchmarkInfrastructureError",
    "MemoryBenchmarkStreamResult",
    "MemoryBenchmarkTaskExecution",
    "run_memory_benchmark_stream",
]
