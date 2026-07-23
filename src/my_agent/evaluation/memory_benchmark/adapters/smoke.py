"""Deterministic local smoke tasks for the memory benchmark workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import argparse
import json
import os
import subprocess
import sys

from my_agent.config import AgentConfig
from my_agent.evaluation.memory_benchmark.adapters.base import execute_official_scorer
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    BenchmarkActionState,
    DockerContainer,
    DockerRuntime,
    benchmark_action_main,
    benchmark_action_tool_config,
    benchmark_action_tools_hash,
    finalize_action_log,
    prepare_runtime_action_log,
    write_benchmark_action_files,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    BenchmarkTask,
    OfficialEvaluatorResult,
    PreparedBenchmarkTask,
)
from my_agent.evaluation.memory_benchmark.backends import (
    AgentCliFourTierBackend,
    Mem0Backend,
    NoMemoryBackend,
    memory_stream_project_key,
)
from my_agent.evaluation.memory_benchmark.runner import (
    MemoryBenchmarkStreamResult,
    run_memory_benchmark_stream,
)
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256


SMOKE_SCHEMA_VERSION = "memory-benchmark-smoke-v1"
SMOKE_SOURCE_REVISION = "b" * 40
SMOKE_EVALUATOR_HASH = canonical_sha256(
    {"schema_version": SMOKE_SCHEMA_VERSION, "evaluator": "exact-files-v1"}
)
SMOKE_AGENT_INSTRUCTIONS = """# Memory benchmark smoke task

Use the `benchmark_action` tool for every shell action. Commands run inside `/workspace` in an isolated container.
Inspect the public input files, create the requested output, and do not edit `.agentcli` or `benchmark_action.py`.
The hidden evaluator runs once after your final answer.
"""


@dataclass(frozen=True)
class _SmokeTaskSpec:
    task_id: str
    family: str
    instruction: str
    input_files: Mapping[str, str]
    expected_files: Mapping[str, str]
    absent_files: tuple[str, ...] = ()


_SPECS = (
    _SmokeTaskSpec(
        "smoke-config-parse-source",
        "config_parse",
        "Read app.env and create result.json containing host as a string and port as an integer.",
        {"app.env": "host=api.internal\nport=8080\n"},
        {"result.json": '{"host":"api.internal","port":8080}\n'},
    ),
    _SmokeTaskSpec(
        "smoke-retry-source",
        "retry_backoff",
        "Read retry.json and write schedule.txt with one exponential backoff delay per line, starting at base_seconds.",
        {"retry.json": '{"attempts":4,"base_seconds":1}\n'},
        {"schedule.txt": "1\n2\n4\n8\n"},
    ),
    _SmokeTaskSpec(
        "smoke-rename-source",
        "batch_rename",
        "Rename every draft-*.tmp file to archived-*.txt while preserving each file's contents.",
        {"draft-alpha.tmp": "alpha\n", "draft-beta.tmp": "beta\n"},
        {"archived-alpha.txt": "alpha\n", "archived-beta.txt": "beta\n"},
        ("draft-alpha.tmp", "draft-beta.tmp"),
    ),
    _SmokeTaskSpec(
        "smoke-log-source",
        "log_summary",
        "Read events.log and create summary.txt with sorted service=count lines for ERROR records only.",
        {
            "events.log": (
                "INFO api started\nERROR worker timeout\nERROR api bad-request\n"
                "ERROR worker retry-exhausted\n"
            )
        },
        {"summary.txt": "api=1\nworker=2\n"},
    ),
    _SmokeTaskSpec(
        "smoke-config-parse-variant",
        "config_parse",
        "Read service.env and create parsed.json containing endpoint as a string and workers as an integer.",
        {"service.env": "endpoint=jobs.internal\nworkers=6\n"},
        {"parsed.json": '{"endpoint":"jobs.internal","workers":6}\n'},
    ),
    _SmokeTaskSpec(
        "smoke-retry-variant",
        "retry_backoff",
        "Read policy.json and write delays.txt with one exponential backoff delay per line, starting at base_seconds.",
        {"policy.json": '{"attempts":4,"base_seconds":3}\n'},
        {"delays.txt": "3\n6\n12\n24\n"},
    ),
    _SmokeTaskSpec(
        "smoke-rename-variant",
        "batch_rename",
        "Rename every image-*.raw file to processed-*.dat while preserving each file's contents.",
        {"image-one.raw": "one\n", "image-two.raw": "two\n"},
        {"processed-one.dat": "one\n", "processed-two.dat": "two\n"},
        ("image-one.raw", "image-two.raw"),
    ),
    _SmokeTaskSpec(
        "smoke-log-variant",
        "log_summary",
        "Read warnings.log and create counts.txt with sorted component=count lines for WARN records only.",
        {
            "warnings.log": (
                "WARN cache stale\nINFO api ready\nWARN api slow\nWARN cache miss\n"
            )
        },
        {"counts.txt": "api=1\ncache=2\n"},
    ),
)


class SmokeAdapter:
    """Generate eight deterministic tasks executed in a locked Docker container."""

    name = "smoke"

    def __init__(
        self,
        *,
        data_root: str | Path,
        run_id: str = "smoke",
        runtime_root: str | Path | None = None,
        container_image: str,
        container_digest: str,
        docker_runtime: DockerRuntime | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        if not str(run_id).strip():
            raise ValueError("smoke run_id must be non-empty")
        self.run_id = str(run_id)
        self.runtime_root = Path(
            runtime_root or self.data_root / ".runtime" / self.run_id
        ).expanduser().resolve()
        self.container_image = str(container_image)
        self.container_digest = str(container_digest)
        self.docker_runtime = docker_runtime or DockerRuntime()
        self._containers: dict[Path, DockerContainer] = {}

    def load_tasks(self, *, limit: int) -> Sequence[BenchmarkTask]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= len(_SPECS):
            raise ValueError(f"smoke limit must be between 1 and {len(_SPECS)}")
        tasks = tuple(_benchmark_task(index, spec) for index, spec in enumerate(_SPECS[:limit], 1))
        prepare_smoke_suite(self.data_root, tasks=tasks)
        return tasks

    def prepare_task(
        self,
        task: BenchmarkTask,
        *,
        task_dir: Path,
        seed: int,
    ) -> PreparedBenchmarkTask:
        spec = _spec_for_task(task.task_id)
        repo = task_dir / "repo"
        workspace = repo / "workspace"
        workspace.mkdir(parents=True)
        for relative_path, content in spec.input_files.items():
            target = workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        runtime_log = (
            self.runtime_root
            / f"seed_{seed}"
            / task.benchmark
            / task.task_id
            / "actions.jsonl"
        )
        prepare_runtime_action_log(runtime_log)
        action_log = task_dir / "actions.jsonl"
        adapter_state = task_dir / "adapter_state.json"
        official_result = task_dir / "official_result.json"
        public_state = repo / ".agentcli" / "benchmark_state.json"
        container: DockerContainer | None = None
        try:
            container = self.docker_runtime.create_container(
                image=self.container_image,
                expected_digest=self.container_digest,
                run_id=self.run_id,
                seed=seed,
                benchmark=task.benchmark,
                task_id=task.task_id,
                bind_mounts={workspace.resolve(): "/workspace"},
                working_directory="/workspace",
            )
            write_benchmark_action_files(
                repo,
                BenchmarkActionState(
                    container_name=container.name,
                    runtime_action_log_path=runtime_log,
                    timeout_seconds=120,
                    max_output_chars=4_000,
                ),
            )
            _write_bytes_atomic(
                repo / "AGENT.md",
                SMOKE_AGENT_INSTRUCTIONS.encode("utf-8"),
            )
            _write_json_atomic(
                adapter_state,
                {
                    "schema_version": SMOKE_SCHEMA_VERSION,
                    "task_id": task.task_id,
                    "evaluator_hash": SMOKE_EVALUATOR_HASH,
                    "workspace": str(workspace.resolve()),
                    "container_name": container.name,
                    "expected_files": dict(spec.expected_files),
                    "absent_files": list(spec.absent_files),
                },
            )
            self._containers[adapter_state.resolve()] = container
        except Exception as exc:
            if container is not None:
                try:
                    self.docker_runtime.cleanup_container(container)
                except Exception as cleanup_exc:  # noqa: BLE001 - preserve prepare cause.
                    exc.add_note(f"container cleanup also failed: {cleanup_exc}")
            raise
        module = "my_agent.evaluation.memory_benchmark.adapters.smoke"
        return PreparedBenchmarkTask(
            task=task,
            repo_path=repo,
            public_prompt=task.instruction,
            agent_test_command=None,
            initial_environment_command=(
                sys.executable,
                "-m",
                module,
                "check-ready",
                "--state",
                str(adapter_state),
            ),
            hidden_evaluator_command=(
                sys.executable,
                "-m",
                module,
                "score",
                "--state",
                str(adapter_state),
                "--result",
                str(official_result),
            ),
            env_overrides={},
            action_log_path=action_log,
            runtime_action_log_path=runtime_log,
            adapter_state_path=adapter_state,
            public_tool_state_path=public_state,
            official_result_path=official_result,
        )

    def finalize_task_artifacts(self, prepared: PreparedBenchmarkTask) -> None:
        finalize_action_log(prepared.runtime_action_log_path, prepared.action_log_path)

    def cleanup_task(self, prepared: PreparedBenchmarkTask) -> None:
        failure: BaseException | None = None
        try:
            self.finalize_task_artifacts(prepared)
        except BaseException as exc:  # preserve cleanup even when artifact finalization fails.
            failure = exc
        container = self._containers.pop(prepared.adapter_state_path.resolve(), None)
        if container is not None:
            try:
                self.docker_runtime.cleanup_container(container)
            except BaseException as exc:
                if failure is not None:
                    failure.add_note(f"container cleanup also failed: {exc}")
                else:
                    failure = exc
        if failure is not None:
            raise failure


def prepare_smoke_suite(
    data_root: str | Path,
    *,
    tasks: Sequence[BenchmarkTask] | None = None,
) -> Path:
    target = Path(data_root).expanduser().resolve() / "smoke" / "tasks.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = tuple(tasks or (_benchmark_task(index, spec) for index, spec in enumerate(_SPECS, 1)))
    payload = b"".join(canonical_json_bytes(task.to_dict()) + b"\n" for task in selected)
    _write_bytes_atomic(target, payload)
    return target


def run_smoke_benchmark(
    *,
    base_config: AgentConfig,
    output_dir: str | Path,
    data_root: str | Path,
    actor_identity_hash: str,
    mem0_config: Mapping[str, Any],
    container_image: str,
    container_digest: str,
    seed: int = 42,
    maintenance_interval_tasks: int = 4,
    docker_runtime: DockerRuntime | None = None,
) -> Mapping[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"smoke output already exists: {root}")
    root.mkdir(parents=True)
    run_id = root.name
    adapter = SmokeAdapter(
        data_root=data_root,
        run_id=run_id,
        runtime_root=root / ".runtime",
        container_image=container_image,
        container_digest=container_digest,
        docker_runtime=docker_runtime,
    )
    tasks = tuple(adapter.load_tasks(limit=8))
    protocol_hash = canonical_sha256(
        {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "task_ids": [task.task_id for task in tasks],
            "task_hashes": [task.content_hash for task in tasks],
            "seed": seed,
            "maintenance_interval_tasks": maintenance_interval_tasks,
            "container_image": container_image,
            "container_digest": container_digest,
        }
    )
    streams: dict[str, MemoryBenchmarkStreamResult] = {}
    backend_hashes: dict[str, str] = {}
    for arm in ("no_memory", "agentcli_four_tier", "mem0"):
        stream_dir = root / "arms" / arm / f"seed_{seed}" / "smoke"
        project_key = memory_stream_project_key(
            run_id=run_id,
            seed=seed,
            benchmark="smoke",
            arm=arm,
        )
        if arm == "no_memory":
            backend = NoMemoryBackend(
                stream_memory_dir=stream_dir / "memory",
                stream_project_key=project_key,
            )
            backend_config = {"arm": arm, "memory_evolver_mode": "off"}
        elif arm == "agentcli_four_tier":
            backend = AgentCliFourTierBackend(
                stream_memory_dir=stream_dir / "memory",
                stream_project_key=project_key,
                maintenance_interval_tasks=maintenance_interval_tasks,
            )
            backend_config = {
                "arm": arm,
                "memory_evolver_mode": "formal",
                "maintenance_interval_tasks": maintenance_interval_tasks,
            }
        else:
            backend = Mem0Backend(
                stream_memory_dir=stream_dir / "memory",
                stream_project_key=project_key,
                mem0_config=mem0_config,
            )
            backend_config = {
                "arm": arm,
                "memory_evolver_mode": "off",
                "mem0_config_hash": canonical_sha256(dict(mem0_config)),
            }
        backend_hash = canonical_sha256(backend_config)
        backend_hashes[arm] = backend_hash
        streams[arm] = run_memory_benchmark_stream(
            tasks=tasks,
            adapter=adapter,
            backend=backend,
            base_config=base_config,
            output_dir=stream_dir,
            run_id=run_id,
            seed=seed,
            stream_memory_dir=stream_dir / "memory",
            stream_project_key=project_key,
            protocol_hash=protocol_hash,
            actor_identity_hash=actor_identity_hash,
            tools_hash=smoke_action_tools_hash(),
            backend_config_hash=backend_hash,
            max_steps=base_config.max_steps,
            command_timeout=base_config.command_timeout,
        )

    arm_reports = {
        arm: _validate_smoke_arm(arm, stream)
        for arm, stream in streams.items()
    }
    report = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "status": (
            "passed"
            if all(item["status"] == "passed" for item in arm_reports.values())
            else "failed"
        ),
        "seed": seed,
        "task_count": len(tasks),
        "protocol_hash": protocol_hash,
        "tools_hash": smoke_action_tools_hash(),
        "actor_identity_hash": actor_identity_hash,
        "backend_config_hashes": backend_hashes,
        "arms": arm_reports,
    }
    _write_json_atomic(root / "smoke_report.json", report)
    return report


def _validate_smoke_arm(
    arm: str,
    stream: MemoryBenchmarkStreamResult,
) -> Mapping[str, Any]:
    executions = stream.executions
    checks: dict[str, bool] = {"eight_tasks_completed": len(executions) == 8}
    if arm == "no_memory":
        checks.update(
            {
                "zero_selected": all(
                    execution.task_result.memory["selected_count"] == 0
                    for execution in executions
                ),
                "zero_written": all(
                    execution.task_result.memory["written_count"] == 0
                    for execution in executions
                ),
                "zero_repository_growth": all(
                    execution.memory_after.entry_count == 0
                    and execution.memory_after.repository_bytes == 0
                    for execution in executions
                ),
            }
        )
    elif arm == "agentcli_four_tier":
        first_half = executions[:4]
        second_half = executions[4:]
        checks.update(
            {
                "starts_empty": bool(executions)
                and executions[0].memory_before.entry_count == 0,
                "writer_committed": any(
                    execution.backend_finalize.status == "committed"
                    for execution in first_half
                ),
                "prior_memory_selected": any(
                    execution.task_result.memory["selected_count"] > 0
                    for execution in second_half
                ),
                "repository_revision_continuity": all(
                    current.memory_before.revision == previous.memory_after.revision
                    for previous, current in zip(
                        executions,
                        executions[1:],
                        strict=False,
                    )
                ),
                "maintenance_ran": sum(
                    int(execution.manifest_result.metrics.get("maintenance_runs", 0) or 0)
                    for execution in executions
                )
                >= 1,
                "maintenance_failures_zero": sum(
                    int(
                        execution.manifest_result.metrics.get(
                            "maintenance_failures", 0
                        )
                        or 0
                    )
                    for execution in executions
                )
                == 0,
            }
        )
    else:
        checks.update(
            {
                "first_search_empty": bool(executions)
                and executions[0].context.candidate_count == 0,
                "first_add_succeeded": bool(executions)
                and bool(executions[0].backend_finalize.written_ids),
                "later_search_nonempty": any(
                    execution.context.candidate_count > 0
                    for execution in executions[4:]
                ),
                "selection_budget_respected": all(
                    execution.context.selected_count <= 20
                    and execution.context.selected_content_tokens <= 1_800
                    for execution in executions
                ),
            }
        )
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "results_path": str(stream.results_path),
    }


def smoke_action_tool_config() -> dict[str, Any]:
    return benchmark_action_tool_config()


def smoke_action_tools_hash() -> str:
    return benchmark_action_tools_hash()


def smoke_action_main(
    argv: Sequence[str] | None = None,
    *,
    command_runner: Callable[..., Any] | None = None,
    cwd: str | Path | None = None,
) -> int:
    kwargs: dict[str, Any] = {"cwd": cwd}
    if command_runner is not None:
        kwargs["command_runner"] = command_runner
    return benchmark_action_main(argv, **kwargs)


def smoke_cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ...smoke")
    commands = parser.add_subparsers(dest="command", required=True)
    ready = commands.add_parser("check-ready")
    ready.add_argument("--state", required=True)
    score = commands.add_parser("score")
    score.add_argument("--state", required=True)
    score.add_argument("--result", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "check-ready":
        return 0 if _smoke_environment_ready(Path(args.state)) else 2
    return execute_official_scorer(
        lambda: _score_smoke_task(Path(args.state)),
        official_result_path=args.result,
    )


def _score_smoke_task(state_path: Path) -> OfficialEvaluatorResult:
    state = _load_json_mapping(state_path)
    if state.get("schema_version") != SMOKE_SCHEMA_VERSION:
        raise ValueError("unsupported smoke adapter state")
    workspace = Path(str(state["workspace"])).resolve()
    expected = state.get("expected_files")
    absent = state.get("absent_files")
    if not isinstance(expected, Mapping) or not isinstance(absent, list):
        raise ValueError("invalid smoke evaluator state")
    resolved = all(
        (workspace / str(relative)).is_file()
        and (workspace / str(relative)).read_text(encoding="utf-8") == str(content)
        for relative, content in expected.items()
    ) and all(not (workspace / str(relative)).exists() for relative in absent)
    return OfficialEvaluatorResult(
        task_id=str(state["task_id"]),
        evaluator_hash=str(state["evaluator_hash"]),
        resolved=resolved,
        reward=1.0 if resolved else 0.0,
    )


def _smoke_environment_ready(state_path: Path) -> bool:
    state = _load_json_mapping(state_path)
    workspace = Path(str(state.get("workspace", "")))
    container_name = str(state.get("container_name", "")).strip()
    if (
        state.get("schema_version") != SMOKE_SCHEMA_VERSION
        or not workspace.is_dir()
        or not container_name
    ):
        return False
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "sh", "-lc", "test -d /workspace"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return int(result.returncode) == 0


def _benchmark_task(index: int, spec: _SmokeTaskSpec) -> BenchmarkTask:
    public_source = {
        "task_id": spec.task_id,
        "family": spec.family,
        "instruction": spec.instruction,
        "input_files": dict(spec.input_files),
    }
    return BenchmarkTask(
        benchmark="smoke",
        subset="memory",
        task_id=spec.task_id,
        order_index=index,
        task_group="smoke:memory",
        instruction=spec.instruction,
        split="test",
        source_revision=SMOKE_SOURCE_REVISION,
        content_hash=canonical_sha256(public_source),
        environment_spec={"family": spec.family, "input_files": sorted(spec.input_files)},
        evaluator_spec={
            "name": "agentcli-memory-smoke",
            "version": SMOKE_SCHEMA_VERSION,
            "hash": SMOKE_EVALUATOR_HASH,
        },
        tags=("smoke", spec.family, "source" if index <= 4 else "variant"),
    )


def _spec_for_task(task_id: str) -> _SmokeTaskSpec:
    for spec in _SPECS:
        if spec.task_id == task_id:
            return spec
    raise ValueError(f"unknown smoke task: {task_id}")


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_atomic(path, canonical_json_bytes(dict(payload)) + b"\n")


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


if __name__ == "__main__":
    raise SystemExit(smoke_cli_main())


__all__ = [
    "SMOKE_EVALUATOR_HASH",
    "SMOKE_SCHEMA_VERSION",
    "SmokeAdapter",
    "prepare_smoke_suite",
    "run_smoke_benchmark",
    "smoke_action_main",
    "smoke_action_tool_config",
    "smoke_action_tools_hash",
    "smoke_cli_main",
]
