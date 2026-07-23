"""LifelongAgentBench OS adapter using the locked official task semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import ast
import hashlib
import json
import os
import sys

from my_agent.evaluation.memory_benchmark.adapters.base import execute_official_scorer
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    BenchmarkActionState,
    DockerContainer,
    DockerRuntime,
    execute_container_command,
    finalize_action_log,
    prepare_runtime_action_log,
    write_benchmark_action_files,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    BenchmarkTask,
    OfficialEvaluatorResult,
    PreparedBenchmarkTask,
)
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256, require_sha256


LIFELONG_OS_SCHEMA_VERSION = "memory-benchmark-lifelong-os-v1"
LIFELONG_OS_EVALUATOR_NAME = "lifelong-agent-bench-os"


@dataclass(frozen=True)
class _OfficialOSTask:
    task_id: str
    source_index: int
    instruction: str
    initialization_script: str
    evaluation_script: str
    skills: tuple[str, ...]
    content_hash: str


class LifelongOSAdapter:
    """Prepare one fresh official OS container per benchmark task."""

    name = "lifelong_os"

    def __init__(
        self,
        *,
        task_data_path: str | Path,
        source: Mapping[str, Any],
        run_id: str,
        runtime_root: str | Path,
        command_timeout_seconds: int = 120,
        max_output_chars: int = 4_000,
        docker_runtime: DockerRuntime | None = None,
    ) -> None:
        if not str(run_id).strip():
            raise ValueError("Lifelong OS run_id must be non-empty")
        self.task_data_path = Path(task_data_path).expanduser().resolve()
        self.source_revision = _required_string(source, "revision")
        self.task_data_revision = _required_string(source, "task_data_revision")
        self.task_data_sha256 = _required_string(source, "task_data_sha256")
        self.container_image = _required_string(source, "container_image")
        self.container_digest = _required_string(source, "container_digest")
        self.evaluator_entrypoint = _required_string(source, "evaluator_entrypoint")
        require_sha256(self.task_data_sha256, field_name="task_data_sha256")
        require_sha256(self.container_digest, field_name="container_digest")
        self.run_id = str(run_id)
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.command_timeout_seconds = _positive_int(
            command_timeout_seconds,
            field_name="command_timeout_seconds",
        )
        self.max_output_chars = _positive_int(
            max_output_chars,
            field_name="max_output_chars",
        )
        self.docker_runtime = docker_runtime or DockerRuntime()
        self.evaluator_hash = canonical_sha256(
            {
                "schema_version": LIFELONG_OS_SCHEMA_VERSION,
                "source_revision": self.source_revision,
                "entrypoint": self.evaluator_entrypoint,
                "semantics": "execute evaluation_command_item and resolve on exit code 0",
            }
        )
        self._records: dict[str, _OfficialOSTask] = {}
        self._containers: dict[Path, DockerContainer] = {}

    def load_tasks(self, *, limit: int) -> Sequence[BenchmarkTask]:
        limit = _positive_int(limit, field_name="limit")
        if not self.task_data_path.is_file():
            raise FileNotFoundError(f"Lifelong OS task data not found: {self.task_data_path}")
        actual_hash = _sha256_file(self.task_data_path)
        if actual_hash != self.task_data_sha256:
            raise ValueError(
                "Lifelong OS task data hash mismatch: "
                f"expected {self.task_data_sha256}, got {actual_hash}"
            )
        rows = _load_rows(self.task_data_path)
        if limit > len(rows):
            raise ValueError(
                f"Lifelong OS limit {limit} exceeds available task count {len(rows)}"
            )
        records = tuple(_parse_official_task(row) for row in rows[:limit])
        task_ids = tuple(record.task_id for record in records)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Lifelong OS source contains duplicate sample_index values")
        self._records = {record.task_id: record for record in records}
        return tuple(
            self._benchmark_task(record, order_index=order_index)
            for order_index, record in enumerate(records, 1)
        )

    def prepare_task(
        self,
        task: BenchmarkTask,
        *,
        task_dir: Path,
        seed: int,
    ) -> PreparedBenchmarkTask:
        record = self._records.get(task.task_id)
        if record is None:
            raise ValueError(f"Lifelong OS task was not loaded by this adapter: {task.task_id}")
        repo = task_dir / "repo"
        repo.mkdir(parents=True)
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
            )
            write_benchmark_action_files(
                repo,
                BenchmarkActionState(
                    container_name=container.name,
                    runtime_action_log_path=runtime_log,
                    timeout_seconds=self.command_timeout_seconds,
                    max_output_chars=self.max_output_chars,
                ),
            )
            _write_json_atomic(
                adapter_state,
                {
                    "schema_version": LIFELONG_OS_SCHEMA_VERSION,
                    "task_id": task.task_id,
                    "evaluator_hash": self.evaluator_hash,
                    "container_name": container.name,
                    "initialization_script": record.initialization_script,
                    "evaluation_script": record.evaluation_script,
                    "command_timeout_seconds": self.command_timeout_seconds,
                    "max_output_chars": self.max_output_chars,
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
        module = "my_agent.evaluation.memory_benchmark.adapters.lifelong_os"
        return PreparedBenchmarkTask(
            task=task,
            repo_path=repo,
            public_prompt=task.instruction,
            agent_test_command=None,
            initial_environment_command=(
                sys.executable,
                "-m",
                module,
                "initialize",
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
        except BaseException as exc:
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

    def _benchmark_task(
        self,
        record: _OfficialOSTask,
        *,
        order_index: int,
    ) -> BenchmarkTask:
        return BenchmarkTask(
            benchmark="lifelong_os",
            subset="os",
            task_id=record.task_id,
            order_index=order_index,
            task_group="lifelong_os:os",
            instruction=record.instruction,
            split="test",
            source_revision=self.source_revision,
            content_hash=record.content_hash,
            environment_spec={
                "source_sample_index": record.source_index,
                "task_data_revision": self.task_data_revision,
                "container_image": self.container_image,
                "container_digest": self.container_digest,
                "initialization_hash": canonical_sha256(record.initialization_script),
            },
            evaluator_spec={
                "name": LIFELONG_OS_EVALUATOR_NAME,
                "version": self.source_revision,
                "entrypoint": self.evaluator_entrypoint,
                "hash": self.evaluator_hash,
                "evaluation_command_hash": canonical_sha256(record.evaluation_script),
            },
            tags=("lifelong-agent-bench", "os", *(f"skill:{skill}" for skill in record.skills)),
        )


def lifelong_os_cli_main(
    argv: Sequence[str] | None = None,
    *,
    command_runner: Callable[..., Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m ...lifelong_os")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--state", required=True)
    score = commands.add_parser("score")
    score.add_argument("--state", required=True)
    score.add_argument("--result", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    runner_kwargs: dict[str, Any] = {}
    if command_runner is not None:
        runner_kwargs["command_runner"] = command_runner
    if args.command == "initialize":
        try:
            state = _load_adapter_state(Path(args.state))
        except (FileNotFoundError, OSError, ValueError):
            return 2
        result = execute_container_command(
            str(state["container_name"]),
            str(state["initialization_script"]) or ":",
            timeout_seconds=int(state["command_timeout_seconds"]),
            max_output_chars=int(state["max_output_chars"]),
            login_shell=False,
            **runner_kwargs,
        )
        return 0 if result.ok else 2
    return execute_official_scorer(
        lambda: _score_lifelong_os(
            _load_adapter_state(Path(args.state)),
            command_runner=command_runner,
        ),
        official_result_path=args.result,
    )


def _score_lifelong_os(
    state: Mapping[str, Any],
    *,
    command_runner: Callable[..., Any] | None,
) -> OfficialEvaluatorResult:
    runner_kwargs: dict[str, Any] = {}
    if command_runner is not None:
        runner_kwargs["command_runner"] = command_runner
    result = execute_container_command(
        str(state["container_name"]),
        str(state["evaluation_script"]),
        timeout_seconds=int(state["command_timeout_seconds"]),
        max_output_chars=int(state["max_output_chars"]),
        login_shell=False,
        **runner_kwargs,
    )
    if result.timed_out or result.returncode not in {0, 1}:
        raise RuntimeError(
            "official Lifelong OS evaluator failed with "
            f"return code {result.returncode}"
        )
    resolved = result.returncode == 0
    return OfficialEvaluatorResult(
        task_id=str(state["task_id"]),
        evaluator_hash=str(state["evaluator_hash"]),
        resolved=resolved,
        reward=1.0 if resolved else 0.0,
    )


def _load_adapter_state(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "task_id",
        "evaluator_hash",
        "container_name",
        "initialization_script",
        "evaluation_script",
        "command_timeout_seconds",
        "max_output_chars",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("Lifelong OS adapter state does not match the v1 schema")
    if payload.get("schema_version") != LIFELONG_OS_SCHEMA_VERSION:
        raise ValueError("unsupported Lifelong OS adapter state schema")
    for field_name in (
        "task_id",
        "evaluator_hash",
        "container_name",
        "evaluation_script",
    ):
        _required_string(payload, field_name)
    if not isinstance(payload.get("initialization_script"), str):
        raise ValueError("initialization_script must be a string")
    require_sha256(str(payload["evaluator_hash"]), field_name="evaluator_hash")
    _positive_int(payload["command_timeout_seconds"], field_name="command_timeout_seconds")
    _positive_int(payload["max_output_chars"], field_name="max_output_chars")
    return payload


def _load_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    if path.suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Lifelong OS JSONL row {line_number} must be an object")
            rows.append(payload)
        return tuple(rows)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:  # pragma: no cover - exercised by packaging preflight.
            raise RuntimeError(
                "reading official Lifelong OS parquet requires the memory-benchmark extra"
            ) from exc
        return tuple(parquet.read_table(path).to_pylist())
    raise ValueError(f"unsupported Lifelong OS task data format: {path.suffix}")


def _parse_official_task(row: Mapping[str, Any]) -> _OfficialOSTask:
    sample_index = row.get("sample_index")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("Lifelong OS sample_index must be a non-negative integer")
    instruction = _required_string(row, "instruction")
    initialization = _literal_mapping(row.get("initialization_command_item"), "initialization")
    evaluation_info = _literal_mapping(row.get("evaluation_info"), "evaluation_info")
    evaluation = _literal_mapping(
        evaluation_info.get("evaluation_command_item"),
        "evaluation_command_item",
    )
    ground_truth = _literal_mapping(
        evaluation_info.get("ground_truth_command_item"),
        "ground_truth_command_item",
    )
    initialization_script = _bash_script(
        initialization,
        "initialization_command_item",
        allow_empty=True,
    )
    evaluation_script = _bash_script(evaluation, "evaluation_command_item")
    _bash_script(ground_truth, "ground_truth_command_item")
    skills = _string_sequence(row.get("skill_list"), field_name="skill_list")
    normalized = {
        "sample_index": sample_index,
        "instruction": instruction,
        "initialization_command_item": dict(initialization),
        "evaluation_info": {
            "evaluation_command_item": dict(evaluation),
            "ground_truth_command_item": dict(ground_truth),
        },
        "skill_list": list(skills),
        "raw_entry_hash": row.get("raw_entry_hash"),
    }
    return _OfficialOSTask(
        task_id=str(sample_index),
        source_index=sample_index,
        instruction=instruction,
        initialization_script=initialization_script,
        evaluation_script=evaluation_script,
        skills=skills,
        content_hash=canonical_sha256(normalized),
    )


def _literal_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid Lifelong OS {field_name}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Lifelong OS {field_name} must be an object")
    return parsed


def _string_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid Lifelong OS {field_name}") from exc
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise ValueError(f"Lifelong OS {field_name} must be an array")
    result = tuple(str(item).strip() for item in parsed)
    if any(not item for item in result):
        raise ValueError(f"Lifelong OS {field_name} cannot contain empty values")
    return result


def _bash_script(
    command_item: Mapping[str, Any],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if str(command_item.get("command_name", "")).casefold() != "bash":
        raise ValueError(f"Lifelong OS {field_name} must use bash")
    value = command_item.get("script")
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"Lifelong OS {field_name}.script is invalid")
    return value


def _required_string(data: Mapping[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(dict(payload)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(lifelong_os_cli_main())


__all__ = [
    "LIFELONG_OS_EVALUATOR_NAME",
    "LIFELONG_OS_SCHEMA_VERSION",
    "LifelongOSAdapter",
    "lifelong_os_cli_main",
]
