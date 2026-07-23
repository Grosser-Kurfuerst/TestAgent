"""InterCode-Bash adapter preserving the locked official reward semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import math
import os
import sys

from my_agent.evaluation.memory_benchmark.adapters.base import execute_official_scorer
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    ACTION_LOG_SCHEMA_VERSION,
    BenchmarkActionState,
    DockerActionResult,
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


INTERCODE_BASH_SCHEMA_VERSION = "memory-benchmark-intercode-bash-v1"
INTERCODE_BASH_EVALUATOR_NAME = "intercode-bash"

_GIT_RESET_SCRIPT = "git reset --hard; git clean -fd;"
_GIT_STATUS_SCRIPT = "git status --short;"
_EVALUATOR_MAX_OUTPUT_CHARS = 1_000_000


@dataclass(frozen=True)
class _OfficialBashTask:
    task_id: str
    source_index: int
    instruction: str
    gold_command: str | tuple[str, ...]
    content_hash: str


class InterCodeBashAdapter:
    """Prepare fresh agent and hidden evaluation containers for each task."""

    name = "intercode_bash"

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
            raise ValueError("InterCode Bash run_id must be non-empty")
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
                "schema_version": INTERCODE_BASH_SCHEMA_VERSION,
                "source_revision": self.source_revision,
                "entrypoint": self.evaluator_entrypoint,
                "reward": {
                    "base": 0.01,
                    "file_system_diff": 0.33,
                    "file_content_md5": 0.33,
                    "last_observation_tfidf": 0.33,
                    "resolved": "math.isclose(reward, 1.0)",
                },
            }
        )
        self._records: dict[str, _OfficialBashTask] = {}
        self._containers: dict[Path, tuple[DockerContainer, DockerContainer]] = {}

    def load_tasks(self, *, limit: int) -> Sequence[BenchmarkTask]:
        limit = _positive_int(limit, field_name="limit")
        if not self.task_data_path.is_file():
            raise FileNotFoundError(
                f"InterCode Bash task data not found: {self.task_data_path}"
            )
        actual_hash = _sha256_file(self.task_data_path)
        if actual_hash != self.task_data_sha256:
            raise ValueError(
                "InterCode Bash task data hash mismatch: "
                f"expected {self.task_data_sha256}, got {actual_hash}"
            )
        rows = _load_rows(self.task_data_path)
        if limit > len(rows):
            raise ValueError(
                f"InterCode Bash limit {limit} exceeds available task count {len(rows)}"
            )
        records = tuple(
            _parse_official_task(row, source_index=index)
            for index, row in enumerate(rows[:limit])
        )
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
            raise ValueError(
                f"InterCode Bash task was not loaded by this adapter: {task.task_id}"
            )
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
        created: list[DockerContainer] = []
        try:
            agent_container = self.docker_runtime.create_container(
                image=self.container_image,
                expected_digest=self.container_digest,
                run_id=self.run_id,
                seed=seed,
                benchmark=task.benchmark,
                task_id=task.task_id,
            )
            created.append(agent_container)
            evaluation_container = self.docker_runtime.create_container(
                image=self.container_image,
                expected_digest=self.container_digest,
                run_id=self.run_id,
                seed=seed,
                benchmark=task.benchmark,
                task_id=f"{task.task_id}-gold",
            )
            created.append(evaluation_container)
            write_benchmark_action_files(
                repo,
                BenchmarkActionState(
                    container_name=agent_container.name,
                    runtime_action_log_path=runtime_log,
                    timeout_seconds=self.command_timeout_seconds,
                    max_output_chars=self.max_output_chars,
                ),
            )
            _write_json_atomic(
                adapter_state,
                {
                    "schema_version": INTERCODE_BASH_SCHEMA_VERSION,
                    "task_id": task.task_id,
                    "evaluator_hash": self.evaluator_hash,
                    "agent_container_name": agent_container.name,
                    "evaluation_container_name": evaluation_container.name,
                    "query": record.instruction,
                    "gold_command": _gold_command_json(record.gold_command),
                    "runtime_action_log_path": str(runtime_log),
                    "command_timeout_seconds": self.command_timeout_seconds,
                    "evaluator_max_output_chars": _EVALUATOR_MAX_OUTPUT_CHARS,
                },
            )
            self._containers[adapter_state.resolve()] = (
                agent_container,
                evaluation_container,
            )
        except Exception as exc:
            _cleanup_after_prepare_failure(self.docker_runtime, created, exc)
            raise
        module = "my_agent.evaluation.memory_benchmark.adapters.intercode_bash"
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
        failures: list[BaseException] = []
        try:
            self.finalize_task_artifacts(prepared)
        except BaseException as exc:
            failures.append(exc)
        containers = self._containers.pop(prepared.adapter_state_path.resolve(), ())
        for container in containers:
            try:
                self.docker_runtime.cleanup_container(container)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            primary = failures[0]
            for failure in failures[1:]:
                primary.add_note(f"additional cleanup failure: {failure}")
            raise primary

    def _benchmark_task(
        self,
        record: _OfficialBashTask,
        *,
        order_index: int,
    ) -> BenchmarkTask:
        return BenchmarkTask(
            benchmark="intercode_bash",
            subset="nl2bash_fs_1",
            task_id=record.task_id,
            order_index=order_index,
            task_group="intercode_bash:nl2bash_fs_1",
            instruction=record.instruction,
            split="test",
            source_revision=self.source_revision,
            content_hash=record.content_hash,
            environment_spec={
                "source_index": record.source_index,
                "task_data_revision": self.task_data_revision,
                "container_image": self.container_image,
                "container_digest": self.container_digest,
                "reset_command_hash": canonical_sha256(_GIT_RESET_SCRIPT),
            },
            evaluator_spec={
                "name": INTERCODE_BASH_EVALUATOR_NAME,
                "version": self.source_revision,
                "entrypoint": self.evaluator_entrypoint,
                "hash": self.evaluator_hash,
                "gold_command_hash": canonical_sha256(
                    _gold_command_json(record.gold_command)
                ),
            },
            tags=("intercode", "bash", "nl2bash", "filesystem"),
        )


def intercode_bash_cli_main(
    argv: Sequence[str] | None = None,
    *,
    command_runner: Callable[..., Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m ...intercode_bash")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--state", required=True)
    score = commands.add_parser("score")
    score.add_argument("--state", required=True)
    score.add_argument("--result", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "initialize":
        try:
            state = _load_adapter_state(Path(args.state))
            reset = _execute_hidden_command(
                str(state["agent_container_name"]),
                _GIT_RESET_SCRIPT,
                state=state,
                command_runner=command_runner,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return 2
        return 0 if reset.ok else 2
    return execute_official_scorer(
        lambda: _score_intercode_bash(
            _load_adapter_state(Path(args.state)),
            command_runner=command_runner,
        ),
        official_result_path=args.result,
    )


def _score_intercode_bash(
    state: Mapping[str, Any],
    *,
    command_runner: Callable[..., Any] | None,
) -> OfficialEvaluatorResult:
    agent_container = str(state["agent_container_name"])
    evaluation_container = str(state["evaluation_container_name"])
    reset = _execute_hidden_command(
        evaluation_container,
        _GIT_RESET_SCRIPT,
        state=state,
        command_runner=command_runner,
    )
    if not reset.ok:
        raise RuntimeError("failed to reset the InterCode evaluation container")

    gold = _gold_command_from_state(state["gold_command"])
    gold_script = gold if isinstance(gold, str) else ";".join(gold)
    gold_result = _execute_hidden_command(
        evaluation_container,
        gold_script,
        state=state,
        command_runner=command_runner,
    )
    if gold_result.timed_out:
        raise RuntimeError("InterCode gold command timed out")
    gold_observation = gold_result.stdout + gold_result.stderr
    agent_observation = _last_action_observation(
        Path(str(state["runtime_action_log_path"])),
        fallback=str(state["query"]),
    )

    agent_status = _require_hidden_success(
        _execute_hidden_command(
            agent_container,
            _GIT_STATUS_SCRIPT,
            state=state,
            command_runner=command_runner,
        ),
        operation="read agent git status",
    )
    evaluation_status = _require_hidden_success(
        _execute_hidden_command(
            evaluation_container,
            _GIT_STATUS_SCRIPT,
            state=state,
            command_runner=command_runner,
        ),
        operation="read evaluation git status",
    )
    diff_agent = _parse_status(agent_status.stdout + agent_status.stderr)
    diff_evaluation = _parse_status(
        evaluation_status.stdout + evaluation_status.stderr
    )

    reward = 0.01
    missing = set(diff_evaluation) - set(diff_agent)
    extra = set(diff_agent) - set(diff_evaluation)
    reward += round(0.33 * (1 - math.erf(len(missing) + len(extra))), 2)

    file_score = 0.33
    shared_changes = [
        change
        for change in set(diff_agent) & set(diff_evaluation)
        if change[1] in {"A", "??", "C"}
    ]
    if shared_changes:
        same_changes = 0
        for path, _status in shared_changes:
            hash_command = f"md5sum {path}" if "." in path else f"md5deep -r {path}"
            agent_hash = _require_hidden_success(
                _execute_hidden_command(
                    agent_container,
                    hash_command,
                    state=state,
                    command_runner=command_runner,
                ),
                operation=f"hash agent path {path}",
            )
            evaluation_hash = _require_hidden_success(
                _execute_hidden_command(
                    evaluation_container,
                    hash_command,
                    state=state,
                    command_runner=command_runner,
                ),
                operation=f"hash evaluation path {path}",
            )
            agent_output = agent_hash.stdout + agent_hash.stderr
            evaluation_output = evaluation_hash.stdout + evaluation_hash.stderr
            same_changes += int(agent_output == evaluation_output)
        file_score = round(0.33 * (same_changes / len(shared_changes)), 2)
    reward += file_score

    similarity = _tfidf_similarity(agent_observation, gold_observation)
    reward += round(0.33 * similarity, 2)
    resolved = math.isclose(reward, 1.0)
    return OfficialEvaluatorResult(
        task_id=str(state["task_id"]),
        evaluator_hash=str(state["evaluator_hash"]),
        resolved=resolved,
        reward=reward,
    )


def _execute_hidden_command(
    container_name: str,
    command: str,
    *,
    state: Mapping[str, Any],
    command_runner: Callable[..., Any] | None,
) -> DockerActionResult:
    kwargs: dict[str, Any] = {}
    if command_runner is not None:
        kwargs["command_runner"] = command_runner
    return execute_container_command(
        container_name,
        command,
        timeout_seconds=int(state["command_timeout_seconds"]),
        max_output_chars=int(state["evaluator_max_output_chars"]),
        login_shell=False,
        **kwargs,
    )


def _require_hidden_success(
    result: DockerActionResult,
    *,
    operation: str,
) -> DockerActionResult:
    if result.ok:
        return result
    raise RuntimeError(f"InterCode evaluator could not {operation}")


def _tfidf_similarity(agent_observation: str, gold_observation: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - packaging/preflight failure path.
        raise RuntimeError(
            "InterCode Bash scoring requires the memory-benchmark extra"
        ) from exc
    try:
        tfidf = TfidfVectorizer().fit_transform(
            [agent_observation, gold_observation]
        )
        similarity = (tfidf * tfidf.T).toarray()[0][1]
        return float(similarity)
    except Exception:  # noqa: BLE001 - official InterCode equality fallback.
        return 1.0 if agent_observation == gold_observation else 0.0


def _parse_status(status: str) -> list[tuple[str, str]]:
    tokens = status.split()
    if len(tokens) % 2:
        raise ValueError("InterCode git status output has an invalid shape")
    return [(tokens[index + 1], tokens[index]) for index in range(0, len(tokens), 2)]


def _last_action_observation(path: Path, *, fallback: str) -> str:
    if not path.exists():
        return fallback
    last: Mapping[str, Any] | None = None
    expected_sequence = 1
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"InterCode action log row {line_number} must be an object")
        if payload.get("schema_version") != ACTION_LOG_SCHEMA_VERSION:
            raise ValueError("InterCode action log schema mismatch")
        if payload.get("sequence") != expected_sequence:
            raise ValueError("InterCode action log sequence is not contiguous")
        if not isinstance(payload.get("stdout"), str) or not isinstance(
            payload.get("stderr"), str
        ):
            raise ValueError("InterCode action log output must be text")
        last = payload
        expected_sequence += 1
    if last is None:
        return fallback
    return str(last["stdout"]) + str(last["stderr"])


def _load_adapter_state(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "task_id",
        "evaluator_hash",
        "agent_container_name",
        "evaluation_container_name",
        "query",
        "gold_command",
        "runtime_action_log_path",
        "command_timeout_seconds",
        "evaluator_max_output_chars",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("InterCode Bash adapter state does not match the v1 schema")
    if payload.get("schema_version") != INTERCODE_BASH_SCHEMA_VERSION:
        raise ValueError("unsupported InterCode Bash adapter state schema")
    for field_name in (
        "task_id",
        "evaluator_hash",
        "agent_container_name",
        "evaluation_container_name",
        "query",
        "runtime_action_log_path",
    ):
        _required_string(payload, field_name)
    require_sha256(str(payload["evaluator_hash"]), field_name="evaluator_hash")
    runtime_log = Path(str(payload["runtime_action_log_path"]))
    if not runtime_log.is_absolute():
        raise ValueError("runtime_action_log_path must be absolute")
    _gold_command_from_state(payload["gold_command"])
    _positive_int(
        payload["command_timeout_seconds"],
        field_name="command_timeout_seconds",
    )
    _positive_int(
        payload["evaluator_max_output_chars"],
        field_name="evaluator_max_output_chars",
    )
    return payload


def _load_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    if path.suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"InterCode Bash JSONL row {line_number} must be an object"
                )
            rows.append(payload)
        return tuple(rows)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise ValueError("InterCode Bash JSON data must be an array")
        rows = tuple(payload)
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("InterCode Bash JSON rows must be objects")
        return rows
    raise ValueError(f"unsupported InterCode Bash task data format: {path.suffix}")


def _parse_official_task(
    row: Mapping[str, Any],
    *,
    source_index: int,
) -> _OfficialBashTask:
    instruction = _required_string(row, "query")
    gold = _gold_command_from_state(row.get("gold"))
    normalized = {
        "source_index": source_index,
        "query": instruction,
        "gold": _gold_command_json(gold),
    }
    return _OfficialBashTask(
        task_id=str(source_index),
        source_index=source_index,
        instruction=instruction,
        gold_command=gold,
        content_hash=canonical_sha256(normalized),
    )


def _gold_command_from_state(value: object) -> str | tuple[str, ...]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("InterCode Bash gold command must be non-empty")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        commands = tuple(str(item) for item in value)
        if not commands or any(not command.strip() for command in commands):
            raise ValueError("InterCode Bash gold command list is invalid")
        return commands
    raise ValueError("InterCode Bash gold command must be text or an array")


def _gold_command_json(value: str | tuple[str, ...]) -> str | list[str]:
    return value if isinstance(value, str) else list(value)


def _cleanup_after_prepare_failure(
    runtime: DockerRuntime,
    containers: Sequence[DockerContainer],
    original: BaseException,
) -> None:
    for container in reversed(containers):
        try:
            runtime.cleanup_container(container)
        except Exception as cleanup_exc:  # noqa: BLE001 - preserve prepare cause.
            original.add_note(f"container cleanup also failed: {cleanup_exc}")


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
    raise SystemExit(intercode_bash_cli_main())


__all__ = [
    "INTERCODE_BASH_EVALUATOR_NAME",
    "INTERCODE_BASH_SCHEMA_VERSION",
    "InterCodeBashAdapter",
    "intercode_bash_cli_main",
]
