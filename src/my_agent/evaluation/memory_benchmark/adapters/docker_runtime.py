"""Isolated Docker task runtime and public benchmark action wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
import argparse
import json
import os
import re
import subprocess
import sys
import time

from my_agent.policy.identity import canonical_json_bytes, canonical_sha256, require_sha256


ACTION_LOG_SCHEMA_VERSION = "memory-benchmark-action-v1"
ACTION_STATE_SCHEMA_VERSION = "memory-benchmark-action-state-v1"
MANAGED_LABEL = "agentcli.memory_benchmark.managed"
RUN_LABEL = "agentcli.memory_benchmark.run_id"
SEED_LABEL = "agentcli.memory_benchmark.seed"
BENCHMARK_LABEL = "agentcli.memory_benchmark.benchmark"
TASK_LABEL = "agentcli.memory_benchmark.task_id"

_CONTAINER_COMPONENT_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_NOT_FOUND_MARKERS = ("no such object", "not found", "no such container")

BENCHMARK_ACTION_SCRIPT = """from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import benchmark_action_main


if __name__ == "__main__":
    raise SystemExit(benchmark_action_main())
"""

BENCHMARK_AGENT_INSTRUCTIONS = """# Benchmark task environment

Use the `benchmark_action` tool for every shell action that must run in the isolated benchmark container.
Do not edit `.agentcli/benchmark_state.json`, `benchmark_action.py`, or `.agentcli/tools.json`.
The official evaluator is hidden and is run exactly once after you finish.
"""


class DockerRuntimeError(RuntimeError):
    """Raised when Docker isolation or cleanup cannot be proven correct."""


@dataclass(frozen=True)
class DockerContainer:
    container_id: str
    name: str
    image: str
    image_digest: str
    labels: Mapping[str, str]

    def __post_init__(self) -> None:
        for field_name in ("container_id", "name", "image"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        require_sha256(self.image_digest, field_name="image_digest")
        normalized = {str(key): str(value) for key, value in self.labels.items()}
        if normalized.get(MANAGED_LABEL) != "true":
            raise ValueError("Docker container must carry the managed benchmark label")
        object.__setattr__(self, "labels", MappingProxyType(normalized))


@dataclass(frozen=True)
class BenchmarkActionState:
    container_name: str
    runtime_action_log_path: Path
    timeout_seconds: int
    max_output_chars: int

    def __post_init__(self) -> None:
        if not isinstance(self.container_name, str) or not self.container_name.strip():
            raise ValueError("container_name must be non-empty")
        path = Path(self.runtime_action_log_path).expanduser()
        if not path.is_absolute():
            raise ValueError("runtime_action_log_path must be absolute")
        object.__setattr__(self, "runtime_action_log_path", path.resolve())
        for field_name in ("timeout_seconds", "max_output_chars"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_STATE_SCHEMA_VERSION,
            "container_name": self.container_name,
            "runtime_action_log_path": str(self.runtime_action_log_path),
            "timeout_seconds": self.timeout_seconds,
            "max_output_chars": self.max_output_chars,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkActionState":
        expected_fields = {
            "schema_version",
            "container_name",
            "runtime_action_log_path",
            "timeout_seconds",
            "max_output_chars",
        }
        if set(data) != expected_fields or data.get("schema_version") != ACTION_STATE_SCHEMA_VERSION:
            raise ValueError("benchmark action state does not match the v1 schema")
        return cls(
            container_name=str(data["container_name"]),
            runtime_action_log_path=Path(str(data["runtime_action_log_path"])),
            timeout_seconds=_positive_int(data["timeout_seconds"], "timeout_seconds"),
            max_output_chars=_positive_int(data["max_output_chars"], "max_output_chars"),
        )


@dataclass(frozen=True)
class DockerActionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_sec: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class DockerRuntime:
    """Small Docker CLI boundary with immutable-image and label checks."""

    def __init__(
        self,
        *,
        command_runner: Callable[..., Any] = subprocess.run,
        docker_executable: str = "docker",
    ) -> None:
        self.command_runner = command_runner
        self.docker_executable = docker_executable

    def preflight(self) -> str:
        result = self._run(
            [self.docker_executable, "version", "--format", "{{.Server.Version}}"],
            timeout=30,
        )
        self._require_success(result, operation="docker version")
        version = str(result.stdout or "").strip()
        if not version:
            raise DockerRuntimeError("docker version returned an empty server version")
        return version

    def inspect_image(self, image: str) -> str:
        if not isinstance(image, str) or not image.strip():
            raise ValueError("image must be non-empty")
        result = self._run(
            [self.docker_executable, "image", "inspect", "--format", "{{.Id}}", image],
            timeout=30,
        )
        self._require_success(result, operation=f"inspect image {image}")
        digest = str(result.stdout or "").strip()
        require_sha256(digest, field_name="docker image ID")
        return digest

    def create_container(
        self,
        *,
        image: str,
        expected_digest: str,
        run_id: str,
        seed: int,
        benchmark: str,
        task_id: str,
        keepalive_argv: Sequence[str] = ("sleep", "infinity"),
    ) -> DockerContainer:
        require_sha256(expected_digest, field_name="expected_digest")
        actual_digest = self.inspect_image(image)
        if actual_digest != expected_digest:
            raise DockerRuntimeError(
                f"Docker image digest mismatch for {image}: expected {expected_digest}, got {actual_digest}"
            )
        name = benchmark_container_name(
            run_id=run_id,
            seed=seed,
            benchmark=benchmark,
            task_id=task_id,
        )
        labels = {
            MANAGED_LABEL: "true",
            RUN_LABEL: run_id,
            SEED_LABEL: str(seed),
            BENCHMARK_LABEL: benchmark,
            TASK_LABEL: task_id,
        }
        argv = [self.docker_executable, "create", "--name", name]
        for key, value in sorted(labels.items()):
            argv.extend(["--label", f"{key}={value}"])
        argv.append(actual_digest)
        argv.extend(str(part) for part in keepalive_argv)
        created = self._run(argv, timeout=60)
        self._require_success(created, operation=f"create container {name}")
        container_id = str(created.stdout or "").strip()
        container = DockerContainer(
            container_id=container_id,
            name=name,
            image=image,
            image_digest=actual_digest,
            labels=labels,
        )
        try:
            started = self._run(
                [self.docker_executable, "start", container.container_id],
                timeout=60,
            )
            self._require_success(started, operation=f"start container {name}")
        except Exception as exc:
            try:
                self.cleanup_container(container)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve the create/start cause.
                exc.add_note(f"container cleanup also failed: {cleanup_exc}")
            raise
        return container

    def execute_action(
        self,
        container: DockerContainer,
        command: str,
        *,
        action_log_path: str | Path,
        timeout_seconds: int,
        max_output_chars: int,
    ) -> DockerActionResult:
        state = BenchmarkActionState(
            container_name=container.name,
            runtime_action_log_path=Path(action_log_path).expanduser().resolve(),
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        return execute_benchmark_action(
            state,
            command,
            command_runner=self.command_runner,
            docker_executable=self.docker_executable,
        )

    def cleanup_container(self, container: DockerContainer) -> None:
        labels = self._inspect_container_labels(container.container_id)
        if labels is None:
            return
        mismatched = {
            key: (value, labels.get(key))
            for key, value in container.labels.items()
            if labels.get(key) != value
        }
        if mismatched:
            raise DockerRuntimeError(
                f"refusing to remove container with mismatched labels: {mismatched}"
            )
        removed = self._run(
            [self.docker_executable, "rm", "--force", container.container_id],
            timeout=60,
        )
        self._require_success(removed, operation=f"remove container {container.name}")

    def _inspect_container_labels(self, container_id: str) -> dict[str, str] | None:
        result = self._run(
            [
                self.docker_executable,
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                container_id,
            ],
            timeout=30,
        )
        if int(result.returncode) != 0:
            detail = f"{result.stdout or ''}\n{result.stderr or ''}".casefold()
            if any(marker in detail for marker in _NOT_FOUND_MARKERS):
                return None
            self._require_success(result, operation=f"inspect container {container_id}")
        try:
            payload = json.loads(str(result.stdout or "{}"))
        except json.JSONDecodeError as exc:
            raise DockerRuntimeError("docker inspect returned invalid label JSON") from exc
        if not isinstance(payload, Mapping):
            raise DockerRuntimeError("docker inspect labels must be a JSON object")
        return {str(key): str(value) for key, value in payload.items()}

    def _run(self, argv: Sequence[str], *, timeout: int) -> Any:
        return self.command_runner(
            list(argv),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _require_success(result: Any, *, operation: str) -> None:
        if int(result.returncode) == 0:
            return
        detail = str(result.stderr or result.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise DockerRuntimeError(f"{operation} failed with code {result.returncode}{suffix}")


def benchmark_container_name(*, run_id: str, seed: int, benchmark: str, task_id: str) -> str:
    components = ("agentcli-mb", run_id, str(seed), benchmark, task_id)
    normalized = "-".join(_safe_container_component(item) for item in components)
    return normalized[:120].rstrip("-.")


def benchmark_action_tool_config() -> dict[str, Any]:
    return {
        "version": 1,
        "tools": [
            {
                "kind": "command",
                "name": "benchmark_action",
                "description": "Execute one bash action in the isolated benchmark container.",
                "risk": "execute",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
                "command": {
                    "argv": ["python", "benchmark_action.py", "--command", "{command}"],
                    "cwd": ".",
                    "timeout_seconds": 120,
                },
            }
        ],
    }


def benchmark_action_tools_hash() -> str:
    return canonical_sha256(
        {
            "tool_config": benchmark_action_tool_config(),
            "wrapper_sha256": canonical_sha256(BENCHMARK_ACTION_SCRIPT),
        }
    )


def write_benchmark_action_files(repo_path: str | Path, state: BenchmarkActionState) -> None:
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError(f"benchmark task repo does not exist: {repo}")
    if state.runtime_action_log_path.is_relative_to(repo):
        raise ValueError("runtime_action_log_path must be outside the benchmark task repo")
    agentcli_dir = repo / ".agentcli"
    agentcli_dir.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(repo / "benchmark_action.py", BENCHMARK_ACTION_SCRIPT.encode("utf-8"))
    _write_bytes_atomic(repo / "AGENT.md", BENCHMARK_AGENT_INSTRUCTIONS.encode("utf-8"))
    _write_bytes_atomic(
        agentcli_dir / "tools.json",
        canonical_json_bytes(benchmark_action_tool_config()) + b"\n",
    )
    _write_bytes_atomic(
        agentcli_dir / "benchmark_state.json",
        canonical_json_bytes(state.to_dict()) + b"\n",
    )


def benchmark_action_main(
    argv: Sequence[str] | None = None,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    cwd: str | Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="benchmark_action.py")
    parser.add_argument("--command", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    state_path = root / ".agentcli" / "benchmark_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark action state must be a JSON object")
    state = BenchmarkActionState.from_dict(payload)
    result = execute_benchmark_action(
        state,
        args.command,
        command_runner=command_runner,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


def execute_benchmark_action(
    state: BenchmarkActionState,
    command: str,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    docker_executable: str = "docker",
) -> DockerActionResult:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("benchmark action command must be non-empty")
    argv = [docker_executable, "exec", state.container_name, "bash", "-lc", command]
    started = time.monotonic()
    timed_out = False
    try:
        completed = command_runner(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=state.timeout_seconds,
        )
        returncode = int(completed.returncode)
        stdout = str(completed.stdout or "")
        stderr = str(completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr) or (
            f"Command timed out after {state.timeout_seconds}s."
        )
    except FileNotFoundError as exc:
        returncode = 127
        stdout = ""
        stderr = f"Command not found: {exc}"
    elapsed = time.monotonic() - started
    result = DockerActionResult(
        returncode=returncode,
        stdout=_truncate_output(stdout, state.max_output_chars),
        stderr=_truncate_output(stderr, state.max_output_chars),
        timed_out=timed_out,
        elapsed_sec=elapsed,
    )
    _append_action_log(state.runtime_action_log_path, command=command, result=result)
    return result


def prepare_runtime_action_log(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise DockerRuntimeError(
            f"runtime action log already exists; previous arm was not finalized: {target}"
        )
    return target


def finalize_action_log(runtime_path: str | Path, final_path: str | Path) -> Path:
    runtime = Path(runtime_path).expanduser().resolve()
    final = Path(final_path).expanduser().resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if runtime.exists():
            raise DockerRuntimeError("both runtime and final action logs exist")
        return final
    if runtime.exists():
        with runtime.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(runtime, final)
        _fsync_directory(final.parent)
        return final
    _write_bytes_atomic(final, b"")
    return final


def _append_action_log(path: Path, *, command: str, result: DockerActionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence = _action_log_line_count(path) + 1
    record = {
        "schema_version": ACTION_LOG_SCHEMA_VERSION,
        "sequence": sequence,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "elapsed_sec": result.elapsed_sec,
    }
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _action_log_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_container_component(value: object) -> str:
    normalized = _CONTAINER_COMPONENT_RE.sub("-", str(value).strip()).strip("-.")
    return normalized or "task"


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _truncate_output(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "\n...[truncated]"
    if max_chars <= len(marker):
        return marker[-max_chars:]
    keep = max(0, max_chars - len(marker))
    return value[:keep] + marker


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = [
    "ACTION_LOG_SCHEMA_VERSION",
    "ACTION_STATE_SCHEMA_VERSION",
    "BENCHMARK_ACTION_SCRIPT",
    "BenchmarkActionState",
    "DockerActionResult",
    "DockerContainer",
    "DockerRuntime",
    "DockerRuntimeError",
    "benchmark_action_main",
    "benchmark_action_tool_config",
    "benchmark_action_tools_hash",
    "benchmark_container_name",
    "execute_benchmark_action",
    "finalize_action_log",
    "prepare_runtime_action_log",
    "write_benchmark_action_files",
]
