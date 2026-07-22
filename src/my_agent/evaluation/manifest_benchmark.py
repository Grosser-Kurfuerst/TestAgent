from __future__ import annotations

import difflib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from my_agent.config import AgentConfig
from my_agent.context import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MEMORY_CONTEXT_TOKENS,
    DEFAULT_REPO_CONTEXT_BUDGET_TOKENS,
    DEFAULT_SHORT_TERM_TOKENS,
    DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS,
    DEFAULT_TOOL_RESULT_CHARS,
)
from my_agent.evaluation.agent_benchmark import record_benchmark_result
from my_agent.llm import AgentLLM, build_llm
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.experience.retrieval.embedding import (
    EmbeddingRetriever,
    TransformersEmbeddingEncoder,
)
from my_agent.observability.trace_metrics import collect_trace_metrics
from my_agent.observability.tracing import TraceWriter
from my_agent.policy.identity import canonical_sha256
from my_agent.runtime import run_agent
from my_agent.schema import TraceEvent
from my_agent.training.contracts import AuthoritativeTaskOutcome, EvaluatorIdentity


AgentRunnerFn = Callable[..., Any]

_IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
MEMORY_MODE_PER_TASK = "per_task"
MEMORY_MODE_SHARED_STREAM = "shared_stream"
MEMORY_MODE_SHARED_BY_GROUP = "shared_by_group"
MEMORY_MODES = {
    MEMORY_MODE_PER_TASK,
    MEMORY_MODE_SHARED_STREAM,
    MEMORY_MODE_SHARED_BY_GROUP,
}


@dataclass(frozen=True)
class ManifestSettings:
    memory_mode: str = MEMORY_MODE_PER_TASK
    stream_id: str = ""
    task_group_fallback: str = ""


@dataclass(frozen=True)
class MemoryStreamResolution:
    memory_mode: str
    stream_id: str
    memory_dir: Path
    memory_project_key: str


@dataclass(frozen=True)
class CommandResult:
    command: str
    ok: bool
    returncode: int
    output: str = ""
    elapsed_sec: float = 0.0
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "ok": self.ok,
            "returncode": self.returncode,
            "output": self.output[:2000],
            "elapsed_sec": round(self.elapsed_sec, 1),
            "skipped": self.skipped,
        }


@dataclass
class ManifestEvalResult:
    task_id: str
    status: str
    resolved: bool
    task_valid: bool
    failure_type: str
    initial_visible: CommandResult
    source: str = "local"
    task_group: str = ""
    reward: float = 0.0
    evaluator_name: str = ""
    evaluator_version: str = ""
    evaluator_hash: str = ""
    outcome_finalized: bool = True
    evolver_writer_status: str = ""
    written_memory_ids: list[str] = field(default_factory=list)
    repository_revision_after_writer: str = ""
    evolver_cadence_id: str = ""
    evolver_maintenance_status: str = ""
    mode: str = "auto"
    tags: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    resolved_config: dict[str, Any] = field(default_factory=dict)
    expected_changed_files: list[str] = field(default_factory=list)
    expected_changed_files_ok: bool | None = None
    initial_hidden: CommandResult | None = None
    final_visible: CommandResult | None = None
    final_hidden: CommandResult | None = None
    patch_apply_ok: bool = False
    changed_files: list[str] = field(default_factory=list)
    patch_lines: int = 0
    patch_path: str = ""
    trace_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    memory_mode: str = MEMORY_MODE_PER_TASK
    stream_id: str = ""
    memory_dir: str = ""
    memory_project_key: str = ""
    memory_entries_before: int = 0
    memory_entries_after: int = 0
    memory_growth: int = 0
    memory_entries_total_before: int = 0
    memory_entries_total_after: int = 0
    memory_total_growth: int = 0
    agent_steps: int = 0
    agent_done: bool = False
    agent_stop_reason: str = ""
    error: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "resolved": self.resolved,
            "task_valid": self.task_valid,
            "failure_type": self.failure_type,
            "source": self.source,
            "task_group": self.task_group,
            "reward": self.reward,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "evaluator_hash": self.evaluator_hash,
            "outcome_finalized": self.outcome_finalized,
            "evolver_writer_status": self.evolver_writer_status,
            "written_memory_ids": list(self.written_memory_ids),
            "repository_revision_after_writer": self.repository_revision_after_writer,
            "evolver_cadence_id": self.evolver_cadence_id,
            "evolver_maintenance_status": self.evolver_maintenance_status,
            "mode": self.mode,
            "tags": list(self.tags),
            "env_overrides": dict(self.env_overrides),
            "resolved_config": dict(self.resolved_config),
            "expected_changed_files": list(self.expected_changed_files),
            "expected_changed_files_ok": self.expected_changed_files_ok,
            "initial_visible": self.initial_visible.to_dict(),
            "initial_hidden": self.initial_hidden.to_dict() if self.initial_hidden else None,
            "initial_visible_ok": self.initial_visible.ok,
            "initial_hidden_ok": self.initial_hidden.ok if self.initial_hidden else None,
            "final_visible": self.final_visible.to_dict() if self.final_visible else None,
            "final_hidden": self.final_hidden.to_dict() if self.final_hidden else None,
            "visible_ok": self.final_visible.ok if self.final_visible else None,
            "hidden_ok": self.final_hidden.ok if self.final_hidden else None,
            "patch_apply_ok": self.patch_apply_ok,
            "changed_files": list(self.changed_files),
            "patch_lines": self.patch_lines,
            "patch_path": self.patch_path,
            "trace_path": self.trace_path,
            "metrics": dict(self.metrics),
            "memory_mode": self.memory_mode,
            "stream_id": self.stream_id,
            "memory_dir": self.memory_dir,
            "memory_project_key": self.memory_project_key,
            "memory_entries_before": self.memory_entries_before,
            "memory_entries_after": self.memory_entries_after,
            "memory_growth": self.memory_growth,
            "memory_entries_total_before": self.memory_entries_total_before,
            "memory_entries_total_after": self.memory_entries_total_after,
            "memory_total_growth": self.memory_total_growth,
            "agent_steps": self.agent_steps,
            "agent_done": self.agent_done,
            "agent_stop_reason": self.agent_stop_reason,
            "error": self.error,
            "elapsed_sec": round(self.elapsed_sec, 1),
        }


@dataclass(frozen=True)
class ManifestBenchmarkResult:
    results: list[ManifestEvalResult]
    summary: dict[str, Any]
    output_dir: Path
    results_path: Path
    summary_path: Path

    def render(self) -> str:
        return (
            f"Manifest eval: {self.summary.get('resolved', 0)}/{self.summary.get('scored', 0)} resolved "
            f"({self.summary.get('solve_rate', 0.0):.1f}%).\n"
            f"results: {self.results_path}\n"
            f"summary: {self.summary_path}"
        )


def run_manifest_benchmark(
    *,
    tasks_path: str | Path,
    output_dir: str | Path,
    config: AgentConfig,
    mode: str = "auto",
    max_steps: int | None = None,
    command_timeout: int | None = None,
    env: Mapping[str, str] | None = None,
    agent_runner: AgentRunnerFn = run_agent,
) -> ManifestBenchmarkResult:
    manifest_path = Path(tasks_path)
    tasks, manifest_settings = _load_manifest(manifest_path)
    resolved_mode = _formal_manifest_mode(mode, formal=config.memory_evolver_mode == "formal")
    tasks = _prepare_manifest_tasks(
        tasks,
        settings=manifest_settings,
        formal=config.memory_evolver_mode == "formal",
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    work_root = output / "work"
    trace_root = output / "traces"
    patch_root = output / "patches"
    memory_root = output / "memory"
    work_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    patch_root.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")

    timeout = command_timeout if command_timeout is not None else config.command_timeout
    run_config = replace(config, command_timeout=timeout, trace_dir=trace_root)
    cli_env = dict(env or {})
    shared_policy, shared_embedding_retriever = _build_shared_runtime_resources(
        run_config,
        agent_runner=agent_runner,
    )

    results: list[ManifestEvalResult] = []
    for index, task in enumerate(tasks, start=1):
        result = _run_manifest_task(
            task,
            index=index,
            manifest_path=manifest_path,
            output_dir=output,
            work_root=work_root,
            trace_root=trace_root,
            patch_root=patch_root,
            memory_root=memory_root,
            manifest_settings=manifest_settings,
            config=run_config,
            mode=resolved_mode,
            max_steps=max_steps,
            command_timeout=timeout,
            cli_env=cli_env,
            agent_runner=agent_runner,
            shared_policy=shared_policy,
            shared_embedding_retriever=shared_embedding_retriever,
        )
        results.append(result)
        with results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    summary = summarize_manifest_results(results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ManifestBenchmarkResult(
        results=results,
        summary=summary,
        output_dir=output,
        results_path=results_path,
        summary_path=summary_path,
    )


def _build_shared_runtime_resources(
    config: AgentConfig,
    *,
    agent_runner: AgentRunnerFn,
) -> tuple[AgentLLM | None, EmbeddingRetriever | None]:
    if config.memory_evolver_mode != "formal" or agent_runner is not run_agent:
        return None, None
    policy = build_llm(config)
    retriever = None
    if config.memory_evolver_retrieval_backend != "lexical_ablation":
        retriever = EmbeddingRetriever(TransformersEmbeddingEncoder.from_config(config))
    return policy, retriever


def _formal_manifest_mode(mode: str, *, formal: bool) -> str:
    normalized = str(getattr(mode, "value", mode) or "auto").strip().lower()
    if not formal:
        return normalized
    if normalized == "auto":
        return "react"
    if normalized != "react":
        raise ValueError("formal OPD manifest collection currently requires mode=react")
    return normalized


def load_manifest_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks, _settings = _load_manifest(path)
    return tasks


def _load_manifest(path: str | Path) -> tuple[list[dict[str, Any]], ManifestSettings]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if manifest_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{manifest_path}:{line_number} must be a JSON object.")
            rows.append(payload)
        return rows, ManifestSettings()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)], ManifestSettings()
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        settings = ManifestSettings(
            memory_mode=_memory_mode_value(payload.get("memory_mode"), default=MEMORY_MODE_PER_TASK),
            stream_id=str(payload.get("stream_id") or "").strip(),
            task_group_fallback=_task_group_fallback(payload.get("task_group_fallback")),
        )
        return [dict(item) for item in payload["tasks"] if isinstance(item, dict)], settings
    if isinstance(payload, dict):
        return [dict(payload)], ManifestSettings()
    raise ValueError("Manifest must be a JSON object, JSON array, or JSONL objects.")


def _prepare_manifest_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    settings: ManifestSettings,
    formal: bool,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        item = dict(task)
        task_group = str(item.get("task_group") or "").strip()
        if not task_group and settings.task_group_fallback == "source":
            task_group = str(item.get("source") or "").strip()
        if formal and not task_group:
            task_id = str(item.get("id") or f"task_{index}")
            raise ValueError(
                f"formal manifest task {task_id!r} requires a non-empty task_group; "
                "smoke manifests may explicitly set task_group_fallback=source"
            )
        item["task_group"] = task_group
        prepared.append(item)
    return prepared


def _task_group_fallback(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"", "source"}:
        raise ValueError("task_group_fallback must be empty or 'source'")
    return normalized


def _resolve_memory_stream(
    task: Mapping[str, Any],
    *,
    manifest_settings: ManifestSettings,
    manifest_path: Path,
    memory_root: Path,
    safe_id: str,
) -> MemoryStreamResolution:
    mode = _memory_mode_value(task.get("memory_mode"), default=manifest_settings.memory_mode)
    if mode == MEMORY_MODE_PER_TASK:
        stream_id = _first_nonblank(task.get("stream_id"), task.get("group"), task.get("id"), safe_id) or safe_id
        return MemoryStreamResolution(
            memory_mode=mode,
            stream_id=stream_id,
            memory_dir=memory_root / safe_id,
            memory_project_key="",
        )
    if mode == MEMORY_MODE_SHARED_STREAM:
        stream_id = _first_nonblank(
            task.get("stream_id"),
            task.get("group"),
            manifest_settings.stream_id,
            manifest_path.stem,
            "manifest",
        )
        return MemoryStreamResolution(
            memory_mode=mode,
            stream_id=stream_id,
            memory_dir=memory_root / "streams" / _safe_id(stream_id),
            memory_project_key=_memory_project_key(manifest_path, mode, stream_id),
        )

    stream_id = _first_nonblank(task.get("stream_id"), task.get("group"))
    if not stream_id:
        raise ValueError("shared_by_group requires task.stream_id or task.group.")
    return MemoryStreamResolution(
        memory_mode=mode,
        stream_id=stream_id,
        memory_dir=memory_root / "groups" / _safe_id(stream_id),
        memory_project_key=_memory_project_key(manifest_path, mode, stream_id),
    )


def _memory_mode_value(value: object, *, default: str) -> str:
    mode = _first_nonblank(value, default, MEMORY_MODE_PER_TASK)
    if mode not in MEMORY_MODES:
        expected = ", ".join(sorted(MEMORY_MODES))
        raise ValueError(f"Unsupported memory_mode: {mode!r}. Expected one of: {expected}.")
    return mode


def _memory_project_key(manifest_path: Path, mode: str, stream_id: str) -> str:
    return f"manifest:{_safe_id(str(manifest_path.resolve()))}:memory:{mode}:stream:{_safe_id(stream_id)}"


def _memory_counts(memory_dir: Path, *, project_key: str = "") -> dict[str, int]:
    store = ExperienceStore.from_dir(memory_dir)
    store.load()
    total = len(store.all())
    visible = len(store.all(project_key=project_key)) if project_key else total
    return {"total": total, "visible": visible}


def _memory_result_fields(
    *,
    memory_stream: MemoryStreamResolution,
    memory_dir: Path,
    memory_project_key: str,
    before_counts: Mapping[str, int],
    after_counts: Mapping[str, int],
) -> dict[str, Any]:
    before_visible = int(before_counts.get("visible", 0))
    after_visible = int(after_counts.get("visible", 0))
    before_total = int(before_counts.get("total", 0))
    after_total = int(after_counts.get("total", 0))
    return {
        "memory_mode": memory_stream.memory_mode,
        "stream_id": memory_stream.stream_id,
        "memory_dir": str(memory_dir),
        "memory_project_key": memory_project_key,
        "memory_entries_before": before_visible,
        "memory_entries_after": after_visible,
        "memory_growth": after_visible - before_visible,
        "memory_entries_total_before": before_total,
        "memory_entries_total_after": after_total,
        "memory_total_growth": after_total - before_total,
    }


def _first_nonblank(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def summarize_manifest_results(results: Sequence[ManifestEvalResult]) -> dict[str, Any]:
    total = len(results)
    scored = sum(1 for result in results if result.task_valid)
    resolved = sum(1 for result in results if result.resolved)
    failure_counts: dict[str, int] = {}
    memory_modes: dict[str, int] = {}
    stream_keys = _stream_summary_keys(results)
    streams: dict[str, dict[str, Any]] = {}
    for result in results:
        key = result.failure_type or "resolved"
        failure_counts[key] = failure_counts.get(key, 0) + 1
        memory_modes[result.memory_mode] = memory_modes.get(result.memory_mode, 0) + 1
        if result.memory_mode in {MEMORY_MODE_SHARED_STREAM, MEMORY_MODE_SHARED_BY_GROUP}:
            stream_key = stream_keys.get(id(result), result.stream_id or result.task_id)
            stream = streams.setdefault(
                stream_key,
                {
                    "memory_mode": result.memory_mode,
                    "stream_id": result.stream_id,
                    "total": 0,
                    "scored": 0,
                    "resolved": 0,
                    "solve_rate": 0.0,
                    "memory_dir": result.memory_dir,
                    "memory_project_key": result.memory_project_key,
                    "memory_entries_before": result.memory_entries_before,
                    "memory_entries_after": result.memory_entries_after,
                    "memory_growth": 0,
                    "memory_entries_total_before": result.memory_entries_total_before,
                    "memory_entries_total_after": result.memory_entries_total_after,
                    "memory_total_growth": 0,
                },
            )
            stream["total"] = int(stream["total"]) + 1
            stream["scored"] = int(stream["scored"]) + (1 if result.task_valid else 0)
            stream["resolved"] = int(stream["resolved"]) + (1 if result.resolved else 0)
            stream["memory_entries_after"] = result.memory_entries_after
            stream["memory_entries_total_after"] = result.memory_entries_total_after
            stream["memory_growth"] = result.memory_entries_after - int(stream["memory_entries_before"])
            stream["memory_total_growth"] = (
                result.memory_entries_total_after - int(stream["memory_entries_total_before"])
            )
    for stream in streams.values():
        stream_scored = int(stream["scored"])
        stream["solve_rate"] = int(stream["resolved"]) / stream_scored * 100 if stream_scored else 0.0
    return {
        "total": total,
        "scored": scored,
        "resolved": resolved,
        "invalid_initial_pass": failure_counts.get("invalid_initial_pass", 0),
        "visible_test_failed": failure_counts.get("visible_test_failed", 0),
        "hidden_test_failed": failure_counts.get("hidden_test_failed", 0),
        "agent_error": failure_counts.get("agent_error", 0),
        "solve_rate": resolved / scored * 100 if scored else 0.0,
        "end_to_end_rate": resolved / total * 100 if total else 0.0,
        "failure_counts": failure_counts,
        "memory": {
            "modes": memory_modes,
            "total_growth": sum(result.memory_total_growth for result in results),
            "visible_growth": sum(result.memory_growth for result in results),
        },
        "streams": streams,
    }


def _stream_summary_keys(results: Sequence[ManifestEvalResult]) -> dict[int, str]:
    shared = [
        result for result in results
        if result.memory_mode in {MEMORY_MODE_SHARED_STREAM, MEMORY_MODE_SHARED_BY_GROUP}
    ]
    identities_by_base: dict[str, set[tuple[str, str, str, str]]] = {}
    identities_by_mode_base: dict[tuple[str, str], set[tuple[str, str, str, str]]] = {}
    for result in shared:
        base = result.stream_id or result.task_id
        identity = _stream_summary_identity(result)
        identities_by_base.setdefault(base, set()).add(identity)
        identities_by_mode_base.setdefault((result.memory_mode, base), set()).add(identity)

    keys: dict[int, str] = {}
    for result in shared:
        base = result.stream_id or result.task_id
        if len(identities_by_base.get(base, set())) <= 1:
            keys[id(result)] = base
            continue
        mode_base = f"{result.memory_mode}:{base}"
        if len(identities_by_mode_base.get((result.memory_mode, base), set())) <= 1:
            keys[id(result)] = mode_base
            continue
        suffix_payload = json.dumps(_stream_summary_identity(result), ensure_ascii=False, sort_keys=True)
        suffix = hashlib.sha1(suffix_payload.encode("utf-8")).hexdigest()[:12]
        keys[id(result)] = f"{mode_base}:{suffix}"
    return keys


def _stream_summary_identity(result: ManifestEvalResult) -> tuple[str, str, str, str]:
    return (
        result.memory_mode,
        result.stream_id or result.task_id,
        result.memory_dir,
        result.memory_project_key,
    )


def _run_manifest_task(
    task: Mapping[str, Any],
    *,
    index: int,
    manifest_path: Path,
    output_dir: Path,
    work_root: Path,
    trace_root: Path,
    patch_root: Path,
    memory_root: Path,
    manifest_settings: ManifestSettings,
    config: AgentConfig,
    mode: str,
    max_steps: int | None,
    command_timeout: int,
    cli_env: Mapping[str, str],
    agent_runner: AgentRunnerFn,
    shared_policy: AgentLLM | None,
    shared_embedding_retriever: EmbeddingRetriever | None,
) -> ManifestEvalResult:
    started = time.monotonic()
    task_id = str(task.get("id") or f"task_{index}")
    source = str(task.get("source") or "local")
    task_group = str(task.get("task_group") or "").strip()
    tags = _string_list(task.get("tags"))
    expected_changed_files = _string_list(task.get("expected_changed_files"))
    safe_id = _safe_id(task_id)
    source_repo = _resolve_repo(task, manifest_path)
    task_dir = work_root / safe_id
    baseline_repo = task_dir / "baseline"
    initial_repo = task_dir / "initial"
    work_repo = task_dir / "repo"
    clean_repo = task_dir / "clean"
    task_trace_dir = trace_root / safe_id
    memory_stream = _resolve_memory_stream(
        task,
        manifest_settings=manifest_settings,
        manifest_path=manifest_path,
        memory_root=memory_root,
        safe_id=safe_id,
    )
    task_memory_dir = memory_stream.memory_dir
    _copy_repo(source_repo, baseline_repo)
    _copy_repo(source_repo, initial_repo)
    _copy_repo(source_repo, work_repo)
    _init_git_baseline(work_repo)
    task_trace_dir.mkdir(parents=True, exist_ok=True)
    task_memory_dir.mkdir(parents=True, exist_ok=True)

    env_overrides = _env_overrides(cli_env, task.get("env_overrides"))
    command_env = _command_env(env_overrides)
    task_config = _config_for_eval_env(
        config,
        env_overrides,
        trace_dir=task_trace_dir,
        memory_dir=task_memory_dir,
        memory_project_key=memory_stream.memory_project_key,
        command_timeout=command_timeout,
    )
    if task_config.memory_evolver_mode == "formal":
        task_config = replace(
            task_config,
            memory_evolver_collection_round=int(
                task.get("collection_round", task_config.memory_evolver_collection_round)
            ),
            memory_evolver_dataset_split=str(
                task.get("split") or task_config.memory_evolver_dataset_split
            ).strip().lower(),
        )
    before_counts = _memory_counts(task_config.memory_dir, project_key=task_config.memory_project_key)
    agent_test_command = task.get("agent_test_command") or task.get("test_command")
    visible_command = task.get("visible_test_command") or agent_test_command
    hidden_command = task.get("hidden_test_command")
    evaluator_name = "manifest"
    evaluator_version = "manifest-evaluator-v1"
    evaluator_hash = canonical_sha256({
        "visible_test_command": _command_label(visible_command),
        "hidden_test_command": _command_label(hidden_command),
        "expected_changed_files": expected_changed_files,
    })
    initial_visible = run_test_command(visible_command, cwd=initial_repo, timeout=command_timeout, env=command_env)
    initial_hidden = (
        run_test_command(hidden_command, cwd=initial_repo, timeout=command_timeout, env=command_env)
        if hidden_command
        else None
    )
    initial_hidden_ok = initial_hidden.ok if initial_hidden is not None else True
    task_valid = not (initial_visible.ok and initial_hidden_ok)
    if not task_valid:
        after_counts = _memory_counts(task_config.memory_dir, project_key=task_config.memory_project_key)
        memory_fields = _memory_result_fields(
            memory_stream=memory_stream,
            memory_dir=task_config.memory_dir,
            memory_project_key=task_config.memory_project_key,
            before_counts=before_counts,
            after_counts=after_counts,
        )
        return ManifestEvalResult(
            task_id=task_id,
            status="failed",
            resolved=False,
            task_valid=False,
            failure_type="invalid_initial_pass",
            initial_visible=initial_visible,
            source=source,
            task_group=task_group,
            reward=0.0,
            evaluator_name=evaluator_name,
            evaluator_version=evaluator_version,
            evaluator_hash=evaluator_hash,
            mode=mode,
            tags=tags,
            env_overrides=env_overrides,
            resolved_config=_config_snapshot(task_config),
            expected_changed_files=expected_changed_files,
            expected_changed_files_ok=None if not expected_changed_files else False,
            initial_hidden=initial_hidden,
            **memory_fields,
            elapsed_sec=time.monotonic() - started,
        )

    state: Any | None = None
    error = ""
    try:
        runner_kwargs: dict[str, Any] = {
            "repo_path": work_repo,
            "task": str(task.get("task") or ""),
            "test_command": _command_label(agent_test_command or visible_command),
            "config": task_config,
            "max_steps": _task_max_steps(task, max_steps, task_config.max_steps),
            "trace_dir": task_trace_dir,
            "mode": mode,
            "metadata": {
                "source_task": task_id,
                "task_id": task_id,
                "task_type": source or mode,
                "task_group": task_group,
                "stream_id": memory_stream.stream_id,
                "memory_mode": memory_stream.memory_mode,
                "memory_project_key": task_config.memory_project_key,
                "tags": tags,
            },
        }
        if shared_policy is not None:
            runner_kwargs["llm"] = shared_policy
        if shared_embedding_retriever is not None:
            runner_kwargs["memory_embedding_retriever"] = shared_embedding_retriever
        state = agent_runner(**runner_kwargs)
    except Exception as exc:  # noqa: BLE001 - evaluator records task-level agent failures.
        error = f"{type(exc).__name__}: {exc}"

    changed_files = compare_changed_files(baseline_repo, work_repo)
    expected_changed_files_ok = (
        None
        if not expected_changed_files
        else all(path in set(changed_files) for path in expected_changed_files)
    )
    patch_text = build_patch_diff(baseline_repo, work_repo, changed_files)
    patch_lines = len(patch_text.splitlines())
    patch_path = patch_root / f"{safe_id}.diff"
    patch_path.write_text(patch_text, encoding="utf-8")
    patch_apply_ok = False
    final_visible: CommandResult | None = None
    final_hidden: CommandResult | None = None
    if not error:
        _copy_repo(source_repo, clean_repo)
        patch_apply_ok = apply_changed_files(work_repo, clean_repo, changed_files)
        if patch_apply_ok:
            final_visible = run_test_command(visible_command, cwd=clean_repo, timeout=command_timeout, env=command_env)
            final_hidden = (
                run_test_command(hidden_command, cwd=clean_repo, timeout=command_timeout, env=command_env)
                if hidden_command
                else None
            )

    visible_ok = bool(final_visible and final_visible.ok)
    hidden_ok = bool(final_hidden.ok) if final_hidden is not None else True
    resolved = bool(patch_apply_ok and visible_ok and hidden_ok)
    failure_type = _failure_type(
        agent_error=bool(error),
        patch_apply_ok=patch_apply_ok,
        visible_ok=visible_ok,
        hidden_ok=hidden_ok,
        has_hidden=hidden_command is not None,
    )
    authoritative_resolved = resolved
    evolver_writer_status = ""
    written_memory_ids: list[str] = []
    repository_revision_after_writer = ""
    evolver_cadence_id = ""
    evolver_maintenance_status = ""
    if task_config.memory_evolver_mode == "formal":
        episode = getattr(state, "evolver_episode", None) if state is not None else None
        if not isinstance(episode, AgentEpisodeArtifact):
            error = error or "RuntimeError: formal task did not produce AgentEpisodeArtifact"
            resolved = False
            failure_type = "evolver_finalize_failed"
        else:
            try:
                final_store = ExperienceStore.from_dir(task_config.memory_dir)
                final_store.load()
                finalize_writer = TraceWriter(episode.trace_path)
                def trace_sink(event, payload):
                    finalize_writer.append(
                        TraceEvent(
                            event=event,
                            payload=payload,
                            run_id=str(getattr(state, "run_id", task_id)),
                        )
                    )
                active_coordinator = getattr(state, "evolver_coordinator", None)
                if isinstance(active_coordinator, EvolverCoordinator):
                    active_coordinator.set_trace_sink(trace_sink)
                    coordinator = active_coordinator
                else:
                    coordinator = EvolverCoordinator(
                        store=final_store,
                        project_key=episode.session.memory_project_key,
                        policy_identity=episode.session.policy_identity,
                        trace_sink=trace_sink,
                    )
                finalize_result = coordinator.finalize_task(
                    episode,
                    AuthoritativeTaskOutcome(
                        task_id=task_id,
                        task_group=task_group,
                        task_valid=True,
                        resolved=authoritative_resolved,
                        reward=1.0 if authoritative_resolved else 0.0,
                        evaluator=EvaluatorIdentity(
                            evaluator_name,
                            evaluator_version,
                            evaluator_hash,
                        ),
                        outcome_finalized=True,
                    ),
                )
                evolver_writer_status = finalize_result.writer_status
                written_memory_ids = list(finalize_result.written_memory_ids)
                repository_revision_after_writer = finalize_result.repository_revision_after
                evolver_cadence_id = finalize_result.cadence_id or ""
                evolver_maintenance_status = finalize_result.maintenance_status or ""
            except Exception as exc:  # noqa: BLE001 - formal collection must fail closed
                error = f"{type(exc).__name__}: {exc}"
                resolved = False
                failure_type = "evolver_finalize_failed"
    trace_path = str(getattr(state, "trace_path", "") or "")
    metrics = _metrics_for_trace(trace_path, task_trace_dir)
    after_counts = _memory_counts(task_config.memory_dir, project_key=task_config.memory_project_key)
    memory_fields = _memory_result_fields(
        memory_stream=memory_stream,
        memory_dir=task_config.memory_dir,
        memory_project_key=task_config.memory_project_key,
        before_counts=before_counts,
        after_counts=after_counts,
    )
    result = ManifestEvalResult(
        task_id=task_id,
        status="passed" if resolved else "failed",
        resolved=resolved,
        task_valid=True,
        failure_type=failure_type,
        initial_visible=initial_visible,
        source=source,
        task_group=task_group,
        reward=1.0 if authoritative_resolved else 0.0,
        evaluator_name=evaluator_name,
        evaluator_version=evaluator_version,
        evaluator_hash=evaluator_hash,
        evolver_writer_status=evolver_writer_status,
        written_memory_ids=written_memory_ids,
        repository_revision_after_writer=repository_revision_after_writer,
        evolver_cadence_id=evolver_cadence_id,
        evolver_maintenance_status=evolver_maintenance_status,
        mode=mode,
        tags=tags,
        env_overrides=env_overrides,
        resolved_config=_config_snapshot(task_config),
        expected_changed_files=expected_changed_files,
        expected_changed_files_ok=expected_changed_files_ok,
        initial_hidden=initial_hidden,
        final_visible=final_visible,
        final_hidden=final_hidden,
        patch_apply_ok=patch_apply_ok,
        changed_files=changed_files,
        patch_lines=patch_lines,
        patch_path=str(patch_path),
        trace_path=trace_path,
        metrics=metrics,
        **memory_fields,
        agent_steps=int(getattr(state, "steps", 0) or 0),
        agent_done=bool(getattr(state, "done", False)),
        agent_stop_reason=str(getattr(state, "stop_reason", "") or ""),
        error=error,
        elapsed_sec=time.monotonic() - started,
    )
    if state is not None:
        record_benchmark_result(
            state,
            benchmark="manifest",
            task_id=task_id,
            status=result.status,
            scored=result.task_valid,
            test_command=_command_label(visible_command),
            test_output=final_visible.output if final_visible else error,
            task_valid=result.task_valid,
            initial_visible_ok=initial_visible.ok,
            initial_hidden_ok=initial_hidden.ok if initial_hidden else None,
            visible_ok=visible_ok,
            hidden_ok=hidden_ok if hidden_command else None,
            resolved=resolved,
            failure_type=failure_type,
            patch_apply_ok=patch_apply_ok,
            changed_files=changed_files,
            patch_lines=patch_lines,
            visible_test_command=_command_label(visible_command),
            visible_test_output=final_visible.output if final_visible else None,
            initial_visible_output=initial_visible.output,
            memory_mode=result.memory_mode,
            stream_id=result.stream_id,
            memory_dir=result.memory_dir,
            memory_project_key=result.memory_project_key,
            memory_entries_before=result.memory_entries_before,
            memory_entries_after=result.memory_entries_after,
            memory_growth=result.memory_growth,
            memory_entries_total_before=result.memory_entries_total_before,
            memory_entries_total_after=result.memory_entries_total_after,
            memory_total_growth=result.memory_total_growth,
        )
    return result


def run_test_command(
    command: str | Sequence[str] | None,
    *,
    cwd: Path,
    timeout: int,
    env: Mapping[str, str],
) -> CommandResult:
    if command is None or (isinstance(command, str) and not command.strip()):
        return CommandResult(command="", ok=True, returncode=0, skipped=True)
    argv = _command_argv(command)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=dict(env),
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return CommandResult(
            command=_command_label(command),
            ok=completed.returncode == 0,
            returncode=completed.returncode,
            output=output,
            elapsed_sec=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
        return CommandResult(
            command=_command_label(command),
            ok=False,
            returncode=124,
            output=output or f"Command timed out after {timeout}s.",
            elapsed_sec=time.monotonic() - started,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            command=_command_label(command),
            ok=False,
            returncode=127,
            output=f"Command not found: {exc}",
            elapsed_sec=time.monotonic() - started,
        )


def compare_changed_files(baseline: Path, changed: Path) -> list[str]:
    baseline_files = _file_map(baseline)
    changed_files = _file_map(changed)
    paths = sorted(set(baseline_files).union(changed_files))
    result: list[str] = []
    for rel in paths:
        old = baseline_files.get(rel)
        new = changed_files.get(rel)
        if old is None or new is None or old.read_bytes() != new.read_bytes():
            result.append(rel)
    return result


def count_patch_lines(baseline: Path, changed: Path, files: Sequence[str]) -> int:
    return len(build_patch_diff(baseline, changed, files).splitlines())


def build_patch_diff(baseline: Path, changed: Path, files: Sequence[str]) -> str:
    lines: list[str] = []
    for rel in files:
        old_path = baseline / rel
        new_path = changed / rel
        old_text = _read_text(old_path) if old_path.exists() else ""
        new_text = _read_text(new_path) if new_path.exists() else ""
        if old_text is None or new_text is None:
            lines.append(f"Binary files a/{rel} and b/{rel} differ")
            continue
        diff = difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
        lines.extend(diff)
    return "\n".join(lines) + ("\n" if lines else "")


def apply_changed_files(source: Path, target: Path, files: Sequence[str]) -> bool:
    try:
        for rel in files:
            src = source / rel
            dst = target / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif dst.exists():
                dst.unlink()
        return True
    except OSError:
        return False


def _resolve_repo(task: Mapping[str, Any], manifest_path: Path) -> Path:
    repo = task.get("repo")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError("Manifest task requires a non-empty repo field.")
    path = Path(repo).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Task repo not found: {path}")
    return path.resolve()


def _copy_repo(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*_IGNORE_DIRS))


def _init_git_baseline(repo: Path) -> None:
    commands = [
        ["git", "init"],
        ["git", "config", "user.email", "agentcli@example.invalid"],
        ["git", "config", "user.name", "AgentCli Evaluator"],
        ["git", "config", "commit.gpgsign", "false"],
        ["git", "add", "-A"],
        ["git", "commit", "--allow-empty", "--no-gpg-sign", "-m", "baseline"],
    ]
    for command in commands:
        _run_git_baseline_command(repo, command)


def _run_git_baseline_command(repo: Path, command: Sequence[str]) -> None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(repo),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Failed to initialize git baseline: git executable was not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Failed to initialize git baseline: {' '.join(command)} timed out.") from exc
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout or "").strip()
        detail = f": {output}" if output else ""
        raise RuntimeError(f"Failed to initialize git baseline with {' '.join(command)}{detail}")


def _file_map(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _IGNORE_DIRS for part in rel.parts):
            continue
        files[rel.as_posix()] = path
    return files


def _command_argv(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    argv = [str(part) for part in command]
    if not argv:
        raise ValueError("Command argv list must not be empty.")
    return argv


def _command_label(command: object) -> str:
    if command is None:
        return ""
    if isinstance(command, str):
        return command
    if isinstance(command, Sequence):
        return " ".join(shlex.quote(str(part)) for part in command)
    return str(command)


def _env_overrides(cli_env: Mapping[str, str], task_env: object) -> dict[str, str]:
    overrides = {str(key): str(value) for key, value in cli_env.items()}
    if task_env is None:
        return overrides
    if not isinstance(task_env, Mapping):
        raise ValueError("env_overrides must be a JSON object.")
    overrides.update({str(key): str(value) for key, value in task_env.items()})
    return overrides


def _command_env(env_overrides: Mapping[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in env_overrides.items()})
    return env


def _config_for_eval_env(
    config: AgentConfig,
    env_overrides: Mapping[str, str],
    *,
    trace_dir: Path,
    memory_dir: Path,
    memory_project_key: str = "",
    command_timeout: int,
) -> AgentConfig:
    values = _config_env_values(config)
    overrides = {str(key): str(value) for key, value in env_overrides.items()}
    values.update(overrides)
    if "AGENTCLI_MEMORY_DIR" in overrides:
        values["AGENTCLI_MEMORY_DIR"] = overrides["AGENTCLI_MEMORY_DIR"]
    elif "MY_AGENT_MEMORY_DIR" in overrides:
        values["AGENTCLI_MEMORY_DIR"] = overrides["MY_AGENT_MEMORY_DIR"]
    else:
        values["AGENTCLI_MEMORY_DIR"] = str(memory_dir)
    if "AGENTCLI_MEMORY_PROJECT_KEY" in overrides:
        values["AGENTCLI_MEMORY_PROJECT_KEY"] = overrides["AGENTCLI_MEMORY_PROJECT_KEY"]
    elif "MY_AGENT_MEMORY_PROJECT_KEY" in overrides:
        values["AGENTCLI_MEMORY_PROJECT_KEY"] = overrides["MY_AGENT_MEMORY_PROJECT_KEY"]
    elif memory_project_key:
        values["AGENTCLI_MEMORY_PROJECT_KEY"] = memory_project_key
    _normalize_evolver_mode_overrides(values, overrides)
    for agentcli_key, my_agent_key in (
        (
            "AGENTCLI_MEMORY_EVOLVER_CANDIDATE_TOP_K_PER_TIER",
            "MY_AGENT_MEMORY_EVOLVER_CANDIDATE_TOP_K_PER_TIER",
        ),
        (
            "AGENTCLI_MEMORY_EVOLVER_SELECTION_PROMPT_TOKENS",
            "MY_AGENT_MEMORY_EVOLVER_SELECTION_PROMPT_TOKENS",
        ),
        (
            "AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_INTERVAL_TASKS",
            "MY_AGENT_MEMORY_EVOLVER_MAINTENANCE_INTERVAL_TASKS",
        ),
        (
            "AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_MAX_TURNS",
            "MY_AGENT_MEMORY_EVOLVER_MAINTENANCE_MAX_TURNS",
        ),
        ("AGENTCLI_MEMORY_EVOLVER_DATASET_DIR", "MY_AGENT_MEMORY_EVOLVER_DATASET_DIR"),
        (
            "AGENTCLI_MEMORY_EVOLVER_TEACHER_MIN_SCORE",
            "MY_AGENT_MEMORY_EVOLVER_TEACHER_MIN_SCORE",
        ),
        (
            "AGENTCLI_MEMORY_EVOLVER_WRITING_TOP_FRACTION",
            "MY_AGENT_MEMORY_EVOLVER_WRITING_TOP_FRACTION",
        ),
        ("AGENTCLI_EMBEDDING_MODEL", "MY_AGENT_EMBEDDING_MODEL"),
        ("AGENTCLI_EMBEDDING_REVISION", "MY_AGENT_EMBEDDING_REVISION"),
        ("AGENTCLI_POLICY_BACKEND", "MY_AGENT_POLICY_BACKEND"),
        ("AGENTCLI_POLICY_BASE_MODEL", "MY_AGENT_POLICY_BASE_MODEL"),
        ("AGENTCLI_POLICY_BASE_REVISION", "MY_AGENT_POLICY_BASE_REVISION"),
        ("AGENTCLI_POLICY_ADAPTER_PATH", "MY_AGENT_POLICY_ADAPTER_PATH"),
        ("AGENTCLI_POLICY_IDENTITY_MANIFEST", "MY_AGENT_POLICY_IDENTITY_MANIFEST"),
        ("AGENTCLI_POLICY_TOKENIZER_REVISION", "MY_AGENT_POLICY_TOKENIZER_REVISION"),
        ("AGENTCLI_POLICY_CHAT_TEMPLATE", "MY_AGENT_POLICY_CHAT_TEMPLATE"),
        ("AGENTCLI_POLICY_DTYPE", "MY_AGENT_POLICY_DTYPE"),
        ("AGENTCLI_POLICY_DEVICE", "MY_AGENT_POLICY_DEVICE"),
        ("AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER", "MY_AGENT_MEMORY_EVOLVER_TOP_K_PER_TIER"),
        ("AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS", "MY_AGENT_MEMORY_EVOLVER_SELECTED_MAX_ITEMS"),
        ("AGENTCLI_MEMORY_EVOLVER_MIN_SCORE", "MY_AGENT_MEMORY_EVOLVER_MIN_SCORE"),
        ("AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES", "MY_AGENT_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES"),
        ("AGENTCLI_MEMORY_EVOLVER_WRITER", "MY_AGENT_MEMORY_EVOLVER_WRITER"),
        ("AGENTCLI_MEMORY_EVOLVER_WRITER_MODE", "MY_AGENT_MEMORY_EVOLVER_WRITER_MODE"),
        ("AGENTCLI_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE", "MY_AGENT_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE"),
        ("AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_RECORDS", "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_RECORDS"),
        ("AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS", "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS"),
        ("AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS", "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS"),
        ("AGENTCLI_MEMORY_EVOLVER_WRITER_DATASET_PATH", "MY_AGENT_MEMORY_EVOLVER_WRITER_DATASET_PATH"),
    ):
        _prefer_my_agent_override(values, overrides, agentcli_key, my_agent_key)
    resolved = AgentConfig.from_env(env=values, require_env_file=False)
    return replace(
        resolved,
        trace_dir=trace_dir,
        command_timeout=command_timeout,
        memory_evolver_tier_caps=dict(config.memory_evolver_tier_caps),
        memory_evolver_tier_weights=dict(config.memory_evolver_tier_weights),
        tool_env_overrides=overrides,
    )


def _normalize_evolver_mode_overrides(values: dict[str, str], overrides: Mapping[str, str]) -> None:
    if "AGENTCLI_MEMORY_EVOLVER_MODE" in overrides:
        return
    if "MY_AGENT_MEMORY_EVOLVER_MODE" in overrides:
        values["AGENTCLI_MEMORY_EVOLVER_MODE"] = overrides["MY_AGENT_MEMORY_EVOLVER_MODE"]
        return
    if "AGENTCLI_MEMORY_EVOLVER" in overrides:
        values["AGENTCLI_MEMORY_EVOLVER_MODE"] = (
            "retrieve_select" if _bool_from_env(overrides["AGENTCLI_MEMORY_EVOLVER"]) else "off"
        )
        return
    if "MY_AGENT_MEMORY_EVOLVER" in overrides:
        values["AGENTCLI_MEMORY_EVOLVER_MODE"] = (
            "retrieve_select" if _bool_from_env(overrides["MY_AGENT_MEMORY_EVOLVER"]) else "off"
        )


def _prefer_my_agent_override(
    values: dict[str, str],
    overrides: Mapping[str, str],
    agentcli_key: str,
    my_agent_key: str,
) -> None:
    if my_agent_key in overrides and agentcli_key not in overrides:
        values[agentcli_key] = overrides[my_agent_key]


def _config_snapshot(config: AgentConfig) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        if item.name == "api_key":
            snapshot[item.name] = "<redacted>" if value else ""
        elif isinstance(value, Path):
            snapshot[item.name] = str(value)
        elif isinstance(value, tuple):
            snapshot[item.name] = [str(part) if isinstance(part, Path) else part for part in value]
        elif isinstance(value, Mapping):
            snapshot[item.name] = {str(key): str(val) for key, val in value.items()}
        else:
            snapshot[item.name] = value
    return snapshot


def _config_env_values(config: AgentConfig) -> dict[str, str]:
    values = {
        "MY_AGENT_LLM_PROVIDER": config.provider,
        "MY_AGENT_USE_FAKE_LLM": _bool_env(config.use_fake_llm),
        "MY_AGENT_API_KEY": config.api_key,
        "MY_AGENT_MODEL": config.model,
        "MY_AGENT_TEMPERATURE": str(config.temperature),
        "MY_AGENT_MAX_STEPS": str(config.max_steps),
        "MY_AGENT_COMMAND_TIMEOUT": str(config.command_timeout),
        "MY_AGENT_TRACE_DIR": str(config.trace_dir),
        "AGENTCLI_ENABLE_PROJECT_TOOLS": _bool_env(config.enable_project_tools),
        "AGENTCLI_ENABLE_PROJECT_PLUGINS": _bool_env(config.enable_project_plugins),
        "MY_AGENT_MAX_ITERATIONS": str(config.max_iterations),
        "MY_AGENT_MAX_TOOL_CALLS": str(config.max_tool_calls),
        "MY_AGENT_MAX_ELAPSED_SECONDS": str(config.max_elapsed_seconds),
        "MY_AGENT_STAGNATION_WINDOW": str(config.stagnation_window),
        "MY_AGENT_REPEATED_FAILURE_WINDOW": str(config.repeated_failure_window),
        "MY_AGENT_RETAIN_RECENT_TURNS": str(config.retain_recent_user_turns),
        "MY_AGENT_MAX_TOOL_RESULT_CHARS": str(config.max_tool_result_chars),
        "MY_AGENT_MAX_SUMMARY_INPUT_CHARS": str(config.max_summary_input_chars),
        "AGENTCLI_PLAN_TASK_MAX_STEPS": str(config.plan_task_max_steps),
        "AGENTCLI_PLAN_MAX_TASKS": str(config.plan_max_tasks),
        "AGENTCLI_PLAN_MAX_REPLANS": str(config.plan_max_replans),
        "AGENTCLI_AGENT_MODE": config.agent_mode,
        "AGENTCLI_TEAM_WORKERS": str(config.team_worker_count),
        "AGENTCLI_TEAM_MAX_STEPS": str(config.team_max_steps),
        "AGENTCLI_TEAM_MAX_RETRIES": str(config.team_max_retries),
        "AGENTCLI_TEAM_STEP_MAX_STEPS": str(config.team_step_max_steps),
        "AGENTCLI_TEAM_DEPENDENCY_CONTEXT_CHARS": str(config.team_dependency_context_chars),
        "AGENTCLI_TEAM_PARALLEL": _bool_env(config.team_parallel_enabled),
        "AGENTCLI_TEAM_ALLOW_UNAPPROVED_RESULTS": _bool_env(config.team_allow_unapproved_results),
        "AGENTCLI_MEMORY": _bool_env(config.memory_enabled),
        "AGENTCLI_MEMORY_DIR": str(config.memory_dir),
        "AGENTCLI_MEMORY_PROJECT_KEY": config.memory_project_key,
        "AGENTCLI_MEMORY_SHORT_TERM_ENTRIES": str(config.memory_short_term_entries),
        "AGENTCLI_MEMORY_RETRIEVAL_LIMIT": str(config.memory_retrieval_limit),
        "AGENTCLI_MEMORY_COMPRESSION_TRIGGER_RATIO": str(config.memory_compression_trigger_ratio),
        "AGENTCLI_MEMORY_RETAIN_RECENT_TURNS": str(config.memory_retain_recent_turns),
        "AGENTCLI_MEMORY_MAP_CHUNK_SIZE": str(config.memory_map_chunk_size),
        "AGENTCLI_MEMORY_EVOLVER_MODE": config.memory_evolver_mode,
        "AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS": str(config.memory_evolver_selected_max_items),
        "AGENTCLI_HITL": _bool_env(config.hitl_enabled),
        "AGENTCLI_HITL_AUDIT_DIR": str(config.hitl_audit_dir),
        "AGENTCLI_HITL_NON_INTERACTIVE": config.hitl_non_interactive,
        "AGENTCLI_HITL_MEDIUM_RISK_MODE": config.hitl_medium_risk_mode,
        "AGENTCLI_HITL_LLM_JUDGE": _bool_env(config.hitl_llm_judge_enabled),
        "AGENTCLI_MAX_PARALLEL_TOOLS": str(config.max_parallel_tools),
        "AGENTCLI_TOOL_BATCH_TIMEOUT_SECONDS": str(config.tool_batch_timeout_seconds),
        "AGENTCLI_TOOL_SHUTDOWN_GRACE_SECONDS": str(config.tool_shutdown_grace_seconds),
        "AGENTCLI_MAX_PROCESS_OUTPUT_CHARS": str(config.max_process_output_chars),
        "AGENTCLI_PLAN_PARALLEL": _bool_env(config.plan_parallel_enabled),
        "AGENTCLI_PLAN_MAX_PARALLEL_TASKS": str(config.plan_max_parallel_tasks),
        "AGENTCLI_PLAN_TASK_BATCH_TIMEOUT_SECONDS": str(config.plan_task_batch_timeout_seconds),
        "AGENTCLI_TEAM_STEP_BATCH_TIMEOUT_SECONDS": str(config.team_step_batch_timeout_seconds),
        "AGENTCLI_MCP": _bool_env(config.mcp_enabled),
        "AGENTCLI_MCP_STARTUP_WAIT_SECONDS": str(config.mcp_startup_wait_seconds),
        "AGENTCLI_MCP_INITIALIZE_TIMEOUT_SECONDS": str(config.mcp_initialize_timeout_seconds),
        "AGENTCLI_MCP_CALL_TIMEOUT_SECONDS": str(config.mcp_call_timeout_seconds),
        "AGENTCLI_MCP_MAX_STARTUP_WORKERS": str(config.mcp_max_startup_workers),
        "AGENTCLI_MCP_REQUIRE_APPROVAL": _bool_env(config.mcp_require_approval),
        "AGENTCLI_MCP_ENABLE_PROJECT_SERVERS": _bool_env(config.mcp_enable_project_servers),
    }
    if config.base_url:
        values["MY_AGENT_BASE_URL"] = config.base_url
    if config.reasoning_effort:
        values["MY_AGENT_REASONING_EFFORT"] = config.reasoning_effort
    if config.memory_evolver_mode == "formal":
        values.update({
            "AGENTCLI_MEMORY_EVOLVER_CANDIDATE_TOP_K_PER_TIER": str(
                config.memory_evolver_candidate_top_k_per_tier
            ),
            "AGENTCLI_MEMORY_EVOLVER_SELECTION_PROMPT_TOKENS": str(
                config.memory_evolver_selection_prompt_tokens
            ),
            "AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_INTERVAL_TASKS": str(
                config.memory_evolver_maintenance_interval_tasks
            ),
            "AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_MAX_TURNS": str(
                config.memory_evolver_maintenance_max_turns
            ),
            "AGENTCLI_MEMORY_EVOLVER_DATASET_DIR": (
                str(config.memory_evolver_dataset_dir) if config.memory_evolver_dataset_dir else ""
            ),
            "AGENTCLI_MEMORY_EVOLVER_COLLECTION_ROUND": str(
                config.memory_evolver_collection_round
            ),
            "AGENTCLI_MEMORY_EVOLVER_DATASET_SPLIT": config.memory_evolver_dataset_split,
            "AGENTCLI_MEMORY_EVOLVER_TEACHER_MIN_SCORE": str(
                config.memory_evolver_teacher_min_score
            ),
            "AGENTCLI_MEMORY_EVOLVER_WRITING_TOP_FRACTION": str(
                config.memory_evolver_writing_top_fraction
            ),
            "AGENTCLI_OPD_ABLATION": config.opd_ablation,
            "AGENTCLI_MEMORY_EVOLVER_RETRIEVAL_BACKEND": (
                config.memory_evolver_retrieval_backend
            ),
            "AGENTCLI_MEMORY_EVOLVER_SELECTION_BACKEND": (
                config.memory_evolver_selection_backend
            ),
            "AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_ENABLED": _bool_env(
                config.memory_evolver_maintenance_enabled
            ),
            "AGENTCLI_EMBEDDING_MODEL": config.embedding_model,
            "AGENTCLI_EMBEDDING_REVISION": config.embedding_revision,
            "AGENTCLI_POLICY_BACKEND": config.policy_backend,
            "AGENTCLI_POLICY_BASE_MODEL": config.policy_base_model,
            "AGENTCLI_POLICY_BASE_REVISION": config.policy_base_revision,
            "AGENTCLI_POLICY_ADAPTER_PATH": (
                str(config.policy_adapter_path) if config.policy_adapter_path else ""
            ),
            "AGENTCLI_POLICY_IDENTITY_MANIFEST": (
                str(config.policy_identity_manifest) if config.policy_identity_manifest else ""
            ),
            "AGENTCLI_POLICY_TOKENIZER_REVISION": config.policy_tokenizer_revision,
            "AGENTCLI_POLICY_CHAT_TEMPLATE": config.policy_chat_template,
            "AGENTCLI_POLICY_DTYPE": config.policy_dtype,
            "AGENTCLI_POLICY_DEVICE": config.policy_device,
        })
    else:
        values.update({
            "AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER": str(config.memory_evolver_top_k_per_tier),
            "AGENTCLI_MEMORY_EVOLVER_MIN_SCORE": str(config.memory_evolver_min_score),
            "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES": str(
                config.memory_evolver_min_experience_entries
            ),
            "AGENTCLI_MEMORY_EVOLVER_WRITER": _bool_env(config.memory_evolver_writer_enabled),
            "AGENTCLI_MEMORY_EVOLVER_WRITER_MODE": config.memory_evolver_writer_mode,
            "AGENTCLI_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE": str(
                config.memory_evolver_writer_min_confidence
            ),
            "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_RECORDS": str(
                config.memory_evolver_writer_max_records
            ),
            "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS": str(
                config.memory_evolver_writer_max_input_chars
            ),
            "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS": str(
                config.memory_evolver_writer_max_content_chars
            ),
            "AGENTCLI_MEMORY_EVOLVER_WRITER_DATASET_PATH": (
                str(config.memory_evolver_writer_dataset_path)
                if config.memory_evolver_writer_dataset_path
                else ""
            ),
        })
    if config.token_budget is not None:
        values["MY_AGENT_TOKEN_BUDGET"] = str(config.token_budget)
    if config.tool_config_paths:
        values["AGENTCLI_TOOL_CONFIGS"] = os.pathsep.join(str(path) for path in config.tool_config_paths)
    if _config_value_is_explicit(config, "context_window", DEFAULT_CONTEXT_WINDOW, "context_window_explicit"):
        values["MY_AGENT_CONTEXT_WINDOW"] = str(config.context_window)
    if _config_value_is_explicit(
        config,
        "response_reserve_tokens",
        8_000,
        "response_reserve_tokens_explicit",
    ):
        values["MY_AGENT_RESPONSE_RESERVE_TOKENS"] = str(config.response_reserve_tokens)
    if _config_value_is_explicit(
        config,
        "compression_buffer_tokens",
        8_000,
        "compression_buffer_tokens_explicit",
    ):
        values["MY_AGENT_COMPRESSION_BUFFER_TOKENS"] = str(config.compression_buffer_tokens)
    if _config_value_is_explicit(
        config,
        "repo_context_budget_tokens",
        DEFAULT_REPO_CONTEXT_BUDGET_TOKENS,
        "repo_context_budget_tokens_explicit",
    ):
        values["AGENTCLI_REPO_CONTEXT_BUDGET_TOKENS"] = str(config.repo_context_budget_tokens)
    if _config_value_is_explicit(
        config,
        "tool_schema_budget_tokens",
        DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS,
        "tool_schema_budget_tokens_explicit",
    ):
        values["AGENTCLI_TOOL_SCHEMA_BUDGET_TOKENS"] = str(config.tool_schema_budget_tokens)
    if _config_value_is_explicit(
        config,
        "memory_short_term_tokens",
        DEFAULT_SHORT_TERM_TOKENS,
        "memory_short_term_tokens_explicit",
    ):
        values["AGENTCLI_MEMORY_SHORT_TERM_TOKENS"] = str(config.memory_short_term_tokens)
    if _config_value_is_explicit(
        config,
        "memory_context_tokens",
        DEFAULT_MEMORY_CONTEXT_TOKENS,
        "memory_context_tokens_explicit",
    ):
        values["AGENTCLI_MEMORY_CONTEXT_TOKENS"] = str(config.memory_context_tokens)
    if _config_value_is_explicit(
        config,
        "memory_tool_result_chars",
        DEFAULT_TOOL_RESULT_CHARS,
        "memory_tool_result_chars_explicit",
    ):
        values["AGENTCLI_MEMORY_TOOL_RESULT_CHARS"] = str(config.memory_tool_result_chars)
    return values


def _config_value_is_explicit(config: AgentConfig, field_name: str, default: int, explicit_attr: str) -> bool:
    if getattr(config, explicit_attr, False):
        return True
    value = getattr(config, field_name)
    return isinstance(value, int) and not isinstance(value, bool) and value != default


def _bool_env(value: bool) -> str:
    return "1" if value else "0"


def _bool_from_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _task_max_steps(task: Mapping[str, Any], cli_max_steps: int | None, default: int) -> int:
    raw = task.get("max_steps")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    if cli_max_steps is not None:
        return max(1, cli_max_steps)
    return max(1, default)


def _failure_type(
    *,
    agent_error: bool,
    patch_apply_ok: bool,
    visible_ok: bool,
    hidden_ok: bool,
    has_hidden: bool,
) -> str:
    if agent_error:
        return "agent_error"
    if not patch_apply_ok:
        return "patch_apply_failed"
    if not visible_ok:
        return "visible_test_failed"
    if has_hidden and not hidden_ok:
        return "hidden_test_failed"
    return ""


def _metrics_for_trace(trace_path: str, trace_dir: Path) -> dict[str, Any]:
    try:
        target = Path(trace_path) if trace_path else trace_dir
        return collect_trace_metrics(target, recursive=True).to_dict()
    except (FileNotFoundError, ValueError):
        return {}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe[:80] or "task"
