from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Iterable

from my_agent.schema import AgentState, TraceEvent


class TraceWriter:
    def __init__(self, trace_path: str | Path):
        self.path = Path(trace_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @classmethod
    def create(cls, trace_dir: str | Path, run_id: str) -> "TraceWriter":
        directory = Path(trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = _safe_trace_suffix(run_id)
        return cls(directory / f"agent_trace_{timestamp}_{suffix}.jsonl")

    def append(self, event: TraceEvent) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(event.to_json_line() + "\n")


def _safe_trace_suffix(run_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in run_id)
    return (safe or "run")[:64]


def append_benchmark_result(
    trace_path: str | Path,
    *,
    run_id: str,
    benchmark: str,
    task_id: str,
    status: str,
    scored: bool,
    test_command: str,
    test_output: str,
    task_valid: bool | None = None,
    initial_visible_ok: bool | None = None,
    initial_hidden_ok: bool | None = None,
    visible_ok: bool | None = None,
    hidden_ok: bool | None = None,
    resolved: bool | None = None,
    failure_type: str | None = None,
    patch_apply_ok: bool | None = None,
    changed_files: Iterable[str] | None = None,
    patch_lines: int | None = None,
    visible_test_command: str | None = None,
    visible_test_output: str | None = None,
    hidden_test_command: str | None = None,
    hidden_test_output: str | None = None,
    initial_visible_output: str | None = None,
    initial_hidden_output: str | None = None,
    memory_mode: str | None = None,
    stream_id: str | None = None,
    memory_dir: str | None = None,
    memory_project_key: str | None = None,
    memory_entries_before: int | None = None,
    memory_entries_after: int | None = None,
    memory_growth: int | None = None,
    memory_entries_total_before: int | None = None,
    memory_entries_total_after: int | None = None,
    memory_total_growth: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "task_id": task_id,
        "status": status,
        "scored": scored,
        "test_command": test_command,
        "test_output": test_output[:2000],
    }
    _add_optional(
        payload,
        task_valid=task_valid,
        initial_visible_ok=initial_visible_ok,
        initial_hidden_ok=initial_hidden_ok,
        visible_ok=visible_ok,
        hidden_ok=hidden_ok,
        resolved=resolved,
        failure_type=failure_type,
        patch_apply_ok=patch_apply_ok,
        changed_files=list(changed_files) if changed_files is not None else None,
        patch_lines=patch_lines,
        visible_test_command=visible_test_command,
        visible_test_output=_truncate_optional(visible_test_output),
        hidden_test_command=hidden_test_command,
        hidden_test_output=_truncate_optional(hidden_test_output),
        initial_visible_output=_truncate_optional(initial_visible_output),
        initial_hidden_output=_truncate_optional(initial_hidden_output),
        memory_mode=memory_mode,
        stream_id=stream_id,
        memory_dir=memory_dir,
        memory_project_key=memory_project_key,
        memory_entries_before=memory_entries_before,
        memory_entries_after=memory_entries_after,
        memory_growth=memory_growth,
        memory_entries_total_before=memory_entries_total_before,
        memory_entries_total_after=memory_entries_total_after,
        memory_total_growth=memory_total_growth,
    )
    TraceWriter(trace_path).append(TraceEvent(event="benchmark_result", payload=payload, run_id=run_id))


def append_agent_completed(
    writer: TraceWriter,
    state: AgentState,
    *,
    mode: str,
    run_label: str,
    child_trace_paths: Iterable[str | Path] = (),
    status: str | None = None,
) -> None:
    payload = {
        "mode": mode,
        "run_label": run_label,
        "stop_reason": state.stop_reason,
        "steps": state.steps,
        "done": state.done,
        "status": status or agent_status(done=state.done, stop_reason=state.stop_reason),
        "trace_path": str(state.trace_path or writer.path),
        "child_trace_paths": [str(path) for path in child_trace_paths if str(path)],
    }
    writer.append(TraceEvent(event="agent.completed", payload=payload, run_id=state.run_id))


def agent_status(*, done: bool, stop_reason: str) -> str:
    normalized = (stop_reason or "").lower()
    if "cancelled" in normalized or "canceled" in normalized:
        return "cancelled"
    failed_markers = ("failed", "failure", "error", "invalid", "validation", "llm_failed")
    if any(marker in normalized for marker in failed_markers):
        return "failed"
    stopped_markers = ("max_", "budget", "timeout", "timed_out")
    if any(marker in normalized for marker in stopped_markers):
        return "stopped"
    return "completed" if done else "running"


def _add_optional(payload: dict[str, Any], **values: Any) -> None:
    for key, value in values.items():
        if value is not None:
            payload[key] = value


def _truncate_optional(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:2000]
