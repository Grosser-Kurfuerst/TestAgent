from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading

from my_agent.schema import TraceEvent


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
) -> None:
    payload = {
        "benchmark": benchmark,
        "task_id": task_id,
        "status": status,
        "scored": scored,
        "test_command": test_command,
        "test_output": test_output[:2000],
    }
    TraceWriter(trace_path).append(TraceEvent(event="benchmark_result", payload=payload, run_id=run_id))
