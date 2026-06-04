from __future__ import annotations

from datetime import datetime
from pathlib import Path

from my_agent.schema import TraceEvent


class TraceWriter:
    def __init__(self, trace_path: str | Path):
        self.path = Path(trace_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, trace_dir: str | Path, run_id: str) -> "TraceWriter":
        directory = Path(trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = run_id.replace("/", "_")[:8]
        return cls(directory / f"agent_trace_{timestamp}_{suffix}.jsonl")

    def append(self, event: TraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(event.to_json_line() + "\n")
