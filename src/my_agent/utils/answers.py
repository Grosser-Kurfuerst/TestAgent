from __future__ import annotations

from pathlib import Path


def append_trace_to_answer(answer: str, trace_path: str | Path | None) -> str:
    if trace_path and "Trace:" not in answer:
        return f"{answer}\nTrace: {trace_path}"
    return answer

