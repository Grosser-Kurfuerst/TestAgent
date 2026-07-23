"""Common adapter protocol and official scorer execution boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from my_agent.evaluation.memory_benchmark.contracts import (
    BenchmarkTask,
    OfficialEvaluatorResult,
    PreparedBenchmarkTask,
    write_official_result_atomic,
)


class BenchmarkAdapter(Protocol):
    name: str

    def load_tasks(self, *, limit: int) -> Sequence[BenchmarkTask]: ...

    def prepare_task(
        self,
        task: BenchmarkTask,
        *,
        task_dir: Path,
        seed: int,
    ) -> PreparedBenchmarkTask: ...

    def finalize_task_artifacts(self, prepared: PreparedBenchmarkTask) -> None: ...

    def cleanup_task(self, prepared: PreparedBenchmarkTask) -> None: ...


def execute_official_scorer(
    scorer: Callable[[], OfficialEvaluatorResult],
    *,
    official_result_path: str | Path,
) -> int:
    """Execute one scorer, persist only a valid result, and normalize exit codes."""

    target = Path(official_result_path).expanduser().resolve()
    try:
        target.unlink(missing_ok=True)
    except OSError:
        return 3
    try:
        result = scorer()
    except Exception:  # noqa: BLE001 - hidden scorer details must not escape publicly.
        return 2
    if not isinstance(result, OfficialEvaluatorResult):
        return 2
    try:
        write_official_result_atomic(target, result)
    except (OSError, ValueError):
        return 3
    return 0 if result.resolved else 1


__all__ = ["BenchmarkAdapter", "execute_official_scorer"]
