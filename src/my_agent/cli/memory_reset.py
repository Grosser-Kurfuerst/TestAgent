"""Explicit destructive reset for the four-tier memory cutover."""

from __future__ import annotations

import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from filelock import FileLock, Timeout as FileLockTimeout

from my_agent.memory.experience_store import (
    EXPERIENCE_LOCK_FILE,
    EXPERIENCE_STORAGE_FILE,
    LEGACY_LONG_TERM_STORAGE_FILE,
)
from my_agent.memory.store_errors import MemoryStoreLockTimeout


RESET_CONFIRMATION = "RESET_FOUR_TIER_MEMORY"

_LEGACY_LOCK_FILE = ".long_term_memory.lock"
_RESET_FILES = (
    LEGACY_LONG_TERM_STORAGE_FILE,
    EXPERIENCE_STORAGE_FILE,
    "usage_logs.jsonl",
    "memory_attribution.jsonl",
    "maintenance_plan.json",
    "maintenance_plan.json.summary.json",
    "maintenance_summary.json",
    "maintenance_trace.jsonl",
    "maintenance_history.jsonl",
)
_RESET_DIRECTORIES = ("maintenance_backups",)


@dataclass(frozen=True)
class MemoryResetResult:
    memory_dir: Path
    dry_run: bool
    removed: tuple[str, ...]
    absent: tuple[str, ...]


def reset_memory_directory(
    memory_dir: str | Path,
    *,
    confirmation: str,
    dry_run: bool = False,
    lock_timeout_seconds: float = 30.0,
) -> MemoryResetResult:
    """Remove repository and id-coupled artifacts after explicit confirmation."""
    if confirmation != RESET_CONFIRMATION:
        raise ValueError(
            f"reset requires --confirm-reset {RESET_CONFIRMATION}"
        )
    if not isfinite(lock_timeout_seconds) or lock_timeout_seconds < 0:
        raise ValueError("lock_timeout_seconds must be finite and non-negative")
    root = Path(memory_dir)
    if root.exists() and not root.is_dir():
        raise ValueError("memory_dir must be a directory")
    root.mkdir(parents=True, exist_ok=True)

    targets = tuple(root / name for name in (*_RESET_FILES, *_RESET_DIRECTORIES))
    removed = tuple(path.name for path in targets if path.exists() or path.is_symlink())
    absent = tuple(path.name for path in targets if path.name not in set(removed))
    if dry_run:
        return MemoryResetResult(root, True, removed, absent)

    locks = (
        FileLock(root / EXPERIENCE_LOCK_FILE),
        FileLock(root / _LEGACY_LOCK_FILE),
    )
    try:
        with ExitStack() as stack:
            for lock in locks:
                stack.enter_context(lock.acquire(timeout=lock_timeout_seconds))
            for path in targets:
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path)
    except FileLockTimeout as exc:
        raise MemoryStoreLockTimeout(
            "timed out acquiring memory reset locks; stop all runtime, writer, "
            "attribution, and maintenance processes before retrying"
        ) from exc

    return MemoryResetResult(root, False, removed, absent)


__all__ = [
    "MemoryResetResult",
    "RESET_CONFIRMATION",
    "reset_memory_directory",
]
