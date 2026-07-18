"""Common maintenance journal path helpers."""

from __future__ import annotations

from pathlib import Path


def history_lock_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(f".{source.name}.lock")


__all__ = ["history_lock_path"]
