from __future__ import annotations

# ruff: noqa: F401 - imported names are re-exported to split CLI test modules

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from filelock import FileLock

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.cli import build_parser, main
from my_agent.cli import memory_maintenance as maintenance_cli
import my_agent.memory.evolver.maintenance.legacy.transaction as maintenance_transaction
from my_agent.memory.evolver.maintenance.contracts import (
    MaintenanceApplyResult,
    MaintenanceApplyStatus,
)
from my_agent.memory.evolver.maintenance.legacy.validation import load_maintenance_plan
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceTier,
)
from my_agent.memory.experience_store import ExperienceStore
from tests.memory.experience.fixtures import typed_experience


PROJECT_KEY = "manifest:demo:memory:shared_stream:stream:python"
OTHER_PROJECT_KEY = "manifest:demo:memory:shared_stream:stream:other"
AS_OF = "2026-07-12T00:00:00+00:00"
NOW = datetime(2026, 7, 12, tzinfo=timezone.utc)

class _MaintenanceCliCase(unittest.TestCase):
    def _store(self, memory_dir: Path) -> ExperienceStore:
        return ExperienceStore.from_dir(memory_dir)
    def _add_invalidated_tip(self, memory_dir: Path) -> ExperienceStore:
        store = self._store(memory_dir)
        store.add(typed_experience(
            "delete-tip",
            "This parser warning is obsolete.",
            ExperienceTier.TIP,
            project_key=PROJECT_KEY,
            created_at=NOW,
            created_by=ExperienceCreatedBy.WRITER,
            invalidated=True,
        ))
        return store
    def _invoke(self, args: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args, ctx=SimpleNamespace())
        return exit_code, stdout.getvalue(), stderr.getvalue()
    def _base_args(self, memory_dir: Path) -> list[str]:
        return [
            "memory",
            "maintain",
            "--memory-dir",
            str(memory_dir),
            "--memory-project-key",
            PROJECT_KEY,
            "--as-of",
            AS_OF,
        ]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

__all__ = [name for name in globals() if not name.startswith('__')]
