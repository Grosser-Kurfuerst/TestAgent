"""Clean, isolated filesystem layout for M0/M1/M2 reproduction runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import sqlite3

from my_agent.memory.evolver.cadence_schema import EVOLVER_STATE_FILENAME, LEDGER_DDL


RUN_LAYOUT_SCHEMA_VERSION = "opd-run-layout-v1"
EXPERIENCE_REPOSITORY_FILENAME = "experience_memory.jsonl"


@dataclass(frozen=True)
class ReproductionRunLayout:
    run_id: str
    root: Path
    baseline_commit: str

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def repository_path(self) -> Path:
        return self.memory_dir / EXPERIENCE_REPOSITORY_FILENAME

    @property
    def ledger_path(self) -> Path:
        return self.memory_dir / EVOLVER_STATE_FILENAME

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def manifest_path(self) -> Path:
        return self.root / "run_manifest.json"

    def initialize_empty(self) -> None:
        if not self.run_id.strip() or not self.baseline_commit.strip():
            raise ValueError("run_id and baseline_commit must not be empty")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"reproduction run root must be empty: {self.root}")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repository_path.write_text("", encoding="utf-8")
        with sqlite3.connect(self.ledger_path) as connection:
            for statement in LEDGER_DDL:
                connection.execute(statement)
            connection.commit()
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": RUN_LAYOUT_SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "baseline_commit": self.baseline_commit,
                    "repository": str(self.repository_path),
                    "ledger": str(self.ledger_path),
                    "output": str(self.output_dir),
                    "repository_initial_state": "empty",
                    "legacy_memory_migration": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def prepare_isolated_runs(
    root: str | Path,
    *,
    baseline_commit: str,
    run_ids: Iterable[str] = ("m0", "m1", "m2"),
) -> tuple[ReproductionRunLayout, ...]:
    base = Path(root)
    identifiers = tuple(str(item).strip() for item in run_ids)
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("run_ids must be unique non-empty strings")
    layouts = tuple(
        ReproductionRunLayout(run_id=run_id, root=base / run_id, baseline_commit=baseline_commit)
        for run_id in identifiers
    )
    for layout in layouts:
        layout.initialize_empty()
    return layouts


__all__ = [
    "EXPERIENCE_REPOSITORY_FILENAME",
    "RUN_LAYOUT_SCHEMA_VERSION",
    "ReproductionRunLayout",
    "prepare_isolated_runs",
]
