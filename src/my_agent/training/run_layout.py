"""Clean, isolated filesystem layout for M0/M1/M2 reproduction runs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Iterable
import json
import platform
import sqlite3

from my_agent.memory.evolver.cadence_schema import EVOLVER_STATE_FILENAME, LEDGER_DDL


RUN_LAYOUT_SCHEMA_VERSION = "opd-run-layout-v2"
RUN_ENVIRONMENT_SCHEMA_VERSION = "opd-run-environment-v1"
EXPERIENCE_REPOSITORY_FILENAME = "experience_memory.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPD_RUNTIME_DISTRIBUTIONS = (
    "accelerate",
    "datasets",
    "huggingface-hub",
    "peft",
    "safetensors",
    "sentence-transformers",
    "tokenizers",
    "torch",
    "transformers",
)


@dataclass(frozen=True)
class ReproductionRunLayout:
    run_id: str
    root: Path
    baseline_commit: str
    lockfile_path: Path | None = None

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
        environment = collect_run_environment(self.lockfile_path)
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
                    "environment": environment,
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
    lockfile_path: str | Path | None = None,
) -> tuple[ReproductionRunLayout, ...]:
    base = Path(root)
    identifiers = tuple(str(item).strip() for item in run_ids)
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("run_ids must be unique non-empty strings")
    layouts = tuple(
        ReproductionRunLayout(
            run_id=run_id,
            root=base / run_id,
            baseline_commit=baseline_commit,
            lockfile_path=Path(lockfile_path) if lockfile_path is not None else None,
        )
        for run_id in identifiers
    )
    for layout in layouts:
        layout.initialize_empty()
    return layouts


def collect_run_environment(lockfile_path: str | Path | None = None) -> dict[str, object]:
    """Capture the concrete Python/ML stack and lockfile used by a run."""

    resolved_lockfile = Path(lockfile_path) if lockfile_path is not None else PROJECT_ROOT / "uv.lock"
    resolved_lockfile = resolved_lockfile.expanduser().resolve()
    if not resolved_lockfile.is_file():
        raise FileNotFoundError(f"reproduction run requires a lockfile: {resolved_lockfile}")
    package_versions: dict[str, str | None] = {}
    for distribution in OPD_RUNTIME_DISTRIBUTIONS:
        try:
            package_versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            package_versions[distribution] = None
    return {
        "schema_version": RUN_ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": package_versions,
        "lockfile": {
            "name": resolved_lockfile.name,
            "sha256": f"sha256:{sha256(resolved_lockfile.read_bytes()).hexdigest()}",
        },
    }


__all__ = [
    "EXPERIENCE_REPOSITORY_FILENAME",
    "OPD_RUNTIME_DISTRIBUTIONS",
    "RUN_ENVIRONMENT_SCHEMA_VERSION",
    "RUN_LAYOUT_SCHEMA_VERSION",
    "ReproductionRunLayout",
    "collect_run_environment",
    "prepare_isolated_runs",
]
