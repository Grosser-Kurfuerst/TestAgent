from __future__ import annotations

from hashlib import sha256
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from my_agent.training.run_layout import (
    OPD_RUNTIME_DISTRIBUTIONS,
    RUN_ENVIRONMENT_SCHEMA_VERSION,
    RUN_LAYOUT_SCHEMA_VERSION,
    prepare_isolated_runs,
)


class ReproductionRunLayoutTests(unittest.TestCase):
    def test_m0_m1_m2_receive_empty_isolated_repository_ledger_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lockfile = Path(tmp) / "uv.lock"
            lockfile.write_text("version = 1\n", encoding="utf-8")
            with mock.patch(
                "my_agent.training.run_layout.metadata.version",
                side_effect=lambda name: f"test-{name}",
            ):
                layouts = prepare_isolated_runs(
                    tmp,
                    baseline_commit="baseline-commit",
                    lockfile_path=lockfile,
                )

            self.assertEqual([item.run_id for item in layouts], ["m0", "m1", "m2"])
            self.assertEqual(len({item.repository_path for item in layouts}), 3)
            for layout in layouts:
                self.assertEqual(layout.repository_path.read_text(encoding="utf-8"), "")
                self.assertEqual(list(layout.output_dir.iterdir()), [])
                manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["schema_version"], RUN_LAYOUT_SCHEMA_VERSION)
                self.assertFalse(manifest["legacy_memory_migration"])
                environment = manifest["environment"]
                self.assertEqual(environment["schema_version"], RUN_ENVIRONMENT_SCHEMA_VERSION)
                self.assertTrue(environment["python"]["version"])
                self.assertEqual(
                    environment["packages"],
                    {name: f"test-{name}" for name in OPD_RUNTIME_DISTRIBUTIONS},
                )
                self.assertEqual(
                    environment["lockfile"]["sha256"],
                    f"sha256:{sha256(lockfile.read_bytes()).hexdigest()}",
                )
                with sqlite3.connect(layout.ledger_path) as connection:
                    tables = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }
                self.assertEqual(tables, {
                    "task_completion",
                    "maintenance_cadence",
                    "task_outcome_evidence",
                })

    def test_existing_run_artifacts_are_not_reused_or_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m0"
            root.mkdir()
            (root / "legacy.jsonl").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_isolated_runs(tmp, baseline_commit="baseline", run_ids=("m0",))

    def test_missing_lockfile_fails_before_run_artifacts_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "runs"
            with self.assertRaisesRegex(FileNotFoundError, "requires a lockfile"):
                prepare_isolated_runs(
                    output,
                    baseline_commit="baseline",
                    run_ids=("m0",),
                    lockfile_path=Path(tmp) / "missing.lock",
                )
            self.assertFalse((output / "m0").exists())


if __name__ == "__main__":
    unittest.main()
