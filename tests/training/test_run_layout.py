from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from my_agent.training.run_layout import prepare_isolated_runs


class ReproductionRunLayoutTests(unittest.TestCase):
    def test_m0_m1_m2_receive_empty_isolated_repository_ledger_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layouts = prepare_isolated_runs(tmp, baseline_commit="baseline-commit")

            self.assertEqual([item.run_id for item in layouts], ["m0", "m1", "m2"])
            self.assertEqual(len({item.repository_path for item in layouts}), 3)
            for layout in layouts:
                self.assertEqual(layout.repository_path.read_text(encoding="utf-8"), "")
                self.assertEqual(list(layout.output_dir.iterdir()), [])
                manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
                self.assertFalse(manifest["legacy_memory_migration"])
                with sqlite3.connect(layout.ledger_path) as connection:
                    tables = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }
                self.assertEqual(tables, {"task_completion", "maintenance_cadence"})

    def test_existing_run_artifacts_are_not_reused_or_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m0"
            root.mkdir()
            (root / "legacy.jsonl").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_isolated_runs(tmp, baseline_commit="baseline", run_ids=("m0",))


if __name__ == "__main__":
    unittest.main()
