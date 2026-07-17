from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.cli import main
from my_agent.cli.memory_reset import RESET_CONFIRMATION
from my_agent.config import AgentConfig
from my_agent.memory.evolver import (
    ExperienceTier,
    MaintenanceAction,
    load_maintenance_plan,
)
from my_agent.memory.experience_retrieval import ExperienceRetriever
from my_agent.memory.experience_store import EXPERIENCE_STORAGE_FILE, ExperienceStore
from my_agent.memory.manager import MemoryManager
from my_agent.memory.store_errors import MemoryStoreLoadError
from tests.memory.experience_fixtures import typed_experience


PROJECT_KEY = "manifest:maintenance:e2e"
NOW = datetime(2026, 7, 16, tzinfo=timezone.utc)
FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "evolver_maintenance"


class MaintenanceEndToEndTests(unittest.TestCase):
    def _invoke(self, args: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args, ctx=SimpleNamespace())
        return code, stdout.getvalue(), stderr.getvalue()

    def _seed(self, memory_dir: Path) -> ExperienceStore:
        memory_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            FIXTURE_DIR / "experience_memory.jsonl",
            memory_dir / EXPERIENCE_STORAGE_FILE,
        )
        store = ExperienceStore.from_dir(memory_dir)
        store.load_strict_snapshot()
        return store

    def _base_args(self, memory_dir: Path) -> list[str]:
        return [
            "memory",
            "maintain",
            "--memory-dir",
            str(memory_dir),
            "--memory-project-key",
            PROJECT_KEY,
            "--as-of",
            NOW.isoformat(),
        ]

    def test_typed_dry_run_apply_backup_and_retrieval_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            store = self._seed(memory_dir)
            before = store.load_strict_snapshot()
            first_plan = root / "first.json"
            second_plan = root / "second.json"

            first = self._invoke(
                self._base_args(memory_dir)
                + ["--dry-run", "--output", str(first_plan)]
            )
            second = self._invoke(
                self._base_args(memory_dir)
                + ["--dry-run", "--output", str(second_plan)]
            )

            self.assertEqual(first[0], 0, first[2])
            self.assertEqual(second[0], 0, second[2])
            self.assertEqual(first_plan.read_bytes(), second_plan.read_bytes())
            self.assertEqual(store.path.read_bytes(), before.raw_bytes)
            plan = load_maintenance_plan(first_plan)
            self.assertEqual(
                {key: plan.summary[key] for key in ("delete", "merge", "promote")},
                {"delete": 1, "merge": 1, "promote": 1},
            )
            self.assertTrue(
                all(
                    isinstance(memory.tier, ExperienceTier)
                    for operation in plan.operations
                    for memory in operation.replacements + operation.additions
                )
            )

            apply = self._invoke(
                self._base_args(memory_dir)
                + [
                    "--apply",
                    "--plan",
                    str(first_plan),
                    "--output",
                    str(first_plan),
                ]
            )
            self.assertEqual(apply[0], 0, apply[2])
            after = store.load_strict_snapshot()
            by_id = {memory.id: memory for memory in after.memories}
            self.assertNotIn("delete-tip", by_id)
            self.assertNotIn("merge-b", by_id)
            source = by_id["promote-tip"]
            self.assertIn(source.promoted_to, by_id)
            self.assertEqual(by_id[source.promoted_to].tier, ExperienceTier.SKILL)
            backups = list((memory_dir / "maintenance_backups").glob("*.experience_memory.jsonl"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), before.raw_bytes)

            hits = ExperienceRetriever(now=NOW).retrieve_candidates(
                "focused failure",
                store=store,
                project_key=PROJECT_KEY,
                top_k_per_tier=10,
            )
            self.assertIn(source.promoted_to, {hit.entry.id for hit in hits})
            history = [
                json.loads(line)
                for line in (memory_dir / "maintenance_history.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["record_type"] for row in history], ["intent", "completion"])
            self.assertTrue(
                any(op.action == MaintenanceAction.PROMOTE for op in plan.operations)
            )

    def test_corrupt_typed_fixture_fails_closed_without_rewriting_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            store_path = memory_dir / EXPERIENCE_STORAGE_FILE
            shutil.copyfile(
                FIXTURE_DIR / "experience_memory_corrupt.jsonl",
                store_path,
            )
            before = store_path.read_bytes()

            exit_code, _, stderr = self._invoke(self._base_args(memory_dir))

            self.assertEqual(exit_code, 1)
            self.assertIn("invalid experience JSONL at line 2", stderr)
            self.assertEqual(store_path.read_bytes(), before)
            self.assertFalse((memory_dir / "maintenance_plan.json").exists())


class ResetAndLegacyCutoverTests(unittest.TestCase):
    def _invoke(self, args: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args, ctx=SimpleNamespace())
        return code, stdout.getvalue(), stderr.getvalue()

    def test_runtime_fails_closed_when_legacy_store_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            legacy = memory_dir / "long_term_memory.jsonl"
            legacy.write_text('{"id":"legacy"}\n', encoding="utf-8")
            config = AgentConfig(
                provider="fake",
                api_key="",
                base_url=None,
                model="fake",
                temperature=0.0,
                max_steps=8,
                command_timeout=60,
                trace_dir=Path(tmp) / "traces",
                use_fake_llm=True,
                memory_dir=memory_dir,
                memory_project_key=PROJECT_KEY,
            )

            with self.assertRaisesRegex(MemoryStoreLoadError, "explicit four-tier memory reset"):
                MemoryManager.from_config(
                    config=config,
                    llm=None,
                    repo_path=Path(tmp),
                )

            self.assertTrue(legacy.exists())
            self.assertFalse((memory_dir / "experience_memory.jsonl").exists())

    def test_explicit_reset_clears_repository_and_id_coupled_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = ExperienceStore.from_dir(memory_dir)
            store.add(typed_experience(
                "old-id",
                "Old typed memory.",
                ExperienceTier.TIP,
                project_key=PROJECT_KEY,
            ))
            (memory_dir / "long_term_memory.jsonl").write_text("legacy\n", encoding="utf-8")
            for name in (
                "usage_logs.jsonl",
                "memory_attribution.jsonl",
                "maintenance_plan.json",
                "maintenance_plan.json.summary.json",
                "maintenance_history.jsonl",
                "maintenance_trace.jsonl",
                "evolver_state.sqlite3",
            ):
                (memory_dir / name).write_text("old-id\n", encoding="utf-8")
            backup_dir = memory_dir / "maintenance_backups"
            backup_dir.mkdir()
            (backup_dir / "old-id.experience_memory.jsonl").write_text("old-id\n")
            args = [
                "memory",
                "reset",
                "--memory-dir",
                str(memory_dir),
                "--confirm-reset",
                RESET_CONFIRMATION,
            ]

            dry_run = self._invoke(args + ["--dry-run"])
            self.assertEqual(dry_run[0], 0, dry_run[2])
            self.assertTrue(store.path.exists())
            applied = self._invoke(args)
            self.assertEqual(applied[0], 0, applied[2])

            for name in (
                "long_term_memory.jsonl",
                "experience_memory.jsonl",
                "usage_logs.jsonl",
                "memory_attribution.jsonl",
                "maintenance_plan.json",
                "maintenance_plan.json.summary.json",
                "maintenance_history.jsonl",
                "maintenance_trace.jsonl",
                "evolver_state.sqlite3",
                "maintenance_backups",
            ):
                self.assertFalse((memory_dir / name).exists(), name)
            snapshot = ExperienceStore.from_dir(memory_dir).load_strict_snapshot()
            self.assertEqual(snapshot.memories, ())
            self.assertNotIn("old-id", snapshot.raw_bytes.decode("utf-8"))

    def test_reset_rejects_missing_confirmation_without_deleting_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            legacy = memory_dir / "long_term_memory.jsonl"
            legacy.write_text("legacy\n", encoding="utf-8")
            result = self._invoke([
                "memory",
                "reset",
                "--memory-dir",
                str(memory_dir),
                "--confirm-reset",
                "WRONG",
            ])
            self.assertEqual(result[0], 1)
            self.assertIn(RESET_CONFIRMATION, result[2])
            self.assertTrue(legacy.exists())


if __name__ == "__main__":
    unittest.main()
