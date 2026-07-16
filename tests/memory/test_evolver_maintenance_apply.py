from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

import my_agent.memory.evolver.transaction as maintenance_transaction
from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    ExperienceTier,
    MaintenanceApplyStatus,
    MaintenanceConfig,
    SkillPayload,
    apply_maintenance_plan,
    build_maintenance_plan,
)
from my_agent.memory.experience_store import ExperienceStore
from tests.memory.experience_fixtures import typed_experience


PROJECT_KEY = "manifest:maintenance:apply"
NOW = datetime(2026, 7, 16, tzinfo=timezone.utc)


def _manual(memory_id: str):
    return typed_experience(
        memory_id,
        f"Manual skill {memory_id}",
        ExperienceTier.SKILL,
        project_key=PROJECT_KEY,
        created_at=NOW,
        created_by=ExperienceCreatedBy.MANUAL,
    )


def _invalidated(memory_id: str = "delete-tip"):
    return typed_experience(
        memory_id,
        f"Obsolete tip {memory_id}",
        ExperienceTier.TIP,
        project_key=PROJECT_KEY,
        created_at=NOW,
        created_by=ExperienceCreatedBy.WRITER,
        invalidated=True,
    )


def _promotable(memory_id: str = "promote-tip"):
    return typed_experience(
        memory_id,
        "Inspect the focused failure before editing.",
        ExperienceTier.TIP,
        project_key=PROJECT_KEY,
        created_at=NOW,
        source_task="task-7",
        run_id="run-7",
        stream_id="stream-7",
        created_by=ExperienceCreatedBy.WRITER,
        writer_confidence=0.81,
        attribution_value=0.2,
        attribution_confidence=0.9,
        candidate_count=6,
        selected_count=4,
        not_selected_count=2,
        attribution_updated_at=NOW,
    )


class MaintenanceApplyTests(unittest.TestCase):
    def _store(self, root: Path) -> ExperienceStore:
        return ExperienceStore.from_dir(root)

    def _plan(self, store: ExperienceStore, *, config: MaintenanceConfig | None = None):
        snapshot = store.load_strict_snapshot()
        return build_maintenance_plan(
            entries=snapshot.memories,
            attribution={},
            repository_revision=snapshot.revision,
            project_key=PROJECT_KEY,
            as_of=NOW,
            config=config,
        )

    def _paths(self, root: Path) -> tuple[Path, Path]:
        return root / "maintenance_backups", root / "maintenance_history.jsonl"

    def test_noop_plan_does_not_rewrite_repository_or_create_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_manual("manual"))
            plan = self._plan(store)
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            with patch.object(
                store,
                "_persist_memories",
                wraps=store._persist_memories,
            ) as persist:
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.NOOP)
            self.assertFalse(result.mutation_committed)
            persist.assert_not_called()
            self.assertEqual(store.path.read_bytes(), before)
            self.assertFalse(backup_dir.exists())
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual([row["record_type"] for row in history], ["completion"])

    def test_delete_merge_and_promote_commit_in_one_repository_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            entries = (
                _invalidated(),
                typed_experience(
                    "merge-a",
                    "Run focused tests, then run the full suite.",
                    ExperienceTier.SKILL,
                    payload=SkillPayload(
                        category="testing",
                        technique="test loop",
                        preconditions=("tests exist",),
                        steps=("focused",),
                    ),
                    project_key=PROJECT_KEY,
                    created_at=NOW,
                    created_by=ExperienceCreatedBy.WRITER,
                ),
                typed_experience(
                    "merge-b",
                    "Run focused tests then run the full suite.",
                    ExperienceTier.SKILL,
                    payload=SkillPayload(
                        category="testing",
                        technique="test loop",
                        preconditions=("suite exists",),
                        steps=("full",),
                    ),
                    project_key=PROJECT_KEY,
                    created_at=NOW,
                    created_by=ExperienceCreatedBy.WRITER,
                ),
                _promotable(),
                _manual("unrelated"),
            )
            for entry in entries:
                store.add(entry)
            before_snapshot = store.load_strict_snapshot()
            unrelated_before = store.get("unrelated")
            plan = self._plan(store)
            backup_dir, history_path = self._paths(root)

            with patch.object(
                store,
                "_persist_memories",
                wraps=store._persist_memories,
            ) as persist:
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.COMMITTED)
            self.assertTrue(result.mutation_committed)
            self.assertEqual((result.deleted, result.merged, result.promoted), (1, 1, 1))
            persist.assert_called_once()
            self.assertTrue(result.after_revision != before_snapshot.revision)
            self.assertTrue(result.backup_path.endswith(".experience_memory.jsonl"))
            self.assertEqual(Path(result.backup_path).read_bytes(), before_snapshot.raw_bytes)
            after = {memory.id: memory for memory in store.load_strict_snapshot().memories}
            self.assertNotIn("delete-tip", after)
            self.assertNotIn("merge-b", after)
            self.assertIn("merge-a", after)
            source = after["promote-tip"]
            self.assertIn(source.promoted_to, after)
            target = after[source.promoted_to]
            self.assertEqual(target.tier, ExperienceTier.SKILL)
            self.assertEqual(target.parent_id, source.id)
            self.assertEqual(target.parent_tier, ExperienceTier.TIP)
            self.assertEqual(target.created_at, NOW)
            self.assertEqual(store.get("unrelated"), unrelated_before)

    def test_stale_plan_fails_before_backup_or_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_invalidated())
            plan = self._plan(store)
            store.add(_manual("later"))
            current = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            result = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertFalse(result.mutation_committed)
            self.assertEqual(result.audit_error_stage, "validation")
            self.assertEqual(store.path.read_bytes(), current)
            self.assertFalse(backup_dir.exists())

    def test_backup_can_restore_the_exact_pre_apply_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_invalidated())
            snapshot = store.load_strict_snapshot()
            plan = self._plan(store)
            backup_dir, history_path = self._paths(root)
            result = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            restored_path = root / "restored_experience_memory.jsonl"
            restored_path.write_bytes(Path(result.backup_path).read_bytes())
            restored = ExperienceStore(restored_path).load_strict_snapshot()
            self.assertEqual(restored.raw_bytes, snapshot.raw_bytes)
            self.assertEqual(restored.revision, snapshot.revision)
            self.assertEqual(restored.memories, snapshot.memories)

    def test_backup_failure_is_pre_commit_and_leaves_repository_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_invalidated())
            plan = self._plan(store)
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            with patch.object(
                maintenance_transaction,
                "_write_backup_atomic",
                side_effect=OSError("backup unavailable"),
            ):
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "backup")
            self.assertEqual(store.path.read_bytes(), before)

    def test_post_commit_history_failure_is_non_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_invalidated())
            plan = self._plan(store)
            backup_dir, history_path = self._paths(root)
            original = maintenance_transaction._append_maintenance_history
            calls = 0

            def fail_completion(path, record, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("history completion failed")
                return original(path, record, **kwargs)

            with patch.object(
                maintenance_transaction,
                "_append_maintenance_history",
                side_effect=fail_completion,
            ):
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

            self.assertEqual(
                result.status,
                MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
            )
            self.assertTrue(result.mutation_committed)
            self.assertFalse(result.should_retry)
            self.assertNotIn("delete-tip", {memory.id for memory in store.all()})

    def test_store_lock_timeout_fails_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holder = self._store(root)
            holder.add(_invalidated())
            plan = self._plan(holder)
            applier = self._store(root)
            backup_dir, history_path = self._paths(root)
            results = []

            def apply_in_thread() -> None:
                results.append(apply_maintenance_plan(
                    store=applier,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                    lock_timeout_seconds=0.05,
                ))

            with holder.exclusive_process_lock():
                thread = threading.Thread(target=apply_in_thread)
                thread.start()
                thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(results[0].status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(results[0].audit_error_stage, "lock")
            self.assertTrue(results[0].should_retry)
            self.assertFalse(backup_dir.exists())
            self.assertFalse(history_path.exists())


if __name__ == "__main__":
    unittest.main()
