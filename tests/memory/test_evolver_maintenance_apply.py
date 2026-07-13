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

import my_agent.memory.evolver.maintenance as maintenance_module
import my_agent.memory.evolver.transaction as maintenance_transaction
from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    MaintenanceApplyStatus,
    MaintenanceConfig,
    MemoryAttributionRecord,
    apply_maintenance_plan,
    build_experience_entry,
    build_maintenance_plan,
)
from my_agent.memory.long_term import LongTermMemoryStore, MemoryStoreLoadError
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType


PROJECT_KEY = "manifest:demo:memory:shared_stream:stream:python"
NOW = datetime(2026, 7, 11, tzinfo=timezone.utc)


def _experience(
    memory_id: str,
    *,
    content: str,
    tier: str = "skill",
    project_key: str = PROJECT_KEY,
    metadata: dict | None = None,
) -> MemoryEntry:
    return build_experience_entry(
        id=memory_id,
        content=content,
        tier=tier,
        project_key=project_key,
        created_at=NOW,
        created_by=ExperienceCreatedBy.WRITER,
        source_task="task-1",
        extra_metadata=metadata,
    )


def _fact(memory_id: str, *, content: str, project_key: str) -> MemoryEntry:
    return MemoryEntry.build(
        id=memory_id,
        content=content,
        type=MemoryType.FACT,
        scope=MemoryScope.PROJECT,
        source="manual",
        token_count=4,
        project_key=project_key,
        created_at=NOW,
    )


def _record(
    memory_id: str,
    *,
    tier: str,
    value: float = 0.3,
    confidence: float = 0.9,
    selected_count: int = 5,
) -> MemoryAttributionRecord:
    return MemoryAttributionRecord(
        memory_id=memory_id,
        tier=tier,
        memory_project_key=PROJECT_KEY,
        candidate_count=8,
        selected_count=selected_count,
        not_selected_count=3,
        value=value,
        confidence=confidence,
        last_used=NOW.isoformat(),
    )


def _refresh_plan_id(payload: dict) -> None:
    operations = tuple(
        maintenance_module.MaintenanceOperation.from_dict(item)
        for item in payload["operations"]
    )
    config = MaintenanceConfig.from_dict(payload["config"]).to_dict()
    payload["plan_id"] = maintenance_module._plan_id(
        repository_revision=payload["repository_revision"],
        project_key=payload["memory_project_key"],
        as_of=payload["as_of"],
        config=config,
        input_summary=payload["input_summary"],
        operations=operations,
        summary=payload["summary"],
    )


class MaintenanceApplyTests(unittest.TestCase):
    def _store(self, root: Path) -> LongTermMemoryStore:
        return LongTermMemoryStore.from_dir(root)

    def _paths(self, root: Path) -> tuple[Path, Path]:
        return root / "maintenance_backups", root / "maintenance_history.jsonl"

    def _plan(
        self,
        store: LongTermMemoryStore,
        *,
        attribution: dict | None = None,
        config: MaintenanceConfig | None = None,
    ):
        snapshot = store.load_strict_snapshot()
        return build_maintenance_plan(
            entries=snapshot.entries,
            attribution=attribution or {},
            repository_revision=snapshot.revision,
            project_key=PROJECT_KEY,
            as_of=NOW,
            config=config,
        )

    def test_noop_plan_does_not_rewrite_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(build_experience_entry(
                id="manual-skill",
                content="Manually protected skill",
                tier="skill",
                project_key=PROJECT_KEY,
                created_at=NOW,
                created_by=ExperienceCreatedBy.MANUAL,
            ))
            plan = self._plan(store)
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            with patch.object(store, "_persist", wraps=store._persist) as persist:
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.NOOP)
            self.assertFalse(result.mutation_committed)
            self.assertEqual(result.before_revision, result.after_revision)
            self.assertEqual(store.path.read_bytes(), before)
            persist.assert_not_called()
            self.assertFalse(backup_dir.exists())
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual([item["record_type"] for item in history], ["completion"])

    def test_noop_history_failure_remains_successful_and_non_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(build_experience_entry(
                id="manual-skill",
                content="Manually protected skill",
                tier="skill",
                project_key=PROJECT_KEY,
                created_at=NOW,
                created_by=ExperienceCreatedBy.MANUAL,
            ))
            plan = self._plan(store)
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            with patch.object(
                maintenance_transaction,
                "append_maintenance_history",
                side_effect=OSError("history unavailable"),
            ):
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.NOOP)
            self.assertFalse(result.audit_complete)
            self.assertFalse(result.should_retry)
            self.assertEqual(result.audit_error_stage, "history_completion")
            self.assertEqual(store.path.read_bytes(), before)

    def test_apply_delete_merge_and_promote_in_one_atomic_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            entries = [
                _experience(
                    "delete-tip",
                    tier="tip",
                    content="Invalidated parser warning",
                    metadata={"maintenance_invalidated": True},
                ),
                _experience(
                    "merge-anchor",
                    content="Run focused parser tests before editing",
                    metadata={"steps": ["focused tests"]},
                ),
                _experience(
                    "merge-source",
                    content="Run focused parser test before edits",
                    metadata={"steps": ["full suite"]},
                ),
                _experience(
                    "promote-tip",
                    tier="tip",
                    content="Inspect the focused parser failure first.",
                    metadata={"category": "debugging", "confidence": 0.8},
                ),
                _fact("plain", content="ordinary fact", project_key=PROJECT_KEY),
                _fact("other", content="other project fact", project_key="other-project"),
            ]
            for entry in entries:
                store.add(entry)
            unrelated_before = {
                entry.id: entry.to_dict()
                for entry in store.all()
                if entry.id in {"plain", "other"}
            }
            record = _record("promote-tip", tier="tip")
            plan = self._plan(
                store,
                attribution={(record.memory_id, record.tier, PROJECT_KEY): record},
                config=MaintenanceConfig(merge_threshold_skill=0.5),
            )
            before_bytes = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            with patch.object(store, "_persist", wraps=store._persist) as persist:
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.COMMITTED)
            self.assertTrue(result.mutation_committed)
            self.assertTrue(result.audit_complete)
            self.assertFalse(result.should_retry)
            persist.assert_called_once()
            self.assertEqual(Path(result.backup_path).read_bytes(), before_bytes)
            after = {entry.id: entry for entry in store.load_strict_snapshot().entries}
            self.assertNotIn("delete-tip", after)
            self.assertNotIn("merge-source", after)
            self.assertIn("merge-anchor", after)
            self.assertIn("promote-tip", after)
            self.assertEqual(after["plain"].to_dict(), unrelated_before["plain"])
            self.assertEqual(after["other"].to_dict(), unrelated_before["other"])
            promoted_id = after["promote-tip"].metadata["maintenance_promoted_to"]
            self.assertIn(promoted_id, after)
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual([item["record_type"] for item in history], ["intent", "completion"])

    def test_revision_conflict_fails_without_changing_current_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            plan = self._plan(store)
            store.add(_fact("later", content="later fact", project_key=PROJECT_KEY))
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
            self.assertFalse(result.should_retry)
            self.assertEqual(result.audit_error_stage, "validation")
            self.assertEqual(store.path.read_bytes(), current)
            self.assertFalse(backup_dir.exists())

    def test_source_precondition_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            original = self._plan(store)
            payload = original.to_dict()
            payload["operations"][0]["source_preconditions"]["delete-tip"][
                "fingerprint"
            ] = "f" * 64
            _refresh_plan_id(payload)
            plan = maintenance_module.MaintenancePlan.from_dict(payload)
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            result = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "validation")
            self.assertEqual(store.path.read_bytes(), before)

    def test_backup_intent_and_persist_failures_leave_memory_unchanged(self) -> None:
        stages = ("backup", "audit_intent", "persist")
        for stage in stages:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = self._store(root)
                store.add(_experience(
                    "delete-tip",
                    tier="tip",
                    content="Invalidated tip",
                    metadata={"maintenance_invalidated": True},
                ))
                plan = self._plan(store)
                before = store.path.read_bytes()
                before_entries = [entry.to_dict() for entry in store.all()]
                backup_dir, history_path = self._paths(root)

                if stage == "backup":
                    context = patch.object(
                        maintenance_transaction,
                        "_write_backup_atomic",
                        side_effect=OSError("backup failed"),
                    )
                elif stage == "audit_intent":
                    context = patch.object(
                        maintenance_transaction,
                        "append_maintenance_history",
                        side_effect=OSError("history failed"),
                    )
                else:
                    context = patch.object(store, "_persist", side_effect=OSError("disk full"))

                with context:
                    result = apply_maintenance_plan(
                        store=store,
                        plan=plan,
                        backup_dir=backup_dir,
                        history_path=history_path,
                    )

                self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
                self.assertFalse(result.mutation_committed)
                self.assertTrue(result.should_retry)
                self.assertEqual(result.audit_error_stage, stage)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertEqual([entry.to_dict() for entry in store.all()], before_entries)

    def test_completion_failure_reports_committed_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            plan = self._plan(store)
            backup_dir, history_path = self._paths(root)
            original_append = maintenance_transaction.append_maintenance_history

            def fail_completion(path, record):
                if record.get("record_type") == "completion":
                    raise OSError("completion unavailable")
                return original_append(path, record)

            with patch.object(
                maintenance_transaction,
                "append_maintenance_history",
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
            self.assertFalse(result.audit_complete)
            self.assertFalse(result.should_retry)
            self.assertEqual(result.audit_error_stage, "history_completion")
            self.assertNotIn("delete-tip", {entry.id for entry in store.all()})
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual([item["record_type"] for item in history], ["intent", "audit_error"])

    def test_retryable_persist_failure_can_resume_same_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            plan = self._plan(store)
            backup_dir, history_path = self._paths(root)

            with patch.object(store, "_persist", side_effect=OSError("transient")):
                first = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )
            self.assertEqual(first.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertTrue(first.should_retry)
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual([item["record_type"] for item in history], ["intent"])

            second = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )
            third = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            self.assertEqual(second.status, MaintenanceApplyStatus.COMMITTED)
            self.assertEqual(third, second)
            self.assertNotIn("delete-tip", {entry.id for entry in store.all()})
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual([item["record_type"] for item in history], ["intent", "completion"])

    def test_completed_plan_remains_idempotent_after_later_store_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            plan = self._plan(store)
            backup_dir, history_path = self._paths(root)
            first = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )
            store.add(_fact("later", content="later fact", project_key=PROJECT_KEY))

            repeated = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            self.assertEqual(first.status, MaintenanceApplyStatus.COMMITTED)
            self.assertEqual(repeated, first)
            self.assertIn("later", {entry.id for entry in store.all()})
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual([item["record_type"] for item in history], ["intent", "completion"])

    def test_post_replace_verification_failure_is_never_reported_pre_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            plan = self._plan(store)
            backup_dir, history_path = self._paths(root)
            original_load = store._load_strict_snapshot_locked
            calls = 0

            def fail_after_replace():
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise MemoryStoreLoadError("verification unavailable")
                return original_load()

            with patch.object(
                store,
                "_load_strict_snapshot_locked",
                side_effect=fail_after_replace,
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
            self.assertEqual(result.audit_error_stage, "verify")
            self.assertNotIn("delete-tip", {entry.id for entry in store.all()})

    def test_plan_id_is_tamper_evident_and_cannot_escape_backup_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            plan = self._plan(store)
            for tampered in ("../escaped", "maint-" + "0" * 24):
                with self.subTest(plan_id=tampered):
                    payload = plan.to_dict()
                    payload["plan_id"] = tampered
                    with self.assertRaises(ValueError):
                        maintenance_module.MaintenancePlan.from_dict(payload)

            with self.assertRaises(ValueError):
                maintenance_module._maintenance_backup_path(
                    root / "backups",
                    "../escaped",
                )

    def test_apply_revalidates_mutable_plan_payloads(self) -> None:
        tamper_cases = (
            "source_precondition",
            "replacement",
            "addition",
            "promotion_created_at",
            "promotion_source",
            "promotion_created_by",
            "promotion_source_task",
            "promotion_lineage",
            "summary",
        )
        for tamper_case in tamper_cases:
            with self.subTest(tamper_case=tamper_case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = self._store(root)
                entries = (
                    _experience(
                        "merge-anchor",
                        content="Run focused parser tests before editing",
                        metadata={"steps": ["focused tests"]},
                    ),
                    _experience(
                        "merge-source",
                        content="Run focused parser test before edits",
                        metadata={"steps": ["full suite"]},
                    ),
                    _experience(
                        "promote-tip",
                        tier="tip",
                        content="Inspect the focused parser failure first.",
                        metadata={"category": "debugging", "confidence": 0.8},
                    ),
                )
                for entry in entries:
                    store.add(entry)
                record = _record("promote-tip", tier="tip")
                plan = self._plan(
                    store,
                    attribution={(record.memory_id, record.tier, PROJECT_KEY): record},
                    config=MaintenanceConfig(merge_threshold_skill=0.5),
                )
                before = store.path.read_bytes()
                backup_dir, history_path = self._paths(root)

                if tamper_case == "source_precondition":
                    plan.operations[0].source_preconditions[
                        plan.operations[0].source_ids[0]
                    ]["project_key"] = "other-project"
                elif tamper_case == "replacement":
                    merge = next(
                        operation
                        for operation in plan.operations
                        if operation.action.value == "merge"
                    )
                    merge.replacements[0]["content"] = "unreviewed replacement"
                elif tamper_case == "summary":
                    plan.summary["promote"] += 1
                else:
                    promotion = next(
                        operation
                        for operation in plan.operations
                        if operation.action.value == "promote"
                    )
                    target = promotion.additions[0]
                    if tamper_case == "addition":
                        target["metadata"]["maintenance_operation_id"] = "op-" + "0" * 24
                    elif tamper_case == "promotion_created_at":
                        target["created_at"] = "1999-01-01T00:00:00+00:00"
                    elif tamper_case == "promotion_source":
                        target["source"] = "manual"
                    elif tamper_case == "promotion_created_by":
                        target["metadata"]["created_by"] = "writer"
                    elif tamper_case == "promotion_source_task":
                        target["metadata"]["source_task"] = "forged-task"
                    else:
                        target["metadata"]["maintenance_parent_id"] = "forged-parent"

                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

                self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
                self.assertEqual(result.audit_error_stage, "plan_validation")
                self.assertFalse(result.should_retry)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertFalse(backup_dir.exists())
                self.assertFalse(history_path.exists())

    def test_apply_lock_timeout_returns_pre_commit_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holder = self._store(root)
            holder.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            plan = self._plan(holder)
            applier = self._store(root)
            before = holder.path.read_bytes()
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
                thread.join(2)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(results[0].audit_error_stage, "lock")
            self.assertTrue(results[0].should_retry)
            self.assertEqual(holder.path.read_bytes(), before)

    def test_apply_lock_timeout_must_be_finite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            store.add(_experience(
                "delete-tip",
                tier="tip",
                content="Invalidated tip",
                metadata={"maintenance_invalidated": True},
            ))
            plan = self._plan(store)
            backup_dir, history_path = self._paths(root)

            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(lock_timeout=value), self.assertRaises(ValueError):
                    apply_maintenance_plan(
                        store=store,
                        plan=plan,
                        backup_dir=backup_dir,
                        history_path=history_path,
                        lock_timeout_seconds=value,
                    )


if __name__ == "__main__":
    unittest.main()
