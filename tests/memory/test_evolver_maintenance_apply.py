from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock

from tests._path import add_src_to_path

add_src_to_path()

import my_agent.memory.evolver.maintenance as maintenance_module
import my_agent.memory.evolver.artifacts as maintenance_artifacts
import my_agent.memory.evolver.contracts as maintenance_contracts
import my_agent.memory.evolver.planner as maintenance_planner
import my_agent.memory.evolver.transaction as maintenance_transaction
import my_agent.memory.evolver.validation as maintenance_validation
from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    MaintenanceAction,
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
    payload["plan_id"] = maintenance_contracts._plan_id(
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

    def _replace_plan_operations(self, plan, operations):
        payload = plan.to_dict()
        payload["operations"] = [operation.to_dict() for operation in operations]
        payload["summary"] = maintenance_contracts._operation_summary(operations)
        _refresh_plan_id(payload)
        return maintenance_module.MaintenancePlan.from_dict(payload)

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
                "_append_maintenance_history",
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
                        "_append_maintenance_history",
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

    def test_transaction_rejects_store_lock_history_and_backup_aliases(self) -> None:
        cases = (
            "history_store",
            "history_store_tmp",
            "history_lock",
            "history_derived_lock",
            "history_backup",
            "history_backup_tmp",
            "backup_hardlink_store",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
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
                backup_dir, default_history = self._paths(root)
                backup_path = maintenance_transaction._maintenance_backup_path(
                    backup_dir,
                    plan.plan_id,
                )
                history_path = default_history
                if case == "history_store":
                    history_path = store.path
                elif case == "history_store_tmp":
                    history_path = store.path.with_suffix(store.path.suffix + ".tmp")
                elif case == "history_lock":
                    history_path = store.lock_path
                elif case == "history_derived_lock":
                    history_path = root / "long_term_memory"
                elif case == "history_backup":
                    history_path = backup_path
                elif case == "history_backup_tmp":
                    history_path = backup_path.with_suffix(backup_path.suffix + ".tmp")
                else:
                    backup_dir.mkdir(parents=True)
                    os.link(store.path, backup_path)

                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

                self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
                self.assertEqual(result.audit_error_stage, "artifact_validation")
                self.assertFalse(result.mutation_committed)
                self.assertFalse(result.should_retry)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertIn("delete-tip", {entry.id for entry in store.all()})
                if case == "history_backup":
                    self.assertFalse(backup_path.exists())
                if case != "backup_hardlink_store":
                    self.assertFalse(default_history.exists())

    def test_transaction_rechecks_supplied_full_artifact_graph(self) -> None:
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
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)
            plan_output = root / "reviewed_plan.json"
            graph = maintenance_artifacts._resolve_maintenance_artifact_graph(
                store_path=store.path,
                store_lock_path=store.lock_path,
                history_path=history_path,
                backup_dir=backup_dir,
                plan_id=plan.plan_id,
                memory_dir=root,
                plan_output_path=plan_output,
                summary_path=root / "summary.json",
                trace_path=root / "trace.jsonl",
            )
            plan_output.symlink_to(
                store.path.with_suffix(store.path.suffix + ".tmp")
            )

            result = maintenance_transaction._apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
                artifact_graph=graph,
            )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "artifact_validation")
            self.assertFalse(result.mutation_committed)
            self.assertEqual(store.path.read_bytes(), before)
            self.assertIn("delete-tip", {entry.id for entry in store.all()})
            self.assertFalse(history_path.exists())
            self.assertFalse(backup_dir.exists())

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
            original_append = maintenance_transaction._append_maintenance_history

            def fail_completion(path, record, *, lock_timeout_seconds=30.0):
                if record.get("record_type") == "completion":
                    raise OSError("completion unavailable")
                return original_append(
                    path,
                    record,
                    lock_timeout_seconds=lock_timeout_seconds,
                )

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

    def test_repeated_audit_error_stage_uses_latest_append_only_event(self) -> None:
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
            committed = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            first = maintenance_transaction.record_post_commit_audit_error(
                history_path=history_path,
                plan=plan,
                result=committed,
                stage="summary",
                error=OSError("first summary failure"),
            )
            maintenance_transaction.record_post_commit_audit_error(
                history_path=history_path,
                plan=plan,
                result=first,
                stage="summary",
                error=RuntimeError("second summary failure"),
            )
            repeated = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            self.assertEqual(
                repeated.status,
                MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
            )
            self.assertEqual(repeated.audit_error_stage, "summary")
            self.assertEqual(repeated.audit_error, "RuntimeError")
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual(
                [item["record_type"] for item in history],
                ["intent", "completion", "audit_error", "audit_error"],
            )

    def test_post_commit_audit_error_uses_requested_history_lock_timeout(self) -> None:
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
            committed = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            with patch.object(
                maintenance_transaction,
                "_best_effort_history",
            ) as best_effort_history:
                updated = maintenance_transaction.record_post_commit_audit_error(
                    history_path=history_path,
                    plan=plan,
                    result=committed,
                    stage="summary",
                    error=OSError("summary unavailable"),
                    lock_timeout_seconds=0.05,
                )

            self.assertEqual(
                updated.status,
                MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
            )
            self.assertFalse(updated.should_retry)
            self.assertEqual(
                best_effort_history.call_args.kwargs["lock_timeout_seconds"],
                0.05,
            )

    def test_concurrent_audit_finalizers_append_complete_events(self) -> None:
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
            committed = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )
            barrier = threading.Barrier(3)
            results = []

            def finalize(index: int) -> None:
                barrier.wait()
                results.append(
                    maintenance_transaction.record_post_commit_audit_error(
                        history_path=history_path,
                        plan=plan,
                        result=committed,
                        stage="trace",
                        error=OSError(f"trace failure {index}"),
                    )
                )

            threads = [threading.Thread(target=finalize, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(2)

            self.assertEqual(len(results), 2)
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            self.assertEqual(
                [item["record_type"] for item in history],
                ["intent", "completion", "audit_error", "audit_error"],
            )
            repeated = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
            )
            self.assertEqual(
                repeated.status,
                MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
            )
            self.assertEqual(repeated.audit_error_stage, "trace")

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
                maintenance_transaction._maintenance_backup_path(
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

    def test_rehashed_delete_cannot_bypass_snapshot_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            protected = _experience(
                "protected-tip",
                tier="tip",
                content="A protected writer-authored tip",
                metadata={"maintenance_protected": True},
            )
            store.add(protected)
            reviewed = self._plan(store)
            keep = reviewed.operations[0]
            delete_id = maintenance_contracts._operation_id(
                action=MaintenanceAction.DELETE,
                source_ids=keep.source_ids,
                target_ids=(),
                replacements=(),
                additions=(),
            )
            delete = maintenance_module.MaintenanceOperation(
                operation_id=delete_id,
                action=MaintenanceAction.DELETE,
                source_ids=keep.source_ids,
                source_tiers=keep.source_tiers,
                source_preconditions=keep.source_preconditions,
                reason_codes=("explicitly_invalidated",),
                evidence=keep.evidence,
                remove_ids=keep.source_ids,
            )
            forged = self._replace_plan_operations(reviewed, (delete,))
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            result = apply_maintenance_plan(
                store=store,
                plan=forged,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "validation")
            self.assertEqual(store.path.read_bytes(), before)
            self.assertIn("protected-tip", {entry.id for entry in store.all()})
            self.assertFalse(backup_dir.exists())
            self.assertFalse(history_path.exists())

    def test_rehashed_merge_recomputes_complete_link_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            left = _experience(
                "unrelated-a",
                content="Always validate database migrations in a staging environment.",
            )
            right = _experience(
                "unrelated-b",
                content="Use a lexer when parsing nested programming language syntax.",
            )
            store.add(left)
            store.add(right)
            reviewed = self._plan(store)
            evidence = {
                entry.id: maintenance_module.maintenance_evidence_for_entry(
                    entry,
                    attribution={},
                    project_key=PROJECT_KEY,
                )
                for entry in (left, right)
            }
            merge = maintenance_planner._merge_operation(
                [(left, evidence[left.id]), (right, evidence[right.id])],
                as_of=NOW,
            )
            payload = merge.to_dict()
            payload["redundancy_score"] = 1.0
            payload["replacements"][0]["metadata"]["maintenance_redundancy_min"] = 1.0
            forged_merge = maintenance_module.MaintenanceOperation.from_dict(payload)
            forged = self._replace_plan_operations(reviewed, (forged_merge,))
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)

            # Without source content the parser can only verify the claimed
            # threshold; the lock-held apply must recompute actual redundancy.
            maintenance_validation.parse_maintenance_plan(forged.to_dict())
            result = apply_maintenance_plan(
                store=store,
                plan=forged,
                backup_dir=backup_dir,
                history_path=history_path,
            )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "validation")
            self.assertEqual(store.path.read_bytes(), before)
            self.assertFalse(backup_dir.exists())
            self.assertFalse(history_path.exists())

    def test_parser_rejects_promotion_that_does_not_meet_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            tip = _experience(
                "low-evidence-tip",
                tier="tip",
                content="Inspect parser errors before editing.",
            )
            store.add(tip)
            reviewed = self._plan(store)
            evidence = maintenance_module.maintenance_evidence_for_entry(
                tip,
                attribution={},
                project_key=PROJECT_KEY,
            )
            promotion, _ = maintenance_planner._promotion_operation(
                tip,
                evidence,
                repository_entries=[tip],
                as_of=NOW,
            )
            forged = self._replace_plan_operations(reviewed, (promotion,))

            with self.assertRaisesRegex(ValueError, "not eligible"):
                maintenance_validation.parse_maintenance_plan(forged.to_dict())

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

    def test_history_lock_timeout_is_retryable_and_pre_commit(self) -> None:
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
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)
            history_path.write_text("", encoding="utf-8")
            history_lock = FileLock(
                str(maintenance_transaction._history_lock_path(history_path))
            )

            with history_lock:
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                    lock_timeout_seconds=0.05,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "history_lock")
            self.assertEqual(result.audit_error, "MaintenanceHistoryLockTimeout")
            self.assertTrue(result.should_retry)
            self.assertFalse(result.mutation_committed)
            self.assertEqual(store.path.read_bytes(), before)
            self.assertIn("delete-tip", {entry.id for entry in store.all()})
            self.assertFalse(backup_dir.exists())
            self.assertEqual(history_path.read_bytes(), b"")

    def test_missing_history_lock_timeout_precedes_mutation_backup(self) -> None:
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
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)
            history_lock = FileLock(
                str(maintenance_transaction._history_lock_path(history_path))
            )

            with history_lock:
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                    lock_timeout_seconds=0.05,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "history_lock")
            self.assertEqual(result.audit_error, "MaintenanceHistoryLockTimeout")
            self.assertTrue(result.should_retry)
            self.assertFalse(result.mutation_committed)
            self.assertEqual(store.path.read_bytes(), before)
            self.assertFalse(history_path.exists())
            self.assertFalse(backup_dir.exists())

    def test_missing_history_lock_timeout_precedes_noop_completion(self) -> None:
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
            history_lock = FileLock(
                str(maintenance_transaction._history_lock_path(history_path))
            )

            with history_lock:
                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                    lock_timeout_seconds=0.05,
                )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "history_lock")
            self.assertEqual(result.audit_error, "MaintenanceHistoryLockTimeout")
            self.assertTrue(result.should_retry)
            self.assertFalse(result.mutation_committed)
            self.assertEqual(store.path.read_bytes(), before)
            self.assertFalse(history_path.exists())
            self.assertFalse(backup_dir.exists())

    def test_invalid_history_is_nonretryable_history_load_failure(self) -> None:
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
            before = store.path.read_bytes()
            backup_dir, history_path = self._paths(root)
            history_path.write_text("{bad json}\n", encoding="utf-8")

            result = apply_maintenance_plan(
                store=store,
                plan=plan,
                backup_dir=backup_dir,
                history_path=history_path,
                lock_timeout_seconds=0.05,
            )

            self.assertEqual(result.status, MaintenanceApplyStatus.PRE_COMMIT_FAILED)
            self.assertEqual(result.audit_error_stage, "history_load")
            self.assertEqual(result.audit_error, "MaintenancePlanError")
            self.assertFalse(result.should_retry)
            self.assertFalse(result.mutation_committed)
            self.assertEqual(store.path.read_bytes(), before)
            self.assertFalse(backup_dir.exists())

    def test_typed_history_corruption_fails_closed_during_history_load(self) -> None:
        corruptions = (
            (
                "string_boolean",
                lambda record: (
                    record.__setitem__("mutation_committed", "false"),
                    record["result"].__setitem__("mutation_committed", "false"),
                ),
            ),
            (
                "string_count",
                lambda record: record["result"].__setitem__("before_count", "1"),
            ),
            (
                "invalid_revision",
                lambda record: record.__setitem__("after_revision", 7),
            ),
            (
                "contradictory_status",
                lambda record: (
                    record.__setitem__("mutation_committed", False),
                    record["result"].__setitem__("mutation_committed", False),
                ),
            ),
            (
                "non_string_id",
                lambda record: record["result"].__setitem__("removed_ids", [1]),
            ),
            (
                "unexpected_field",
                lambda record: record.__setitem__("task", "sensitive input"),
            ),
        )
        for name, corrupt in corruptions:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = self._store(root)
                store.add(_experience(
                    "delete-tip",
                    tier="tip",
                    content="Invalidated tip",
                    metadata={"maintenance_invalidated": True},
                ))
                plan = self._plan(store)
                snapshot = store.load_strict_snapshot()
                before = store.path.read_bytes()
                backup_dir, history_path = self._paths(root)
                backup_path = maintenance_transaction._maintenance_backup_path(
                    backup_dir,
                    plan.plan_id,
                )
                committed = maintenance_transaction._maintenance_apply_result(
                    plan=plan,
                    status=MaintenanceApplyStatus.COMMITTED,
                    mutation_committed=True,
                    audit_complete=True,
                    should_retry=False,
                    before_revision=snapshot.revision,
                    after_revision="sha256:" + "0" * 64,
                    before_count=1,
                    after_count=0,
                    removed_ids=("delete-tip",),
                    backup_path=str(backup_path),
                )
                record = maintenance_transaction._completion_history_record(
                    plan,
                    committed,
                )
                corrupt(record)
                history_path.write_text(
                    json.dumps(record) + "\n",
                    encoding="utf-8",
                )

                result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=backup_dir,
                    history_path=history_path,
                )

                self.assertEqual(
                    result.status,
                    MaintenanceApplyStatus.PRE_COMMIT_FAILED,
                )
                self.assertEqual(result.audit_error_stage, "history_load")
                self.assertEqual(result.audit_error, "MaintenancePlanError")
                self.assertFalse(result.should_retry)
                self.assertFalse(result.mutation_committed)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertIn("delete-tip", {entry.id for entry in store.all()})
                self.assertFalse(backup_path.exists())

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
