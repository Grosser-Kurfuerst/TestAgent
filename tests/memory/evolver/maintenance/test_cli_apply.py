from __future__ import annotations

# ruff: noqa: F403, F405 - shared CLI support exports frozen fixtures

from tests.memory.evolver.maintenance.cli_support import *

class MaintenanceCliApplyTests(_MaintenanceCliCase):
    def test_apply_reviewed_plan_writes_history_backup_and_completed_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            dry_trace = memory_dir / "dry_trace.jsonl"
            dry_args = self._base_args(memory_dir) + [
                "--output",
                str(plan_path),
                "--trace-output",
                str(dry_trace),
            ]
            self.assertEqual(self._invoke(dry_args)[0], 0)
            apply_trace = memory_dir / "apply_trace.jsonl"

            exit_code, stdout, stderr = self._invoke(
                self._base_args(memory_dir)
                + [
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(plan_path),
                    "--trace-output",
                    str(apply_trace),
                    "--apply",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr, "")
            self.assertIn("Status: committed", stdout)
            self.assertNotIn(
                "delete-tip",
                {entry.id for entry in store.load_strict_snapshot().memories},
            )
            summary = json.loads(
                Path(str(plan_path) + ".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "committed")
            self.assertTrue(summary["mutation_committed"])
            self.assertFalse(summary["should_retry"])
            history = _read_jsonl(memory_dir / "maintenance_history.jsonl")
            self.assertEqual(
                [record["record_type"] for record in history],
                ["intent", "completion"],
            )
            self.assertEqual(len(list((memory_dir / "maintenance_backups").glob("*.jsonl"))), 1)
            events = _read_jsonl(apply_trace)
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "memory.maintenance_started",
                    "memory.maintenance_proposed",
                    "memory.maintenance_completed",
                ],
            )
            completed = events[-1]["payload"]
            for key in (
                "plan_id",
                "status",
                "mutation_committed",
                "audit_complete",
                "should_retry",
                "before_revision",
                "after_revision",
                "before_count",
                "after_count",
                "removed_ids",
                "updated_ids",
                "added_ids",
                "backup_path",
            ):
                self.assertIn(key, completed)
    def test_apply_without_plan_persists_artifact_before_calling_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "generated_before_apply.json"
            original_apply = maintenance_cli.apply_maintenance_plan

            def checking_apply(**kwargs):
                self.assertTrue(plan_path.exists())
                self.assertEqual(load_maintenance_plan(plan_path), kwargs["plan"])
                return original_apply(**kwargs)

            with patch.object(
                maintenance_cli,
                "apply_maintenance_plan",
                side_effect=checking_apply,
            ):
                exit_code, _, _ = self._invoke(
                    self._base_args(memory_dir)
                    + ["--output", str(plan_path), "--apply"]
                )

            self.assertEqual(exit_code, 0)
            self.assertNotIn(
                "delete-tip",
                {entry.id for entry in store.load_strict_snapshot().memories},
            )
    def test_reviewed_plan_project_mismatch_fails_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir) + ["--output", str(plan_path)]
                )[0],
                0,
            )
            before = store.path.read_bytes()
            mismatch_trace = memory_dir / "mismatch_trace.jsonl"

            exit_code, _, stderr = self._invoke([
                "memory",
                "maintain",
                "--memory-dir",
                str(memory_dir),
                "--memory-project-key",
                OTHER_PROJECT_KEY,
                "--plan",
                str(plan_path),
                "--trace-output",
                str(mismatch_trace),
                "--apply",
            ])

            self.assertEqual(exit_code, 1)
            self.assertIn("status=pre_commit_failed", stderr)
            self.assertIn("memory_project_key does not match", stderr)
            self.assertEqual(store.path.read_bytes(), before)
            event = _read_jsonl(mismatch_trace)[0]
            self.assertEqual(event["event"], "memory.maintenance_failed")
            self.assertEqual(event["payload"]["stage"], "validation")
            self.assertNotIn("This parser warning", json.dumps(event))
    def test_revision_conflict_returns_nonzero_and_failed_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir) + ["--output", str(plan_path)]
                )[0],
                0,
            )
            store.add(typed_experience(
                "later-skill",
                "A later unrelated skill.",
                ExperienceTier.SKILL,
                project_key=PROJECT_KEY,
                created_at=NOW,
            ))
            trace_path = memory_dir / "conflict_trace.jsonl"

            exit_code, _, stderr = self._invoke(
                self._base_args(memory_dir)
                + [
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(plan_path),
                    "--trace-output",
                    str(trace_path),
                    "--apply",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertIn("status=pre_commit_failed", stderr)
            summary = json.loads(
                Path(str(plan_path) + ".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "pre_commit_failed")
            self.assertFalse(summary["mutation_committed"])
            events = _read_jsonl(trace_path)
            self.assertEqual(events[-1]["event"], "memory.maintenance_failed")
            self.assertNotIn(
                "memory.maintenance_completed",
                {event["event"] for event in events},
            )
            self.assertIn("delete-tip", {entry.id for entry in store.all()})
            self.assertIn("later-skill", {entry.id for entry in store.all()})
    def test_strict_load_and_invalid_config_fail_with_safe_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            store = self._store(memory_dir)
            store.path.write_text("{secret-invalid-memory\n", encoding="utf-8")
            trace_path = memory_dir / "strict_failure_trace.jsonl"

            exit_code, _, stderr = self._invoke(
                self._base_args(memory_dir) + ["--trace-output", str(trace_path)]
            )

            self.assertEqual(exit_code, 1)
            self.assertIn("stage=strict_load", stderr)
            event_text = trace_path.read_text(encoding="utf-8")
            self.assertNotIn("secret-invalid-memory", event_text)
            event = _read_jsonl(trace_path)[0]
            self.assertEqual(event["event"], "memory.maintenance_failed")
            self.assertEqual(event["payload"]["stage"], "strict_load")

        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            self._add_invalidated_tip(memory_dir)
            exit_code, _, stderr = self._invoke(
                self._base_args(memory_dir) + ["--merge-threshold", "1.5"]
            )
            self.assertEqual(exit_code, 1)
            self.assertIn("stage=validation", stderr)
    def test_typed_history_corruption_is_reported_as_history_load_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir) + ["--output", str(plan_path)]
                )[0],
                0,
            )
            plan = load_maintenance_plan(plan_path)
            snapshot = store.load_strict_snapshot()
            backup_dir = memory_dir / "maintenance_backups"
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
            record["mutation_committed"] = "false"
            record["result"]["mutation_committed"] = "false"
            history_path = memory_dir / "maintenance_history.jsonl"
            history_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            exit_code, _, stderr = self._invoke(
                self._base_args(memory_dir)
                + [
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(plan_path),
                    "--apply",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertIn("stage=history_load", stderr)
            summary = json.loads(
                Path(str(plan_path) + ".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "pre_commit_failed")
            self.assertEqual(summary["audit_error_stage"], "history_load")
            self.assertFalse(summary["mutation_committed"])
            self.assertFalse(summary["should_retry"])
            self.assertIn("delete-tip", {entry.id for entry in store.all()})
            self.assertFalse(backup_path.exists())
    def test_reviewed_plan_schema_error_and_explicit_missing_attribution_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            self._add_invalidated_tip(memory_dir)
            invalid_plan = memory_dir / "invalid_plan.json"
            invalid_plan.write_text(
                json.dumps({"schema_version": 999}),
                encoding="utf-8",
            )

            exit_code, _, stderr = self._invoke(
                self._base_args(memory_dir)
                + ["--plan", str(invalid_plan), "--apply"]
            )
            self.assertEqual(exit_code, 1)
            self.assertIn("stage=plan_load", stderr)

        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            self._add_invalidated_tip(memory_dir)
            exit_code, _, stderr = self._invoke(
                self._base_args(memory_dir)
                + ["--attribution", str(memory_dir / "missing.jsonl")]
            )
            self.assertEqual(exit_code, 1)
            self.assertIn("FileNotFoundError", stderr)
    def test_lock_timeout_returns_nonzero_pre_commit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir) + ["--output", str(plan_path)]
                )[0],
                0,
            )
            trace_path = memory_dir / "lock_trace.jsonl"
            with store.exclusive_process_lock():
                exit_code, _, stderr = self._invoke(
                    self._base_args(memory_dir)
                    + [
                        "--plan",
                        str(plan_path),
                        "--trace-output",
                        str(trace_path),
                        "--lock-timeout-seconds",
                        "0.01",
                        "--apply",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("stage=lock", stderr)
            summary = json.loads(
                (memory_dir / "maintenance_plan.json.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "pre_commit_failed")
            self.assertTrue(summary["should_retry"])
            event = _read_jsonl(trace_path)[0]
            self.assertEqual(event["event"], "memory.maintenance_failed")
            self.assertEqual(event["payload"]["stage"], "lock")
    def test_history_lock_timeout_is_reported_retryable_by_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            before = store.path.read_bytes()
            history_path = memory_dir / "maintenance_history.jsonl"
            history_path.write_text("", encoding="utf-8")
            history_lock_path = history_path.with_name(f".{history_path.name}.lock")
            trace_path = memory_dir / "history_lock_trace.jsonl"

            with FileLock(str(history_lock_path)):
                exit_code, _, stderr = self._invoke(
                    self._base_args(memory_dir)
                    + [
                        "--trace-output",
                        str(trace_path),
                        "--lock-timeout-seconds",
                        "0.05",
                        "--apply",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("stage=history_lock", stderr)
            self.assertEqual(store.path.read_bytes(), before)
            summary = json.loads(
                (memory_dir / "maintenance_plan.json.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "pre_commit_failed")
            self.assertEqual(summary["audit_error_stage"], "history_lock")
            self.assertEqual(summary["audit_error"], "MaintenanceHistoryLockTimeout")
            self.assertTrue(summary["should_retry"])
            self.assertFalse(summary["mutation_committed"])
            event = _read_jsonl(trace_path)[-1]
            self.assertEqual(event["event"], "memory.maintenance_failed")
            self.assertEqual(event["payload"]["stage"], "history_lock")
            self.assertTrue(event["payload"]["should_retry"])
