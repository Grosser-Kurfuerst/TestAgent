from __future__ import annotations

# ruff: noqa: F403, F405 - shared CLI support exports frozen fixtures

from tests.memory.evolver.maintenance.cli_support import *

class MaintenanceCliAuditTests(_MaintenanceCliCase):
    def test_committed_with_audit_error_returns_zero_and_warns_not_to_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir) + ["--output", str(plan_path)]
                )[0],
                0,
            )
            plan = load_maintenance_plan(plan_path)
            result = MaintenanceApplyResult(
                plan_id=plan.plan_id,
                status=MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
                mutation_committed=True,
                audit_complete=False,
                should_retry=False,
                before_revision=plan.repository_revision,
                after_revision="sha256:after",
                before_count=1,
                after_count=0,
                kept=0,
                deleted=1,
                merged=0,
                promoted=0,
                removed_ids=("delete-tip",),
                backup_path=str(memory_dir / "maintenance_backups" / "backup.jsonl"),
                audit_error_stage="history_completion",
                audit_error="OSError",
            )
            trace_path = memory_dir / "audit_error_trace.jsonl"

            with patch.object(
                maintenance_cli,
                "apply_maintenance_plan",
                return_value=result,
            ):
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

            self.assertEqual(exit_code, 0)
            self.assertIn("DO NOT RETRY", stderr)
            summary = json.loads(
                Path(str(plan_path) + ".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "committed_with_audit_error")
            self.assertFalse(summary["should_retry"])
            events = _read_jsonl(trace_path)
            self.assertEqual(events[-1]["event"], "memory.maintenance_completed")
            self.assertNotIn(
                "memory.maintenance_failed",
                {event["event"] for event in events},
            )
    def test_completed_trace_failure_does_not_turn_commit_into_retryable_failure(self) -> None:
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
            original_append = maintenance_cli.TraceWriter.append

            def fail_completed(writer, event):
                if event.event == "memory.maintenance_completed":
                    raise OSError("trace unavailable")
                return original_append(writer, event)

            with patch.object(
                maintenance_cli.TraceWriter,
                "append",
                autospec=True,
                side_effect=fail_completed,
            ):
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

            self.assertEqual(exit_code, 0)
            self.assertIn("DO NOT RETRY", stderr)
            self.assertNotIn(
                "delete-tip",
                {entry.id for entry in store.load_strict_snapshot().memories},
            )
            summary = json.loads(
                Path(str(plan_path) + ".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "committed_with_audit_error")
            self.assertFalse(summary["audit_complete"])
            self.assertEqual(summary["audit_error_stage"], "trace")
            self.assertFalse(summary["trace_complete"])
            self.assertFalse(summary["should_retry"])
            history = _read_jsonl(memory_dir / "maintenance_history.jsonl")
            self.assertEqual(
                [record["record_type"] for record in history],
                ["intent", "completion", "audit_error"],
            )
            self.assertEqual(history[-1]["status"], "committed_with_audit_error")
            self.assertEqual(history[-1]["audit_error_stage"], "trace")
    def test_summary_failure_after_commit_emits_audit_error_completion(self) -> None:
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
            trace_path = memory_dir / "summary_failure_trace.jsonl"

            with patch.object(
                maintenance_cli,
                "_write_summary",
                side_effect=OSError("summary unavailable"),
            ):
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

            self.assertEqual(exit_code, 0)
            self.assertIn("DO NOT RETRY", stderr)
            self.assertNotIn(
                "delete-tip",
                {entry.id for entry in store.load_strict_snapshot().memories},
            )
            events = _read_jsonl(trace_path)
            self.assertEqual(events[-1]["event"], "memory.maintenance_completed")
            completed = events[-1]["payload"]
            self.assertEqual(completed["status"], "committed_with_audit_error")
            self.assertTrue(completed["mutation_committed"])
            self.assertFalse(completed["audit_complete"])
            self.assertFalse(completed["should_retry"])
            self.assertEqual(completed["audit_error_stage"], "summary")
            history = _read_jsonl(memory_dir / "maintenance_history.jsonl")
            self.assertEqual(
                [record["record_type"] for record in history],
                ["intent", "completion", "audit_error"],
            )
            self.assertEqual(history[-1]["status"], "committed_with_audit_error")
            self.assertEqual(history[-1]["audit_error_stage"], "summary")
    def test_post_commit_finalizer_uses_cli_lock_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir) + ["--output", str(plan_path)]
                )[0],
                0,
            )
            original_record = maintenance_cli.record_post_commit_audit_error
            observed_timeouts: list[float] = []

            def record_with_timeout(**kwargs):
                observed_timeouts.append(kwargs["lock_timeout_seconds"])
                return original_record(**kwargs)

            with (
                patch.object(
                    maintenance_cli,
                    "_write_summary",
                    side_effect=OSError("summary unavailable"),
                ),
                patch.object(
                    maintenance_cli,
                    "record_post_commit_audit_error",
                    side_effect=record_with_timeout,
                ),
            ):
                exit_code, _, stderr = self._invoke(
                    self._base_args(memory_dir)
                    + [
                        "--plan",
                        str(plan_path),
                        "--output",
                        str(plan_path),
                        "--lock-timeout-seconds",
                        "0.05",
                        "--apply",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("DO NOT RETRY", stderr)
            self.assertEqual(observed_timeouts, [0.05])
    def test_unexpected_post_commit_helpers_use_independent_emergency_finalizer(self) -> None:
        for helper_name in ("_summary_for_apply", "_render_summary"):
            with self.subTest(helper=helper_name), tempfile.TemporaryDirectory() as tmp:
                memory_dir = Path(tmp) / "memory"
                store = self._add_invalidated_tip(memory_dir)
                plan_path = memory_dir / "reviewed_plan.json"
                self.assertEqual(
                    self._invoke(
                        self._base_args(memory_dir)
                        + [
                            "--output",
                            str(plan_path),
                            "--trace-output",
                            str(memory_dir / "dry_trace.jsonl"),
                        ]
                    )[0],
                    0,
                )
                trace_path = memory_dir / f"{helper_name}_trace.jsonl"

                with patch.object(
                    maintenance_cli,
                    helper_name,
                    side_effect=RuntimeError("unexpected CLI failure"),
                ):
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

                self.assertEqual(exit_code, 0)
                self.assertIn("DO NOT RETRY", stderr)
                self.assertNotIn(
                    "delete-tip",
                    {entry.id for entry in store.load_strict_snapshot().memories},
                )
                summary = json.loads(
                    Path(str(plan_path) + ".summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(summary["status"], "committed_with_audit_error")
                self.assertTrue(summary["mutation_committed"])
                self.assertFalse(summary["audit_complete"])
                self.assertFalse(summary["should_retry"])
                events = _read_jsonl(trace_path)
                self.assertEqual(events[-1]["event"], "memory.maintenance_completed")
                self.assertEqual(
                    events[-1]["payload"]["status"],
                    "committed_with_audit_error",
                )
                self.assertNotIn(
                    "memory.maintenance_failed",
                    {event["event"] for event in events},
                )
    def test_shared_helper_failure_after_commit_still_writes_emergency_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir)
                    + [
                        "--output",
                        str(plan_path),
                        "--trace-output",
                        str(memory_dir / "dry_trace.jsonl"),
                    ]
                )[0],
                0,
            )
            trace_path = memory_dir / "shared_helper_trace.jsonl"
            original_apply = maintenance_cli.apply_maintenance_plan
            original_safe_int = maintenance_cli._safe_int
            committed = False

            def apply_then_break_helpers(**kwargs):
                nonlocal committed
                result = original_apply(**kwargs)
                committed = result.mutation_committed
                return result

            def safe_int_until_commit(value):
                if committed:
                    raise RuntimeError("shared conversion helper failed")
                return original_safe_int(value)

            with patch.object(
                maintenance_cli,
                "apply_maintenance_plan",
                side_effect=apply_then_break_helpers,
            ), patch.object(
                maintenance_cli,
                "_safe_int",
                side_effect=safe_int_until_commit,
            ):
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

            self.assertEqual(exit_code, 0)
            self.assertIn("DO NOT RETRY", stderr)
            self.assertNotIn(
                "delete-tip",
                {entry.id for entry in store.load_strict_snapshot().memories},
            )
            summary = json.loads(
                Path(str(plan_path) + ".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "committed_with_audit_error")
            self.assertFalse(summary["audit_complete"])
            self.assertFalse(summary["should_retry"])
            events = _read_jsonl(trace_path)
            self.assertEqual(events[-1]["event"], "memory.maintenance_completed")
            self.assertEqual(
                events[-1]["payload"]["status"],
                "committed_with_audit_error",
            )
            self.assertNotIn(
                "memory.maintenance_failed",
                {event["event"] for event in events},
            )
    def test_emergency_audit_sinks_fail_independently_without_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir)
                    + [
                        "--output",
                        str(plan_path),
                        "--trace-output",
                        str(memory_dir / "dry_trace.jsonl"),
                    ]
                )[0],
                0,
            )
            trace_path = memory_dir / "combined_failure_trace.jsonl"
            original_apply = maintenance_cli.apply_maintenance_plan
            original_append = maintenance_cli.TraceWriter.append
            original_safe_int = maintenance_cli._safe_int
            committed = False
            completed_attempts = 0

            def apply_then_break_helpers(**kwargs):
                nonlocal committed
                result = original_apply(**kwargs)
                committed = result.mutation_committed
                return result

            def safe_int_until_commit(value):
                if committed:
                    raise RuntimeError("shared conversion helper failed")
                return original_safe_int(value)

            def fail_completed_trace(writer, event):
                nonlocal completed_attempts
                if event.event == "memory.maintenance_completed":
                    completed_attempts += 1
                    raise OSError("completed trace unavailable")
                return original_append(writer, event)

            with patch.object(
                maintenance_cli,
                "apply_maintenance_plan",
                side_effect=apply_then_break_helpers,
            ), patch.object(
                maintenance_cli,
                "_safe_int",
                side_effect=safe_int_until_commit,
            ), patch.object(
                maintenance_cli,
                "_write_summary",
                side_effect=OSError("summary unavailable"),
            ) as write_summary, patch.object(
                maintenance_cli.TraceWriter,
                "append",
                autospec=True,
                side_effect=fail_completed_trace,
            ):
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

            self.assertEqual(exit_code, 0)
            self.assertIn("DO NOT RETRY", stderr)
            self.assertGreaterEqual(write_summary.call_count, 2)
            self.assertEqual(completed_attempts, 1)
            self.assertNotIn(
                "delete-tip",
                {entry.id for entry in store.load_strict_snapshot().memories},
            )
            events = _read_jsonl(trace_path)
            self.assertEqual(
                [event["event"] for event in events],
                ["memory.maintenance_started", "memory.maintenance_proposed"],
            )
            self.assertNotIn(
                "memory.maintenance_failed",
                {event["event"] for event in events},
            )
