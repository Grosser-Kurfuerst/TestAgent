from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.cli import build_parser, main
from my_agent.cli import memory_maintenance as maintenance_cli
from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    MaintenanceApplyResult,
    MaintenanceApplyStatus,
    build_experience_entry,
    load_maintenance_plan,
)
from my_agent.memory.long_term import LongTermMemoryStore
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType


PROJECT_KEY = "manifest:demo:memory:shared_stream:stream:python"
OTHER_PROJECT_KEY = "manifest:demo:memory:shared_stream:stream:other"
AS_OF = "2026-07-12T00:00:00+00:00"
NOW = datetime(2026, 7, 12, tzinfo=timezone.utc)


class MaintenanceCliTests(unittest.TestCase):
    def _store(self, memory_dir: Path) -> LongTermMemoryStore:
        return LongTermMemoryStore.from_dir(memory_dir)

    def _add_invalidated_tip(self, memory_dir: Path) -> LongTermMemoryStore:
        store = self._store(memory_dir)
        store.add(build_experience_entry(
            id="delete-tip",
            content="This parser warning is obsolete.",
            tier="tip",
            project_key=PROJECT_KEY,
            created_at=NOW,
            created_by=ExperienceCreatedBy.WRITER,
            extra_metadata={"maintenance_invalidated": True},
        ))
        return store

    def _invoke(self, args: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args, ctx=SimpleNamespace())
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def _base_args(self, memory_dir: Path) -> list[str]:
        return [
            "memory",
            "maintain",
            "--memory-dir",
            str(memory_dir),
            "--memory-project-key",
            PROJECT_KEY,
            "--as-of",
            AS_OF,
        ]

    def test_parser_registers_single_project_maintenance_and_defaults_to_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "memory",
            "maintain",
            "--memory-dir",
            "/tmp/memory",
            "--memory-project-key",
            PROJECT_KEY,
        ])

        self.assertEqual(args.command, "memory")
        self.assertEqual(args.memory_command, "maintain")
        self.assertFalse(args.apply)
        self.assertFalse(args.dry_run)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([
                    "memory",
                    "maintain",
                    "--memory-dir",
                    "/tmp/memory",
                    "--memory-project-key",
                    " ",
                ])
            with self.assertRaises(SystemExit):
                parser.parse_args([
                    "memory",
                    "maintain",
                    "--memory-dir",
                    "/tmp/memory",
                    "--memory-project-key",
                    PROJECT_KEY,
                    "--all-projects",
                ])
            with self.assertRaises(SystemExit):
                parser.parse_args([
                    "memory",
                    "maintain",
                    "--memory-dir",
                    "/tmp/memory",
                    "--memory-project-key",
                    PROJECT_KEY,
                    "--include-global",
                ])
            for value in ("nan", "inf", "-inf"):
                with self.subTest(lock_timeout=value), self.assertRaises(SystemExit):
                    parser.parse_args([
                        "memory",
                        "maintain",
                        "--memory-dir",
                        "/tmp/memory",
                        "--memory-project-key",
                        PROJECT_KEY,
                        "--lock-timeout-seconds",
                        value,
                    ])

    def test_default_dry_run_writes_plan_summary_and_trace_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            before = store.path.read_bytes()

            exit_code, stdout, stderr = self._invoke(self._base_args(memory_dir))

            plan_path = memory_dir / "maintenance_plan.json"
            summary_path = Path(str(plan_path) + ".summary.json")
            trace_path = memory_dir / "maintenance_trace.jsonl"
            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr, "")
            self.assertIn("Mode: dry_run", stdout)
            self.assertEqual(store.path.read_bytes(), before)
            self.assertTrue(plan_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertFalse((memory_dir / "maintenance_history.jsonl").exists())
            self.assertFalse((memory_dir / "maintenance_backups").exists())

            plan = load_maintenance_plan(plan_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            events = _read_jsonl(trace_path)
            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["plan_id"], plan.plan_id)
            self.assertEqual(summary["delete"], 1)
            self.assertFalse(summary["mutation_committed"])
            self.assertEqual(
                [event["event"] for event in events],
                ["memory.maintenance_started", "memory.maintenance_proposed"],
            )
            started = events[0]["payload"]
            self.assertEqual(started["mode"], "dry_run")
            self.assertEqual(started["policy"], plan.policy)
            self.assertEqual(started["scope_mode"], "single_project")
            self.assertEqual(started["memory_project_key"], PROJECT_KEY)
            self.assertEqual(started["repository_revision"], plan.repository_revision)
            self.assertEqual(started["entries_total"], 1)
            self.assertEqual(started["experiences_considered"], 1)
            self.assertEqual(started["as_of"], AS_OF)
            proposed = events[1]["payload"]
            self.assertEqual(proposed["plan_id"], plan.plan_id)
            self.assertEqual(proposed["delete"], 1)
            self.assertEqual(proposed["source_entries_removed"], 1)
            self.assertIsInstance(proposed["operation_summaries"], list)
            self.assertTrue(events[0]["run_id"].startswith("maintenance-maint-"))

    def test_default_as_of_and_threshold_overrides_are_serialized_in_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            self._add_invalidated_tip(memory_dir)
            before_date = datetime.now(timezone.utc).date()

            exit_code, _, _ = self._invoke([
                "memory",
                "maintain",
                "--memory-dir",
                str(memory_dir),
                "--memory-project-key",
                PROJECT_KEY,
                "--delete-value-threshold",
                "-0.2",
                "--delete-min-confidence",
                "0.8",
                "--delete-min-candidate-count",
                "7",
                "--stale-after-days",
                "120",
                "--merge-threshold",
                "0.91",
                "--merge-max-cluster-size",
                "4",
                "--promote-value-threshold",
                "0.3",
                "--promote-min-confidence",
                "0.85",
                "--promote-min-selected-count",
                "6",
                "--max-promotions",
                "3",
            ])

            self.assertEqual(exit_code, 0)
            plan = load_maintenance_plan(memory_dir / "maintenance_plan.json")
            as_of = datetime.fromisoformat(plan.as_of)
            self.assertIn(as_of.date(), {before_date, datetime.now(timezone.utc).date()})
            self.assertEqual(
                (as_of.hour, as_of.minute, as_of.second, as_of.microsecond),
                (0, 0, 0, 0),
            )
            self.assertEqual(as_of.utcoffset(), timezone.utc.utcoffset(as_of))
            self.assertEqual(plan.config["delete_value_threshold"], -0.2)
            self.assertEqual(plan.config["delete_min_confidence"], 0.8)
            self.assertEqual(plan.config["delete_min_candidate_count"], 7)
            self.assertEqual(plan.config["stale_after_days"], 120)
            self.assertEqual(plan.config["merge_threshold_tip"], 0.91)
            self.assertEqual(plan.config["merge_threshold_skill"], 0.91)
            self.assertEqual(plan.config["merge_threshold_tool"], 0.91)
            self.assertEqual(plan.config["merge_max_cluster_size"], 4)
            self.assertEqual(plan.config["promote_value_threshold"], 0.3)
            self.assertEqual(plan.config["promote_min_confidence"], 0.85)
            self.assertEqual(plan.config["promote_min_selected_count"], 6)
            self.assertEqual(plan.config["max_promotions"], 3)

    def test_artifact_paths_cannot_alias_store_or_each_other(self) -> None:
        cases = (
            "output_direct",
            "trace_direct",
            "output_symlink",
            "trace_hardlink",
            "output_trace_same",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                memory_dir = Path(tmp) / "memory"
                store = self._add_invalidated_tip(memory_dir)
                before = store.path.read_bytes()
                alias = memory_dir / "artifact_alias.jsonl"
                args = self._base_args(memory_dir)
                if case == "output_direct":
                    args += ["--output", str(store.path)]
                elif case == "trace_direct":
                    args += ["--trace-output", str(store.path)]
                elif case == "output_symlink":
                    alias.symlink_to(store.path)
                    args += ["--output", str(alias)]
                elif case == "trace_hardlink":
                    os.link(store.path, alias)
                    args += ["--trace-output", str(alias)]
                else:
                    args += [
                        "--output",
                        str(alias),
                        "--trace-output",
                        str(alias),
                    ]

                exit_code, _, stderr = self._invoke(args)

                self.assertEqual(exit_code, 1)
                self.assertIn("maintenance paths must be distinct", stderr)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertFalse((memory_dir / "maintenance_trace.jsonl").exists())
                self.assertFalse((memory_dir / "maintenance_plan.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            dangerous_path = Path(tmp) / "not-created"
            exit_code, _, stderr = self._invoke([
                "memory",
                "maintain",
                "--memory-dir",
                str(dangerous_path),
                "--memory-project-key",
                PROJECT_KEY,
                "--output",
                str(dangerous_path),
            ])
            self.assertEqual(exit_code, 1)
            self.assertIn("maintenance paths must be distinct", stderr)
            self.assertFalse(dangerous_path.exists())

    def test_apply_rejects_history_aliasing_concrete_backup_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            before = store.path.read_bytes()
            plan_path = memory_dir / "reviewed_plan.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir) + ["--output", str(plan_path)]
                )[0],
                0,
            )
            plan = load_maintenance_plan(plan_path)
            backup_dir = memory_dir / "backups"
            concrete_backup = (
                backup_dir / f"{plan.plan_id}.long_term_memory.jsonl"
            )

            exit_code, _, stderr = self._invoke(
                self._base_args(memory_dir)
                + [
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(plan_path),
                    "--backup-dir",
                    str(backup_dir),
                    "--history-output",
                    str(concrete_backup),
                    "--apply",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertIn("status=pre_commit_failed", stderr)
            self.assertEqual(store.path.read_bytes(), before)
            self.assertIn("delete-tip", {entry.id for entry in store.all()})
            self.assertFalse(concrete_backup.exists())
            summary = json.loads(
                Path(str(plan_path) + ".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["audit_error_stage"], "artifact_validation")

    def test_reviewed_plan_alias_is_reused_without_rewrite(self) -> None:
        for alias_kind in ("direct", "symlink", "hardlink"):
            with self.subTest(alias_kind=alias_kind), tempfile.TemporaryDirectory() as tmp:
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
                before = plan_path.read_bytes()
                output_path = plan_path
                if alias_kind == "symlink":
                    output_path = memory_dir / "plan_symlink.json"
                    output_path.symlink_to(plan_path)
                elif alias_kind == "hardlink":
                    output_path = memory_dir / "plan_hardlink.json"
                    os.link(plan_path, output_path)

                with patch.object(
                    maintenance_cli,
                    "write_maintenance_plan",
                    wraps=maintenance_cli.write_maintenance_plan,
                ) as write_plan:
                    exit_code, _, stderr = self._invoke(
                        self._base_args(memory_dir)
                        + [
                            "--plan",
                            str(plan_path),
                            "--output",
                            str(output_path),
                            "--trace-output",
                            str(memory_dir / "apply_trace.jsonl"),
                            "--apply",
                        ]
                    )

                self.assertEqual(exit_code, 0)
                self.assertEqual(stderr, "")
                write_plan.assert_not_called()
                self.assertEqual(plan_path.read_bytes(), before)
                self.assertNotIn(
                    "delete-tip",
                    {entry.id for entry in store.load_strict_snapshot().entries},
                )

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
                {entry.id for entry in store.load_strict_snapshot().entries},
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
                {entry.id for entry in store.load_strict_snapshot().entries},
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
            store.add(MemoryEntry.build(
                id="later-fact",
                content="A later unrelated fact.",
                type=MemoryType.FACT,
                scope=MemoryScope.PROJECT,
                source="manual",
                token_count=4,
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
            self.assertIn("later-fact", {entry.id for entry in store.all()})

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
                {entry.id for entry in store.load_strict_snapshot().entries},
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
                {entry.id for entry in store.load_strict_snapshot().entries},
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
                    {entry.id for entry in store.load_strict_snapshot().entries},
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
                {entry.id for entry in store.load_strict_snapshot().entries},
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
                {entry.id for entry in store.load_strict_snapshot().entries},
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


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
