from __future__ import annotations

# ruff: noqa: F403, F405 - shared CLI support exports frozen fixtures

from tests.memory.evolver.maintenance.cli_support import *

class MaintenanceCliPlanningTests(_MaintenanceCliCase):
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
    def test_reviewed_plan_dry_run_rejects_stale_repository_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = self._add_invalidated_tip(memory_dir)
            plan_path = memory_dir / "reviewed.json"
            self.assertEqual(
                self._invoke(
                    self._base_args(memory_dir) + ["--output", str(plan_path)]
                )[0],
                0,
            )
            summary_path = Path(str(plan_path) + ".summary.json")
            summary_path.unlink()
            (memory_dir / "maintenance_trace.jsonl").unlink()

            snapshot = store.load_strict_snapshot()
            store.replace_all_atomically(
                (replace(snapshot.memories[0], run_id="new-run"),),
                expected_revision=snapshot.revision,
            )
            before = store.path.read_bytes()

            exit_code, _, stderr = self._invoke(
                self._base_args(memory_dir)
                + [
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(plan_path),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertIn("reviewed maintenance plan is stale", stderr)
            self.assertEqual(store.path.read_bytes(), before)
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8"))["status"],
                "pre_commit_failed",
            )
            events = _read_jsonl(memory_dir / "maintenance_trace.jsonl")
            self.assertEqual(
                [event["event"] for event in events],
                ["memory.maintenance_failed"],
            )
            self.assertEqual(events[0]["payload"]["status"], "pre_commit_failed")
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
                backup_dir / f"{plan.plan_id}.experience_memory.jsonl"
            )
            summary_path = Path(str(plan_path) + ".summary.json")
            summary_before = summary_path.read_bytes()

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
            self.assertIn("stage=artifact_validation", stderr)
            self.assertEqual(summary_path.read_bytes(), summary_before)
    def test_derived_backup_tmp_rejects_direct_symlink_and_hardlink_outputs(self) -> None:
        for alias_kind in ("direct", "symlink", "hardlink"):
            with self.subTest(alias_kind=alias_kind), tempfile.TemporaryDirectory() as tmp:
                memory_dir = Path(tmp) / "memory"
                store = self._add_invalidated_tip(memory_dir)
                before = store.path.read_bytes()
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
                plan = load_maintenance_plan(plan_path)
                backup_dir = memory_dir / "backups"
                backup_tmp = (
                    backup_dir
                    / f"{plan.plan_id}.experience_memory.jsonl.tmp"
                )
                output_path = backup_tmp
                sentinel = b""
                if alias_kind == "symlink":
                    output_path = memory_dir / "backup_tmp_symlink.json"
                    output_path.symlink_to(backup_tmp)
                elif alias_kind == "hardlink":
                    backup_dir.mkdir(parents=True)
                    sentinel = b"backup tmp sentinel\n"
                    backup_tmp.write_bytes(sentinel)
                    output_path = memory_dir / "backup_tmp_hardlink.json"
                    os.link(backup_tmp, output_path)

                exit_code, _, stderr = self._invoke(
                    self._base_args(memory_dir)
                    + [
                        "--plan",
                        str(plan_path),
                        "--output",
                        str(output_path),
                        "--trace-output",
                        str(memory_dir / "apply_trace.jsonl"),
                        "--history-output",
                        str(memory_dir / "apply_history.jsonl"),
                        "--backup-dir",
                        str(backup_dir),
                        "--apply",
                    ]
                )

                self.assertEqual(exit_code, 1)
                self.assertIn("maintenance paths must be distinct", stderr)
                self.assertIn("stage=artifact_validation", stderr)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertIn("delete-tip", {entry.id for entry in store.all()})
                self.assertFalse((memory_dir / "apply_trace.jsonl").exists())
                self.assertFalse((memory_dir / "apply_history.jsonl").exists())
                if alias_kind == "hardlink":
                    self.assertEqual(backup_tmp.read_bytes(), sentinel)
                else:
                    self.assertFalse(backup_tmp.exists())
    def test_store_tmp_and_history_lock_sidecars_cannot_be_cli_artifacts(self) -> None:
        cases = ("output_store_tmp", "output_history_lock", "history_store_tmp")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                memory_dir = Path(tmp) / "memory"
                store = self._add_invalidated_tip(memory_dir)
                before = store.path.read_bytes()
                store_tmp = store.path.with_suffix(store.path.suffix + ".tmp")
                history = memory_dir / "maintenance_history.jsonl"
                history_lock = history.with_name(f".{history.name}.lock")
                args = self._base_args(memory_dir)
                if case == "output_store_tmp":
                    args += ["--output", str(store_tmp)]
                elif case == "output_history_lock":
                    args += ["--output", str(history_lock)]
                else:
                    args += ["--history-output", str(store_tmp), "--apply"]

                exit_code, _, stderr = self._invoke(args)

                self.assertEqual(exit_code, 1)
                self.assertIn("maintenance paths must be distinct", stderr)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertIn("delete-tip", {entry.id for entry in store.all()})
                self.assertFalse(store_tmp.exists())
                self.assertFalse(history_lock.exists())
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
                    {entry.id for entry in store.load_strict_snapshot().memories},
                )
