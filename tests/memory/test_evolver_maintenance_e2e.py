from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.cli import main
from my_agent.memory.evolver import (
    DEFAULT_TIER_WEIGHTS,
    ExperienceCreatedBy,
    ExperienceSelector,
    ExperienceTier,
    MaintenanceAction,
    UsageLogEntry,
    UsageLogger,
    build_experience_entry,
    experience_tier,
    load_maintenance_plan,
)
from my_agent.memory.long_term import LongTermMemoryStore, memory_dedup_key
from my_agent.memory.retrieval import MemoryRetriever


PROJECT_KEY = "manifest:phase6-e2e:memory:shared_stream:stream:python"
AS_OF = datetime(2026, 7, 13, tzinfo=timezone.utc)
AS_OF_TEXT = AS_OF.isoformat()
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "evolver_maintenance"


class MaintenanceEndToEndTests(unittest.TestCase):
    def _invoke(self, args: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(args, ctx=SimpleNamespace())
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def _copy_fixture(self, root: Path) -> tuple[Path, Path]:
        memory_dir = root / "memory"
        memory_dir.mkdir(parents=True)
        shutil.copyfile(
            FIXTURE_DIR / "long_term_memory.jsonl",
            memory_dir / "long_term_memory.jsonl",
        )
        attribution_path = memory_dir / "memory_attribution.jsonl"
        shutil.copyfile(FIXTURE_DIR / "memory_attribution.jsonl", attribution_path)
        return memory_dir, attribution_path

    def _base_args(self, memory_dir: Path, attribution_path: Path) -> list[str]:
        return [
            "memory",
            "maintain",
            "--memory-dir",
            str(memory_dir),
            "--memory-project-key",
            PROJECT_KEY,
            "--attribution",
            str(attribution_path),
            "--as-of",
            AS_OF_TEXT,
        ]

    def test_reviewed_plan_apply_is_deterministic_audited_and_retrievable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir, attribution_path = self._copy_fixture(root)
            store = LongTermMemoryStore.from_dir(memory_dir)
            before_snapshot = store.load_strict_snapshot()
            before_bytes = before_snapshot.raw_bytes
            before = {entry.id: entry.to_dict() for entry in before_snapshot.entries}
            resident_reader = LongTermMemoryStore.from_dir(memory_dir)
            self.assertIn(
                "mem-negative-tip",
                {entry.id for entry in resident_reader.search_candidates(project_key=PROJECT_KEY)},
            )
            attribution = {
                row["memory_id"]: row
                for row in _read_jsonl(attribution_path)
            }
            for row in attribution.values():
                self._assert_phase5_attribution_record(row)
            first_plan_path = root / "maintenance_plan_first.json"
            second_plan_path = root / "maintenance_plan_second.json"

            first_result = self._invoke(
                self._base_args(memory_dir, attribution_path)
                + [
                    "--dry-run",
                    "--output",
                    str(first_plan_path),
                    "--trace-output",
                    str(root / "dry_run_first_trace.jsonl"),
                ]
            )
            second_result = self._invoke(
                self._base_args(memory_dir, attribution_path)
                + [
                    "--dry-run",
                    "--output",
                    str(second_plan_path),
                    "--trace-output",
                    str(root / "dry_run_second_trace.jsonl"),
                ]
            )

            self.assertEqual(first_result[0], 0, first_result[2])
            self.assertEqual(second_result[0], 0, second_result[2])
            self.assertEqual(store.path.read_bytes(), before_bytes)
            self.assertEqual(first_plan_path.read_bytes(), second_plan_path.read_bytes())

            plan = load_maintenance_plan(first_plan_path)
            self.assertEqual(plan.input_summary["entries_total"], 9)
            self.assertEqual(plan.input_summary["experiences_considered"], 7)
            self.assertEqual(
                {key: plan.summary[key] for key in ("keep", "delete", "merge", "promote")},
                {"keep": 3, "delete": 1, "merge": 1, "promote": 1},
            )
            merge = next(
                operation
                for operation in plan.operations
                if operation.action == MaintenanceAction.MERGE
            )
            promotion = next(
                operation
                for operation in plan.operations
                if operation.action == MaintenanceAction.PROMOTE
            )
            self.assertEqual(set(merge.source_ids), {"mem-dup-a", "mem-dup-b"})
            self.assertEqual(promotion.source_ids, ("mem-promotable-tip",))

            apply_trace = root / "apply_trace.jsonl"
            history_path = root / "maintenance_history.jsonl"
            backup_dir = root / "maintenance_backups"
            apply_result = self._invoke(
                self._base_args(memory_dir, attribution_path)
                + [
                    "--plan",
                    str(first_plan_path),
                    "--output",
                    str(first_plan_path),
                    "--trace-output",
                    str(apply_trace),
                    "--history-output",
                    str(history_path),
                    "--backup-dir",
                    str(backup_dir),
                    "--apply",
                ]
            )

            self.assertEqual(apply_result[0], 0, apply_result[2])
            self.assertIn("Status: committed", apply_result[1])
            summary = json.loads(
                Path(str(first_plan_path) + ".summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "committed")
            self.assertTrue(summary["mutation_committed"])
            self.assertTrue(summary["audit_complete"])
            self.assertFalse(summary["should_retry"])

            backups = list(backup_dir.glob("*.jsonl"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), before_bytes)
            history = _read_jsonl(history_path)
            self.assertEqual(
                [record["record_type"] for record in history],
                ["intent", "completion"],
            )
            self.assertEqual(history[-1]["status"], "committed")
            self.assertTrue(history[-1]["mutation_committed"])
            events = _read_jsonl(apply_trace)
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "memory.maintenance_started",
                    "memory.maintenance_proposed",
                    "memory.maintenance_completed",
                ],
            )
            self.assertEqual(events[-1]["payload"]["status"], "committed")

            after_snapshot = store.load_strict_snapshot()
            after = {entry.id: entry for entry in after_snapshot.entries}
            self.assertNotIn("mem-negative-tip", after)
            merged_anchor_id = merge.target_ids[0]
            merged_source_id = next(
                source_id for source_id in merge.source_ids if source_id != merged_anchor_id
            )
            self.assertIn(merged_anchor_id, after)
            self.assertNotIn(merged_source_id, after)
            merged_anchor = after[merged_anchor_id]
            self.assertEqual(
                set(merged_anchor.metadata["maintenance_source_ids"]),
                set(merge.source_ids),
            )
            self.assertEqual(
                merged_anchor.metadata["maintenance_source_fingerprints"],
                {
                    source_id: before[source_id]["fingerprint"]
                    for source_id in merge.source_ids
                },
            )
            expected_merge_evidence = {
                item["memory_id"]: item for item in merge.evidence
            }
            self.assertEqual(
                merged_anchor.metadata["maintenance_source_evidence"],
                expected_merge_evidence,
            )
            for source_id in merge.source_ids:
                for key in (
                    "value",
                    "confidence",
                    "candidate_count",
                    "selected_count",
                    "not_selected_count",
                ):
                    self.assertEqual(
                        expected_merge_evidence[source_id][key],
                        attribution[source_id][key],
                    )
            self.assertEqual(
                merged_anchor.metadata["maintenance_operation_id"],
                merge.operation_id,
            )

            source_tip = after["mem-promotable-tip"]
            promoted_id = promotion.target_ids[0]
            promoted = after[promoted_id]
            self.assertEqual(source_tip.metadata["maintenance_promoted_to"], promoted_id)
            self.assertEqual(promoted.created_at, AS_OF)
            self.assertEqual(promoted.fingerprint, source_tip.fingerprint)
            self.assertEqual(experience_tier(source_tip), ExperienceTier.TIP)
            self.assertEqual(experience_tier(promoted), ExperienceTier.SKILL)
            self.assertEqual(promoted.metadata["maintenance_parent_id"], source_tip.id)
            self.assertEqual(
                promoted.metadata["maintenance_source_fingerprints"],
                {source_tip.id: source_tip.fingerprint},
            )
            expected_promotion_evidence = {
                source_tip.id: promotion.evidence[0]
            }
            self.assertEqual(
                promoted.metadata["maintenance_source_evidence"],
                expected_promotion_evidence,
            )
            for key in (
                "value",
                "confidence",
                "candidate_count",
                "selected_count",
                "not_selected_count",
            ):
                self.assertEqual(
                    expected_promotion_evidence[source_tip.id][key],
                    attribution[source_tip.id][key],
                )
            self.assertEqual(
                promoted.metadata["maintenance_parent_value"],
                attribution[source_tip.id]["value"],
            )
            self.assertEqual(
                promoted.metadata["maintenance_parent_confidence"],
                attribution[source_tip.id]["confidence"],
            )
            self.assertEqual(
                promoted.metadata["maintenance_operation_id"],
                promotion.operation_id,
            )

            for memory_id in (
                "mem-high-skill",
                "mem-low-confidence",
                "mem-manual",
                "mem-other-project",
                "mem-ordinary-fact",
            ):
                self.assertEqual(after[memory_id].to_dict(), before[memory_id])

            raw_lines = store.path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(all(isinstance(json.loads(line), dict) for line in raw_lines))
            self.assertEqual(
                len({entry.id for entry in after_snapshot.entries}),
                len(after_snapshot.entries),
            )
            self.assertEqual(
                len({memory_dedup_key(entry) for entry in after_snapshot.entries}),
                len(after_snapshot.entries),
            )

            retriever = MemoryRetriever(now=AS_OF)
            selector = ExperienceSelector(
                tier_weights={"trajectory": 0.9, "tip": 0.8, "skill": 1.0, "tool": 1.2},
                tier_caps={"trajectory": 5, "tip": 5, "skill": 5, "tool": 5},
                selected_max_items=10,
            )
            negative_hits = retriever.retrieve(
                before["mem-negative-tip"]["content"],
                short_term=None,
                long_term=resident_reader,
                project_key=PROJECT_KEY,
                limit=20,
            )
            negative_selection = selector.select(
                query=before["mem-negative-tip"]["content"],
                hits=negative_hits,
                max_tokens=2_000,
            )
            self.assertNotIn("mem-negative-tip", {hit.entry.id for hit in negative_hits})
            self.assertNotIn(
                "mem-negative-tip",
                {candidate.id for candidate in negative_selection.candidates},
            )

            merge_hits = retriever.retrieve(
                before[merged_source_id]["content"],
                short_term=None,
                long_term=resident_reader,
                project_key=PROJECT_KEY,
                limit=20,
            )
            merge_selection = selector.select(
                query=before[merged_source_id]["content"],
                hits=merge_hits,
                max_tokens=2_000,
            )
            self.assertIn(merged_anchor_id, {hit.entry.id for hit in merge_hits})
            self.assertNotIn(merged_source_id, {hit.entry.id for hit in merge_hits})
            self.assertIn(
                merged_anchor_id,
                {candidate.id for candidate in merge_selection.candidates},
            )

            promotion_hits = retriever.retrieve(
                source_tip.content,
                short_term=None,
                long_term=resident_reader,
                project_key=PROJECT_KEY,
                limit=20,
            )
            promotion_selection = selector.select(
                query=source_tip.content,
                hits=promotion_hits,
                max_tokens=2_000,
            )
            promoted_candidate = next(
                candidate
                for candidate in promotion_selection.candidates
                if candidate.id == promoted_id
            )
            self.assertEqual(promoted_candidate.tier, ExperienceTier.SKILL)
            self.assertNotIn(
                source_tip.id,
                {candidate.id for candidate in promotion_selection.candidates},
            )

    def test_phase5_cli_attribution_is_consumed_by_phase6_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            store = LongTermMemoryStore.from_dir(memory_dir)
            store.add(build_experience_entry(
                id="phase5-negative-tip",
                content="Delete diagnostics before understanding the failing assertion.",
                tier="tip",
                project_key=PROJECT_KEY,
                created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                created_by=ExperienceCreatedBy.WRITER,
            ))
            usage_path = root / "usage_logs.jsonl"
            usage_rows = []
            for index in range(8):
                selected = index < 2
                usage_rows.append(UsageLogEntry(
                    task_id=f"phase5-task-{index}",
                    task_type="phase5_to_phase6",
                    timestamp=f"2026-07-01T00:00:0{index}+00:00",
                    run_id=f"phase5-run-{index}",
                    stream_id="python",
                    memory_project_key=PROJECT_KEY,
                    memory_mode="shared_stream",
                    retrieved_candidates={"tip": ["phase5-negative-tip"]},
                    selected_memory_ids=(
                        {"tip": ["phase5-negative-tip"]}
                        if selected
                        else {"tip": []}
                    ),
                    env_reward=0.0 if selected else 1.0,
                    success=not selected,
                    status="complete",
                ))
            UsageLogger(usage_path).overwrite(usage_rows)
            attribution_path = root / "memory_attribution.jsonl"

            score_result = self._invoke([
                "data",
                "score-memory-attribution",
                "--memory-dir",
                str(memory_dir),
                "--memory-project-key",
                PROJECT_KEY,
                "--usage-log",
                str(usage_path),
                "--output",
                str(attribution_path),
            ])

            self.assertEqual(score_result[0], 0, score_result[2])
            attribution = _read_jsonl(attribution_path)
            self.assertEqual(len(attribution), 1)
            self.assertEqual(attribution[0]["memory_project_key"], PROJECT_KEY)
            self.assertLessEqual(attribution[0]["value"], -0.05)
            plan_path = root / "maintenance_plan.json"

            maintain_result = self._invoke([
                "memory",
                "maintain",
                "--memory-dir",
                str(memory_dir),
                "--memory-project-key",
                PROJECT_KEY,
                "--attribution",
                str(attribution_path),
                "--as-of",
                AS_OF_TEXT,
                "--output",
                str(plan_path),
                "--trace-output",
                str(root / "maintenance_trace.jsonl"),
                "--history-output",
                str(root / "maintenance_history.jsonl"),
                "--backup-dir",
                str(root / "maintenance_backups"),
                "--apply",
            ])

            self.assertEqual(maintain_result[0], 0, maintain_result[2])
            plan = load_maintenance_plan(plan_path)
            self.assertEqual(plan.memory_project_key, PROJECT_KEY)
            self.assertEqual(plan.summary["delete"], 1)
            self.assertNotIn(
                "phase5-negative-tip",
                {entry.id for entry in store.search_candidates(project_key=PROJECT_KEY)},
            )

    def _assert_phase5_attribution_record(self, row: dict) -> None:
        candidate_count = int(row["candidate_count"])
        selected_count = int(row["selected_count"])
        not_selected_count = int(row["not_selected_count"])
        self.assertEqual(candidate_count, selected_count + not_selected_count)
        self.assertEqual(len(row["selected_task_ids"]), selected_count)
        self.assertEqual(len(row["not_selected_task_ids"]), not_selected_count)
        self.assertEqual(row["groups"], row["stream_ids"])
        for key in (
            "success_when_selected",
            "success_when_candidate_not_selected",
            "reward_when_selected",
            "reward_when_candidate_not_selected",
        ):
            self.assertIsNotNone(row[key])

        expected_confidence = min(1.0, sqrt(selected_count) / sqrt(8))
        self.assertAlmostEqual(row["confidence"], expected_confidence, places=6)
        selected_reward = float(row["reward_when_selected"])
        control_reward = float(row["reward_when_candidate_not_selected"])
        pool_reward = (
            selected_count * selected_reward
            + not_selected_count * control_reward
        ) / candidate_count
        raw_value = (
            (selected_reward - pool_reward)
            * selected_count
            / candidate_count
        )
        expected_value = (
            DEFAULT_TIER_WEIGHTS[row["tier"]]
            * raw_value
            * expected_confidence
        )
        self.assertAlmostEqual(row["value"], expected_value, places=6)

    def test_corrupt_unrelated_line_blocks_maintenance_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            memory_dir.mkdir()
            memory_path = memory_dir / "long_term_memory.jsonl"
            shutil.copyfile(
                FIXTURE_DIR / "long_term_memory_corrupt.jsonl",
                memory_path,
            )
            before = memory_path.read_bytes()
            plan_path = root / "corrupt_plan.json"

            exit_code, _, stderr = self._invoke([
                "memory",
                "maintain",
                "--memory-dir",
                str(memory_dir),
                "--memory-project-key",
                PROJECT_KEY,
                "--as-of",
                AS_OF_TEXT,
                "--output",
                str(plan_path),
                "--trace-output",
                str(root / "corrupt_trace.jsonl"),
                "--dry-run",
            ])

            self.assertEqual(exit_code, 1)
            self.assertIn("MemoryStoreLoadError", stderr)
            self.assertEqual(memory_path.read_bytes(), before)
            self.assertFalse(plan_path.exists())


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
