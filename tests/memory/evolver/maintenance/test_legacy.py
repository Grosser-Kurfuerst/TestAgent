from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver.maintenance.contracts import (
    MaintenanceAction,
    MaintenanceConfig,
    MaintenancePlanError,
    maintenance_plan_json,
    write_maintenance_plan,
)
from my_agent.memory.evolver.maintenance.legacy.planner import (
    load_project_attribution,
    lookup_experiences,
    maintenance_evidence_for_entry,
    redundancy_score,
)
from my_agent.memory.evolver.maintenance.legacy.service import build_maintenance_plan
from my_agent.memory.evolver.maintenance.legacy.validation import (
    load_maintenance_plan,
    parse_maintenance_plan,
)
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceTrajectoryStep,
    ExperienceTier,
    SkillPayload,
    TrajectoryPayload,
)
from my_agent.memory.experience_store import experience_memories_revision
from my_agent.memory.types import MemoryScope
from my_agent.opd_data.legacy.attribution import MemoryAttributionRecord
from tests.memory.experience.fixtures import typed_experience


PROJECT_KEY = "manifest:maintenance:project"
NOW = datetime(2026, 7, 16, tzinfo=timezone.utc)


def _writer_memory(
    memory_id: str,
    content: str,
    tier: ExperienceTier,
    *,
    project_key: str = PROJECT_KEY,
    created_at: datetime | None = None,
    **kwargs,
):
    return typed_experience(
        memory_id,
        content,
        tier,
        project_key=project_key,
        created_at=created_at or NOW - timedelta(days=10),
        created_by=ExperienceCreatedBy.WRITER,
        **kwargs,
    )


class MaintenanceConfigTests(unittest.TestCase):
    def test_defaults_round_trip_and_invalid_values_fail_closed(self) -> None:
        config = MaintenanceConfig()
        self.assertEqual(MaintenanceConfig.from_dict(config.to_dict()), config)
        with self.assertRaises(ValueError):
            MaintenanceConfig(merge_threshold_skill=1.1)
        with self.assertRaises(ValueError):
            MaintenanceConfig(merge_max_cluster_size=1)
        with self.assertRaises(ValueError):
            MaintenanceConfig(protect_manual=False)
        with self.assertRaises(ValueError):
            MaintenanceConfig.from_dict({"unknown": 1})


class TypedMaintenancePlannerTests(unittest.TestCase):
    def test_evidence_reads_flat_attribution_and_provenance_fields(self) -> None:
        memory = _writer_memory(
            "tip-1",
            "Inspect parser failures first.",
            ExperienceTier.TIP,
            source_task="task-7",
            writer_confidence=0.83,
            attribution_value=0.2,
            attribution_confidence=0.75,
            candidate_count=8,
            selected_count=5,
            not_selected_count=3,
            last_used=NOW - timedelta(days=1),
            attribution_updated_at=NOW,
        )

        evidence = maintenance_evidence_for_entry(
            memory,
            attribution={},
            project_key=PROJECT_KEY,
        )

        self.assertEqual(evidence.tier, "tip")
        self.assertEqual(evidence.created_by, "writer")
        self.assertEqual(evidence.source_task, "task-7")
        self.assertEqual(evidence.value, 0.2)
        self.assertEqual(evidence.writer_confidence, 0.83)
        self.assertTrue(evidence.has_attribution)

    def test_artifact_attribution_overrides_flat_snapshot_values(self) -> None:
        memory = _writer_memory("tip-1", "Use strict parsing.", ExperienceTier.TIP)
        record = MemoryAttributionRecord(
            memory_id=memory.id,
            tier=memory.tier.value,
            memory_project_key=PROJECT_KEY,
            candidate_count=6,
            selected_count=4,
            not_selected_count=2,
            value=0.31,
            confidence=0.82,
            last_used=NOW.isoformat(),
        )
        evidence = maintenance_evidence_for_entry(
            memory,
            attribution={(memory.id, memory.tier.value, PROJECT_KEY): record},
            project_key=PROJECT_KEY,
        )
        self.assertEqual(evidence.value, 0.31)
        self.assertEqual(evidence.selected_count, 4)

    def test_lookup_and_redundancy_use_typed_memories(self) -> None:
        left = _writer_memory(
            "skill-a",
            "Run focused tests, then run the full suite.",
            ExperienceTier.SKILL,
        )
        right = _writer_memory(
            "skill-b",
            "Run focused tests then run the full suite.",
            ExperienceTier.SKILL,
        )
        other = typed_experience(
            "other",
            "Run focused tests.",
            ExperienceTier.SKILL,
            project_key="other-project",
        )

        hits = lookup_experiences(
            [left, right, other],
            "focused tests",
            project_key=PROJECT_KEY,
        )

        self.assertEqual({hit.memory.id for hit in hits}, {"skill-a", "skill-b"})
        self.assertEqual(redundancy_score(left, right), 1.0)

    def test_plan_covers_keep_delete_merge_and_promote(self) -> None:
        manual = typed_experience(
            "manual",
            "Keep this manual skill.",
            ExperienceTier.SKILL,
            project_key=PROJECT_KEY,
            created_at=NOW,
            created_by=ExperienceCreatedBy.MANUAL,
        )
        invalidated = _writer_memory(
            "delete-tip",
            "Obsolete parser warning.",
            ExperienceTier.TIP,
            invalidated=True,
        )
        merge_a = _writer_memory(
            "merge-a",
            "Run focused tests, then run the full suite.",
            ExperienceTier.SKILL,
            payload=SkillPayload(
                category="testing",
                technique="focused-to-full",
                preconditions=("tests exist",),
                steps=("run focused tests",),
            ),
        )
        merge_b = _writer_memory(
            "merge-b",
            "Run focused tests then run the full suite.",
            ExperienceTier.SKILL,
            payload=SkillPayload(
                category="testing",
                technique="focused-to-full",
                preconditions=("suite is stable",),
                steps=("run full suite",),
            ),
        )
        promotable = _writer_memory(
            "promote-tip",
            "Inspect the focused failure before editing.",
            ExperienceTier.TIP,
            writer_confidence=0.8,
            attribution_value=0.2,
            attribution_confidence=0.9,
            candidate_count=6,
            selected_count=4,
            not_selected_count=2,
            attribution_updated_at=NOW,
        )
        entries = [manual, invalidated, merge_a, merge_b, promotable]

        plan = build_maintenance_plan(
            entries=entries,
            attribution={},
            repository_revision=experience_memories_revision(entries),
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        self.assertEqual(
            {key: plan.summary[key] for key in ("keep", "delete", "merge", "promote")},
            {"keep": 1, "delete": 1, "merge": 1, "promote": 1},
        )
        merge = next(op for op in plan.operations if op.action == MaintenanceAction.MERGE)
        self.assertTrue(all(item.tier == ExperienceTier.SKILL for item in merge.replacements))
        self.assertEqual(
            merge.replacements[0].payload.steps,
            ("run focused tests", "run full suite"),
        )
        promotion = next(op for op in plan.operations if op.action == MaintenanceAction.PROMOTE)
        self.assertEqual(promotion.replacements[0].promoted_to, promotion.target_ids[0])
        self.assertEqual(promotion.additions[0].tier, ExperienceTier.SKILL)
        self.assertEqual(promotion.additions[0].parent_id, promotable.id)
        self.assertEqual(promotion.additions[0].created_by, ExperienceCreatedBy.MAINTENANCE)

    def test_global_memory_can_only_be_kept_and_other_projects_are_not_considered(self) -> None:
        global_memory = typed_experience(
            "global",
            "Global protected tip.",
            ExperienceTier.TIP,
            scope=MemoryScope.GLOBAL,
            created_by=ExperienceCreatedBy.WRITER,
            invalidated=True,
        )
        other = _writer_memory(
            "other",
            "Other project tip.",
            ExperienceTier.TIP,
            project_key="other-project",
            invalidated=True,
        )
        entries = [global_memory, other]
        plan = build_maintenance_plan(
            entries=entries,
            attribution={},
            repository_revision=experience_memories_revision(entries),
            project_key=PROJECT_KEY,
            as_of=NOW,
        )
        self.assertEqual(len(plan.operations), 1)
        self.assertEqual(plan.operations[0].action, MaintenanceAction.KEEP)
        self.assertEqual(plan.operations[0].source_ids, ("global",))

    def test_successful_trajectory_promotes_to_typed_skill_payload(self) -> None:
        trajectory = _writer_memory(
            "trajectory",
            "Repair the parser and verify the focused regression.",
            ExperienceTier.TRAJECTORY,
            payload=TrajectoryPayload(
                task_description="Repair a parser regression",
                steps=(
                    ExperienceTrajectoryStep(
                        step_num=1,
                        action="inspect failure",
                        result="located strict parser boundary",
                        reward=1.0,
                    ),
                    ExperienceTrajectoryStep(
                        step_num=2,
                        action="discard noisy edit",
                        result="regression remained",
                        reward=-1.0,
                    ),
                ),
                outcome="success",
                key_learnings=("Validate the strict boundary first.",),
                tags=("parser",),
            ),
            attribution_value=0.25,
            attribution_confidence=0.9,
            candidate_count=6,
            selected_count=5,
            not_selected_count=1,
            attribution_updated_at=NOW,
        )

        plan = build_maintenance_plan(
            entries=[trajectory],
            attribution={},
            repository_revision=experience_memories_revision([trajectory]),
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        promotion = next(op for op in plan.operations if op.action == MaintenanceAction.PROMOTE)
        target = promotion.additions[0]
        self.assertIsInstance(target.payload, SkillPayload)
        self.assertEqual(target.content, "Validate the strict boundary first.")
        self.assertEqual(target.payload.category, "trajectory_distillation")
        self.assertEqual(target.payload.preconditions, ("Repair a parser regression",))
        self.assertEqual(
            target.payload.steps,
            ("inspect failure: located strict parser boundary",),
        )
        self.assertEqual(target.parent_tier, ExperienceTier.TRAJECTORY)

    def test_promotion_links_existing_skill_with_same_repository_dedup_key(self) -> None:
        content = "Inspect the focused failure before editing."
        existing = typed_experience(
            "existing-skill",
            content,
            ExperienceTier.SKILL,
            project_key=PROJECT_KEY,
            created_at=NOW - timedelta(days=20),
            created_by=ExperienceCreatedBy.MANUAL,
        )
        tip = _writer_memory(
            "tip",
            content,
            ExperienceTier.TIP,
            attribution_value=0.2,
            attribution_confidence=0.9,
            candidate_count=5,
            selected_count=4,
            not_selected_count=1,
            attribution_updated_at=NOW,
        )
        entries = [existing, tip]

        plan = build_maintenance_plan(
            entries=entries,
            attribution={},
            repository_revision=experience_memories_revision(entries),
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        promotion = next(op for op in plan.operations if op.action == MaintenanceAction.PROMOTE)
        self.assertEqual(promotion.target_ids, (existing.id,))
        self.assertEqual(promotion.additions, ())
        self.assertEqual(promotion.reason_codes, ("promotion_linked_existing_skill",))
        self.assertEqual(promotion.replacements[0].promoted_to, existing.id)

    def test_protected_memory_is_excluded_from_delete_merge_and_promote(self) -> None:
        protected_delete = _writer_memory(
            "protected-delete",
            "Obsolete but protected.",
            ExperienceTier.TIP,
            protected=True,
            invalidated=True,
        )
        protected_promote = _writer_memory(
            "protected-promote",
            "High-value but protected.",
            ExperienceTier.TIP,
            protected=True,
            attribution_value=0.3,
            attribution_confidence=0.95,
            candidate_count=8,
            selected_count=7,
            not_selected_count=1,
            attribution_updated_at=NOW,
        )
        protected_merge = _writer_memory(
            "protected-merge",
            "Run focused tests, then run the full suite.",
            ExperienceTier.SKILL,
            protected=True,
        )
        merge_peer = _writer_memory(
            "merge-peer",
            "Run focused tests then run the full suite.",
            ExperienceTier.SKILL,
        )
        entries = [protected_delete, protected_promote, protected_merge, merge_peer]

        plan = build_maintenance_plan(
            entries=entries,
            attribution={},
            repository_revision=experience_memories_revision(entries),
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        by_source = {
            operation.source_ids: operation
            for operation in plan.operations
        }
        for memory in (protected_delete, protected_promote, protected_merge):
            operation = by_source[(memory.id,)]
            self.assertEqual(operation.action, MaintenanceAction.KEEP)
            self.assertEqual(operation.reason_codes, ("protected_metadata",))
        self.assertFalse(any(op.action == MaintenanceAction.MERGE for op in plan.operations))
        self.assertFalse(any(op.action == MaintenanceAction.PROMOTE for op in plan.operations))

    def test_cross_tier_and_trajectory_candidates_never_merge(self) -> None:
        entries = [
            _writer_memory("tip", "Inspect parser failure.", ExperienceTier.TIP),
            _writer_memory("skill", "Inspect parser failure.", ExperienceTier.SKILL),
            _writer_memory(
                "trajectory-a",
                "Inspect parser failure then repair it.",
                ExperienceTier.TRAJECTORY,
            ),
            _writer_memory(
                "trajectory-b",
                "Inspect parser failure, then repair it.",
                ExperienceTier.TRAJECTORY,
            ),
        ]

        plan = build_maintenance_plan(
            entries=entries,
            attribution={},
            repository_revision=experience_memories_revision(entries),
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        self.assertFalse(any(op.action == MaintenanceAction.MERGE for op in plan.operations))
        self.assertEqual({op.action for op in plan.operations}, {MaintenanceAction.KEEP})

    def test_merge_preserves_anchor_identity_attribution_provenance_and_lineage(self) -> None:
        anchor = _writer_memory(
            "anchor",
            "Run focused tests, then run the full suite.",
            ExperienceTier.SKILL,
            source_task="task-anchor",
            run_id="run-anchor",
            stream_id="stream-anchor",
            writer_confidence=0.87,
            attribution_value=0.4,
            attribution_confidence=0.9,
            candidate_count=8,
            selected_count=6,
            not_selected_count=2,
            success_when_selected=0.8,
            success_when_candidate_not_selected=0.25,
            reward_when_selected=0.75,
            reward_when_candidate_not_selected=0.2,
            last_used=NOW - timedelta(days=1),
            attribution_updated_at=NOW,
            parent_id="parent-tip",
            parent_tier=ExperienceTier.TIP,
        )
        peer = _writer_memory(
            "peer",
            "Run focused tests then run the full suite.",
            ExperienceTier.SKILL,
            attribution_value=0.1,
            attribution_confidence=0.8,
            candidate_count=4,
            selected_count=3,
            not_selected_count=1,
            attribution_updated_at=NOW,
        )
        entries = [anchor, peer]

        plan = build_maintenance_plan(
            entries=entries,
            attribution={},
            repository_revision=experience_memories_revision(entries),
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        merge = next(op for op in plan.operations if op.action == MaintenanceAction.MERGE)
        replacement = merge.replacements[0]
        for field_name in (
            "id",
            "content",
            "fingerprint",
            "created_at",
            "scope",
            "project_key",
            "tier",
            "source_task",
            "run_id",
            "stream_id",
            "writer_confidence",
            "attribution_value",
            "attribution_confidence",
            "candidate_count",
            "selected_count",
            "not_selected_count",
            "success_when_selected",
            "success_when_candidate_not_selected",
            "reward_when_selected",
            "reward_when_candidate_not_selected",
            "last_used",
            "attribution_updated_at",
            "parent_id",
            "parent_tier",
        ):
            self.assertEqual(getattr(replacement, field_name), getattr(anchor, field_name))
        self.assertEqual(replacement.created_by, ExperienceCreatedBy.MAINTENANCE)
        self.assertEqual(replacement.maintenance_operation_id, merge.operation_id)


class MaintenancePlanContractTests(unittest.TestCase):
    def _promotion_plan(self):
        memory = _writer_memory(
            "tip",
            "Inspect errors before editing.",
            ExperienceTier.TIP,
            attribution_value=0.2,
            attribution_confidence=0.9,
            candidate_count=5,
            selected_count=4,
            not_selected_count=1,
            attribution_updated_at=NOW,
        )
        return build_maintenance_plan(
            entries=[memory],
            attribution={},
            repository_revision=experience_memories_revision([memory]),
            project_key=PROJECT_KEY,
            as_of=NOW,
        ), memory

    def test_plan_round_trip_uses_typed_schema_two_operation_payloads(self) -> None:
        plan, memory = self._promotion_plan()
        payload = plan.to_dict()
        mutation = payload["operations"][0]["additions"][0]
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(mutation["schema_version"], 2)
        self.assertEqual(parse_maintenance_plan(payload, repository_entries=[memory]), plan)

    def test_legacy_or_malformed_operation_payload_fails_closed(self) -> None:
        plan, _ = self._promotion_plan()
        payload = plan.to_dict()
        payload["operations"][0]["additions"][0] = {
            "id": "legacy",
            "content": "legacy fact",
            "type": "fact",
            "metadata": {"evolver_tier": "skill"},
        }
        with self.assertRaises(MaintenancePlanError):
            parse_maintenance_plan(payload)

    def test_plan_json_and_file_output_are_deterministic(self) -> None:
        plan, _ = self._promotion_plan()
        rendered = maintenance_plan_json(plan)
        self.assertEqual(rendered, maintenance_plan_json(plan))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_maintenance_plan(plan, Path(tmp) / "plan.json")
            self.assertEqual(path.read_text(encoding="utf-8"), rendered)
            self.assertEqual(load_maintenance_plan(path), plan)

    def test_strict_attribution_loader_rejects_cross_project_records(self) -> None:
        record = MemoryAttributionRecord(
            memory_id="tip",
            tier="tip",
            memory_project_key="other-project",
            candidate_count=1,
            selected_count=1,
            not_selected_count=0,
            value=0.1,
            confidence=0.8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory_attribution.jsonl"
            path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_project_attribution(path, memory_project_key=PROJECT_KEY)


if __name__ == "__main__":
    unittest.main()
