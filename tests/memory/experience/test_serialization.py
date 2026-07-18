from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.experience.serialization import (
    EXPERIENCE_SCHEMA_VERSION,
    experience_canonical_json,
    experience_from_dict,
    experience_to_dict,
)
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceTier,
    ExperienceTrajectoryStep,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType, content_fingerprint


NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


def _payloads():
    return {
        ExperienceTier.TRAJECTORY: TrajectoryPayload(
            task_description="Fix parser",
            steps=(ExperienceTrajectoryStep(1, action="run_tests", reward=1.0),),
            outcome="success",
            total_reward=1.0,
            key_learnings=("Test first",),
            tags=("parser",),
        ),
        ExperienceTier.TIP: TipPayload("testing", "warning", "focused test fails"),
        ExperienceTier.SKILL: SkillPayload(
            "testing",
            "focused-to-full",
            ("focused test exists",),
            ("run focused test", "run full suite"),
        ),
        ExperienceTier.TOOL: ToolPayload(
            "run_tests",
            "bash",
            "pytest {path} -q",
            args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    }


def _memory(tier: ExperienceTier, *, index: int = 0) -> ExperienceMemory:
    content = f"Reusable {tier.value} memory {index}"
    return ExperienceMemory(
        id=f"exp-{tier.value}-{index}",
        content=content,
        tier=tier,
        payload=_payloads()[tier],
        scope=MemoryScope.PROJECT,
        project_key="/repo",
        created_at=NOW,
        token_count=8,
        fingerprint=content_fingerprint(content),
        source_task="task-1",
        run_id="run-1",
        stream_id="stream-1",
        created_by=ExperienceCreatedBy.WRITER,
        writer_confidence=0.9,
        attribution_value=0.25,
        attribution_confidence=0.75,
        candidate_count=3,
        selected_count=2,
        not_selected_count=1,
        success_when_selected=1.0,
        success_when_candidate_not_selected=0.0,
        reward_when_selected=0.8,
        reward_when_candidate_not_selected=0.2,
        last_used=NOW,
        attribution_updated_at=NOW,
        protected=True,
        promoted_to="skill-target",
        maintenance_operation_id="op-1",
        parent_id="parent-1",
        parent_tier=ExperienceTier.TIP,
    )


class ExperienceSerializationTests(unittest.TestCase):
    def test_all_tiers_round_trip_with_concrete_payload_types(self) -> None:
        expected_types = {
            ExperienceTier.TRAJECTORY: TrajectoryPayload,
            ExperienceTier.TIP: TipPayload,
            ExperienceTier.SKILL: SkillPayload,
            ExperienceTier.TOOL: ToolPayload,
        }
        for tier in ExperienceTier:
            with self.subTest(tier=tier.value):
                memory = _memory(tier)
                payload = experience_to_dict(memory)
                restored = experience_from_dict(payload)

                self.assertEqual(restored, memory)
                self.assertIs(type(restored.payload), expected_types[tier])
                self.assertEqual(experience_to_dict(restored), payload)

    def test_serializer_uses_schema_two_and_stable_canonical_json(self) -> None:
        memory = _memory(ExperienceTier.SKILL)
        payload = experience_to_dict(memory)
        canonical = experience_canonical_json(memory)

        self.assertEqual(payload["schema_version"], EXPERIENCE_SCHEMA_VERSION)
        self.assertEqual(json.loads(canonical), payload)
        self.assertEqual(canonical, experience_canonical_json(experience_from_dict(payload)))
        self.assertNotIn("metadata", payload)
        self.assertNotIn("candidate_memory_ids", payload)
        self.assertNotIn("writer_reason", payload)

    def test_optional_top_level_fields_default_canonically(self) -> None:
        payload = experience_to_dict(_memory(ExperienceTier.TIP))
        for key in (
            "source_task",
            "run_id",
            "stream_id",
            "created_by",
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
            "protected",
            "invalidated",
            "promoted_to",
            "maintenance_operation_id",
            "parent_id",
            "parent_tier",
        ):
            payload.pop(key)

        restored = experience_from_dict(payload)

        self.assertEqual(restored.created_by, ExperienceCreatedBy.MANUAL)
        self.assertEqual(restored.writer_confidence, 1.0)
        self.assertEqual(restored.candidate_count, 0)
        self.assertIsNone(restored.last_used)
        self.assertFalse(restored.protected)

    def test_unknown_schema_tier_and_fields_fail_closed(self) -> None:
        base = experience_to_dict(_memory(ExperienceTier.TIP))
        cases = {
            "schema": {**base, "schema_version": 1},
            "tier": {**base, "tier": "future"},
            "top-level": {**base, "writer_reason": "do not persist"},
            "payload": {**base, "payload": {**base["payload"], "unknown": True}},
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    experience_from_dict(payload)

    def test_missing_required_and_tier_specific_fields_fail_closed(self) -> None:
        base = experience_to_dict(_memory(ExperienceTier.SKILL))
        missing_content = dict(base)
        missing_content.pop("content")
        missing_step = dict(base)
        missing_step["payload"] = dict(base["payload"])
        missing_step["payload"].pop("steps")

        for payload in (missing_content, missing_step):
            with self.assertRaises(ValueError):
                experience_from_dict(payload)

    def test_noncanonical_scalar_types_and_datetimes_are_rejected(self) -> None:
        base = experience_to_dict(_memory(ExperienceTier.TIP))
        cases = (
            {**base, "token_count": True},
            {**base, "candidate_count": "3"},
            {**base, "writer_confidence": "0.9"},
            {**base, "writer_confidence": float("nan")},
            {**base, "created_at": "2026-07-15T12:00:00"},
            {**base, "last_used": "2026-07-15T12:00:00"},
            {**base, "protected": 1},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    experience_from_dict(payload)

    def test_serializer_rejects_plain_memory_entry(self) -> None:
        entry = MemoryEntry.build(
            id="fact",
            content="ordinary fact",
            type=MemoryType.FACT,
            scope=MemoryScope.PROJECT,
            source="manual",
            token_count=3,
            project_key="/repo",
            created_at=NOW,
        )
        with self.assertRaises(TypeError):
            experience_to_dict(entry)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
