from __future__ import annotations

import unittest
from dataclasses import fields, replace
from datetime import datetime, timezone

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceTier,
    ExperienceTrajectoryStep,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.types import MemoryContext, MemoryScope, RetrievalHit, content_fingerprint


NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


def _trajectory_payload() -> TrajectoryPayload:
    return TrajectoryPayload(
        task_description="Fix the parser regression",
        steps=(
            ExperienceTrajectoryStep(
                step_num=1,
                observation="Focused parser test fails",
                action="run_tests",
                action_params={"path": "tests/test_parser.py"},
                result="Parser failure reproduced",
                reward=0.0,
            ),
        ),
        outcome="success",
        total_reward=1.0,
        key_learnings=("Run the focused test first",),
        tags=("parser", "testing"),
    )


def _payloads() -> dict[ExperienceTier, object]:
    return {
        ExperienceTier.TRAJECTORY: _trajectory_payload(),
        ExperienceTier.TIP: TipPayload(
            category="testing",
            severity="warning",
            trigger="A focused test fails",
        ),
        ExperienceTier.SKILL: SkillPayload(
            category="testing",
            technique="Focused-to-full validation",
            preconditions=("A focused test target is known",),
            steps=("Run focused tests", "Run the full suite"),
        ),
        ExperienceTier.TOOL: ToolPayload(
            name="run_focused_tests",
            language="bash",
            code="pytest {path} -q",
            args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    }


def _memory(tier: ExperienceTier, payload: object, **changes: object) -> ExperienceMemory:
    content = str(changes.pop("content", f"Reusable {tier.value} guidance"))
    values: dict[str, object] = {
        "id": changes.pop("id", f"exp-{tier.value}"),
        "content": content,
        "tier": tier,
        "payload": payload,
        "scope": MemoryScope.PROJECT,
        "project_key": "/repo",
        "created_at": NOW,
        "token_count": 8,
        "fingerprint": content_fingerprint(content),
        "source_task": "task-1",
        "run_id": "run-1",
        "stream_id": "stream-1",
        "created_by": ExperienceCreatedBy.WRITER,
    }
    values.update(changes)
    return ExperienceMemory(**values)  # type: ignore[arg-type]


class ExperiencePayloadTests(unittest.TestCase):
    def test_all_four_payloads_construct_valid_memories(self) -> None:
        for tier, payload in _payloads().items():
            with self.subTest(tier=tier.value):
                memory = _memory(tier, payload)
                self.assertEqual(memory.tier, tier)
                self.assertIs(memory.payload, payload)

    def test_payload_collections_are_normalized_deduplicated_and_sorted(self) -> None:
        first = ExperienceTrajectoryStep(step_num=2, action=" second ")
        second = ExperienceTrajectoryStep(step_num=1, action=" first ")
        payload = TrajectoryPayload(
            task_description=" task ",
            steps=[first, second],  # type: ignore[arg-type]
            outcome=" success ",
            key_learnings=[" one ", "", "one", "two"],  # type: ignore[arg-type]
            tags=["tag", " tag "],  # type: ignore[arg-type]
        )

        self.assertEqual(payload.task_description, "task")
        self.assertEqual(tuple(step.step_num for step in payload.steps), (1, 2))
        self.assertEqual(payload.key_learnings, ("one", "two"))
        self.assertEqual(payload.tags, ("tag",))

        skill = SkillPayload(
            category=" debugging ",
            technique=" isolate imports ",
            preconditions=[" package moved ", "package moved"],  # type: ignore[arg-type]
            steps=[" compile ", "", "compile", "test"],  # type: ignore[arg-type]
        )
        self.assertEqual(skill.preconditions, ("package moved",))
        self.assertEqual(skill.steps, ("compile", "test"))

    def test_trajectory_steps_require_unique_numbers_and_finite_rewards(self) -> None:
        step = ExperienceTrajectoryStep(step_num=1)
        with self.assertRaises(ValueError):
            TrajectoryPayload("task", (step, step), "success")
        with self.assertRaises(ValueError):
            ExperienceTrajectoryStep(step_num=1, reward=float("nan"))
        with self.assertRaises(ValueError):
            ExperienceTrajectoryStep(step_num=True)  # type: ignore[arg-type]

    def test_tip_skill_and_tool_required_fields_are_enforced(self) -> None:
        invalid_factories = (
            lambda: TipPayload("", "warning", "trigger"),
            lambda: TipPayload("testing", "notice", "trigger"),
            lambda: TipPayload("testing", "warning", ""),
            lambda: SkillPayload("testing", "", (), ("step",)),
            lambda: SkillPayload("testing", "technique", (), ("",)),
            lambda: ToolPayload("", "bash", "echo ok"),
            lambda: ToolPayload("tool", "bash", "", command=""),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(ValueError):
                    factory()

    def test_tool_schema_is_json_safe_and_detached_from_input(self) -> None:
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        payload = ToolPayload("tool", "python", "print('ok')", args_schema=schema)
        schema["properties"]["count"]["type"] = "string"  # type: ignore[index]

        self.assertEqual(payload.args_schema["properties"]["count"]["type"], "integer")
        with self.assertRaises(ValueError):
            ToolPayload("tool", "python", "print('ok')", args_schema={"value": float("inf")})
        with self.assertRaises(ValueError):
            ToolPayload("tool", "python", "print('ok')", args_schema={"value": object()})


class ExperienceMemoryTests(unittest.TestCase):
    def test_tier_and_payload_type_must_match(self) -> None:
        payloads = _payloads()
        tiers = tuple(ExperienceTier)
        for index, tier in enumerate(tiers):
            wrong_tier = tiers[(index + 1) % len(tiers)]
            with self.subTest(tier=tier.value, wrong_tier=wrong_tier.value):
                with self.assertRaises(ValueError):
                    _memory(wrong_tier, payloads[tier])

    def test_scope_project_and_datetime_contracts_fail_closed(self) -> None:
        payload = _payloads()[ExperienceTier.TIP]
        invalid_changes = (
            {"scope": MemoryScope.SESSION},
            {"scope": "project"},
            {"scope": MemoryScope.PROJECT, "project_key": ""},
            {"scope": MemoryScope.GLOBAL, "project_key": "/repo"},
            {"created_at": datetime(2026, 7, 15, 12, 0, 0)},
            {"last_used": datetime(2026, 7, 15, 12, 0, 0)},
            {"attribution_updated_at": datetime(2026, 7, 15, 12, 0, 0)},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    _memory(ExperienceTier.TIP, payload, **changes)

        global_memory = _memory(
            ExperienceTier.TIP,
            payload,
            scope=MemoryScope.GLOBAL,
            project_key="",
        )
        self.assertEqual(global_memory.scope, MemoryScope.GLOBAL)

    def test_common_numeric_fields_reject_invalid_values(self) -> None:
        payload = _payloads()[ExperienceTier.SKILL]
        invalid_changes = (
            {"token_count": -1},
            {"token_count": True},
            {"writer_confidence": float("nan")},
            {"writer_confidence": 1.1},
            {"attribution_value": float("inf")},
            {"attribution_value": -1.1},
            {"attribution_confidence": True},
            {"candidate_count": 1, "selected_count": 0, "not_selected_count": 0},
            {"candidate_count": 1, "selected_count": -1, "not_selected_count": 2},
            {"success_when_selected": 1.1},
            {"reward_when_selected": float("nan")},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    _memory(ExperienceTier.SKILL, payload, **changes)

    def test_content_and_fingerprint_are_normalized_and_checked(self) -> None:
        payload = _payloads()[ExperienceTier.TIP]
        memory = _memory(
            ExperienceTier.TIP,
            payload,
            content="  Run focused tests.  ",
        )
        self.assertEqual(memory.content, "Run focused tests.")

        with self.assertRaises(ValueError):
            _memory(ExperienceTier.TIP, payload, fingerprint="wrong")
        with self.assertRaises(ValueError):
            _memory(ExperienceTier.TIP, payload, content="   ")

    def test_record_is_flat_and_excludes_task_level_metadata_bags(self) -> None:
        field_names = {item.name for item in fields(ExperienceMemory)}
        self.assertFalse(
            field_names
            & {
                "metadata",
                "identity",
                "provenance",
                "quality",
                "maintenance",
                "candidate_memory_ids",
                "selected_memory_ids",
                "source_trace",
                "writer_reason",
                "writer_policy",
                "task_type",
                "outcome_source",
            }
        )
        self.assertTrue({"source_task", "run_id", "stream_id", "created_by"} <= field_names)

    def test_shared_retrieval_types_accept_experience_memory(self) -> None:
        memory = _memory(ExperienceTier.TIP, _payloads()[ExperienceTier.TIP])
        hit = RetrievalHit[ExperienceMemory](
            entry=memory,
            score=1.0,
            matched_terms=("focused",),
            source_weight=1.2,
            time_decay=1.0,
        )
        context = MemoryContext[ExperienceMemory](
            injected_text=memory.content,
            hits=[hit],
            estimated_tokens=memory.token_count,
        )

        self.assertIs(context.hits[0].entry, memory)
        self.assertEqual(replace(memory, protected=True).payload, memory.payload)


if __name__ == "__main__":
    unittest.main()
