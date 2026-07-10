from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    EVOLVER_SCHEMA_VERSION,
    ExperienceCreatedBy,
    ExperienceRecord,
    ExperienceTier,
    ExperienceTrajectoryStep,
    build_experience_entry,
    experience_metadata,
    experience_record_from_entry,
    experience_tier,
    is_experience_entry,
    normalize_experience_tier,
)
from my_agent.memory.long_term import LongTermMemoryStore
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType


NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


def _plain_entry() -> MemoryEntry:
    return MemoryEntry.build(
        id="plain",
        content="ordinary fact",
        type=MemoryType.FACT,
        scope=MemoryScope.PROJECT,
        source="manual",
        token_count=estimate_tokens("ordinary fact"),
        project_key="/repo",
        created_at=NOW,
    )


class EvolverMetadataTests(unittest.TestCase):
    def test_public_memory_package_exports_evolver_types(self) -> None:
        from my_agent.memory import (
            ExperienceCreatedBy as ExportedCreatedBy,
            ExperienceRecord as ExportedRecord,
            ExperienceTier as ExportedTier,
            ExperienceTrajectoryStep as ExportedStep,
            build_experience_entry as exported_build,
            experience_record_from_entry as exported_record_from_entry,
            experience_tier as exported_tier,
            is_experience_entry as exported_is_experience,
        )

        self.assertIs(ExportedCreatedBy, ExperienceCreatedBy)
        self.assertIs(ExportedRecord, ExperienceRecord)
        self.assertIs(ExportedTier, ExperienceTier)
        self.assertIs(ExportedStep, ExperienceTrajectoryStep)
        self.assertIs(exported_build, build_experience_entry)
        self.assertIs(exported_record_from_entry, experience_record_from_entry)
        self.assertIs(exported_tier, experience_tier)
        self.assertIs(exported_is_experience, is_experience_entry)

    def test_all_valid_tiers_build_metadata(self) -> None:
        for tier in ExperienceTier:
            metadata = experience_metadata(tier=tier, source_task="task-1")

            self.assertEqual(metadata["evolver_schema_version"], EVOLVER_SCHEMA_VERSION)
            self.assertEqual(metadata["evolver_tier"], tier.value)
            self.assertEqual(metadata["source_task"], "task-1")
            self.assertEqual(metadata["created_by"], ExperienceCreatedBy.MANUAL.value)

    def test_extra_metadata_cannot_override_core_fields(self) -> None:
        metadata = experience_metadata(
            tier=ExperienceTier.SKILL,
            source_task="real-task",
            created_by=ExperienceCreatedBy.WRITER,
            extra={
                "evolver_schema_version": 999,
                "evolver_tier": "tool",
                "source_task": "fake-task",
                "created_by": "maintenance",
                "category": "debugging",
            },
        )

        self.assertEqual(metadata["evolver_schema_version"], EVOLVER_SCHEMA_VERSION)
        self.assertEqual(metadata["evolver_tier"], "skill")
        self.assertEqual(metadata["source_task"], "real-task")
        self.assertEqual(metadata["created_by"], "writer")
        self.assertEqual(metadata["category"], "debugging")

    def test_plain_entry_is_not_experience(self) -> None:
        entry = _plain_entry()

        self.assertIsNone(experience_tier(entry))
        self.assertFalse(is_experience_entry(entry))
        self.assertIsNone(experience_record_from_entry(entry))

    def test_unknown_tier_is_tolerated_when_reading_but_rejected_when_building(self) -> None:
        entry = MemoryEntry.build(
            id="bad",
            content="unknown tier",
            type=MemoryType.FACT,
            scope=MemoryScope.PROJECT,
            source="evolver:future",
            token_count=estimate_tokens("unknown tier"),
            project_key="/repo",
            metadata={"evolver_tier": "future"},
        )

        self.assertIsNone(normalize_experience_tier("future"))
        self.assertIsNone(experience_tier(entry))
        self.assertIsNone(experience_record_from_entry(entry))
        with self.assertRaises(ValueError):
            experience_metadata(tier="future")
        with self.assertRaises(ValueError):
            build_experience_entry(id="bad", content="bad", tier="future", project_key="/repo")

    def test_invalid_created_by_is_rejected_when_building(self) -> None:
        with self.assertRaises(ValueError):
            experience_metadata(tier=ExperienceTier.TIP, created_by="robot")

    def test_build_experience_entry_uses_fact_type_and_evolver_source(self) -> None:
        entry = build_experience_entry(
            id="exp_1",
            content="Use pytest -q before editing parser modules.",
            tier=ExperienceTier.TIP,
            project_key="/repo",
            source_task="task-1",
            run_id="run-1",
        )

        self.assertEqual(entry.type, MemoryType.FACT)
        self.assertEqual(entry.source, "evolver:tip")
        self.assertEqual(entry.scope, MemoryScope.PROJECT)
        self.assertEqual(entry.project_key, "/repo")
        self.assertEqual(entry.run_id, "run-1")
        self.assertEqual(entry.metadata["evolver_schema_version"], EVOLVER_SCHEMA_VERSION)
        self.assertEqual(entry.metadata["evolver_tier"], "tip")
        self.assertEqual(entry.metadata["source_task"], "task-1")

    def test_build_experience_entry_rejects_empty_content(self) -> None:
        with self.assertRaises(ValueError):
            build_experience_entry(id="empty", content="  ", tier=ExperienceTier.SKILL, project_key="/repo")

    def test_build_experience_entry_preserves_explicit_created_at(self) -> None:
        entry = build_experience_entry(
            id="fixed-time",
            content="Deterministic promoted skill",
            tier=ExperienceTier.SKILL,
            project_key="/repo",
            created_at=NOW,
        )

        self.assertEqual(entry.created_at, NOW)

    def test_experience_trajectory_step_renders_json_safe_dict(self) -> None:
        step = ExperienceTrajectoryStep(
            step_num=1,
            observation="failing test",
            action="run_tests",
            action_params={"command": "pytest tests/test_parser.py -q", "flags": ("-q",)},
            result="1 failed",
            reward=0.5,
        )

        payload = step.to_dict()
        self.assertEqual(payload["step_num"], 1)
        self.assertEqual(payload["action_params"]["flags"], ["-q"])
        json.dumps(payload)

    def test_experience_record_from_entry_restores_known_fields_and_preserves_metadata(self) -> None:
        entry = build_experience_entry(
            id="exp_2",
            content="Always run compileall after package migration.",
            tier="skill",
            project_key="/repo",
            source_task="task-2",
            created_by=ExperienceCreatedBy.MAINTENANCE,
            extra_metadata={"technique": "compile import smoke"},
        )

        record = experience_record_from_entry(entry)

        self.assertIsInstance(record, ExperienceRecord)
        self.assertEqual(record.id, "exp_2")
        self.assertEqual(record.tier, ExperienceTier.SKILL)
        self.assertEqual(record.source_task, "task-2")
        self.assertEqual(record.created_by, ExperienceCreatedBy.MAINTENANCE)
        self.assertEqual(record.project_key, "/repo")
        self.assertEqual(record.metadata["technique"], "compile import smoke")


class EvolverLongTermStoreTests(unittest.TestCase):
    def _store(self, dir_path: Path) -> LongTermMemoryStore:
        return LongTermMemoryStore(dir_path / "long_term_memory.jsonl")

    def test_four_tier_metadata_round_trips_through_store(self) -> None:
        trajectory_step = ExperienceTrajectoryStep(
            step_num=1,
            observation="test failed",
            action="run_tests",
            action_params={"command": "pytest tests/test_parser.py -q"},
            result="fixed after parser change",
            reward=1.0,
        ).to_dict()
        cases = [
            (
                ExperienceTier.TRAJECTORY,
                "trajectory experience",
                {
                    "task_description": "Fix parser test",
                    "steps": [trajectory_step],
                    "outcome": "success",
                    "total_reward": 1.0,
                    "key_learnings": ["Parser strips comments before tokenization"],
                    "tags": ["parser", "pytest"],
                    "usage_count": 3,
                    "success_count": 2,
                    "last_used": "2026-06-18T12:00:00+00:00",
                },
            ),
            (
                ExperienceTier.TIP,
                "tip experience",
                {"category": "testing", "severity": "warning", "trigger": "pytest import errors"},
            ),
            (
                ExperienceTier.SKILL,
                "skill experience",
                {
                    "category": "debugging",
                    "technique": "Bisect failing import",
                    "preconditions": "module moved to package",
                    "steps": ["run compileall", "check __init__.py re-exports"],
                },
            ),
            (
                ExperienceTier.TOOL,
                "tool experience",
                {
                    "name": "pytest_single_file",
                    "language": "bash",
                    "code": "pytest {test_path} -q",
                    "input_description": "test_path: path to a pytest file",
                    "output_description": "pytest failure summary",
                    "tool_name": "run_tests",
                    "command": "pytest tests/test_parser.py -q",
                    "args_schema": {"test_path": "str"},
                    "repo_context": "run from repo root",
                    "template": "pytest {test_path} -q",
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            for index, (tier, content, metadata) in enumerate(cases):
                entry = build_experience_entry(
                    id=f"exp_{index}",
                    content=content,
                    tier=tier,
                    project_key="/repo",
                    source_task=f"task-{index}",
                    extra_metadata=metadata,
                )
                stored, created = store.add(entry)
                self.assertTrue(created)
                self.assertIs(stored, entry)

            reloaded = self._store(Path(tmp))
            reloaded.load()

        restored = {entry.metadata["evolver_tier"]: entry for entry in reloaded.all(project_key="/repo")}
        self.assertEqual(set(restored), {"trajectory", "tip", "skill", "tool"})
        for tier, _, metadata in cases:
            entry = restored[tier.value]
            record = experience_record_from_entry(entry)
            self.assertIsNotNone(record)
            self.assertEqual(record.tier, tier)
            for key, value in metadata.items():
                self.assertEqual(entry.metadata[key], value)

    def test_legacy_payload_without_metadata_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            payload = {
                "id": "legacy",
                "content": "legacy fact",
                "type": "fact",
                "scope": "project",
                "source": "manual",
                "created_at": NOW.isoformat(),
                "token_count": 4,
                "project_key": "/repo",
            }
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

            store = LongTermMemoryStore(path)
            store.load()

        entries = store.all(project_key="/repo")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].metadata, {})
        self.assertFalse(is_experience_entry(entries[0]))

    def test_dedup_and_visibility_semantics_are_unchanged_for_experience_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            first = build_experience_entry(
                id="exp_a",
                content="same content",
                tier=ExperienceTier.TIP,
                project_key="/repo-a",
            )
            duplicate = build_experience_entry(
                id="exp_dup",
                content="same content",
                tier=ExperienceTier.TIP,
                project_key="/repo-a",
            )
            other_project = build_experience_entry(
                id="exp_b",
                content="same content",
                tier=ExperienceTier.TIP,
                project_key="/repo-b",
            )
            global_entry = build_experience_entry(
                id="exp_global",
                content="global experience",
                tier=ExperienceTier.SKILL,
                project_key="",
                scope=MemoryScope.GLOBAL,
            )

            stored_first, created_first = store.add(first)
            stored_duplicate, created_duplicate = store.add(duplicate)
            _, created_other_project = store.add(other_project)
            _, created_global = store.add(global_entry)

            self.assertTrue(created_first)
            self.assertFalse(created_duplicate)
            self.assertEqual(stored_duplicate.id, stored_first.id)
            self.assertTrue(created_other_project)
            self.assertTrue(created_global)
            visible_a = {entry.id for entry in store.all(project_key="/repo-a")}
            visible_b = {entry.id for entry in store.all(project_key="/repo-b")}
            self.assertEqual(visible_a, {"exp_a", "exp_global"})
            self.assertEqual(visible_b, {"exp_b", "exp_global"})

    def test_same_content_can_coexist_across_experience_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            tip = build_experience_entry(
                id="tip",
                content="Use the project parser before editing manifests.",
                tier=ExperienceTier.TIP,
                project_key="/repo",
            )
            skill = build_experience_entry(
                id="skill",
                content=tip.content,
                tier=ExperienceTier.SKILL,
                project_key="/repo",
            )
            duplicate_skill = build_experience_entry(
                id="skill-duplicate",
                content=tip.content,
                tier=ExperienceTier.SKILL,
                project_key="/repo",
            )

            _, tip_created = store.add(tip)
            _, skill_created = store.add(skill)
            stored_duplicate, duplicate_created = store.add(duplicate_skill)

            self.assertTrue(tip_created)
            self.assertTrue(skill_created)
            self.assertFalse(duplicate_created)
            self.assertEqual(stored_duplicate.id, "skill")
            self.assertEqual({entry.id for entry in store.all()}, {"tip", "skill"})

    def test_global_experiences_dedupe_by_tier_but_not_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            first = build_experience_entry(
                id="global-tip-a",
                content="Global guidance",
                tier=ExperienceTier.TIP,
                project_key="/repo-a",
                scope=MemoryScope.GLOBAL,
            )
            duplicate = build_experience_entry(
                id="global-tip-b",
                content=first.content,
                tier=ExperienceTier.TIP,
                project_key="/repo-b",
                scope=MemoryScope.GLOBAL,
            )
            skill = build_experience_entry(
                id="global-skill",
                content=first.content,
                tier=ExperienceTier.SKILL,
                project_key="/repo-b",
                scope=MemoryScope.GLOBAL,
            )

            store.add(first)
            duplicate_entry, duplicate_created = store.add(duplicate)
            _, skill_created = store.add(skill)

            self.assertFalse(duplicate_created)
            self.assertEqual(duplicate_entry.id, "global-tip-a")
            self.assertTrue(skill_created)


if __name__ == "__main__":
    unittest.main()
