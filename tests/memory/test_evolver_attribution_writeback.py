from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    ExperienceTier,
    MemoryAttributionRecord,
    selection_score,
)
from my_agent.memory.evolver.attribution import write_back_attribution
from my_agent.memory.experience_store import ExperienceStore
from my_agent.memory.types import RetrievalHit
from tests.memory.experience_fixtures import typed_experience


UPDATED_AT = "2026-01-01T00:00:00+00:00"
LAST_USED = "2025-12-31T23:59:00+00:00"


def _selection_score(entry, *, retrieval_score: float = 1.0) -> float:
    return selection_score(
        RetrievalHit(
            entry=entry,
            score=retrieval_score,
            matched_terms=("useful",),
            source_weight=1.2,
            time_decay=1.0,
        ),
        tier_weights={entry.tier.value: 1.0},
    )


def _entry(
    memory_id: str,
    tier: str = "skill",
    *,
    project_key: str = "proj-A",
):
    return replace(
        typed_experience(
            memory_id,
            f"useful experience {memory_id}",
            ExperienceTier(tier),
            project_key=project_key,
            source_task="task-1",
            writer_confidence=0.73,
        ),
        run_id="run-1",
        stream_id="stream-1",
        protected=True,
        maintenance_operation_id="maintenance-op-1",
        parent_id="parent-1",
        parent_tier=ExperienceTier.TIP,
    )


def _record(memory_id: str, *, value: float = 0.25, tier: str = "skill", project_key: str = "proj-A"):
    return MemoryAttributionRecord(
        memory_id=memory_id,
        tier=tier,
        memory_project_key=project_key,
        candidate_count=2,
        selected_count=1,
        not_selected_count=1,
        success_when_selected=1.0,
        success_when_candidate_not_selected=0.0,
        reward_when_selected=0.8,
        reward_when_candidate_not_selected=0.2,
        value=value,
        confidence=1.0,
        last_used=LAST_USED,
    )


class AttributionWriteBackTests(unittest.TestCase):
    def test_write_back_only_changes_flat_attribution_fields_and_affects_selection_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            stored, _ = store.add(_entry("mem-1"))
            before = stored
            before_score = _selection_score(before)

            summary = write_back_attribution(
                store=store,
                records=[_record("mem-1", value=0.25)],
                updated_at=UPDATED_AT,
            )

            self.assertEqual(summary.updated, 1)
            after = store.get("mem-1")
            self.assertIsNotNone(after)
            assert after is not None
            for field_name in (
                "id",
                "content",
                "tier",
                "payload",
                "scope",
                "project_key",
                "created_at",
                "fingerprint",
                "source_task",
                "run_id",
                "stream_id",
                "created_by",
                "writer_confidence",
                "protected",
                "invalidated",
                "promoted_to",
                "maintenance_operation_id",
                "parent_id",
                "parent_tier",
            ):
                self.assertEqual(getattr(after, field_name), getattr(before, field_name), field_name)
            self.assertEqual(after.attribution_value, 0.25)
            self.assertEqual(after.attribution_confidence, 1.0)
            self.assertEqual(after.candidate_count, 2)
            self.assertEqual(after.selected_count, 1)
            self.assertEqual(after.not_selected_count, 1)
            self.assertEqual(after.success_when_selected, 1.0)
            self.assertEqual(after.success_when_candidate_not_selected, 0.0)
            self.assertEqual(after.reward_when_selected, 0.8)
            self.assertEqual(after.reward_when_candidate_not_selected, 0.2)
            self.assertEqual(after.last_used, datetime.fromisoformat(LAST_USED))
            self.assertEqual(after.attribution_updated_at, datetime.fromisoformat(UPDATED_AT))
            self.assertGreater(_selection_score(after), before_score)

    def test_low_evidence_write_back_clears_old_value_but_updates_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(replace(
                _entry("mem-1"),
                attribution_value=0.45,
                attribution_confidence=1.0,
                candidate_count=9,
                selected_count=9,
            ))

            summary = write_back_attribution(
                store=store,
                records=[_record("mem-1", value=0.001)],
                min_abs_value_to_write=0.01,
                updated_at=UPDATED_AT,
            )

            self.assertEqual(summary.updated, 1)
            self.assertEqual(summary.skipped_low_evidence, 1)
            after = store.get("mem-1")
            self.assertIsNotNone(after)
            assert after is not None
            self.assertEqual(after.attribution_value, 0.0)
            self.assertEqual(after.attribution_confidence, 1.0)
            self.assertEqual(after.candidate_count, 2)
            self.assertEqual(after.selected_count, 1)

    def test_project_tier_guards_and_all_projects_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(_entry("mem-A", project_key="proj-A"))
            store.add(_entry("mem-B", project_key="proj-B"))
            store.add(_entry("mem-tool", tier="tool", project_key="proj-A"))

            summary = write_back_attribution(
                store=store,
                records=[
                    _record("mem-B", project_key="proj-B"),
                    _record("mem-tool", tier="skill", project_key="proj-A"),
                ],
                project_key="proj-A",
                updated_at=UPDATED_AT,
            )

            self.assertEqual(summary.updated, 0)
            self.assertEqual(summary.skipped_by_project_key, 1)
            self.assertEqual(summary.skipped_tier_mismatch, 1)
            self.assertEqual(store.get("mem-B").candidate_count, 0)  # type: ignore[union-attr]
            self.assertEqual(store.get("mem-tool").candidate_count, 0)  # type: ignore[union-attr]

            all_projects = write_back_attribution(
                store=store,
                records=[_record("mem-B", project_key="proj-B")],
                project_key="proj-A",
                all_projects=True,
                updated_at=UPDATED_AT,
            )

            self.assertEqual(all_projects.updated, 1)
            self.assertEqual(store.get("mem-B").attribution_value, 0.25)  # type: ignore[union-attr]

    def test_persist_failure_restores_memory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(_entry("mem-1"))

            def fail_persist(_memories) -> None:
                raise OSError("disk full")

            store._persist_memories = fail_persist  # type: ignore[method-assign]
            with self.assertRaises(OSError):
                write_back_attribution(
                    store=store,
                    records=[_record("mem-1", value=0.25)],
                    updated_at=UPDATED_AT,
                )

            restored = store.get("mem-1")
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.attribution_value, 0.0)
            self.assertEqual(restored.candidate_count, 0)


if __name__ == "__main__":
    unittest.main()
