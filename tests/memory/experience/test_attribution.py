from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver.selection.legacy import selection_score
from my_agent.memory.experience.models import ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from my_agent.memory.types import RetrievalHit
from my_agent.opd_data.legacy.attribution import (
    MemoryAttributionRecord,
    write_attribution_jsonl,
    write_back_attribution,
)
from tests.memory.experience.fixtures import typed_experience


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
    def test_artifact_and_store_share_canonical_attribution_precision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(_entry("mem-precision"))
            record = MemoryAttributionRecord(
                memory_id="mem-precision",
                tier="skill",
                memory_project_key="proj-A",
                candidate_count=3,
                selected_count=1,
                not_selected_count=2,
                success_when_selected=0.666666666,
                success_when_candidate_not_selected=0.333333333,
                reward_when_selected=0.111111111,
                reward_when_candidate_not_selected=0.222222222,
                value=0.123456789,
                confidence=0.876543219,
                last_used=LAST_USED,
            )
            artifact = Path(tmp) / "memory_attribution.jsonl"

            self.assertEqual(record.value, 0.123456789)
            self.assertEqual(record.confidence, 0.876543219)
            write_attribution_jsonl([record], artifact)
            write_back_attribution(
                store=store,
                records=[record],
                min_abs_value_to_write=0.0,
                updated_at=UPDATED_AT,
            )

            payload = json.loads(artifact.read_text(encoding="utf-8"))
            stored = store.get("mem-precision")
            self.assertIsNotNone(stored)
            assert stored is not None
            expected = {
                "value": 0.123457,
                "confidence": 0.876543,
                "success_when_selected": 0.666667,
                "success_when_candidate_not_selected": 0.333333,
                "reward_when_selected": 0.111111,
                "reward_when_candidate_not_selected": 0.222222,
            }
            for record_field, store_field in (
                ("value", "attribution_value"),
                ("confidence", "attribution_confidence"),
                ("success_when_selected", "success_when_selected"),
                ("success_when_candidate_not_selected", "success_when_candidate_not_selected"),
                ("reward_when_selected", "reward_when_selected"),
                ("reward_when_candidate_not_selected", "reward_when_candidate_not_selected"),
            ):
                self.assertEqual(payload[record_field], expected[record_field])
                self.assertEqual(getattr(stored, store_field), expected[record_field])

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

            record = _record("mem-1", value=0.0099996)
            self.assertEqual(record.value, 0.0099996)
            summary = write_back_attribution(
                store=store,
                records=[record],
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

    def test_write_back_prevalidates_entire_batch_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(_entry("mem-valid"))
            store.add(_entry("mem-invalid"))

            with self.assertRaisesRegex(ValueError, "attribution_value"):
                write_back_attribution(
                    store=store,
                    records=[
                        _record("mem-valid", value=0.25),
                        _record("mem-invalid", value=1.5),
                    ],
                    min_abs_value_to_write=0.0,
                    updated_at=UPDATED_AT,
                )

            for memory_id in ("mem-valid", "mem-invalid"):
                stored = store.get(memory_id)
                self.assertIsNotNone(stored)
                assert stored is not None
                self.assertEqual(stored.attribution_value, 0.0)
                self.assertEqual(stored.candidate_count, 0)


if __name__ == "__main__":
    unittest.main()
