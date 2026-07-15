from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    MemoryAttributionRecord,
    build_experience_entry,
)
from my_agent.memory.evolver.attribution import write_back_attribution
from my_agent.memory.evolver.types import ExperienceTier
from my_agent.memory.long_term import LongTermMemoryStore
from my_agent.memory.types import MemoryScope


def _legacy_selection_score(entry, *, retrieval_score: float = 1.0) -> float:
    value = float(entry.metadata.get("evolver_value", 0.0) or 0.0)
    confidence = entry.metadata.get("evolver_confidence")
    if confidence is None:
        confidence = entry.metadata.get("confidence", 1.0)
    value_weight = max(0.5, min(1.5, 1.0 + value))
    confidence_weight = max(0.5, min(1.2, float(confidence)))
    return retrieval_score * value_weight * confidence_weight


def _entry(
    memory_id: str,
    tier: str = "skill",
    *,
    project_key: str = "proj-A",
    extra_metadata: dict | None = None,
):
    return build_experience_entry(
        id=memory_id,
        content=f"useful experience {memory_id}",
        tier=tier,
        project_key=project_key,
        scope=MemoryScope.PROJECT,
        source_task="task-1",
        extra_metadata=extra_metadata,
    )


def _record(memory_id: str, *, value: float = 0.25, tier: str = "skill", project_key: str = "proj-A"):
    return MemoryAttributionRecord(
        memory_id=memory_id,
        tier=tier,
        memory_project_key=project_key,
        candidate_count=2,
        selected_count=1,
        not_selected_count=1,
        value=value,
        confidence=1.0,
    )


class AttributionWriteBackTests(unittest.TestCase):
    def test_write_back_only_changes_metadata_and_affects_selection_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore(Path(tmp) / "long_term_memory.jsonl")
            store.load()
            stored, _ = store.add(_entry("mem-1", extra_metadata={"confidence": 0.73}))
            before = stored
            before_score = _legacy_selection_score(before)

            summary = write_back_attribution(
                store=store,
                records=[_record("mem-1", value=0.25)],
                updated_at="2026-01-01T00:00:00+00:00",
            )

            self.assertEqual(summary.updated, 1)
            after = store.all(project_key="proj-A")[0]
            self.assertEqual(after.id, before.id)
            self.assertEqual(after.content, before.content)
            self.assertEqual(after.fingerprint, before.fingerprint)
            self.assertEqual(after.created_at, before.created_at)
            self.assertEqual(after.project_key, before.project_key)
            self.assertEqual(after.run_id, before.run_id)
            self.assertEqual(after.metadata["evolver_value"], 0.25)
            self.assertEqual(after.metadata["evolver_confidence"], 1.0)
            self.assertEqual(after.metadata["confidence"], 0.73)
            self.assertEqual(after.metadata["evolver_candidate_count"], 2)
            self.assertEqual(after.metadata["evolver_selected_count"], 1)
            self.assertEqual(after.metadata["evolver_not_selected_count"], 1)
            self.assertGreater(_legacy_selection_score(after), before_score)

    def test_low_evidence_write_back_clears_old_value_but_updates_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore(Path(tmp) / "long_term_memory.jsonl")
            store.load()
            store.add(_entry(
                "mem-1",
                extra_metadata={
                    "evolver_value": 0.45,
                    "evolver_confidence": 1.0,
                    "evolver_candidate_count": 9,
                },
            ))

            summary = write_back_attribution(
                store=store,
                records=[_record("mem-1", value=0.001)],
                min_abs_value_to_write=0.01,
                updated_at="2026-01-01T00:00:00+00:00",
            )

            self.assertEqual(summary.updated, 1)
            self.assertEqual(summary.skipped_low_evidence, 1)
            after = store.all(project_key="proj-A")[0]
            self.assertEqual(after.metadata["evolver_value"], 0.0)
            self.assertEqual(after.metadata["evolver_candidate_count"], 2)
            self.assertEqual(after.metadata["evolver_selected_count"], 1)

    def test_project_filter_and_tier_mismatch_skip_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore(Path(tmp) / "long_term_memory.jsonl")
            store.load()
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
                updated_at="2026-01-01T00:00:00+00:00",
            )

            self.assertEqual(summary.updated, 0)
            self.assertEqual(summary.skipped_by_project_key, 1)
            self.assertEqual(summary.skipped_tier_mismatch, 1)
            by_id = {entry.id: entry for entry in store.all(project_key=None)}
            self.assertNotIn("evolver_value", by_id["mem-B"].metadata)
            self.assertNotIn("evolver_value", by_id["mem-tool"].metadata)

    def test_persist_failure_restores_memory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore(Path(tmp) / "long_term_memory.jsonl")
            store.load()
            store.add(_entry("mem-1"))

            def fail_persist() -> None:
                raise OSError("disk full")

            store._persist = fail_persist  # type: ignore[method-assign]
            with self.assertRaises(OSError):
                write_back_attribution(
                    store=store,
                    records=[_record("mem-1", value=0.25)],
                    updated_at="2026-01-01T00:00:00+00:00",
                )

            restored = store.all(project_key="proj-A")[0]
            self.assertNotIn("evolver_value", restored.metadata)


if __name__ == "__main__":
    unittest.main()
