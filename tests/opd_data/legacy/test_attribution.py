from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.experience.models import ExperienceTier
from my_agent.opd_data.legacy.attribution import (
    AttributionConfig,
    MemoryAttributionRecord,
    attribution_summary,
    load_attribution_jsonl,
    render_attribution_summary,
    score_all_memories,
    score_memory,
    write_attribution_jsonl,
)
from my_agent.opd_data.legacy.usage_log import UsageLogEntry
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType
from tests.memory.experience.fixtures import typed_experience


def _entry(memory_id: str, tier: str, *, project_key: str = "proj-A"):
    return typed_experience(
        memory_id,
        f"experience {memory_id}",
        ExperienceTier(tier),
        project_key=project_key,
        scope=MemoryScope.PROJECT if project_key else MemoryScope.GLOBAL,
    )


def _plain_entry(memory_id: str, *, project_key: str = "proj-A") -> MemoryEntry:
    return MemoryEntry.build(
        id=memory_id,
        content=f"fact {memory_id}",
        type=MemoryType.FACT,
        scope=MemoryScope.PROJECT,
        source="manual",
        token_count=estimate_tokens(f"fact {memory_id}"),
        project_key=project_key,
        metadata={},  # no evolver_tier -> not an experience
    )


def _log(
    *,
    task_id: str,
    task_type: str = "humaneval",
    selected: list[str] | None = None,
    candidates: list[str],
    reward: float,
    project_key: str = "proj-A",
    stream_id: str = "python",
    success: bool | None = None,
    timestamp: str = "",
) -> UsageLogEntry:
    return UsageLogEntry(
        task_id=task_id,
        task_type=task_type,
        timestamp=timestamp,
        stream_id=stream_id,
        memory_project_key=project_key,
        retrieved_candidates={"skill": candidates},
        selected_memory_ids={"skill": selected} if selected is not None else {},
        env_reward=float(reward),
        success=success if success is not None else bool(reward >= 1.0),
        status="complete",
    )


class ScoreMemoryTests(unittest.TestCase):
    def test_value_clip_requires_finite_typed_schema_range(self) -> None:
        for value in (-0.1, 1.01, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "value_clip"):
                AttributionConfig(value_clip=value)

    def test_positive_value_when_selected_reward_higher_than_pool(self) -> None:
        logs = [
            _log(task_id="A", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0),
            _log(task_id="B", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0),
            _log(task_id="C", candidates=["mem-1", "mem-2"], selected=[], reward=0.0),
            _log(task_id="D", candidates=["mem-1", "mem-2"], selected=[], reward=0.0),
        ]
        record = score_memory(memory_id="mem-1", tier="skill", usage_logs=logs)
        self.assertGreater(record.value, 0.0)
        self.assertEqual(record.candidate_count, 4)
        self.assertEqual(record.selected_count, 2)
        self.assertEqual(record.not_selected_count, 2)
        self.assertEqual(record.reward_when_selected, 1.0)
        self.assertEqual(record.reward_when_candidate_not_selected, 0.0)

    def test_negative_value_when_selected_reward_lower_than_pool(self) -> None:
        logs = [
            _log(task_id="A", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=0.0),
            _log(task_id="B", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=0.0),
            _log(task_id="C", candidates=["mem-1", "mem-2"], selected=[], reward=1.0),
            _log(task_id="D", candidates=["mem-1", "mem-2"], selected=[], reward=1.0),
        ]
        record = score_memory(memory_id="mem-1", tier="skill", usage_logs=logs)
        self.assertLess(record.value, 0.0)

    def test_zero_value_with_insufficient_evidence(self) -> None:
        # Only selected, no not-selected control -> value=0 but record emitted.
        logs = [
            _log(task_id="A", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0),
            _log(task_id="B", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0),
        ]
        record = score_memory(memory_id="mem-1", tier="skill", usage_logs=logs)
        self.assertEqual(record.value, 0.0)
        self.assertEqual(record.not_selected_count, 0)
        self.assertIsNone(record.reward_when_candidate_not_selected)

    def test_confidence_grows_with_selected_count_and_caps_at_one(self) -> None:
        base = [
            _log(task_id="C", candidates=["mem-1", "mem-2"], selected=[], reward=0.0),
            _log(task_id="D", candidates=["mem-1", "mem-2"], selected=[], reward=0.0),
        ]
        low = score_memory(memory_id="mem-1", tier="skill", usage_logs=[
            _log(task_id="A", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0),
            *base,
        ])
        high = score_memory(memory_id="mem-1", tier="skill", usage_logs=[
            *[_log(task_id=f"A{i}", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0) for i in range(10)],
            *base,
        ])
        self.assertGreater(high.confidence, low.confidence)
        self.assertLessEqual(high.confidence, 1.0)

    def test_tier_weight_scales_value(self) -> None:
        logs = [
            _log(task_id="A", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0),
            _log(task_id="B", candidates=["mem-1", "mem-2"], selected=[], reward=0.0),
        ]
        skill = score_memory(memory_id="mem-1", tier="skill", usage_logs=logs)
        tool = score_memory(memory_id="mem-1", tier="tool", usage_logs=logs)
        # tool weight (1.2) > skill weight (1.0)
        self.assertGreater(tool.value, skill.value)

    def test_value_clipped_to_configured_clip(self) -> None:
        logs = [
            _log(task_id="A", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0),
            _log(task_id="B", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0),
            _log(task_id="C", candidates=["mem-1", "mem-2"], selected=[], reward=0.0),
            _log(task_id="D", candidates=["mem-1", "mem-2"], selected=[], reward=0.0),
        ]
        record = score_memory(
            memory_id="mem-1", tier="tool", usage_logs=logs,
            config=AttributionConfig(value_clip=0.01),
        )
        self.assertLessEqual(record.value, 0.01)
        self.assertGreaterEqual(record.value, -0.01)

    def test_task_type_groups_are_independent(self) -> None:
        # mem-1 selected-and-successful in humaneval; selected-and-failing in sql.
        # The two task_type contributions must be averaged, not maxed.
        logs = [
            _log(task_id="h1", task_type="humaneval", candidates=["mem-1", "x"], selected=["mem-1"], reward=1.0),
            _log(task_id="h2", task_type="humaneval", candidates=["mem-1", "x"], selected=[], reward=0.0),
            _log(task_id="s1", task_type="sql", candidates=["mem-1", "x"], selected=["mem-1"], reward=0.0),
            _log(task_id="s2", task_type="sql", candidates=["mem-1", "x"], selected=[], reward=1.0),
        ]
        record = score_memory(memory_id="mem-1", tier="skill", usage_logs=logs)
        # humaneval contribution positive, sql contribution negative -> they cancel.
        self.assertEqual(record.value, 0.0)
        self.assertEqual(record.task_types, ("humaneval", "sql"))

    def test_incomplete_logs_excluded(self) -> None:
        started = _log(task_id="A", candidates=["mem-1", "mem-2"], selected=["mem-1"], reward=1.0)
        started = UsageLogEntry(
            task_id=started.task_id, task_type=started.task_type, status="started",
            stream_id=started.stream_id, memory_project_key=started.memory_project_key,
            retrieved_candidates=started.retrieved_candidates,
            selected_memory_ids=started.selected_memory_ids,
        )
        complete = _log(task_id="B", candidates=["mem-1", "mem-2"], selected=[], reward=0.0)
        record = score_memory(memory_id="mem-1", tier="skill", usage_logs=[started, complete])
        self.assertEqual(record.candidate_count, 1)

    def test_last_used_is_latest_stable_selected_timestamp(self) -> None:
        logs = [
            _log(
                task_id="A",
                candidates=["mem-1"],
                selected=["mem-1"],
                reward=1.0,
                timestamp="2026-07-01T00:00:00+00:00",
            ),
            _log(
                task_id="B",
                candidates=["mem-1"],
                selected=["mem-1"],
                reward=1.0,
                timestamp="2026-07-03T00:00:00+00:00",
            ),
            _log(
                task_id="C",
                candidates=["mem-1"],
                selected=[],
                reward=0.0,
                timestamp="2026-07-05T00:00:00+00:00",
            ),
        ]

        record = score_memory(memory_id="mem-1", tier="skill", usage_logs=logs)

        self.assertEqual(record.last_used, "2026-07-03T00:00:00+00:00")

    def test_last_used_remains_empty_without_selected_timestamp(self) -> None:
        record = score_memory(
            memory_id="mem-1",
            tier="skill",
            usage_logs=[
                _log(task_id="A", candidates=["mem-1"], selected=["mem-1"], reward=1.0),
                _log(task_id="B", candidates=["mem-1"], selected=[], reward=0.0),
            ],
        )

        self.assertEqual(record.last_used, "")


class ScoreAllMemoriesTests(unittest.TestCase):
    def test_only_memories_in_pool_are_scored(self) -> None:
        entries = [_entry("mem-1", "skill"), _entry("mem-other", "skill")]
        logs = [
            _log(task_id="A", candidates=["mem-1", "mem-x"], selected=["mem-1"], reward=1.0),
            _log(task_id="B", candidates=["mem-1", "mem-x"], selected=[], reward=0.0),
        ]
        # mem-other is not in candidate pool and is not an entry here; mem-x not an entry.
        records = score_all_memories(entries=entries, usage_logs=logs, project_key="proj-A")
        self.assertEqual([r.memory_id for r in records], ["mem-1"])
        self.assertNotIn("mem-other", [r.memory_id for r in records])

    def test_selected_only_memory_is_not_scored_without_candidate_evidence(self) -> None:
        entries = [_entry("mem-1", "skill")]
        logs = [
            _log(task_id="A", candidates=[], selected=["mem-1"], reward=1.0),
        ]

        records = score_all_memories(entries=entries, usage_logs=logs, project_key="proj-A")

        self.assertEqual(records, [])

    def test_non_experience_entries_are_rejected_at_typed_boundary(self) -> None:
        plain = _plain_entry("plain-1")
        logs = [_log(task_id="A", candidates=["plain-1", "mem-2"], selected=["plain-1"], reward=1.0)]
        with self.assertRaisesRegex(TypeError, "ExperienceMemory"):
            score_all_memories(
                entries=[plain],  # type: ignore[list-item]
                usage_logs=logs,
                project_key="proj-A",
            )

    def test_typed_cutover_matches_pre_cutover_golden_scores(self) -> None:
        entries = [
            _entry("mem-golden", "tool"),
            _entry("mem-low", "skill"),
        ]
        logs = [
            _log(
                task_id="g1", candidates=["mem-golden"], selected=["mem-golden"],
                reward=1.0, success=True, timestamp="2026-01-01T00:00:00+00:00",
            ),
            _log(
                task_id="g2", candidates=["mem-golden"], selected=["mem-golden"],
                reward=0.5, success=True, timestamp="2026-01-02T00:00:00+00:00",
            ),
            _log(
                task_id="g3", candidates=["mem-golden"], selected=["mem-golden"],
                reward=0.0, success=False, timestamp="2026-01-03T00:00:00+00:00",
            ),
            _log(
                task_id="g4", candidates=["mem-golden"], selected=[],
                reward=0.25, success=True, timestamp="2026-01-04T00:00:00+00:00",
            ),
            _log(
                task_id="g5", candidates=["mem-golden"], selected=[],
                reward=-0.25, success=False, timestamp="2026-01-05T00:00:00+00:00",
            ),
            _log(
                task_id="l1", task_type="sql", candidates=["mem-low"],
                selected=["mem-low"], reward=1.0, success=True,
            ),
        ]

        records = score_all_memories(entries=entries, usage_logs=logs, project_key="proj-A")

        self.assertEqual(records[0].value, 0.0881816307401944)
        self.assertEqual(records[0].confidence, 0.6123724356957945)
        self.assertEqual(
            [record.to_dict() for record in records],
            [
                {
                    "memory_id": "mem-golden",
                    "tier": "tool",
                    "memory_project_key": "proj-A",
                    "candidate_count": 5,
                    "selected_count": 3,
                    "not_selected_count": 2,
                    "success_when_selected": 0.666667,
                    "success_when_candidate_not_selected": 0.5,
                    "reward_when_selected": 0.5,
                    "reward_when_candidate_not_selected": 0.0,
                    "value": 0.088182,
                    "confidence": 0.612372,
                    "groups": ["python"],
                    "task_types": ["humaneval"],
                    "stream_ids": ["python"],
                    "selected_task_ids": ["g1", "g2", "g3"],
                    "not_selected_task_ids": ["g4", "g5"],
                    "last_used": "2026-01-03T00:00:00+00:00",
                },
                {
                    "memory_id": "mem-low",
                    "tier": "skill",
                    "memory_project_key": "proj-A",
                    "candidate_count": 1,
                    "selected_count": 1,
                    "not_selected_count": 0,
                    "success_when_selected": 1.0,
                    "success_when_candidate_not_selected": None,
                    "reward_when_selected": 1.0,
                    "reward_when_candidate_not_selected": None,
                    "value": 0.0,
                    "confidence": 0.353553,
                    "groups": ["python"],
                    "task_types": ["sql"],
                    "stream_ids": ["python"],
                    "selected_task_ids": ["l1"],
                    "not_selected_task_ids": [],
                    "last_used": "",
                },
            ],
        )

    def test_project_key_filters_usage_logs_and_entries(self) -> None:
        # Same memory_id appears in two different streams with different outcomes.
        entries_a = [_entry("mem-1", "skill", project_key="proj-A")]
        entries_b = [_entry("mem-1", "skill", project_key="proj-B")]
        logs = [
            # stream A: mem-1 selected + success
            _log(task_id="a1", candidates=["mem-1", "x"], selected=["mem-1"], reward=1.0, project_key="proj-A"),
            _log(task_id="a2", candidates=["mem-1", "x"], selected=[], reward=0.0, project_key="proj-A"),
            # stream B: mem-1 selected + failure (must NOT pollute A's score)
            _log(task_id="b1", candidates=["mem-1", "x"], selected=["mem-1"], reward=0.0, project_key="proj-B"),
            _log(task_id="b2", candidates=["mem-1", "x"], selected=[], reward=1.0, project_key="proj-B"),
        ]
        records_a = score_all_memories(entries=entries_a, usage_logs=logs, project_key="proj-A")
        records_b = score_all_memories(entries=entries_b, usage_logs=logs, project_key="proj-B")
        self.assertGreater(records_a[0].value, 0.0)
        self.assertLess(records_b[0].value, 0.0)
        self.assertEqual(records_a[0].memory_project_key, "proj-A")
        self.assertEqual(records_b[0].memory_project_key, "proj-B")
        # record A reward_when_selected reflects only stream-A logs.
        self.assertEqual(records_a[0].reward_when_selected, 1.0)
        self.assertEqual(records_a[0].reward_when_candidate_not_selected, 0.0)

    def test_entries_from_other_project_excluded(self) -> None:
        entries = [
            _entry("mem-1", "skill", project_key="proj-A"),
            _entry("mem-2", "skill", project_key="proj-B"),
        ]
        logs = [
            _log(task_id="a1", candidates=["mem-1"], selected=["mem-1"], reward=1.0, project_key="proj-A"),
            _log(task_id="a2", candidates=["mem-1"], selected=[], reward=0.0, project_key="proj-A"),
            _log(task_id="b1", candidates=["mem-2"], selected=["mem-2"], reward=1.0, project_key="proj-B"),
        ]
        records = score_all_memories(entries=entries, usage_logs=logs, project_key="proj-A")
        self.assertEqual([r.memory_id for r in records], ["mem-1"])

    def test_global_project_key_consumes_all_logs(self) -> None:
        entries = [_entry("mem-1", "skill", project_key="")]
        logs = [
            _log(task_id="a1", candidates=["mem-1", "x"], selected=["mem-1"], reward=1.0, project_key="proj-X"),
            _log(task_id="a2", candidates=["mem-1", "x"], selected=[], reward=0.0, project_key="proj-Y"),
        ]
        records = score_all_memories(entries=entries, usage_logs=logs, project_key="")
        self.assertEqual([r.memory_id for r in records], ["mem-1"])


class JsonlIoTests(unittest.TestCase):
    def test_write_and_load_roundtrip_byte_stable(self) -> None:
        records = [
            score_memory(memory_id="mem-2", tier="skill", usage_logs=[
                _log(task_id="A", candidates=["mem-2", "x"], selected=["mem-2"], reward=1.0),
                _log(task_id="B", candidates=["mem-2", "x"], selected=[], reward=0.0),
            ]),
            score_memory(memory_id="mem-1", tier="tip", usage_logs=[
                _log(task_id="C", candidates=["mem-1", "x"], selected=["mem-1"], reward=1.0),
                _log(task_id="D", candidates=["mem-1", "x"], selected=[], reward=0.0),
            ]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory_attribution.jsonl"
            write_attribution_jsonl(records, path)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["memory_id"], "mem-1")  # sorted by id
            self.assertEqual(json.loads(lines[1])["memory_id"], "mem-2")

            loaded = load_attribution_jsonl(path)
            self.assertEqual(set(loaded.keys()), {"mem-1", "mem-2"})
            self.assertEqual(loaded["mem-1"].tier, "tip")
            # write again -> identical bytes
            first = path.read_text(encoding="utf-8")
            write_attribution_jsonl(records, path)
            self.assertEqual(path.read_text(encoding="utf-8"), first)

    def test_load_skips_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory_attribution.jsonl"
            path.write_text(
                json.dumps({"memory_id": "mem-1", "tier": "skill", "value": 0.1, "confidence": 0.5}) + "\n"
                + "{bad}\n\n",
                encoding="utf-8",
            )
            loaded = load_attribution_jsonl(path)
            self.assertEqual(list(loaded.keys()), ["mem-1"])


class AttributionSummaryTests(unittest.TestCase):
    def test_summary_uses_configured_evidence_thresholds_and_renders_skips(self) -> None:
        record = MemoryAttributionRecord(
            memory_id="mem-1",
            tier="skill",
            candidate_count=3,
            selected_count=1,
            not_selected_count=2,
            value=0.0,
        )

        summary = attribution_summary(
            [record],
            config=AttributionConfig(min_candidate_count=4),
        )

        self.assertEqual(summary["skipped_low_evidence"], 1)
        self.assertEqual(summary["skipped_by_project_key"], 0)
        rendered = render_attribution_summary(summary)
        self.assertIn("Skipped by project key: 0", rendered)
        self.assertIn("Skipped low evidence: 1", rendered)


if __name__ == "__main__":
    unittest.main()
