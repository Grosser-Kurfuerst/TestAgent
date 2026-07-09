from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    BenchmarkOutcome,
    SelectionSnapshot,
    UsageLogEntry,
    UsageLogger,
    benchmark_outcome_from_trace,
    collect_usage_from_manifest_results,
    flatten_tier_ids,
    group_ids_by_tier,
    read_trace_events,
    selection_from_trace,
    usage_entry_from_manifest_result,
    usage_entry_from_result_row,
    usage_entry_from_trace,
)


def _candidate_event(summaries: list[dict], *, timestamp: str = "") -> dict:
    return {
        "event": "memory.evolver_candidates",
        "time": "2026-01-01T00:00:00",
        "payload": {
            "candidate_count": len(summaries),
            "candidate_summaries": summaries,
            "candidate_ids": [s["id"] for s in summaries],
            "selection_policy": "default_v1",
            "memory_project_key": "manifest:abc:memory:shared_stream:stream:python",
            "timestamp": timestamp,
        },
    }


def _selected_event(ids: list[str], *, timestamp: str = "") -> dict:
    return {
        "event": "memory.evolver_selected",
        "time": "2026-01-01T00:00:01",
        "payload": {
            "selected_count": len(ids),
            "selected_ids": ids,
            "selection_policy": "default_v1",
            "memory_project_key": "manifest:abc:memory:shared_stream:stream:python",
            "timestamp": timestamp,
        },
    }


def _benchmark_result_event(*, resolved: bool, task_id: str = "humaneval-1") -> dict:
    return {
        "event": "benchmark_result",
        "time": "2026-01-01T00:05:00",
        "payload": {
            "task_id": task_id,
            "resolved": resolved,
            "status": "resolved" if resolved else "failed",
            "failure_type": "" if resolved else "assertion_failed",
            "memory_project_key": "manifest:abc:memory:shared_stream:stream:python",
            "stream_id": "python",
            "memory_mode": "shared_stream",
            "source": "humaneval",
        },
    }


class UsageLogEntryTests(unittest.TestCase):
    def test_all_candidate_ids_preserves_order_and_dedups(self) -> None:
        entry = UsageLogEntry(
            task_id="t1",
            task_type="humaneval",
            retrieved_candidates={"skill": ["mem-1", "mem-2", "mem-1"], "tip": ["mem-3", "mem-2"]},
        )
        self.assertEqual(entry.all_candidate_ids(), ["mem-1", "mem-2", "mem-3"])

    def test_all_selected_ids_preserves_order_and_dedups(self) -> None:
        entry = UsageLogEntry(
            task_id="t1",
            task_type="humaneval",
            selected_memory_ids={"skill": ["mem-1", "mem-1"], "tool": ["mem-4"]},
        )
        self.assertEqual(entry.all_selected_ids(), ["mem-1", "mem-4"])

    def test_is_complete_default_and_explicit(self) -> None:
        self.assertTrue(UsageLogEntry(task_id="t", task_type="x").is_complete)
        self.assertTrue(UsageLogEntry(task_id="t", task_type="x", status="complete").is_complete)
        self.assertFalse(UsageLogEntry(task_id="t", task_type="x", status="started").is_complete)

    def test_roundtrip_to_dict_from_dict_is_stable(self) -> None:
        entry = UsageLogEntry(
            task_id="t1",
            task_type="humaneval",
            timestamp="2026-01-01T00:00:00",
            retrieved_candidates={"skill": ["mem-1"]},
            selected_memory_ids={"skill": ["mem-1"]},
            env_reward=1.0,
            success=True,
            status="complete",
            tags=("smoke",),
            memory_project_key="manifest:abc:memory:shared_stream:stream:python",
            stream_id="python",
        )
        restored = UsageLogEntry.from_dict(json.loads(json.dumps(entry.to_dict(), ensure_ascii=False)))
        self.assertEqual(restored, entry)
        self.assertEqual(restored.tags, ("smoke",))

    def test_timestamp_default_is_empty_not_now(self) -> None:
        entry = UsageLogEntry(task_id="t", task_type="x")
        self.assertEqual(entry.timestamp, "")


class UsageLoggerTests(unittest.TestCase):
    def test_append_creates_parent_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "usage_logs.jsonl"
            logger = UsageLogger(path)
            logger.append(UsageLogEntry(task_id="t1", task_type="humaneval", success=True))
            logger.append(UsageLogEntry(task_id="t2", task_type="humaneval"))

            entries = logger.load_all()
            self.assertEqual([e.task_id for e in entries], ["t1", "t2"])
            self.assertEqual(entries[0].success, True)

    def test_load_all_skips_corrupt_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage_logs.jsonl"
            path.write_text(
                json.dumps(UsageLogEntry(task_id="t1", task_type="x").to_dict()) + "\n"
                + "{not valid json}\n"
                + "\n"
                + json.dumps(UsageLogEntry(task_id="t2", task_type="x").to_dict()) + "\n",
                encoding="utf-8",
            )
            entries = UsageLogger(path).load_all()
            self.assertEqual([e.task_id for e in entries], ["t1", "t2"])

    def test_started_then_complete_merges_same_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage_logs.jsonl"
            logger = UsageLogger(path)
            started = UsageLogEntry(
                task_id="t1",
                task_type="humaneval",
                status="started",
                retrieved_candidates={"skill": ["mem-1"]},
                selected_memory_ids={"skill": ["mem-1"]},
                memory_project_key="proj-A",
            )
            logger.append(started)
            outcome = started.merge_outcome(env_reward=1.0, success=True)
            logger.append(outcome)
            entries = logger.load_all()
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0].is_complete)
            self.assertTrue(entries[0].success)
            self.assertEqual(entries[0].retrieved_candidates, {"skill": ["mem-1"]})

    def test_started_complete_isolated_by_memory_project_key(self) -> None:
        # Same task_id, different streams must NOT merge across projects.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage_logs.jsonl"
            logger = UsageLogger(path)
            started_a = UsageLogEntry(
                task_id="t1", task_type="humaneval", status="started",
                memory_project_key="proj-A",
                retrieved_candidates={"skill": ["mem-A"]},
            )
            started_b = UsageLogEntry(
                task_id="t1", task_type="humaneval", status="started",
                memory_project_key="proj-B",
                retrieved_candidates={"skill": ["mem-B"]},
            )
            logger.append(started_a)
            logger.append(started_b)
            logger.append(started_a.merge_outcome(env_reward=1.0, success=True))
            logger.append(started_b.merge_outcome(env_reward=0.0, success=False))
            entries = logger.load_all()
            self.assertEqual(len(entries), 2)
            by_proj = {e.memory_project_key: e for e in entries}
            self.assertTrue(by_proj["proj-A"].success)
            self.assertFalse(by_proj["proj-B"].success)

    def test_overwrite_replaces_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage_logs.jsonl"
            logger = UsageLogger(path)
            logger.append(UsageLogEntry(task_id="old", task_type="x"))
            count = logger.overwrite([UsageLogEntry(task_id="new", task_type="y")])
            self.assertEqual(count, 1)
            self.assertEqual([e.task_id for e in logger.load_all()], ["new"])

    def test_load_for_memory_filters_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage_logs.jsonl"
            logger = UsageLogger(path)
            logger.append(UsageLogEntry(
                task_id="t1", task_type="x",
                retrieved_candidates={"skill": ["mem-1"]},
            ))
            logger.append(UsageLogEntry(
                task_id="t2", task_type="x",
                selected_memory_ids={"skill": ["mem-1"]},
            ))
            logger.append(UsageLogEntry(task_id="t3", task_type="x"))
            entries = logger.load_for_memory("mem-1")
            self.assertEqual([e.task_id for e in entries], ["t1", "t2"])


class GroupByTierTests(unittest.TestCase):
    def test_group_ids_by_tier_from_summaries(self) -> None:
        grouped = group_ids_by_tier([
            {"id": "mem-1", "tier": "skill"},
            {"id": "mem-2", "tier": "tip"},
            {"id": "mem-3"},  # no tier -> unknown
        ])
        self.assertEqual(grouped, {"skill": ["mem-1"], "tip": ["mem-2"], "unknown": ["mem-3"]})

    def test_flatten_tier_ids_dedups_preserves_order(self) -> None:
        self.assertEqual(
            flatten_tier_ids({"skill": ["mem-1", "mem-2"], "tip": ["mem-2", "mem-3"]}),
            ["mem-1", "mem-2", "mem-3"],
        )


class TraceJoinTests(unittest.TestCase):
    def test_selection_from_trace_uses_last_non_empty_selected_and_backfills_tier(self) -> None:
        events = [
            _candidate_event([
                {"id": "mem-1", "tier": "skill"},
                {"id": "mem-2", "tier": "tip"},
            ]),
            _selected_event([]),  # empty selection, should not become the chosen snapshot
            _candidate_event([
                {"id": "mem-1", "tier": "skill"},
                {"id": "mem-3", "tier": "trajectory"},
            ], timestamp="2026-01-01T00:00:30"),
            _selected_event(["mem-1"], timestamp="2026-01-01T00:00:31"),
        ]
        snapshot = selection_from_trace(events)
        self.assertIsInstance(snapshot, SelectionSnapshot)
        self.assertEqual(snapshot.candidate_count, 2)
        self.assertEqual(snapshot.retrieved_candidates, {"skill": ["mem-1"], "trajectory": ["mem-3"]})
        # selected tier must be backfilled from the candidate map for this run.
        self.assertEqual(snapshot.selected_memory_ids, {"skill": ["mem-1"]})
        self.assertEqual(snapshot.selection_policy, "default_v1")
        self.assertEqual(snapshot.timestamp, "2026-01-01T00:00:31")

    def test_selection_from_trace_candidates_only_empty_selected(self) -> None:
        events = [_candidate_event([{"id": "mem-1", "tier": "skill"}]), _selected_event([])]
        snapshot = selection_from_trace(events)
        self.assertEqual(snapshot.selected_memory_ids, {})
        self.assertEqual(snapshot.retrieved_candidates, {"skill": ["mem-1"]})

    def test_benchmark_outcome_from_trace_returns_last_result(self) -> None:
        events = [_benchmark_result_event(resolved=False), _benchmark_result_event(resolved=True, task_id="humaneval-2")]
        outcome = benchmark_outcome_from_trace(events)
        self.assertIsInstance(outcome, BenchmarkOutcome)
        self.assertTrue(outcome.resolved)
        self.assertEqual(outcome.task_id, "humaneval-2")

    def test_usage_entry_from_result_row_builds_complete_entry(self) -> None:
        events = [
            _candidate_event([{"id": "mem-1", "tier": "skill"}, {"id": "mem-2", "tier": "tip"}]),
            _selected_event(["mem-1"]),
            _benchmark_result_event(resolved=True),
        ]
        row = {
            "task_id": "humaneval-1",
            "resolved": True,
            "status": "resolved",
            "source": "humaneval",
            "mode": "auto",
            "tags": ["smoke"],
            "stream_id": "python",
            "memory_mode": "shared_stream",
            "memory_project_key": "manifest:abc:memory:shared_stream:stream:python",
            "trace_path": "",
        }
        entry = usage_entry_from_result_row(row, trace_events=events)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.task_id, "humaneval-1")
        self.assertEqual(entry.task_type, "humaneval")  # source preferred over mode
        self.assertEqual(entry.retrieved_candidates, {"skill": ["mem-1"], "tip": ["mem-2"]})
        self.assertEqual(entry.selected_memory_ids, {"skill": ["mem-1"]})
        self.assertTrue(entry.success)
        self.assertEqual(entry.env_reward, 1.0)
        self.assertEqual(entry.memory_project_key, "manifest:abc:memory:shared_stream:stream:python")
        self.assertEqual(entry.tags, ("smoke",))
        self.assertTrue(entry.is_complete)

    def test_usage_entry_outcome_falls_back_to_trace_benchmark_result(self) -> None:
        events = [
            _candidate_event([{"id": "mem-1", "tier": "skill"}]),
            _selected_event(["mem-1"]),
            _benchmark_result_event(resolved=False),
        ]
        # Row without resolved/status -> outcome must come from benchmark_result, not finish_called.
        row = {
            "task_id": "humaneval-1",
            "source": "humaneval",
            "memory_project_key": "manifest:abc:memory:shared_stream:stream:python",
            "trace_path": "",
        }
        entry = usage_entry_from_result_row(row, trace_events=events)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertFalse(entry.success)
        self.assertEqual(entry.env_reward, 0.0)
        self.assertEqual(entry.failure_type, "assertion_failed")

    def test_usage_entry_returns_none_when_no_outcome_signal(self) -> None:
        events = [_candidate_event([{"id": "mem-1", "tier": "skill"}]), _selected_event(["mem-1"])]
        row = {"task_id": "humaneval-1", "source": "humaneval", "trace_path": ""}
        self.assertIsNone(usage_entry_from_result_row(row, trace_events=events))

    def test_usage_entry_does_not_treat_runtime_stop_reason_as_success(self) -> None:
        # A trace with selection but NO benchmark_result, and a manifest row with
        # only runtime stop_reason fields, must NOT be marked successful.
        events = [_candidate_event([{"id": "mem-1", "tier": "skill"}]), _selected_event(["mem-1"])]
        row = {"task_id": "humaneval-1", "agent_stop_reason": "finish_called", "trace_path": ""}
        self.assertIsNone(usage_entry_from_result_row(row, trace_events=events))

    def test_usage_entry_from_trace_uses_trace_outcome_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            trace_path.write_text(
                "\n".join(json.dumps(e) for e in [
                    _candidate_event([{"id": "mem-1", "tier": "skill"}]),
                    _selected_event(["mem-1"]),
                    _benchmark_result_event(resolved=True),
                ]),
                encoding="utf-8",
            )
            entry = usage_entry_from_trace(trace_path)
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertTrue(entry.success)

    def test_read_trace_events_skips_blank_and_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            trace_path.write_text(
                json.dumps(_candidate_event([{"id": "mem-1", "tier": "skill"}])) + "\n"
                + "{bad}\n\n"
                + json.dumps(_selected_event(["mem-1"])) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(read_trace_events(trace_path)), 2)


class CollectUsageTests(unittest.TestCase):
    def test_collect_usage_from_manifest_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            trace_path.write_text(
                "\n".join(json.dumps(e) for e in [
                    _candidate_event([{"id": "mem-1", "tier": "skill"}]),
                    _selected_event(["mem-1"]),
                    _benchmark_result_event(resolved=True),
                ]),
                encoding="utf-8",
            )
            results_path = Path(tmp) / "results.jsonl"
            results_path.write_text(
                json.dumps({
                    "task_id": "humaneval-1",
                    "resolved": True,
                    "status": "resolved",
                    "source": "humaneval",
                    "memory_project_key": "manifest:abc:memory:shared_stream:stream:python",
                    "trace_path": str(trace_path),
                }) + "\n"
                + json.dumps({"task_id": "humaneval-2", "agent_stop_reason": "finish_called"}) + "\n",
                encoding="utf-8",
            )
            entries = collect_usage_from_manifest_results(results_path)
            self.assertEqual([e.task_id for e in entries], ["humaneval-1"])

    def test_repeated_collection_produces_byte_stable_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            trace_events = [
                _candidate_event([{"id": "mem-1", "tier": "skill"}], timestamp="2026-01-01T00:00:00"),
                _selected_event(["mem-1"], timestamp="2026-01-01T00:00:01"),
                _benchmark_result_event(resolved=True),
            ]
            trace_path.write_text("\n".join(json.dumps(e) for e in trace_events), encoding="utf-8")
            row = {
                "task_id": "humaneval-1",
                "resolved": True,
                "status": "resolved",
                "source": "humaneval",
                "memory_project_key": "manifest:abc:memory:shared_stream:stream:python",
                "trace_path": str(trace_path),
            }
            first = json.dumps(usage_entry_from_result_row(row).__dict__, ensure_ascii=False, sort_keys=True)
            second = json.dumps(usage_entry_from_result_row(row).__dict__, ensure_ascii=False, sort_keys=True)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()