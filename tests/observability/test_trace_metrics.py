from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.observability.trace_metrics import collect_trace_metrics, format_trace_metrics


class TraceMetricsEvolverTests(unittest.TestCase):
    def test_collects_evolver_candidate_and_selected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_candidates",
                            "payload": {
                                "candidate_count": 3,
                                "tiers": {"tip": 2, "skill": 1},
                                "selection_policy": "rule_tier_weighted_v1",
                            },
                        },
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_selected",
                            "payload": {
                                "selected_count": 2,
                                "selected_ids": ["tip-1", "skill-1"],
                                "tiers": {"tip": 1, "skill": 1},
                                "selection_policy": "rule_tier_weighted_v1",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.evolver_candidate_events, 1)
        self.assertEqual(metrics.evolver_selected_events, 1)
        self.assertEqual(metrics.evolver_candidates_total, 3)
        self.assertEqual(metrics.evolver_selected_total, 2)
        self.assertEqual(metrics.evolver_selected_by_tier, {"skill": 1, "tip": 1})
        self.assertEqual(metrics.evolver_selection_policies, {"rule_tier_weighted_v1": 1})
        self.assertEqual(metrics.to_dict()["evolver_selected_total"], 2)
        self.assertIn("Evolver selection: candidate_events=1", format_trace_metrics(metrics))

    def test_recursive_metrics_include_child_evolver_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parent = base / "parent.jsonl"
            child = base / "child.jsonl"
            parent.write_text(
                json.dumps(
                    {
                        "run_id": "parent",
                        "event": "agent.completed",
                        "payload": {"stop_reason": "plan_completed", "child_trace_paths": [str(child)]},
                    }
                ),
                encoding="utf-8",
            )
            child.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "child",
                            "event": "memory.evolver_candidates",
                            "payload": {"candidate_ids": ["a", "b"]},
                        },
                        {
                            "run_id": "child",
                            "event": "memory.evolver_selected",
                            "payload": {
                                "selected_ids": ["a"],
                                "tiers": {"tool": 1},
                                "selection_policy": "rule_tier_weighted_v1",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            shallow = collect_trace_metrics(parent, recursive=False)
            recursive = collect_trace_metrics(parent, recursive=True)

        self.assertEqual(shallow.evolver_candidates_total, 0)
        self.assertEqual(recursive.trace_files, 2)
        self.assertEqual(recursive.evolver_candidates_total, 2)
        self.assertEqual(recursive.evolver_selected_total, 1)
        self.assertEqual(recursive.evolver_selected_by_tier, {"tool": 1})

    def test_old_trace_defaults_evolver_metrics_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                json.dumps({"run_id": "run-1", "event": "run.completed", "payload": {"stop_reason": "done"}}),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace)

        self.assertEqual(metrics.evolver_candidate_events, 0)
        self.assertEqual(metrics.evolver_selected_events, 0)
        self.assertEqual(metrics.evolver_candidates_total, 0)
        self.assertEqual(metrics.evolver_selected_total, 0)
        self.assertEqual(metrics.evolver_selected_by_tier, {})
        self.assertEqual(metrics.evolver_selection_policies, {})
        self.assertEqual(metrics.evolver_writer_started_events, 0)
        self.assertEqual(metrics.evolver_writer_saved_events, 0)
        self.assertEqual(metrics.evolver_writer_saved_total, 0)
        self.assertEqual(metrics.evolver_writer_saved_by_tier, {})
        self.assertEqual(metrics.evolver_writer_failed_events, 0)
        self.assertEqual(metrics.maintenance_runs, 0)
        self.assertEqual(metrics.maintenance_applied_runs, 0)
        self.assertEqual(metrics.maintenance_failures, 0)
        self.assertEqual(metrics.maintenance_committed_with_audit_error, 0)

    def test_collects_evolver_writer_saved_total_and_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_writer_started",
                            "payload": {
                                "mode": "fallback",
                                "outcome": "success",
                                "selected_count": 2,
                                "candidate_count": 5,
                            },
                        },
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_writer_saved",
                            "payload": {
                                "saved_count": 2,
                                "duplicate_count": 1,
                                "saved_records": [{"id": "exp_1", "tier": "skill"}, {"id": "exp_2", "tier": "tool"}],
                                "tiers": {"skill": 1, "tool": 1},
                                "writer_policy": "fallback_runtime_v1",
                            },
                        },
                        {
                            "run_id": "run-1",
                            "event": "memory.evolver_writer_failed",
                            "payload": {"phase": "unknown", "error": "ValueError: boom"},
                        },
                    ]
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.evolver_writer_started_events, 1)
        self.assertEqual(metrics.evolver_writer_saved_events, 1)
        self.assertEqual(metrics.evolver_writer_saved_total, 2)
        self.assertEqual(metrics.evolver_writer_saved_by_tier, {"skill": 1, "tool": 1})
        self.assertEqual(metrics.evolver_writer_failed_events, 1)
        self.assertEqual(metrics.to_dict()["evolver_writer_saved_total"], 2)
        self.assertIn("Evolver writer: started_events=1", format_trace_metrics(metrics))
        self.assertIn("saved=2", format_trace_metrics(metrics))

    def test_recursive_metrics_include_child_evolver_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parent = base / "parent.jsonl"
            child = base / "child.jsonl"
            parent.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "parent",
                            "event": "memory.evolver_writer_started",
                            "payload": {"mode": "fallback"},
                        },
                        {
                            "run_id": "parent",
                            "event": "agent.completed",
                            "payload": {"stop_reason": "plan_completed", "child_trace_paths": [str(child)]},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            child.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {
                            "run_id": "child",
                            "event": "memory.evolver_writer_saved",
                            "payload": {
                                "saved_count": 3,
                                "tiers": {"tip": 2, "skill": 1},
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            shallow = collect_trace_metrics(parent, recursive=False)
            recursive = collect_trace_metrics(parent, recursive=True)

        self.assertEqual(shallow.evolver_writer_started_events, 1)
        self.assertEqual(shallow.evolver_writer_saved_total, 0)
        self.assertEqual(recursive.trace_files, 2)
        self.assertEqual(recursive.evolver_writer_started_events, 1)
        self.assertEqual(recursive.evolver_writer_saved_total, 3)
        self.assertEqual(recursive.evolver_writer_saved_by_tier, {"skill": 1, "tip": 2})

    def test_writer_saved_count_falls_back_to_saved_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "event": "memory.evolver_writer_saved",
                        "payload": {
                            "saved_records": [{"id": "exp_1", "tier": "tip"}, {"id": "exp_2", "tier": "skill"}],
                            "tiers": {"tip": 1, "skill": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.evolver_writer_saved_total, 2)
        self.assertEqual(metrics.evolver_writer_saved_by_tier, {"skill": 1, "tip": 1})

    def test_collects_memory_maintenance_metrics_from_trace_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "maintenance_trace.jsonl"
            events = [
                {
                    "run_id": "maintenance-1",
                    "event": "memory.maintenance_started",
                    "payload": {"mode": "apply"},
                },
                {
                    "run_id": "maintenance-1",
                    "event": "memory.maintenance_proposed",
                    "payload": {
                        "keep": 4,
                        "delete": 1,
                        "merge": 2,
                        "promote": 1,
                        "source_entries_removed": 3,
                        "entries_added": 2,
                    },
                },
                {
                    "run_id": "maintenance-1",
                    "event": "memory.maintenance_completed",
                    "payload": {
                        "status": "committed_with_audit_error",
                        "mutation_committed": True,
                    },
                },
                {
                    "run_id": "maintenance-2",
                    "event": "memory.maintenance_started",
                    "payload": {"mode": "apply"},
                },
                {
                    "run_id": "maintenance-2",
                    "event": "memory.maintenance_failed",
                    "payload": {
                        "status": "pre_commit_failed",
                        "stage": "validation",
                    },
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )

            metrics = collect_trace_metrics(trace, recursive=False)

        self.assertEqual(metrics.maintenance_runs, 2)
        self.assertEqual(metrics.maintenance_applied_runs, 1)
        self.assertEqual(metrics.maintenance_keep, 4)
        self.assertEqual(metrics.maintenance_delete, 1)
        self.assertEqual(metrics.maintenance_merge, 2)
        self.assertEqual(metrics.maintenance_promote, 1)
        self.assertEqual(metrics.maintenance_removed_entries, 3)
        self.assertEqual(metrics.maintenance_added_entries, 2)
        self.assertEqual(metrics.maintenance_failures, 1)
        self.assertEqual(metrics.maintenance_committed_with_audit_error, 1)
        self.assertEqual(metrics.to_dict()["maintenance_removed_entries"], 3)
        self.assertIn("Memory maintenance: runs=2", format_trace_metrics(metrics))
        self.assertIn("Maintenance actions: keep=4", format_trace_metrics(metrics))


if __name__ == "__main__":
    unittest.main()
