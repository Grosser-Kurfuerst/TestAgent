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


if __name__ == "__main__":
    unittest.main()
