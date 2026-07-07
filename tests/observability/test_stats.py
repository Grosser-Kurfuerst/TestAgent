from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.stats import collect_trace_stats, format_trace_stats
from my_agent.evaluation.trace_metrics import collect_trace_metrics


class TraceStatsTests(unittest.TestCase):
    def test_collects_tool_success_tests_edits_and_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {"run_id": "run-1", "event": "run.started", "payload": {}},
                _tool_event("run-1", "read_file", True),
                _tool_event("run-1", "replace_in_file", True),
                _tool_event("run-1", "run_tests", True),
                _tool_event("run-1", "grep", False, blocked=True),
            ]
            trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

            stats = collect_trace_stats(trace)

        self.assertEqual(stats.trace_files, 1)
        self.assertEqual(stats.runs, 1)
        self.assertEqual(stats.tool_calls, 4)
        self.assertEqual(stats.successful_tool_calls, 3)
        self.assertEqual(stats.blocked_tool_calls, 1)
        self.assertEqual(stats.test_runs, 1)
        self.assertEqual(stats.passed_test_runs, 1)
        self.assertEqual(stats.edit_count, 1)
        self.assertEqual(stats.tool_distribution["run_tests"], 1)
        self.assertIn("Tool success rate: 3/4", format_trace_stats(stats))
        self.assertEqual(stats.evolver_candidate_events, 0)
        self.assertEqual(stats.evolver_selected_events, 0)

    def test_collects_evolver_selection_for_nonrecursive_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {
                    "run_id": "run-1",
                    "event": "memory.evolver_candidates",
                    "payload": {"candidate_count": 4, "selection_policy": "rule_tier_weighted_v1"},
                },
                {
                    "run_id": "run-1",
                    "event": "memory.evolver_selected",
                    "payload": {
                        "selected_count": 2,
                        "tiers": {"tip": 1, "tool": 1},
                        "selection_policy": "rule_tier_weighted_v1",
                    },
                },
            ]
            trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

            stats = collect_trace_stats(trace)

        self.assertEqual(stats.evolver_candidate_events, 1)
        self.assertEqual(stats.evolver_selected_events, 1)
        self.assertEqual(stats.evolver_candidates_total, 4)
        self.assertEqual(stats.evolver_selected_total, 2)
        self.assertEqual(stats.evolver_selected_by_tier, {"tip": 1, "tool": 1})
        self.assertEqual(stats.evolver_selection_policies, {"rule_tier_weighted_v1": 1})
        self.assertEqual(stats.to_dict()["evolver_candidates_total"], 4)
        self.assertIn("Evolver selection: candidate_events=1", format_trace_stats(stats))

    def test_recursive_metrics_follow_child_traces_and_tokens_by_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parent = base / "parent.jsonl"
            child = base / "children" / "child.jsonl"
            child.parent.mkdir()
            parent_events = [
                {
                    "run_id": "parent",
                    "event": "llm.completed",
                    "payload": {
                        "phase": "plan_planner",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                    },
                },
                {
                    "run_id": "parent",
                    "event": "agent.completed",
                    "payload": {
                        "stop_reason": "plan_completed",
                        "child_trace_paths": [str(child)],
                    },
                },
            ]
            child_events = [
                _tool_event("child", "run_tests", True),
                {
                    "run_id": "child",
                    "event": "llm.completed",
                    "payload": {
                        "phase": "react",
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                    },
                },
                {"run_id": "child", "event": "run.completed", "payload": {"stop_reason": "finish_called"}},
            ]
            parent.write_text("\n".join(json.dumps(event) for event in parent_events), encoding="utf-8")
            child.write_text("\n".join(json.dumps(event) for event in child_events), encoding="utf-8")

            shallow = collect_trace_metrics(parent, recursive=False)
            recursive = collect_trace_metrics(parent, recursive=True)

        self.assertEqual(shallow.trace_files, 1)
        self.assertEqual(shallow.test_runs, 0)
        self.assertEqual(recursive.trace_files, 2)
        self.assertEqual(recursive.test_runs, 1)
        self.assertEqual(recursive.llm_iterations, 2)
        self.assertEqual(recursive.total_tokens, 19)
        self.assertEqual(recursive.tokens_by_phase["plan_planner"]["total_tokens"], 13)
        self.assertEqual(recursive.tokens_by_phase["react"]["llm_iterations"], 1)
        self.assertEqual(recursive.no_test_finish, 0)


def _tool_event(run_id: str, tool: str, ok: bool, blocked: bool = False) -> dict[str, object]:
    return {
        "run_id": run_id,
        "event": "tool.completed",
        "payload": {
            "id": f"call_{tool}",
            "name": tool,
            "ok": ok,
            "content": "",
            "blocked": blocked,
            "error_code": "",
        },
    }


if __name__ == "__main__":
    unittest.main()
