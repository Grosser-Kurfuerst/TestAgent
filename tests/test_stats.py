from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.stats import collect_trace_stats, format_trace_stats


class TraceStatsTests(unittest.TestCase):
    def test_collects_tool_success_tests_edits_and_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            events = [
                {"run_id": "run-1", "event": "plan", "payload": {}},
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


def _tool_event(run_id: str, tool: str, ok: bool, blocked: bool = False) -> dict[str, object]:
    return {
        "run_id": run_id,
        "event": "tool_call",
        "payload": {
            "call": {"tool": tool, "arguments": {}, "reason": "test"},
            "result": {"ok": ok, "output": "", "blocked": blocked, "reason": ""},
        },
    }


if __name__ == "__main__":
    unittest.main()
