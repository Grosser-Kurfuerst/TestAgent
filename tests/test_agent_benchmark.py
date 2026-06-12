from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.evaluation.agent_benchmark import (
    EvalResult,
    is_transient_llm_error,
    run_import_test_fallback,
    run_pytest_or_fallback,
    summarize_results,
)


class AgentBenchmarkResultTests(unittest.TestCase):
    def test_eval_result_to_dict_drops_passed_field(self) -> None:
        result = EvalResult(task_id="1", status="passed", test_output="ok")

        payload = result.to_dict()

        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["scored"])
        self.assertNotIn("passed", payload)

    def test_summary_uses_status_and_scored(self) -> None:
        summary = summarize_results(
            [
                EvalResult(task_id="1", status="passed", scored=True),
                EvalResult(task_id="2", status="failed", scored=True),
                EvalResult(task_id="3", status="error", scored=True),
                EvalResult(task_id="4", status="transient_error", scored=False),
                EvalResult(task_id="5", status="transient_error", scored=True),
            ]
        )

        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["scored"], 4)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["error"], 1)
        self.assertEqual(summary["transient_excluded"], 1)
        self.assertEqual(summary["transient_counted"], 1)
        self.assertAlmostEqual(summary["solve_rate"], 25.0)
        self.assertAlmostEqual(summary["end_to_end_rate"], 20.0)

    def test_transient_error_detection(self) -> None:
        self.assertTrue(is_transient_llm_error(RuntimeError("LLM response message content was empty.")))
        self.assertTrue(is_transient_llm_error(RuntimeError("HTTP 429 Too Many Requests")))
        self.assertTrue(is_transient_llm_error(RuntimeError("request timeout")))
        self.assertFalse(is_transient_llm_error(ValueError("bad task row")))


class AgentBenchmarkPytestTests(unittest.TestCase):
    def test_pytest_missing_uses_fallback(self) -> None:
        calls: list[Path] = []

        def fallback(repo_path: Path) -> tuple[bool, str]:
            calls.append(repo_path)
            return True, "fallback ok"

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            completed = SimpleNamespace(returncode=1, stdout="", stderr="No module named pytest")
            with mock.patch("my_agent.evaluation.agent_benchmark.subprocess.run", return_value=completed):
                passed, output = run_pytest_or_fallback(repo, fallback)

        self.assertTrue(passed)
        self.assertEqual(output, "fallback ok")
        self.assertEqual(calls, [repo])

    def test_import_fallback_does_not_reuse_cached_solution_module(self) -> None:
        stale_solution = ModuleType("solution")
        stale_solution.one = lambda: 0  # type: ignore[attr-defined]
        previous_solution = sys.modules.get("solution")
        sys.modules["solution"] = stale_solution
        try:
            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                (repo / "tests").mkdir()
                (repo / "solution.py").write_text("def one() -> int:\n    return 1\n", encoding="utf-8")
                (repo / "tests" / "test_solution.py").write_text(
                    "from solution import one\n\n"
                    "def test_solution() -> None:\n"
                    "    assert one() == 1\n",
                    encoding="utf-8",
                )

                passed, output = run_import_test_fallback(repo, "test_solution")

            self.assertTrue(passed, output)
            self.assertIs(sys.modules.get("solution"), stale_solution)
        finally:
            if previous_solution is None:
                sys.modules.pop("solution", None)
            else:
                sys.modules["solution"] = previous_solution


if __name__ == "__main__":
    unittest.main()
