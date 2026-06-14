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
    BenchmarkSpec,
    EvalResult,
    is_transient_llm_error,
    load_results_file,
    run_import_test_fallback,
    run_benchmark_with_config,
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

    def test_load_results_file_supports_current_and_legacy_status_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(
                '{"task_id":"old","passed":true}\n'
                '{"task_id":"new","status":"failed","scored":true}\n',
                encoding="utf-8",
            )

            results = load_results_file(path)

        self.assertEqual([result.status for result in results], ["passed", "failed"])
        self.assertEqual(summarize_results(results)["passed"], 1)

    def test_run_benchmark_can_summarize_full_results_file_on_resume(self) -> None:
        spec = BenchmarkSpec(
            name="demo",
            display_name="Demo",
            test_command="python -m pytest -q",
            build_repo=lambda row, base_dir: base_dir / str(row["task_id"]),
            task_id=lambda row: str(row["task_id"]),
            task_prompt=lambda row: str(row["task"]),
            evaluate_solution=lambda repo_path: (True, "ok"),
        )

        def fake_agent_runner(**kwargs):
            return SimpleNamespace(steps=1, done=True, stop_reason="finish_called", trace_path=None, run_id="")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "results.jsonl").write_text(
                '{"task_id":"0","status":"failed","scored":true}\n',
                encoding="utf-8",
            )

            run = run_benchmark_with_config(
                config=None,
                output_dir=output_dir,
                spec=spec,
                load_rows=lambda split: [{"task_id": "0", "task": "old"}, {"task_id": "1", "task": "new"}],
                split="test",
                start=1,
                limit=1,
                max_steps=1,
                llm_retries=0,
                retry_delay_sec=0,
                write_summary=True,
                summary_scope="results_file",
                agent_runner=fake_agent_runner,
            )
            written_summary = (output_dir / "summary.json").read_text(encoding="utf-8")

        self.assertEqual(run.summary["total"], 2)
        self.assertEqual(run.summary["passed"], 1)
        self.assertEqual(run.summary["failed"], 1)
        self.assertIn('"total": 2', written_summary)


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
