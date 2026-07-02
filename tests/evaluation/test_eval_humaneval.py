from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests._path import add_src_to_path


add_src_to_path()

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_humaneval.py"
SPEC = importlib.util.spec_from_file_location("eval_humaneval_script", SCRIPT_PATH)
assert SPEC is not None
eval_humaneval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = eval_humaneval
SPEC.loader.exec_module(eval_humaneval)


ROW = {
    "task_id": "HumanEval/1",
    "prompt": 'def add(a: int, b: int) -> int:\n    """Return a plus b."""\n',
    "canonical_solution": "    return a + b\n",
    "test": (
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
        "    assert candidate(-1, 1) == 0\n"
    ),
    "entry_point": "add",
}


def _state(
    steps: int = 3,
    done: bool = True,
    stop_reason: str = "finish_called",
    trace_path: Path | None = None,
    run_id: str = "run-1",
):
    return SimpleNamespace(steps=steps, done=done, stop_reason=stop_reason, trace_path=trace_path, run_id=run_id)


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class EvalHumanEvalRepoTests(unittest.TestCase):
    def test_build_humaneval_repo_writes_solution_and_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = eval_humaneval.build_humaneval_repo(ROW, Path(tmp))

            solution = (repo / "solution.py").read_text(encoding="utf-8")
            test_file = (repo / "tests" / "test_solution.py").read_text(encoding="utf-8")

        self.assertIn("def add", solution)
        self.assertIn("pass", solution)
        self.assertIn("from solution import add", test_file)
        self.assertIn("check(add)", test_file)
        self.assertTrue(repo.name.startswith("humaneval_HumanEval_1"))

    def test_evaluate_solution_passes_for_correct_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = eval_humaneval.build_humaneval_repo(ROW, Path(tmp))
            (repo / "solution.py").write_text(
                ROW["prompt"].rstrip() + "\n" + ROW["canonical_solution"],
                encoding="utf-8",
            )

            passed, output = eval_humaneval.evaluate_solution(repo)

        self.assertTrue(passed, output)

    def test_evaluate_solution_fails_for_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = eval_humaneval.build_humaneval_repo(ROW, Path(tmp))

            passed, output = eval_humaneval.evaluate_solution(repo)

        self.assertFalse(passed)
        self.assertTrue(output)

    def test_fallback_evaluate_solution_passes_for_correct_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = eval_humaneval.build_humaneval_repo(ROW, Path(tmp))
            (repo / "solution.py").write_text(
                ROW["prompt"].rstrip() + "\n" + ROW["canonical_solution"],
                encoding="utf-8",
            )

            passed, output = eval_humaneval.evaluate_solution_fallback(repo)

        self.assertTrue(passed, output)


class EvalHumanEvalResultTests(unittest.TestCase):
    def test_passed_result_uses_status_and_drops_passed_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(eval_humaneval, "run_agent", return_value=_state()):
                with mock.patch.object(eval_humaneval, "evaluate_solution", return_value=(True, "ok")):
                    result = eval_humaneval.run_one_task(
                        ROW,
                        Path(tmp),
                        config=None,
                        max_steps=5,
                        llm_retries=0,
                    )

        self.assertEqual(result.status, "passed")
        self.assertTrue(result.scored)
        payload = result.to_dict()
        self.assertNotIn("passed", payload)
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["scored"])

    def test_failed_result_is_scored_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(eval_humaneval, "run_agent", return_value=_state()):
                with mock.patch.object(eval_humaneval, "evaluate_solution", return_value=(False, "assertion failed")):
                    result = eval_humaneval.run_one_task(
                        ROW,
                        Path(tmp),
                        config=None,
                        max_steps=5,
                        llm_retries=0,
                    )

        self.assertEqual(result.status, "failed")
        self.assertTrue(result.scored)
        self.assertEqual(result.test_output, "assertion failed")

    def test_passed_result_appends_benchmark_result_to_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            with mock.patch.object(eval_humaneval, "run_agent", return_value=_state(trace_path=trace, run_id="run-1")):
                with mock.patch.object(eval_humaneval, "evaluate_solution", return_value=(True, "ok")):
                    result = eval_humaneval.run_one_task(
                        ROW,
                        Path(tmp),
                        config=None,
                        max_steps=5,
                        llm_retries=0,
                    )
            events = _read_events(trace)

        self.assertEqual(result.status, "passed")
        self.assertEqual(events[-1]["event"], "benchmark_result")
        self.assertEqual(events[-1]["run_id"], "run-1")
        self.assertEqual(events[-1]["payload"]["benchmark"], "humaneval")
        self.assertEqual(events[-1]["payload"]["task_id"], "HumanEval/1")
        self.assertEqual(events[-1]["payload"]["status"], "passed")
        self.assertTrue(events[-1]["payload"]["scored"])
        self.assertEqual(events[-1]["payload"]["test_command"], "python -m pytest -q")

    def test_failed_result_appends_benchmark_result_to_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            with mock.patch.object(eval_humaneval, "run_agent", return_value=_state(trace_path=trace, run_id="run-2")):
                with mock.patch.object(eval_humaneval, "evaluate_solution", return_value=(False, "assertion failed")):
                    result = eval_humaneval.run_one_task(
                        ROW,
                        Path(tmp),
                        config=None,
                        max_steps=5,
                        llm_retries=0,
                    )
            events = _read_events(trace)

        self.assertEqual(result.status, "failed")
        self.assertEqual(events[-1]["event"], "benchmark_result")
        self.assertEqual(events[-1]["payload"]["status"], "failed")
        self.assertIn("assertion failed", events[-1]["payload"]["test_output"])

    def test_non_transient_exception_is_scored_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(eval_humaneval, "run_agent", side_effect=ValueError("bad config")):
                result = eval_humaneval.run_one_task(
                    ROW,
                    Path(tmp),
                    config=None,
                    max_steps=5,
                    llm_retries=1,
                    retry_delay_sec=0,
                )

        self.assertEqual(result.status, "error")
        self.assertTrue(result.scored)
        self.assertIn("ValueError: bad config", result.error)
        self.assertEqual(result.attempts, 1)

    def test_transient_error_retries_then_can_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                eval_humaneval,
                "run_agent",
                side_effect=[RuntimeError("LLM response message content was empty."), _state()],
            ) as run_mock:
                with mock.patch.object(eval_humaneval, "evaluate_solution", return_value=(True, "ok")):
                    result = eval_humaneval.run_one_task(
                        ROW,
                        Path(tmp),
                        config=None,
                        max_steps=5,
                        llm_retries=1,
                        retry_delay_sec=0,
                    )

        self.assertEqual(result.status, "passed")
        self.assertTrue(result.scored)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(run_mock.call_count, 2)

    def test_transient_error_excluded_after_retries_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                eval_humaneval,
                "run_agent",
                side_effect=RuntimeError("LLM response message content was empty."),
            ):
                result = eval_humaneval.run_one_task(
                    ROW,
                    Path(tmp),
                    config=None,
                    max_steps=5,
                    llm_retries=1,
                    retry_delay_sec=0,
                )

        self.assertEqual(result.status, "transient_error")
        self.assertFalse(result.scored)
        self.assertEqual(result.attempts, 2)
        self.assertIn("LLM response message content was empty", result.error)

    def test_transient_error_can_be_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                eval_humaneval,
                "run_agent",
                side_effect=RuntimeError("LLM response message content was empty."),
            ):
                result = eval_humaneval.run_one_task(
                    ROW,
                    Path(tmp),
                    config=None,
                    max_steps=5,
                    llm_retries=0,
                    retry_delay_sec=0,
                    count_transient_errors=True,
                )

        self.assertEqual(result.status, "transient_error")
        self.assertTrue(result.scored)

    def test_retry_rebuilds_task_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                eval_humaneval,
                "build_humaneval_repo",
                side_effect=[Path(tmp) / "attempt1", Path(tmp) / "attempt2"],
            ) as build_mock:
                with mock.patch.object(
                    eval_humaneval,
                    "run_agent",
                    side_effect=[RuntimeError("LLM response message content was empty."), _state()],
                ):
                    with mock.patch.object(eval_humaneval, "evaluate_solution", return_value=(True, "ok")):
                        result = eval_humaneval.run_one_task(
                            ROW,
                            Path(tmp),
                            config=None,
                            max_steps=5,
                            llm_retries=1,
                            retry_delay_sec=0,
                        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(build_mock.call_count, 2)


class EvalHumanEvalSummaryTests(unittest.TestCase):
    def test_summary_uses_status_and_scored(self) -> None:
        results = [
            eval_humaneval.EvalResult(task_id="1", status="passed", scored=True),
            eval_humaneval.EvalResult(task_id="2", status="failed", scored=True),
            eval_humaneval.EvalResult(task_id="3", status="error", scored=True),
            eval_humaneval.EvalResult(task_id="4", status="transient_error", scored=False),
            eval_humaneval.EvalResult(task_id="5", status="transient_error", scored=True),
        ]

        summary = eval_humaneval.summarize_results(results)

        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["scored"], 4)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["error"], 1)
        self.assertEqual(summary["transient_excluded"], 1)
        self.assertEqual(summary["transient_counted"], 1)
        self.assertAlmostEqual(summary["solve_rate"], 25.0)
        self.assertAlmostEqual(summary["end_to_end_rate"], 20.0)

    def test_prepare_results_file_overwrites_only_on_fresh_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text("old\n", encoding="utf-8")

            eval_humaneval._prepare_results_file(path, start=0)
            self.assertEqual(path.read_text(encoding="utf-8"), "")

            path.write_text("keep\n", encoding="utf-8")
            eval_humaneval._prepare_results_file(path, start=2)
            self.assertEqual(path.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
