from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_mbpp.py"
SPEC = importlib.util.spec_from_file_location("eval_mbpp_script", SCRIPT_PATH)
assert SPEC is not None
eval_mbpp = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = eval_mbpp
SPEC.loader.exec_module(eval_mbpp)


ROW = {
    "task_id": "1",
    "text": "Write a function that returns one.",
    "code": "def one() -> int:\n    return 1\n",
    "test_list": ["assert one() == 1"],
}


def _state(steps: int = 3, done: bool = True, stop_reason: str = "finish_called"):
    return SimpleNamespace(steps=steps, done=done, stop_reason=stop_reason)


class EvalMbppResultTests(unittest.TestCase):
    def test_passed_result_uses_status_and_drops_passed_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(eval_mbpp, "run_agent", return_value=_state()):
                with mock.patch.object(eval_mbpp, "evaluate_solution", return_value=(True, "ok")):
                    result = eval_mbpp.run_one_task(
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
            with mock.patch.object(eval_mbpp, "run_agent", return_value=_state()):
                with mock.patch.object(eval_mbpp, "evaluate_solution", return_value=(False, "assertion failed")):
                    result = eval_mbpp.run_one_task(
                        ROW,
                        Path(tmp),
                        config=None,
                        max_steps=5,
                        llm_retries=0,
                    )

        self.assertEqual(result.status, "failed")
        self.assertTrue(result.scored)
        self.assertEqual(result.test_output, "assertion failed")

    def test_non_transient_exception_is_scored_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(eval_mbpp, "run_agent", side_effect=ValueError("bad config")):
                result = eval_mbpp.run_one_task(
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
                eval_mbpp,
                "run_agent",
                side_effect=[RuntimeError("LLM response message content was empty."), _state()],
            ) as run_mock:
                with mock.patch.object(eval_mbpp, "evaluate_solution", return_value=(True, "ok")):
                    result = eval_mbpp.run_one_task(
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
                eval_mbpp,
                "run_agent",
                side_effect=RuntimeError("LLM response message content was empty."),
            ):
                result = eval_mbpp.run_one_task(
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
                eval_mbpp,
                "run_agent",
                side_effect=RuntimeError("LLM response message content was empty."),
            ):
                result = eval_mbpp.run_one_task(
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
                eval_mbpp,
                "build_mbpp_repo",
                side_effect=[Path(tmp) / "attempt1", Path(tmp) / "attempt2"],
            ) as build_mock:
                with mock.patch.object(
                    eval_mbpp,
                    "run_agent",
                    side_effect=[RuntimeError("LLM response message content was empty."), _state()],
                ):
                    with mock.patch.object(eval_mbpp, "evaluate_solution", return_value=(True, "ok")):
                        result = eval_mbpp.run_one_task(
                            ROW,
                            Path(tmp),
                            config=None,
                            max_steps=5,
                            llm_retries=1,
                            retry_delay_sec=0,
                        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(build_mock.call_count, 2)


class EvalMbppSummaryTests(unittest.TestCase):
    def test_summary_uses_status_and_scored(self) -> None:
        results = [
            eval_mbpp.EvalResult(task_id="1", status="passed", scored=True),
            eval_mbpp.EvalResult(task_id="2", status="failed", scored=True),
            eval_mbpp.EvalResult(task_id="3", status="error", scored=True),
            eval_mbpp.EvalResult(task_id="4", status="transient_error", scored=False),
            eval_mbpp.EvalResult(task_id="5", status="transient_error", scored=True),
        ]

        summary = eval_mbpp.summarize_results(results)

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

            eval_mbpp._prepare_results_file(path, start=0)
            self.assertEqual(path.read_text(encoding="utf-8"), "")

            path.write_text("keep\n", encoding="utf-8")
            eval_mbpp._prepare_results_file(path, start=2)
            self.assertEqual(path.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
