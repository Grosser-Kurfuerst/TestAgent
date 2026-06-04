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

from my_agent.llm import FakeLLM
from my_agent.runtime import run_agent


def write_runtime_repo(repo: Path) -> None:
    (repo / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    \"\"\"Return a minus b.\"\"\"\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import add, subtract\n\n"
        "class CalculatorTests(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n\n"
        "    def test_subtract(self):\n"
        "        self.assertEqual(subtract(5, 3), 2)\n",
        encoding="utf-8",
    )


def read_trace(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RuntimeTests(unittest.TestCase):
    def test_run_agent_rejects_zero_max_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_runtime_repo(repo)

            with self.assertRaisesRegex(ValueError, "max_steps must be >= 1"):
                run_agent(repo_path=repo, task="Fix subtract.", max_steps=0, trace_dir=repo / "traces")

    def test_fake_llm_repairs_sample_bug_and_writes_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="Fix the subtract function so it returns the first number minus the second number.",
                test_command="python -m unittest discover -s tests -q",
                max_steps=8,
                trace_dir=base / "traces",
            )

            self.assertTrue(state.done)
            self.assertEqual(state.stop_reason, "finish_called")
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))
            self.assertTrue(any(record.call.tool == "run_tests" and record.result.ok for record in state.tool_history))
            self.assertTrue(any(record.call.tool == "finish" for record in state.tool_history))
            self.assertIn("Tests: passed", state.final_answer)

            self.assertIsNotNone(state.trace_path)
            events = read_trace(state.trace_path)
            event_names = [str(event["event"]) for event in events]
            for expected in ("repo_indexed", "plan", "tool_call", "verify", "final_summary"):
                self.assertIn(expected, event_names)
            self.assertEqual({event["run_id"] for event in events}, {state.run_id})

    def test_invalid_actor_json_retries_then_consumes_valid_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            valid_read = json.dumps(
                {
                    "tool": "read_file",
                    "arguments": {"path": "calculator.py"},
                    "reason": "Inspect the file after the invalid output recovery.",
                }
            )

            state = run_agent(
                repo_path=repo,
                task="Fix subtract.",
                test_command="python -m unittest discover -s tests -q",
                llm=FakeLLM(actor_responses=["this is not json", valid_read]),
                max_steps=8,
                trace_dir=base / "traces",
            )

            self.assertTrue(state.done)
            self.assertEqual(state.stop_reason, "finish_called")
            self.assertEqual(state.tool_history[0].call.tool, "invalid_tool_call")
            self.assertEqual(state.tool_history[1].call.tool, "read_file")
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))

            events = read_trace(state.trace_path)
            tool_events = [event for event in events if event["event"] == "tool_call"]
            self.assertIn("No JSON object found", str(tool_events[0]["payload"]))

    def test_second_invalid_actor_json_stops_with_protocol_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="Fix subtract.",
                test_command="python -m unittest discover -s tests -q",
                llm=FakeLLM(actor_responses=["first bad output", "second bad output"]),
                max_steps=4,
                trace_dir=base / "traces",
            )

            self.assertTrue(state.done)
            self.assertEqual(state.stop_reason, "invalid_tool_call")
            self.assertEqual(state.steps, 2)
            self.assertEqual([record.call.tool for record in state.tool_history], ["invalid_tool_call", "invalid_tool_call"])
            self.assertIn("invalid_tool_call", state.final_answer)
            self.assertIn("return a + b", (repo / "calculator.py").read_text(encoding="utf-8"))

    def test_finish_after_failed_tests_is_not_marked_successful(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            run_tests = json.dumps(
                {
                    "tool": "run_tests",
                    "arguments": {"command": "python -m unittest discover -s tests -q"},
                    "reason": "Run tests before editing.",
                }
            )
            finish = json.dumps(
                {
                    "tool": "finish",
                    "arguments": {"summary": "Done even though tests failed."},
                    "reason": "Finish too early.",
                }
            )

            state = run_agent(
                repo_path=repo,
                task="Fix subtract.",
                test_command="python -m unittest discover -s tests -q",
                llm=FakeLLM(actor_responses=[run_tests, finish]),
                max_steps=4,
                trace_dir=base / "traces",
            )

            self.assertTrue(state.done)
            self.assertEqual(state.stop_reason, "finished_with_failed_tests")
            self.assertIn("Tests: failed", state.final_answer)
            self.assertFalse(state.tool_history[0].result.ok)

    def test_max_steps_stops_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="Fix subtract.",
                test_command="python -m unittest discover -s tests -q",
                max_steps=2,
                trace_dir=base / "traces",
            )

            self.assertTrue(state.done)
            self.assertEqual(state.stop_reason, "max_steps_reached")
            self.assertEqual(state.steps, 2)
            self.assertIn("max_steps_reached", state.final_answer)
            self.assertIn("return a + b", (repo / "calculator.py").read_text(encoding="utf-8"))

    def test_runtime_index_skips_protected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            (repo / "credentials.json").write_text('{"token": "needle_secret"}\n', encoding="utf-8")

            state = run_agent(
                repo_path=repo,
                task="Find needle_secret without leaking credentials.",
                test_command="python -m unittest discover -s tests -q",
                max_steps=1,
                trace_dir=base / "traces",
            )

            self.assertNotIn("needle_secret", state.repo_context)
            self.assertNotIn("credentials.json", state.repo_context)


if __name__ == "__main__":
    unittest.main()
