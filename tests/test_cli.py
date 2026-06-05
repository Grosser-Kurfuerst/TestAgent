from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.cli import DEFAULT_TASK_FILE, format_task, load_task, main


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


class CliTests(unittest.TestCase):
    def test_load_sample_task(self) -> None:
        task = load_task(DEFAULT_TASK_FILE)

        self.assertEqual(task["id"], "sample_subtract_bug")
        self.assertEqual(task["repo"], "examples/sample_repo")
        self.assertIn("subtract", task["task"])
        self.assertEqual(task["test_command"], "python -m unittest discover -s tests -q")

    def test_format_task_includes_required_fields(self) -> None:
        output = format_task(load_task(DEFAULT_TASK_FILE))

        self.assertIn("id: sample_subtract_bug", output)
        self.assertIn("repo: examples/sample_repo", output)
        self.assertIn("test_command: python -m unittest discover -s tests -q", output)

    def test_cli_load_task_prints_task(self) -> None:
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            exit_code = main(["load-task", "--task-file", str(DEFAULT_TASK_FILE)])

        self.assertEqual(exit_code, 0)
        self.assertIn("sample_subtract_bug", stream.getvalue())
        self.assertIn("Fix the subtract function", stream.getvalue())

    def test_cli_run_executes_fake_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            task_file = base / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "id": "tmp",
                        "source": "local",
                        "repo": str(repo),
                        "task": "Fix the subtract function so it returns a minus b.",
                        "test_command": "python -m unittest discover -s tests -q",
                    }
                ),
                encoding="utf-8",
            )
            trace_dir = base / "traces"
            stream = io.StringIO()
            env_file = _write_env_file(base, "MY_AGENT_LLM_PROVIDER=fake\n")

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with contextlib.redirect_stdout(stream):
                    exit_code = main(["run", "--task-file", str(task_file), "--trace-dir", str(trace_dir)])

            self.assertEqual(exit_code, 0)
            self.assertIn("# Plan", stream.getvalue())
            self.assertIn("# Review", stream.getvalue())
            self.assertIn("# Final summary", stream.getvalue())
            self.assertIn("Tests: passed", stream.getvalue())
            self.assertIn("Risks:", stream.getvalue())
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))
            self.assertEqual(len(list(trace_dir.glob("*.jsonl"))), 1)

    def test_cli_run_without_api_key_returns_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            task_file = base / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "id": "tmp",
                        "source": "local",
                        "repo": str(repo),
                        "task": "Fix the subtract function so it returns a minus b.",
                        "test_command": "python -m unittest discover -s tests -q",
                    }
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            env_file = _write_env_file(base, "MY_AGENT_LLM_PROVIDER=openai\n")

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with contextlib.redirect_stderr(stderr):
                    exit_code = main(["run", "--task-file", str(task_file), "--trace-dir", str(base / "traces")])

            self.assertEqual(exit_code, 1)
            self.assertIn("No API key configured", stderr.getvalue())

    def test_cli_config_without_dotenv_returns_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / ".env"
            stderr = io.StringIO()

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", missing):
                with contextlib.redirect_stderr(stderr):
                    exit_code = main(["config"])

            self.assertEqual(exit_code, 1)
            self.assertIn("Configuration file not found", stderr.getvalue())

    def test_cli_stats_prints_trace_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            task_file = base / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "id": "tmp",
                        "source": "local",
                        "repo": str(repo),
                        "task": "Fix the subtract function so it returns a minus b.",
                        "test_command": "python -m unittest discover -s tests -q",
                    }
                ),
                encoding="utf-8",
            )
            trace_dir = base / "traces"
            run_output = io.StringIO()
            env_file = _write_env_file(base, "MY_AGENT_LLM_PROVIDER=fake\n")

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with contextlib.redirect_stdout(run_output):
                    self.assertEqual(main(["run", "--task-file", str(task_file), "--trace-dir", str(trace_dir)]), 0)

            stats_output = io.StringIO()
            with contextlib.redirect_stdout(stats_output):
                exit_code = main(["stats", "--trace", str(trace_dir)])

            self.assertEqual(exit_code, 0)
            output = stats_output.getvalue()
            self.assertIn("Tool success rate:", output)
            self.assertIn("Test pass rate: 1/1", output)
            self.assertIn("Edit count: 1", output)
            self.assertIn("- run_tests: 1", output)

    def test_cli_index_prints_repository_context(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "examples" / "sample_repo"
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            exit_code = main(["index", "--repo", str(repo), "--query", "subtract"])

        output = stream.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertNotIn("placeholder", output.lower())
        self.assertIn("# Repository tree", output)
        self.assertIn("# Symbol index", output)
        self.assertIn("calculator.py", output)
        self.assertIn("function subtract", output)

    def test_cli_retrieve_prints_related_context(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "examples" / "sample_repo"
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            exit_code = main(["retrieve", "--repo", str(repo), "--query", "subtract", "--top-k", "1"])

        output = stream.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("## calculator.py", output)
        self.assertNotIn("## tests/test_calculator.py", output)
        self.assertIn("subtract", output)
        self.assertIn("score=", output)

    def test_cli_rejects_non_positive_top_k(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "examples" / "sample_repo"
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(["retrieve", "--repo", str(repo), "--query", "subtract", "--top-k", "0"])

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("top_k must be >= 1", stderr.getvalue())

    def test_sample_repo_fixture_exists(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "examples" / "sample_repo"

        self.assertTrue((repo / "calculator.py").exists())
        self.assertTrue((repo / "tests" / "test_calculator.py").exists())
        self.assertTrue((repo / "AGENT.md").exists())


def _write_env_file(base: Path, content: str) -> Path:
    env_file = base / ".env"
    env_file.write_text(content, encoding="utf-8")
    return env_file


if __name__ == "__main__":
    unittest.main()
