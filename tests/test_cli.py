from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.cli import DEFAULT_TASK_FILE, format_task, load_task, main


class CliTests(unittest.TestCase):
    def test_load_sample_task(self) -> None:
        task = load_task(DEFAULT_TASK_FILE)

        self.assertEqual(task["id"], "sample_subtract_bug")
        self.assertEqual(task["repo"], "examples/sample_repo")
        self.assertIn("subtract", task["task"])
        self.assertEqual(task["test_command"], "pytest -q")

    def test_format_task_includes_required_fields(self) -> None:
        output = format_task(load_task(DEFAULT_TASK_FILE))

        self.assertIn("id: sample_subtract_bug", output)
        self.assertIn("repo: examples/sample_repo", output)
        self.assertIn("test_command: pytest -q", output)

    def test_cli_load_task_prints_task(self) -> None:
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            exit_code = main(["load-task", "--task-file", str(DEFAULT_TASK_FILE)])

        self.assertEqual(exit_code, 0)
        self.assertIn("sample_subtract_bug", stream.getvalue())
        self.assertIn("Fix the subtract function", stream.getvalue())

    def test_cli_run_placeholder_loads_task(self) -> None:
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            exit_code = main(["run", "--task-file", str(DEFAULT_TASK_FILE)])

        self.assertEqual(exit_code, 0)
        self.assertIn("Phase 4 placeholder", stream.getvalue())
        self.assertIn("sample_subtract_bug", stream.getvalue())

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


if __name__ == "__main__":
    unittest.main()
