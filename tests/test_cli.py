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

    def test_sample_repo_fixture_exists(self) -> None:
        repo = Path(__file__).resolve().parents[1] / "examples" / "sample_repo"

        self.assertTrue((repo / "calculator.py").exists())
        self.assertTrue((repo / "tests" / "test_calculator.py").exists())
        self.assertTrue((repo / "AGENT.md").exists())


if __name__ == "__main__":
    unittest.main()
