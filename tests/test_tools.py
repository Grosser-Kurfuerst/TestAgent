from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.schema import ToolResult
from my_agent.tools import HookViolation, RepoTools, ToolRegistry, validate_test_command


def write_calculator_repo(repo: Path) -> None:
    (repo / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import subtract\n\n"
        "class CalculatorTests(unittest.TestCase):\n"
        "    def test_subtract(self):\n"
        "        self.assertEqual(subtract(5, 3), 2)\n",
        encoding="utf-8",
    )


class ToolRegistryTests(unittest.TestCase):
    def test_registry_runs_registered_tool_and_lists_descriptions(self) -> None:
        registry = ToolRegistry()
        registry.register("echo", "echo a message", lambda args: ToolResult(ok=True, output=args["message"]))

        self.assertEqual(registry.tool_names, ["echo"])
        self.assertIn("echo a message", registry.descriptions())
        self.assertEqual(registry.run("echo", {"message": "hello"}).output, "hello")

    def test_registry_unknown_tool_fails(self) -> None:
        result = ToolRegistry().run("missing", {})

        self.assertFalse(result.ok)
        self.assertIn("Unknown tool", result.output)

    def test_registry_converts_exceptions_to_tool_results(self) -> None:
        registry = ToolRegistry()

        def fail(_: dict[str, object]) -> ToolResult:
            raise RuntimeError("boom")

        registry.register("fail", "raise an exception", fail)
        result = registry.run("fail", {})

        self.assertFalse(result.ok)
        self.assertIn("Tool error: RuntimeError: boom", result.output)

    def test_registry_marks_hook_violations_as_blocked(self) -> None:
        registry = ToolRegistry()

        def blocked(_: dict[str, object]) -> ToolResult:
            raise HookViolation("blocked")

        registry.register("blocked", "raise hook violation", blocked)
        result = registry.run("blocked", {})

        self.assertFalse(result.ok)
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "blocked")


class RepoToolsTests(unittest.TestCase):
    def test_read_search_and_retrieve_tools_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_calculator_repo(repo)
            tools = RepoTools(repo)

            listed = tools.run("list_files", {"path": "."})
            read = tools.run("read_file", {"path": "calculator.py", "limit": 40})
            grep = tools.run("grep", {"pattern": "subtract", "path": "."})
            retrieved = tools.run("retrieve_context", {"query": "subtract", "top_k": 1})

            self.assertTrue(listed.ok)
            self.assertIn("calculator.py", listed.output)
            self.assertTrue(read.ok)
            self.assertIn("file truncated", read.output)
            self.assertTrue(grep.ok)
            self.assertIn("calculator.py:4", grep.output)
            self.assertTrue(retrieved.ok)
            self.assertIn("## calculator.py", retrieved.output)

    def test_run_rejects_non_object_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = RepoTools(Path(tmp)).run("finish", [])  # type: ignore[arg-type]

            self.assertFalse(result.ok)
            self.assertIn("Tool arguments must be a JSON object", result.output)

    def test_write_and_replace_return_diff_and_hook_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_calculator_repo(repo)
            tools = RepoTools(repo)

            replaced = tools.run(
                "replace_in_file",
                {
                    "path": "calculator.py",
                    "old": "def subtract(a, b):\n    return a + b\n",
                    "new": "def subtract(a, b):\n    return a - b\n",
                },
            )
            written = tools.run("write_file", {"path": "notes.txt", "content": "done\n"})

            self.assertTrue(replaced.ok)
            self.assertIn("--- a/calculator.py", replaced.output)
            self.assertIn("+    return a - b", replaced.output)
            self.assertIn("Hook note", replaced.output)
            self.assertTrue(written.ok)
            self.assertIn("--- a/notes.txt", written.output)
            self.assertIn("+done", written.output)

    def test_replace_in_file_rejects_missing_duplicate_and_bad_snippets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "sample.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
            tools = RepoTools(repo)

            missing = tools.run("replace_in_file", {"path": "sample.py", "old": "y = 2", "new": "z = 3"})
            duplicate = tools.run("replace_in_file", {"path": "sample.py", "old": "x = 1", "new": "x = 2"})
            bad = tools.run("replace_in_file", {"path": "sample.py", "old": 1, "new": "x = 2"})

            self.assertFalse(missing.ok)
            self.assertIn("old text not found", missing.output)
            self.assertFalse(duplicate.ok)
            self.assertIn("old text occurs 2 times", duplicate.output)
            self.assertFalse(bad.ok)
            self.assertIn("requires string old and new", bad.output)

    def test_security_hooks_block_unsafe_paths_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            outside = base / "outside.py"
            outside.write_text("outside", encoding="utf-8")
            (repo / "node_modules").mkdir()
            tools = RepoTools(repo)

            results = [
                tools.run("read_file", {"path": "../outside.py"}),
                tools.run("read_file", {"path": str(outside)}),
                tools.run("write_file", {"path": ".env", "content": "SECRET=1\n"}),
                tools.run("write_file", {"path": ".git/config", "content": "config\n"}),
                tools.run("list_files", {"path": "node_modules"}),
                tools.run("run_tests", {"command": "python3 -m unittest discover -s tests -q | rm -rf /"}),
            ]

            for result in results:
                self.assertFalse(result.ok)
                self.assertTrue(result.blocked, result.output)

    def test_run_tests_blocks_external_discover_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            outside_tests = base / "outside_tests"
            marker = base / "executed.txt"
            repo.mkdir()
            outside_tests.mkdir()
            (outside_tests / "test_external.py").write_text(
                "import unittest\n"
                "from pathlib import Path\n\n"
                "class ExternalTests(unittest.TestCase):\n"
                "    def test_external_execution(self):\n"
                f"        Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
                encoding="utf-8",
            )

            result = RepoTools(repo).run("run_tests", {"command": f"python3 -m unittest discover -s {outside_tests} -q"})

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertIn("escapes repository root", result.output)
            self.assertFalse(marker.exists())

    def test_test_command_path_validation_is_runner_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            outside = base / "outside_tests"
            repo.mkdir()
            outside.mkdir()

            self.assertEqual(validate_test_command("pytest -s", repo_root=repo), ["pytest", "-s"])
            with self.assertRaisesRegex(HookViolation, "escapes repository root"):
                validate_test_command(f"pytest {outside}", repo_root=repo)
            with self.assertRaisesRegex(HookViolation, "escapes repository root"):
                validate_test_command(f"pytest --junitxml={outside / 'report.xml'}", repo_root=repo)
            with self.assertRaisesRegex(HookViolation, "escapes repository root"):
                validate_test_command(f"pytest --junitxml {outside / 'report.xml'}", repo_root=repo)
            with self.assertRaisesRegex(HookViolation, "escapes repository root"):
                validate_test_command(f"pytest --custom-output={outside / 'report.xml'}", repo_root=repo)
            with self.assertRaisesRegex(HookViolation, "escapes repository root"):
                validate_test_command(f"pytest --cov-report=html:{outside / 'coverage'}", repo_root=repo)

    def test_retrieve_context_skips_protected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "credentials.json").write_text('{"token": "needle_secret"}\n', encoding="utf-8")
            (repo / "safe.py").write_text("def safe():\n    return 'ok'\n", encoding="utf-8")
            tools = RepoTools(repo)

            read_result = tools.run("read_file", {"path": "credentials.json"})
            retrieve_result = tools.run("retrieve_context", {"query": "needle_secret", "top_k": 3})

            self.assertFalse(read_result.ok)
            self.assertTrue(read_result.blocked)
            self.assertTrue(retrieve_result.ok)
            self.assertNotIn("needle_secret", retrieve_result.output)
            self.assertNotIn("credentials.json", retrieve_result.output)
            self.assertIn("No relevant files found", retrieve_result.output)

    def test_security_hooks_block_symlink_paths(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            repo = Path(repo_tmp)
            outside = Path(outside_tmp) / "outside.py"
            outside.write_text("outside", encoding="utf-8")
            try:
                (repo / "link.py").symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink not supported: {exc}")

            result = RepoTools(repo).run("read_file", {"path": "link.py"})

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertIn("symlink", result.output)

    def test_run_tests_reports_stdout_stderr_and_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            tests = repo / "tests"
            tests.mkdir()
            test_file = tests / "test_sample.py"
            test_file.write_text(
                "import unittest\n\n"
                "class SampleTests(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertEqual(1 + 1, 2)\n",
                encoding="utf-8",
            )
            tools = RepoTools(repo)

            passed = tools.run("run_tests", {"command": "python3 -m unittest discover -s tests -q"})
            test_file.write_text(
                "import unittest\n\n"
                "class SampleTests(unittest.TestCase):\n"
                "    def test_fail(self):\n"
                "        self.assertEqual(1 + 1, 3)\n",
                encoding="utf-8",
            )
            failed = tools.run("run_tests", {"command": "python3 -m unittest discover -s tests -q"})

            self.assertTrue(passed.ok, passed.output)
            self.assertIn("exit_status: 0", passed.output)
            self.assertIn("stdout:", passed.output)
            self.assertIn("stderr:", passed.output)
            self.assertFalse(failed.ok)
            self.assertIn("exit_status: 1", failed.output)
            self.assertIn("Hook note", failed.output)

    def test_git_diff_and_finish_do_not_throw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            tools = RepoTools(repo)

            diff = tools.run("git_diff", {})
            finish = tools.run("finish", {"summary": "complete"})

            self.assertIsInstance(diff.output, str)
            self.assertTrue(finish.ok)
            self.assertEqual(finish.output, "complete")

    def test_grep_rejects_invalid_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "sample.py").write_text("x = 1\n", encoding="utf-8")
            result = RepoTools(repo).run("grep", {"pattern": "[", "path": "."})

            self.assertFalse(result.ok)
            self.assertIn("Invalid regex", result.output)


if __name__ == "__main__":
    unittest.main()
