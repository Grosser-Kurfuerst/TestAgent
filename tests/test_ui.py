from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.ui import AgentRepl, PlainRenderer


def fake_config(trace_dir: Path | None = None) -> AgentConfig:
    return AgentConfig(
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        max_steps=8,
        command_timeout=60,
        trace_dir=trace_dir or Path("traces"),
        use_fake_llm=True,
    )


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
        "from calculator import subtract\n\n"
        "class CalculatorTests(unittest.TestCase):\n"
        "    def test_subtract(self):\n"
        "        self.assertEqual(subtract(5, 3), 2)\n",
        encoding="utf-8",
    )


class ReplTests(unittest.TestCase):
    def test_repl_banner_help_tools_context_and_quit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "sample.py").write_text("x = 1\n", encoding="utf-8")
            output = io.StringIO()
            errors = io.StringIO()
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(repo / "traces"),
                trace_dir=repo / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO("/help\n/tools\n/context\n/compact focus\n/clear\n/trace\n/quit\n"),
            )

            exit_code = repl.run(show_banner=True)

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("my-agent 0.1.0", text)
        self.assertIn("/tools", text)
        self.assertIn("read_file", text)
        self.assertIn("compression trigger", text)
        self.assertIn("No conversation history was compacted", text)
        self.assertIn("Conversation context cleared", text)
        self.assertIn("Latest trace: none", text)
        self.assertEqual(errors.getvalue(), "")

    def test_repl_task_renders_tools_and_updates_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_runtime_repo(repo)
            output = io.StringIO()
            errors = io.StringIO()
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(repo / "traces"),
                trace_dir=repo / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO("Fix subtract.\n/context\n/quit\n"),
            )

            exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("tool started: retrieve_context", text)
        self.assertIn("tool completed: run_tests", text)
        self.assertIn("Updated subtract", text)
        self.assertIn("conversation:", text)
        self.assertNotIn("conversation: 1 estimated tokens", text)
        self.assertEqual(errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
