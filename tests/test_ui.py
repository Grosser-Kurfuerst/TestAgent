from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.plan import AgentMode, PlanState, PlanStatus, PlanTask, TaskStatus
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
    (repo / "AGENT.md").write_text(
        "# Test rules\n\n- Run `python -m unittest discover -s tests -q` after changing code.\n",
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
        self.assertIn("TuraCLI", text)
        self.assertIn("version: 0.1.0", text)
        self.assertIn("/tools", text)
        self.assertIn("/plan", text)
        self.assertIn("/mode", text)
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

    def test_plan_command_runs_plan_agent(self) -> None:
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
                input_stream=io.StringIO("/plan 先检查 calculator.py，再修复 subtract，并运行测试\n/trace\n/quit\n"),
            )

            exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("plan started:", text)
        self.assertIn("task_1", text)
        self.assertIn("running", text)
        self.assertIn("succeeded", text)
        self.assertIn("plan succeeded:", text)
        self.assertIn("Plan succeeded", text)
        self.assertIn("Latest trace:", text)
        self.assertEqual(errors.getvalue(), "")

    def test_plan_command_repairs_surrogateescape_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_runtime_repo(repo)
            output = io.StringIO()
            errors = io.StringIO()
            raw_command = "/plan " + "先检查calculator.py 再进行测试".encode("utf-8").decode(
                "ascii", errors="surrogateescape"
            )
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(repo / "traces"),
                trace_dir=repo / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO(raw_command + "\n/quit\n"),
            )

            exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        self.assertIn("plan started:", output.getvalue())
        self.assertIn("plan succeeded:", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_mode_command_changes_routing_mode(self) -> None:
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
                input_stream=io.StringIO("/mode plan\nDo task\n/quit\n"),
            )
            final_state = SimpleNamespace(trace_path=repo / "traces" / "trace.jsonl", final_answer="done")

            with mock.patch("my_agent.ui.repl.run_agent", return_value=final_state) as run_agent_mock:
                exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_agent_mock.call_args.kwargs["mode"], AgentMode.PLAN)
        self.assertIn("Mode set to plan.", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_renderer_displays_plan_status_icons(self) -> None:
        output = io.StringIO()
        renderer = PlainRenderer(output=output)
        plan = PlanState.create(goal="goal", summary="summary", tasks=[PlanTask("task_1", "Inspect", "Inspect")])
        plan.status = PlanStatus.SUCCEEDED

        renderer.plan_started(plan)
        plan.tasks[0].status = TaskStatus.RUNNING
        renderer.plan_task_updated(plan.tasks[0], plan_id=plan.id)
        plan.tasks[0].status = TaskStatus.SUCCEEDED
        renderer.plan_completed(plan)

        text = output.getvalue()
        self.assertIn("○ task_1", text)
        self.assertIn("▶ task_1 running", text)
        self.assertIn("plan succeeded:", text)

    def test_renderer_has_ascii_status_fallback(self) -> None:
        output = io.StringIO()
        renderer = PlainRenderer(output=output, unicode_icons=False)
        task = PlanTask("task_1", "Inspect", "Inspect", status=TaskStatus.FAILED)

        renderer.plan_task_updated(task, plan_id="plan_1")

        self.assertIn("[failed] task_1 failed Inspect", output.getvalue())


if __name__ == "__main__":
    unittest.main()
