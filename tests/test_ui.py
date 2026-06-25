from __future__ import annotations

import io
import time
import threading
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
from my_agent.hitl import HitlToolRegistry
from my_agent.plan import AgentMode, PlanState, PlanStatus, PlanTask, TaskStatus
from my_agent.team import ExecutionStep, StepStatus, TeamState, TeamStatus
from my_agent.ui import AgentRepl, PlainRenderer


def fake_config(trace_dir: Path | None = None, **overrides: object) -> AgentConfig:
    resolved_trace_dir = trace_dir or Path("traces")
    values = {
        "provider": "fake",
        "api_key": "",
        "base_url": None,
        "model": "fake",
        "temperature": 0.0,
        "max_steps": 8,
        "command_timeout": 60,
        "trace_dir": resolved_trace_dir,
        "memory_dir": resolved_trace_dir.parent / "memory",
        "use_fake_llm": True,
    }
    values.update(overrides)
    return AgentConfig(**values)


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
        self.assertIn("/memory", text)
        self.assertIn("/save", text)
        self.assertIn("/plan", text)
        self.assertIn("/team", text)
        self.assertIn("/hitl", text)
        self.assertIn("/mode", text)
        self.assertIn("team", text)
        self.assertIn("read_file", text)
        self.assertIn("compression trigger", text)
        self.assertIn("No conversation history was compacted", text)
        self.assertIn("Extracted 0 facts, cleared 0 short-term entries", text)
        self.assertIn("Latest trace: none", text)
        self.assertEqual(errors.getvalue(), "")

    def test_hitl_command_toggles_session_approval_and_tools_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "sample.py").write_text("x = 1\n", encoding="utf-8")
            output = io.StringIO()
            errors = io.StringIO()
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(repo / "traces", hitl_enabled=False),
                trace_dir=repo / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO("/hitl\n/hitl on\n/tools\n/hitl off\n/quit\n"),
            )
            self.assertIsInstance(repl._tools.registry, HitlToolRegistry)
            repl._load_tools = mock.Mock(side_effect=AssertionError("hitl toggle must not reload tools"))  # type: ignore[method-assign]

            exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("HITL approval is off", text)
        self.assertIn("HITL approval enabled.", text)
        self.assertIn("approval", text)
        self.assertIn("write_file", text)
        self.assertIn("ask", text)
        self.assertIn("HITL approval disabled. Cleared HITL approve-all grants.", text)
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
        self.assertIn("short-term:", text)
        self.assertEqual(errors.getvalue(), "")

    def test_memory_save_memory_clear_commands_share_persistent_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            output = io.StringIO()
            errors = io.StringIO()
            config = fake_config(base / "traces", memory_dir=base / "memory")
            repl = AgentRepl(
                repo_path=repo,
                config=config,
                trace_dir=base / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO(
                    "/save 用户偏好：回答中文，先给结论\n"
                    "/memory\n"
                    "读取 calculator.py\n"
                    "/clear\n"
                    "/memory\n"
                    "/quit\n"
                ),
            )

            exit_code = repl.run(show_banner=False)

            self.assertEqual(exit_code, 0)
            text = output.getvalue()
            self.assertIn("Saved memory:", text)
            self.assertIn("Memory", text)
            self.assertIn("用户偏好：回答中文，先给结论", text)
            self.assertIn("Extracted", text)
            self.assertIn("cleared", text)
            self.assertIn("short-term: 0 entries", text)
            self.assertEqual(errors.getvalue(), "")

            reopened_output = io.StringIO()
            reopened = AgentRepl(
                repo_path=repo,
                config=config,
                trace_dir=base / "traces",
                renderer=PlainRenderer(output=reopened_output, errors=io.StringIO()),
                input_stream=io.StringIO("/memory\n/quit\n"),
            )
            reopened.run(show_banner=False)
            self.assertIn("用户偏好：回答中文，先给结论", reopened_output.getvalue())

    def test_clear_reports_fact_extraction_failure_but_still_clears(self) -> None:
        class FailingFactLLM:
            supports_tools = True

            def chat(self, messages: list[object], tools: list[dict[str, object]] | None = None) -> object:
                raise RuntimeError("fact extraction unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            (repo / "sample.py").write_text("x = 1\n", encoding="utf-8")
            output = io.StringIO()
            errors = io.StringIO()
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(base / "traces", memory_dir=base / "memory"),
                trace_dir=base / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO("/clear\n/memory\n/quit\n"),
            )
            repl._memory.compressor.llm = FailingFactLLM()  # type: ignore[assignment]
            repl._memory.append_user_message("用户偏好：回答中文")

            exit_code = repl.run(show_banner=False)

            self.assertEqual(exit_code, 0)
            text = output.getvalue()
            self.assertIn("Fact extraction failed; cleared 1 short-term entries.", text)
            self.assertIn("short-term: 0 entries", text)
            self.assertEqual(errors.getvalue(), "")

    def test_repl_extracts_session_facts_when_input_stream_ends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            output = io.StringIO()
            errors = io.StringIO()
            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(base / "traces", memory_dir=base / "memory"),
                trace_dir=base / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO(""),
            )
            reasons: list[str] = []

            def record_extract(*, reason: str, run_id: str = "") -> list[object]:
                reasons.append(reason)
                return []

            repl._memory.extract_facts = record_extract  # type: ignore[method-assign]

            exit_code = repl.run(show_banner=False)

            self.assertEqual(exit_code, 0)
            self.assertEqual(reasons, ["session_end"])
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

    def test_team_command_runs_team_agent(self) -> None:
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
                input_stream=io.StringIO("/team 先检查 calculator.py，再修复 subtract，并运行测试\n/trace\n/quit\n"),
            )

            exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("team started:", text)
        self.assertIn("reviewing", text)
        self.assertIn("completed", text)
        self.assertIn("team succeeded:", text)
        self.assertIn("Team succeeded", text)
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

    def test_cancel_command_requests_current_task_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_runtime_repo(repo)
            output = io.StringIO()
            errors = io.StringIO()
            started = threading.Event()

            def fake_run_agent(**kwargs: object) -> SimpleNamespace:
                token = kwargs["cancellation_token"]
                started.set()
                deadline = time.monotonic() + 2
                while not token.is_cancelled() and time.monotonic() < deadline:  # type: ignore[attr-defined]
                    time.sleep(0.001)
                return SimpleNamespace(trace_path=repo / "traces" / "trace.jsonl", final_answer="Cancelled.")

            repl = AgentRepl(
                repo_path=repo,
                config=fake_config(repo / "traces"),
                trace_dir=repo / "traces",
                renderer=PlainRenderer(output=output, errors=errors),
                input_stream=io.StringIO("Long task\n/cancel\n/quit\n"),
            )

            with mock.patch("my_agent.ui.repl.run_agent", side_effect=fake_run_agent):
                exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertTrue(started.is_set())
        self.assertIn("Cancellation requested.", text)
        self.assertIn("Cancelled.", text)
        self.assertEqual(errors.getvalue(), "")

    def test_mode_command_accepts_team_mode(self) -> None:
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
                input_stream=io.StringIO("/mode team\nDo task\n/quit\n"),
            )
            final_state = SimpleNamespace(trace_path=repo / "traces" / "trace.jsonl", final_answer="done")

            with mock.patch("my_agent.ui.repl.run_agent", return_value=final_state) as run_agent_mock:
                exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_agent_mock.call_args.kwargs["mode"], AgentMode.TEAM)
        self.assertIn("Mode set to team.", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_team_command_without_task_sets_next_mode(self) -> None:
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
                input_stream=io.StringIO("/team\nDo task\n/quit\n"),
            )
            final_state = SimpleNamespace(trace_path=repo / "traces" / "trace.jsonl", final_answer="done")

            with mock.patch("my_agent.ui.repl.run_agent", return_value=final_state) as run_agent_mock:
                exit_code = repl.run(show_banner=False)

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_agent_mock.call_args.kwargs["mode"], AgentMode.TEAM)
        self.assertIn("Next task will use Multi-Agent team orchestration.", output.getvalue())
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

    def test_renderer_displays_team_status_icons(self) -> None:
        output = io.StringIO()
        renderer = PlainRenderer(output=output)
        team = TeamState.create(goal="goal", summary="summary", steps=[ExecutionStep("step_1", "Inspect", "Inspect")])
        team.status = TeamStatus.RUNNING

        renderer.team_started(team)
        team.steps[0].status = StepStatus.REVIEWING
        renderer.team_step_updated(team.steps[0], team_id=team.id)
        team.steps[0].status = StepStatus.COMPLETED
        team.status = TeamStatus.SUCCEEDED
        renderer.team_completed(team)

        text = output.getvalue()
        self.assertIn("team started:", text)
        self.assertIn("○ step_1", text)
        self.assertIn("? step_1 reviewing", text)
        self.assertIn("team succeeded:", text)

    def test_renderer_has_ascii_status_fallback(self) -> None:
        output = io.StringIO()
        renderer = PlainRenderer(output=output, unicode_icons=False)
        task = PlanTask("task_1", "Inspect", "Inspect", status=TaskStatus.FAILED)

        renderer.plan_task_updated(task, plan_id="plan_1")

        self.assertIn("[failed] task_1 failed Inspect", output.getvalue())


if __name__ == "__main__":
    unittest.main()
