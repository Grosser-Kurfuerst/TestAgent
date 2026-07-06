from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests._path import add_src_to_path
from tests.ui.test_mcp_repl import mcp_server_config, write_fake_mcp_server_with_logs, write_mcp_config

add_src_to_path()

from my_agent.cli import DEFAULT_TASK_FILE, build_parser, format_task, load_task, main
from my_agent.cli.common import config_from_env as cli_config_from_env


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

    def test_build_parser_includes_chat_command(self) -> None:
        args = build_parser().parse_args(
            ["chat", "--repo", ".", "--mode", "team", "--test-command", "python -m unittest", "--hitl", "--no-banner"]
        )

        self.assertEqual(args.command, "chat")
        self.assertEqual(args.repo, ".")
        self.assertEqual(args.mode, "team")
        self.assertEqual(args.test_command, "python -m unittest")
        self.assertTrue(args.hitl)
        self.assertTrue(args.no_banner)

    def test_build_parser_includes_run_mode(self) -> None:
        args = build_parser().parse_args(["run", "--task-file", str(DEFAULT_TASK_FILE), "--mode", "team", "--no-hitl"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.mode, "team")
        self.assertFalse(args.hitl)

    def test_build_parser_run_mode_defaults_to_config_mode(self) -> None:
        args = build_parser().parse_args(["run", "--task-file", str(DEFAULT_TASK_FILE)])

        self.assertEqual(args.command, "run")
        self.assertIsNone(args.mode)

    def test_cli_config_prints_team_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = _write_env_file(
                Path(tmp),
                "MY_AGENT_LLM_PROVIDER=fake\nAGENTCLI_TEAM_WORKERS=4\nAGENTCLI_AGENT_MODE=team\n",
            )
            stream = io.StringIO()

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with contextlib.redirect_stdout(stream):
                    exit_code = main(["config"])

        self.assertEqual(exit_code, 0)
        output = json.loads(stream.getvalue())
        self.assertEqual(output["agent_mode"], "team")
        self.assertEqual(output["team_worker_count"], 4)
        self.assertEqual(output["team_max_steps"], 12)
        self.assertFalse(output["hitl_enabled"])
        self.assertEqual(output["hitl_medium_risk_mode"], "ask")

    def test_cli_config_prints_allowed_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = _write_env_file(Path(tmp), "MY_AGENT_LLM_PROVIDER=fake\n")
            stream = io.StringIO()
            env = {
                "AGENTCLI_AGENT_MODE": "team",
                "AGENTCLI_TEAM_WORKERS": "5",
                "AGENTCLI_TEAM_MAX_RETRIES": "0",
                "AGENTCLI_HITL": "1",
                "AGENTCLI_HITL_MEDIUM_RISK_MODE": "allow",
                "AGENTCLI_PLAN_PARALLEL": "0",
                "AGENTCLI_PLAN_TASK_BATCH_TIMEOUT_SECONDS": "30",
                "AGENTCLI_TEAM_STEP_BATCH_TIMEOUT_SECONDS": "40",
                "MY_AGENT_MAX_PARALLEL_TOOLS": "2",
                "AGENTCLI_REPO_CONTEXT_BUDGET_TOKENS": "21000",
                "MY_AGENT_TOOL_SCHEMA_BUDGET_TOKENS": "13000",
                "AGENTCLI_MEMORY_PROJECT_KEY": "stream:cli",
                "AGENTCLI_MEMORY_EVOLVER_MODE": "retrieve_select",
                "AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER": "7",
                "AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS": "6",
                "AGENTCLI_MEMORY_EVOLVER_MIN_SCORE": "0.25",
                "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES": "3",
            }

            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                    with contextlib.redirect_stdout(stream):
                        exit_code = main(["config"])

        self.assertEqual(exit_code, 0)
        output = json.loads(stream.getvalue())
        self.assertEqual(output["agent_mode"], "team")
        self.assertEqual(output["team_worker_count"], 5)
        self.assertEqual(output["team_max_retries"], 0)
        self.assertTrue(output["hitl_enabled"])
        self.assertEqual(output["hitl_medium_risk_mode"], "allow")
        self.assertFalse(output["plan_parallel_enabled"])
        self.assertEqual(output["plan_task_batch_timeout_seconds"], 30)
        self.assertEqual(output["team_step_batch_timeout_seconds"], 40)
        self.assertEqual(output["max_parallel_tools"], 2)
        self.assertEqual(output["repo_context_budget_tokens"], 21_000)
        self.assertEqual(output["tool_schema_budget_tokens"], 13_000)
        self.assertEqual(output["memory_project_key"], "stream:cli")
        self.assertEqual(output["memory_evolver_mode"], "retrieve_select")
        self.assertEqual(output["memory_evolver_top_k_per_tier"], 7)
        self.assertEqual(output["memory_evolver_selected_max_items"], 6)
        self.assertEqual(output["memory_evolver_min_score"], 0.25)
        self.assertEqual(output["memory_evolver_min_experience_entries"], 3)

    def test_cli_config_from_env_allows_evolver_environment_overrides(self) -> None:
        config = cli_config_from_env(
            env={
                "MY_AGENT_LLM_PROVIDER": "fake",
                "AGENTCLI_MEMORY_EVOLVER": "1",
                "MY_AGENT_MEMORY_EVOLVER_TOP_K_PER_TIER": "8",
                "AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS": "5",
                "MY_AGENT_MEMORY_EVOLVER_MIN_SCORE": "0.4",
                "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES": "2",
            },
            require_env_file=False,
        )

        self.assertEqual(config.memory_evolver_mode, "retrieve_select")
        self.assertEqual(config.memory_evolver_top_k_per_tier, 8)
        self.assertEqual(config.memory_evolver_selected_max_items, 5)
        self.assertEqual(config.memory_evolver_min_score, 0.4)
        self.assertEqual(config.memory_evolver_min_experience_entries, 2)

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
            self.assertIn("tests=passed", stream.getvalue())
            self.assertIn("Updated subtract", stream.getvalue())
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))
            self.assertEqual(len(list(trace_dir.glob("*.jsonl"))), 1)

    def test_cli_run_reports_tool_schema_capping_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            task_file = base / "task.json"
            task_file.write_text(json.dumps({"repo": str(repo), "task": "Do task"}), encoding="utf-8")
            env_file = _write_env_file(base, "MY_AGENT_LLM_PROVIDER=fake\n")
            stdout = io.StringIO()
            stderr = io.StringIO()

            def capped_run_agent(**kwargs: object) -> object:
                event_sink = kwargs["event_sink"]
                event_sink(  # type: ignore[operator]
                    SimpleNamespace(
                        event="tools.schema_capped",
                        payload={
                            "included_count": 9,
                            "omitted_count": 1,
                            "omitted": ["oversized_project_tool"],
                        },
                    )
                )
                return SimpleNamespace(plan="", review="", final_answer="done", trace_path="trace.jsonl", stop_reason="assistant_final")

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with mock.patch("my_agent.cli.run_agent", side_effect=capped_run_agent):
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exit_code = main(["run", "--task-file", str(task_file), "--trace-dir", str(base / "traces")])

            self.assertEqual(exit_code, 0)
            self.assertIn("Tool schema budget applied: 9 exposed, 1 omitted", stderr.getvalue())
            self.assertIn("oversized_project_tool", stderr.getvalue())

    def test_cli_run_mode_plan_executes_fake_plan_runtime(self) -> None:
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
                        "task": "先检查 calculator.py，再修复 subtract，并运行测试",
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
                    exit_code = main(
                        ["run", "--task-file", str(task_file), "--trace-dir", str(trace_dir), "--mode", "plan"]
                    )

            self.assertEqual(exit_code, 0)
            output = stream.getvalue()
            self.assertIn("Plan generated by FakeLLM", output)
            self.assertIn("status=succeeded", output)
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(list(trace_dir.rglob("*.jsonl"))), 2)

    def test_cli_run_mode_team_executes_fake_team_runtime_and_stats(self) -> None:
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
                        "task": "先检查 calculator.py 和测试，再修复 subtract，并运行测试",
                        "test_command": "python -m unittest discover -s tests -q",
                    }
                ),
                encoding="utf-8",
            )
            trace_dir = base / "traces"
            stream = io.StringIO()
            env_file = _write_env_file(
                base,
                "MY_AGENT_LLM_PROVIDER=fake\nMY_AGENT_MAX_STEPS=2\nAGENTCLI_TEAM_MAX_STEPS=12\n",
            )

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with contextlib.redirect_stdout(stream):
                    exit_code = main(
                        ["run", "--task-file", str(task_file), "--trace-dir", str(trace_dir), "--mode", "team"]
                    )

            self.assertEqual(exit_code, 0)
            output = stream.getvalue()
            self.assertIn("Team plan generated by FakeLLM", output)
            self.assertIn("Team review: status=succeeded", output)
            self.assertIn("Team succeeded", output)
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(list(trace_dir.rglob("*.jsonl"))), 2)

            stats_output = io.StringIO()
            with contextlib.redirect_stdout(stats_output):
                stats_exit = main(["stats", "--trace", str(trace_dir)])

            self.assertEqual(stats_exit, 0)
            self.assertIn("Tool success rate:", stats_output.getvalue())

    def test_cli_run_without_mode_uses_agent_mode_from_config(self) -> None:
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
                        "task": "先检查 calculator.py 和测试，再修复 subtract，并运行测试",
                        "test_command": "python -m unittest discover -s tests -q",
                    }
                ),
                encoding="utf-8",
            )
            trace_dir = base / "traces"
            stream = io.StringIO()
            env_file = _write_env_file(base, "MY_AGENT_LLM_PROVIDER=fake\nAGENTCLI_AGENT_MODE=team\n")

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with contextlib.redirect_stdout(stream):
                    exit_code = main(["run", "--task-file", str(task_file), "--trace-dir", str(trace_dir)])

            self.assertEqual(exit_code, 0)
            output = stream.getvalue()
            self.assertIn("Team plan generated by FakeLLM", output)
            self.assertIn("Team succeeded", output)

    def test_cli_chat_accepts_inline_fake_env_without_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            missing_env = base / ".env.missing"
            env = {
                "MY_AGENT_LLM_PROVIDER": "fake",
                "AGENTCLI_MEMORY_DIR": str(base / "memory"),
            }

            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("my_agent.config.DEFAULT_ENV_FILE", missing_env):
                    with mock.patch("my_agent.cli.AgentRepl") as repl_cls:
                        repl_cls.return_value.run.return_value = 0
                        exit_code = main(["chat", "--repo", str(repo), "--hitl", "--no-banner"])

            self.assertEqual(exit_code, 0)
            config = repl_cls.call_args.kwargs["config"]
            self.assertEqual(config.provider, "fake")
            self.assertTrue(config.use_fake_llm)
            self.assertEqual(config.memory_dir, base / "memory")
            self.assertTrue(config.hitl_enabled)

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

    def test_cli_run_keyboard_interrupt_cancels_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            task_file = base / "task.json"
            task_file.write_text(
                json.dumps({"repo": str(repo), "task": "Long task"}),
                encoding="utf-8",
            )
            env_file = _write_env_file(base, "MY_AGENT_LLM_PROVIDER=fake\n")
            stderr = io.StringIO()
            seen_tokens: list[object] = []

            def interrupting_run_agent(**kwargs: object) -> object:
                seen_tokens.append(kwargs["cancellation_token"])
                raise KeyboardInterrupt()

            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with mock.patch("my_agent.cli.run_agent", side_effect=interrupting_run_agent):
                    with contextlib.redirect_stderr(stderr):
                        exit_code = main(["run", "--task-file", str(task_file), "--trace-dir", str(base / "traces")])

        self.assertEqual(exit_code, 130)
        self.assertIn("Cancelled.", stderr.getvalue())
        self.assertTrue(seen_tokens)
        self.assertTrue(seen_tokens[0].is_cancelled())

    def test_cli_run_sigint_returns_even_when_worker_does_not_exit_after_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            task_file = base / "task.json"
            task_file.write_text(
                json.dumps({"repo": str(repo), "task": "Long task"}),
                encoding="utf-8",
            )
            env_file = _write_env_file(
                base,
                "MY_AGENT_LLM_PROVIDER=fake\nAGENTCLI_TOOL_SHUTDOWN_GRACE_SECONDS=0\n",
            )
            stderr = io.StringIO()
            seen_tokens: list[object] = []

            def non_cooperative_run_agent(**kwargs: object) -> object:
                token = kwargs["cancellation_token"]
                seen_tokens.append(token)
                while not token.is_cancelled():
                    time.sleep(0.001)
                time.sleep(1)
                return SimpleNamespace(plan="", review="", final_answer="", trace_path="", stop_reason="cancelled")

            timer = threading.Timer(0.02, lambda: os.kill(os.getpid(), signal.SIGINT))
            with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                with mock.patch("my_agent.cli.run_agent", side_effect=non_cooperative_run_agent):
                    timer.start()
                    try:
                        with contextlib.redirect_stderr(stderr):
                            started = time.monotonic()
                            exit_code = main(["run", "--task-file", str(task_file), "--trace-dir", str(base / "traces")])
                            elapsed = time.monotonic() - started
                    finally:
                        timer.cancel()

        self.assertEqual(exit_code, 130)
        self.assertLess(elapsed, 0.5)
        self.assertIn("Cancelled.", stderr.getvalue())
        self.assertTrue(seen_tokens)
        self.assertTrue(seen_tokens[0].is_cancelled())

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
        repo = Path(__file__).resolve().parents[2] / "examples" / "sample_repo"
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
        repo = Path(__file__).resolve().parents[2] / "examples" / "sample_repo"
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            exit_code = main(["retrieve", "--repo", str(repo), "--query", "subtract", "--top-k", "1"])

        output = stream.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("## calculator.py", output)
        self.assertNotIn("## tests/test_calculator.py", output)
        self.assertIn("subtract", output)
        self.assertIn("score=", output)

    def test_cli_tools_list_loads_enabled_project_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config_dir = repo / ".agentcli"
            config_dir.mkdir()
            (config_dir / "tools.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "tools": [
                            {
                                "name": "show_python_version",
                                "description": "Show Python version.",
                                "kind": "command",
                                "risk": "execute",
                                "enabled": True,
                                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                                "command": {"argv": ["python", "--version"], "timeout_seconds": 10, "cwd": "."},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stream = io.StringIO()

            with mock.patch.dict(
                os.environ,
                {"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                clear=True,
            ):
                with contextlib.redirect_stdout(stream):
                    exit_code = main(["tools", "list", "--repo", str(repo)])

        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("show_python_version", output)
        self.assertIn("config:project", output)

    def test_cli_tools_list_closes_mcp_servers_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "mcp-pid.txt"
            server = repo / "sticky_mcp_server.py"
            server.write_text(
                textwrap.dedent(
                    f"""
                    import json
                    import os
                    from pathlib import Path
                    import sys
                    import time

                    Path({str(marker)!r}).write_text(str(os.getpid()), encoding="utf-8")
                    tools = [
                        {{
                            "name": "echo",
                            "description": "Echo.",
                            "inputSchema": {{"type": "object", "properties": {{}}, "additionalProperties": False}},
                        }}
                    ]
                    for line in sys.stdin:
                        message = json.loads(line)
                        method = message.get("method")
                        request_id = message.get("id")
                        if method == "initialize":
                            result = {{"protocolVersion": "2024-11-05", "capabilities": {{"tools": {{}}}}}}
                        elif method == "tools/list":
                            result = {{"tools": tools}}
                        elif method == "notifications/initialized":
                            continue
                        else:
                            result = {{}}
                        print(json.dumps({{"jsonrpc": "2.0", "id": request_id, "result": result}}), flush=True)
                        if method == "tools/list":
                            break
                    while True:
                        time.sleep(1)
                    """
                ),
                encoding="utf-8",
            )
            config_dir = repo / ".paicli"
            config_dir.mkdir()
            (config_dir / "mcp.json").write_text(
                json.dumps({"mcpServers": {"sticky": {"command": sys.executable, "args": [str(server)]}}}),
                encoding="utf-8",
            )
            stream = io.StringIO()

            with mock.patch.dict(
                os.environ,
                {"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                clear=True,
            ):
                with contextlib.redirect_stdout(stream):
                    exit_code = main(["tools", "list", "--repo", str(repo)])

            pid = int(marker.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("mcp__sticky__echo", stream.getvalue())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_cli_mcp_status_logs_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "starts.txt"
            script = write_fake_mcp_server_with_logs(repo)
            write_mcp_config(repo, {"fake": mcp_server_config(script, env={"START_MARKER": str(marker)})})
            status_stream = io.StringIO()
            logs_stream = io.StringIO()
            reload_stream = io.StringIO()

            with mock.patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(status_stream):
                    status_exit = main(["mcp", "status", "--repo", str(repo)])
                with contextlib.redirect_stdout(logs_stream):
                    logs_exit = main(["mcp", "logs", "fake", "--repo", str(repo)])
                with contextlib.redirect_stdout(reload_stream):
                    reload_exit = main(["mcp", "reload", "--repo", str(repo)])
            marker_text = marker.read_text(encoding="utf-8")

        self.assertEqual(status_exit, 0)
        self.assertEqual(logs_exit, 0)
        self.assertEqual(reload_exit, 0)
        self.assertIn("fake\tready\tstdio\t1", status_stream.getvalue())
        self.assertIn("fake stderr ready", logs_stream.getvalue())
        self.assertIn("Reloaded MCP servers.", reload_stream.getvalue())
        self.assertIn("fake\tready\tstdio\t1", reload_stream.getvalue())
        self.assertEqual(marker_text, "3")

    def test_cli_mcp_disabled_does_not_start_servers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "starts.txt"
            script = write_fake_mcp_server_with_logs(repo)
            write_mcp_config(repo, {"fake": mcp_server_config(script, env={"START_MARKER": str(marker)})})
            status_stream = io.StringIO()
            logs_stream = io.StringIO()

            with mock.patch.dict(os.environ, {"AGENTCLI_MCP": "0"}, clear=True):
                with contextlib.redirect_stdout(status_stream):
                    status_exit = main(["mcp", "status", "--repo", str(repo)])
                with contextlib.redirect_stdout(logs_stream):
                    logs_exit = main(["mcp", "logs", "fake", "--repo", str(repo)])

        self.assertEqual(status_exit, 0)
        self.assertEqual(logs_exit, 0)
        self.assertIn("MCP servers\n- disabled", status_stream.getvalue())
        self.assertIn("MCP logs: fake\nMCP is disabled.", logs_stream.getvalue())
        self.assertFalse(marker.exists())

    def test_cli_run_and_tools_use_same_dynamic_tool_environment_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            task_file = base / "task.json"
            task_file.write_text(
                json.dumps({"id": "tmp", "source": "local", "repo": str(repo), "task": "List tools."}),
                encoding="utf-8",
            )
            env_file = _write_env_file(base, "MY_AGENT_LLM_PROVIDER=fake\n")
            final_state = SimpleNamespace(plan="p", review="r", final_answer="f", trace_path=base / "trace.jsonl")
            stream = io.StringIO()

            with mock.patch.dict(
                os.environ,
                {"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                clear=True,
            ):
                with mock.patch("my_agent.config.DEFAULT_ENV_FILE", env_file):
                    with mock.patch("my_agent.cli.run_agent", return_value=final_state) as run_agent_mock:
                        with contextlib.redirect_stdout(stream):
                            exit_code = main(["run", "--task-file", str(task_file)])

        self.assertEqual(exit_code, 0)
        config = run_agent_mock.call_args.kwargs["config"]
        self.assertTrue(config.enable_project_tools)
        self.assertEqual(config.tool_config_paths, ())

    def test_cli_rejects_non_positive_top_k(self) -> None:
        repo = Path(__file__).resolve().parents[2] / "examples" / "sample_repo"
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(["retrieve", "--repo", str(repo), "--query", "subtract", "--top-k", "0"])

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("top_k must be >= 1", stderr.getvalue())

    def test_sample_repo_fixture_exists(self) -> None:
        repo = Path(__file__).resolve().parents[2] / "examples" / "sample_repo"

        self.assertTrue((repo / "calculator.py").exists())
        self.assertTrue((repo / "tests" / "test_calculator.py").exists())
        self.assertTrue((repo / "AGENT.md").exists())


def _write_env_file(base: Path, content: str) -> Path:
    env_file = base / ".env"
    if "AGENTCLI_MEMORY_DIR" not in content and "MY_AGENT_MEMORY_DIR" not in content:
        content += f"AGENTCLI_MEMORY_DIR={base / 'memory'}\n"
    env_file.write_text(content, encoding="utf-8")
    return env_file


if __name__ == "__main__":
    unittest.main()
