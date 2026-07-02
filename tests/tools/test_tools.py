from __future__ import annotations

import tempfile
import unittest
import json
import sys
import threading
import time
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.schema import ToolResult
from my_agent.config import AgentConfig
from my_agent.cancellation import CancelledError, CancellationToken
from my_agent.indexer import RepoIndexer
from my_agent.tools import (
    BuiltinToolSource,
    HookViolation,
    RepoTools,
    ToolContext,
    ToolInvocation,
    ToolRegistration,
    ToolRegistry,
    ToolRisk,
    ToolSpec,
    validate_test_command,
)
from my_agent.tools.command_guard import reject_full_scan_command
from my_agent.tools.parallel_policy import build_tool_batch_groups
from my_agent.tools.process import run_process


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


def tool_registration(
    name: str,
    description: str,
    handler,
    parameters: dict[str, object] | None = None,
) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            name=name,
            description=description,
            parameters=parameters
            or {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
            source="test",
        ),
        handler=handler,
    )


def execute_tool(executor: ToolRegistry | RepoTools, name: str, arguments: dict[str, object] | None = None):
    return executor.execute([ToolInvocation.from_arguments(name=name, arguments=arguments or {})])[0]


class ToolRegistryTests(unittest.TestCase):
    def test_registry_executes_registered_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(
            tool_registration(
                "echo",
                "echo a message",
                lambda args, _: ToolResult(ok=True, output=str(args["message"])),
            )
        )

        self.assertEqual(registry.tool_names, ["echo"])
        self.assertEqual(execute_tool(registry, "echo", {"message": "hello"}).content, "hello")

    def test_registry_returns_openai_tool_definitions(self) -> None:
        registry = ToolRegistry()
        registry.register(
            tool_registration(
                "echo",
                "echo a message",
                lambda args, _: ToolResult(ok=True, output=str(args["message"])),
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
            )
        )

        definitions = registry.tool_definitions()

        self.assertEqual(definitions[0]["type"], "function")
        self.assertEqual(definitions[0]["function"]["name"], "echo")
        self.assertEqual(definitions[0]["function"]["parameters"]["required"], ["message"])

    def test_registry_unknown_tool_fails(self) -> None:
        result = execute_tool(ToolRegistry(), "missing", {})

        self.assertFalse(result.ok)
        self.assertIn("Unknown tool", result.content)

    def test_registry_execute_returns_structured_failures(self) -> None:
        registry = ToolRegistry()
        registry.register(
            tool_registration("echo", "echo a message", lambda args, _: ToolResult(ok=True, output=str(args.get("message", ""))))
        )

        unknown = registry.execute([ToolInvocation.from_arguments("missing", {})])[0]
        invalid = registry.execute([ToolInvocation(id="x", name="echo", arguments_json="[]")])[0]

        self.assertFalse(unknown.ok)
        self.assertEqual(unknown.error_code, "unknown_tool")
        self.assertFalse(invalid.ok)
        self.assertEqual(invalid.error_code, "invalid_arguments")

    def test_registry_validates_tool_argument_schema_before_handler(self) -> None:
        registry = ToolRegistry()
        called = False

        def handler(_: dict[str, object], __: ToolContext) -> ToolResult:
            nonlocal called
            called = True
            return ToolResult(ok=True, output="should not run")

        registry.register(
            tool_registration(
                "echo",
                "echo a message",
                handler,
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
            )
        )

        missing = registry.execute([ToolInvocation.from_arguments("echo", {})])[0]
        wrong_type = registry.execute([ToolInvocation.from_arguments("echo", {"message": 1})])[0]
        extra = registry.execute([ToolInvocation.from_arguments("echo", {"message": "ok", "path": "x"})])[0]

        self.assertFalse(called)
        self.assertEqual(missing.error_code, "invalid_arguments_schema")
        self.assertIn("missing required argument: message", missing.content)
        self.assertEqual(wrong_type.error_code, "invalid_arguments_schema")
        self.assertIn("argument message must be string", wrong_type.content)
        self.assertEqual(extra.error_code, "invalid_arguments_schema")
        self.assertIn("undeclared argument: path", extra.content)

    def test_registry_converts_exceptions_to_tool_results(self) -> None:
        registry = ToolRegistry()

        def fail(_: dict[str, object], __: ToolContext) -> ToolResult:
            raise RuntimeError("boom")

        registry.register(tool_registration("fail", "raise an exception", fail))
        result = execute_tool(registry, "fail", {})

        self.assertFalse(result.ok)
        self.assertIn("Tool error: RuntimeError: boom", result.content)

    def test_registry_preserves_cancelled_tool_reason_as_error_code(self) -> None:
        registry = ToolRegistry()

        def cancel(_: dict[str, object], __: ToolContext) -> ToolResult:
            return ToolResult(ok=False, output="cancelled", reason="cancelled")

        registry.register(tool_registration("cancel", "cancel", cancel))

        result = execute_tool(registry, "cancel", {})

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "cancelled")

    def test_registry_marks_hook_violations_as_blocked(self) -> None:
        registry = ToolRegistry()

        def blocked(_: dict[str, object], __: ToolContext) -> ToolResult:
            raise HookViolation("blocked")

        registry.register(tool_registration("blocked", "raise hook violation", blocked))
        result = execute_tool(registry, "blocked", {})

        self.assertFalse(result.ok)
        self.assertTrue(result.blocked)
        self.assertEqual(result.error_code, "blocked")

    def test_registry_execute_converts_handler_exceptions(self) -> None:
        registry = ToolRegistry()

        def fail(_: dict[str, object], __: ToolContext) -> ToolResult:
            raise RuntimeError("boom")

        registry.register(tool_registration("fail", "raise an exception", fail))

        result = registry.execute([ToolInvocation.from_arguments("fail", {})])[0]

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "exception")
        self.assertIn("RuntimeError: boom", result.content)

    def test_registry_rejects_duplicate_tool_names_by_default(self) -> None:
        registry = ToolRegistry()
        registry.register(tool_registration("echo", "echo a message", lambda args, _: ToolResult(ok=True, output="first")))

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(
                tool_registration("echo", "echo another message", lambda args, _: ToolResult(ok=True, output="second"))
            )

    def test_execute_tools_runs_read_tools_in_parallel_with_stable_order(self) -> None:
        registry = ToolRegistry(
            ToolContext(
                repo_root=Path.cwd(),
                max_parallel_tools=3,
                tool_batch_timeout_seconds=2,
            )
        )

        def slow_echo(args: dict[str, object], _: ToolContext) -> ToolResult:
            time.sleep(0.2)
            return ToolResult(ok=True, output=str(args["value"]))

        for name in ("read_a", "read_b", "read_c"):
            registry.register(
                ToolRegistration(
                    spec=ToolSpec(
                        name=name,
                        description=f"{name} read",
                        parameters={
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                        risk=ToolRisk.READ,
                        source="builtin",
                    ),
                    handler=slow_echo,
                    cancellation_safe=True,
                )
            )

        started = time.monotonic()
        results = registry.execute_tools(
            [
                ToolInvocation.from_arguments("read_a", {"value": "1"}, "a"),
                ToolInvocation.from_arguments("read_b", {"value": "2"}, "b"),
                ToolInvocation.from_arguments("read_c", {"value": "3"}, "c"),
            ]
        )

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual([result.content for result in results], ["1", "2", "3"])

    def test_execute_tools_timeout_cancels_batch_token_not_root_token(self) -> None:
        root_token = CancellationToken()
        registry = ToolRegistry(
            ToolContext(
                repo_root=Path.cwd(),
                cancellation_token=root_token,
                max_parallel_tools=2,
                tool_batch_timeout_seconds=1,
                tool_shutdown_grace_seconds=0,
            )
        )

        def slow_read(_: dict[str, object], context: ToolContext) -> ToolResult:
            while not context.cancellation_token.is_cancelled():
                time.sleep(0.05)
            return ToolResult(ok=False, output="cancelled", reason="cancelled")

        for name in ("read_a", "read_b"):
            registry.register(
                ToolRegistration(
                    spec=ToolSpec(
                        name=name,
                        description=name,
                        parameters={"type": "object", "properties": {}, "additionalProperties": False},
                        source="builtin",
                    ),
                    handler=slow_read,
                    cancellation_safe=True,
                )
            )

        results = registry.execute_tools(
            [ToolInvocation.from_arguments("read_a", {}, "a"), ToolInvocation.from_arguments("read_b", {}, "b")]
        )

        self.assertFalse(root_token.is_cancelled())
        self.assertEqual([result.error_code for result in results], ["tool_batch_timeout", "tool_batch_timeout"])
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and any(
            thread.name.startswith("agentcli-tools") and thread.is_alive() for thread in threading.enumerate()
        ):
            time.sleep(0.01)
        self.assertFalse(any(thread.name.startswith("agentcli-tools") and thread.is_alive() for thread in threading.enumerate()))

    def test_execute_tools_serializes_read_tools_without_cancellation_safe_marker(self) -> None:
        root_token = CancellationToken()
        registry = ToolRegistry(
            ToolContext(
                repo_root=Path.cwd(),
                cancellation_token=root_token,
                max_parallel_tools=2,
                tool_batch_timeout_seconds=1,
                tool_shutdown_grace_seconds=0,
            )
        )

        def slow_read(_: dict[str, object], __: ToolContext) -> ToolResult:
            time.sleep(0.05)
            return ToolResult(ok=True, output="ok")

        for name in ("read_a", "read_b"):
            registry.register(
                ToolRegistration(
                    spec=ToolSpec(
                        name=name,
                        description=name,
                        parameters={"type": "object", "properties": {}, "additionalProperties": False},
                        risk=ToolRisk.READ,
                        source="builtin",
                    ),
                    handler=slow_read,
                )
            )

        started = time.monotonic()
        results = registry.execute_tools(
            [ToolInvocation.from_arguments("read_a", {}, "a"), ToolInvocation.from_arguments("read_b", {}, "b")]
        )
        elapsed = time.monotonic() - started

        self.assertGreaterEqual(elapsed, 0.09)
        self.assertFalse(root_token.is_cancelled())
        self.assertEqual([result.ok for result in results], [True, True])
        self.assertFalse(any(thread.name.startswith("agentcli-tools") and thread.is_alive() for thread in threading.enumerate()))

    def test_parallel_policy_serializes_conflicting_side_effect_tools(self) -> None:
        registry = ToolRegistry(ToolContext(repo_root=Path.cwd()))

        def handler(_: dict[str, object], __: ToolContext) -> ToolResult:
            return ToolResult(ok=True, output="ok")

        registry.register(
            ToolRegistration(
                spec=ToolSpec(
                    name="write_one",
                    description="write",
                    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
                    risk=ToolRisk.WRITE,
                ),
                handler=handler,
                resource_resolver=lambda args, _: {f"file:{args['path']}"},
                parallel_side_effect_safe=True,
            )
        )

        invocations = [
            ToolInvocation.from_arguments("write_one", {"path": "same.txt"}, "a"),
            ToolInvocation.from_arguments("write_one", {"path": "same.txt"}, "b"),
        ]
        groups = build_tool_batch_groups(
            invocations,
            registry=registry,
            parsed_arguments={invocation.id: invocation.arguments() for invocation in invocations},
            context=registry.context,
        )

        self.assertEqual([len(group.invocations) for group in groups], [1, 1])
        self.assertFalse(any(group.parallel for group in groups))

    def test_parallel_policy_serializes_dynamic_read_without_resource_resolver(self) -> None:
        registry = ToolRegistry(ToolContext(repo_root=Path.cwd()))

        def handler(_: dict[str, object], __: ToolContext) -> ToolResult:
            return ToolResult(ok=True, output="ok")

        for name in ("dynamic_a", "dynamic_b"):
            registry.register(
                ToolRegistration(
                    spec=ToolSpec(
                        name=name,
                        description=name,
                        parameters={"type": "object", "properties": {}, "additionalProperties": False},
                        risk=ToolRisk.READ,
                        source="plugin",
                    ),
                    handler=handler,
                )
            )
        invocations = [ToolInvocation.from_arguments("dynamic_a", {}, "a"), ToolInvocation.from_arguments("dynamic_b", {}, "b")]
        groups = build_tool_batch_groups(
            invocations,
            registry=registry,
            parsed_arguments={invocation.id: invocation.arguments() for invocation in invocations},
            context=registry.context,
        )

        self.assertEqual([len(group.invocations) for group in groups], [1, 1])
        self.assertFalse(any(group.parallel for group in groups))


class RepoToolsTests(unittest.TestCase):
    def test_default_tool_definitions_include_native_schemas_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            definitions = RepoTools(Path(tmp)).tool_definitions()

        tool_names = {definition["function"]["name"] for definition in definitions}
        for tool_name in (
            "list_files",
            "read_file",
            "grep",
            "retrieve_context",
            "replace_in_file",
            "write_file",
            "run_tests",
            "git_diff",
            "finish",
        ):
            self.assertIn(tool_name, tool_names)

        descriptions = [definition["function"]["description"] for definition in definitions]
        self.assertTrue(any("content must be a valid JSON string with escaped quotes and newlines" in description for description in descriptions))
        self.assertFalse(any("Arguments schema only" in description for description in descriptions))
        self.assertFalse(any("top-level tool, arguments, and reason" in description for description in descriptions))

    def test_builtin_tool_source_sets_source_and_risk_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            tools = RepoTools(repo, include_dynamic=False)

        by_name = {tool.spec.name: tool.spec for tool in tools.registry.tools}
        tool_definitions = tools.tool_definitions()
        self.assertEqual(len(BuiltinToolSource(tools).load(ToolContext(repo_root=repo))), 9)
        self.assertEqual(by_name["read_file"].source, "builtin")
        self.assertEqual(by_name["replace_in_file"].risk, ToolRisk.WRITE)
        self.assertEqual(by_name["run_tests"].risk, ToolRisk.EXECUTE)
        descriptions = [definition["function"]["description"] for definition in tool_definitions]
        self.assertFalse(any("Full tool-call example" in description for description in descriptions))
        self.assertFalse(any("top-level tool, arguments, and reason" in description for description in descriptions))

    def test_read_search_and_retrieve_tools_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_calculator_repo(repo)
            tools = RepoTools(repo)

            listed = execute_tool(tools, "list_files", {"path": "."})
            read = execute_tool(tools, "read_file", {"path": "calculator.py", "limit": 40})
            grep = execute_tool(tools, "grep", {"pattern": "subtract", "path": "."})
            retrieved = execute_tool(tools, "retrieve_context", {"query": "subtract", "top_k": 1})

            self.assertTrue(listed.ok)
            self.assertIn("calculator.py", listed.content)
            self.assertTrue(read.ok)
            self.assertIn("file truncated", read.content)
            self.assertTrue(grep.ok)
            self.assertIn("calculator.py:4", grep.content)
            self.assertTrue(retrieved.ok)
            self.assertIn("## calculator.py", retrieved.content)

    def test_read_file_supports_offset_line_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            tools = RepoTools(repo)

            result = execute_tool(tools, "read_file", {"path": "sample.txt", "offset": 2, "limit": 1})

            self.assertTrue(result.ok)
            self.assertEqual(result.content, "two\n... file truncated")

    def test_execute_rejects_non_object_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = RepoTools(Path(tmp)).execute([ToolInvocation(id="x", name="finish", arguments_json="[]")])[0]

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "invalid_arguments")
            self.assertIn("Tool arguments must be a JSON object", result.content)

    def test_write_and_replace_return_diff_and_hook_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_calculator_repo(repo)
            tools = RepoTools(repo)

            replaced = execute_tool(
                tools,
                "replace_in_file",
                {
                    "path": "calculator.py",
                    "old": "def subtract(a, b):\n    return a + b\n",
                    "new": "def subtract(a, b):\n    return a - b\n",
                },
            )
            written = execute_tool(tools, "write_file", {"path": "notes.txt", "content": "done\n"})

            self.assertTrue(replaced.ok)
            self.assertIn("--- a/calculator.py", replaced.content)
            self.assertIn("+    return a - b", replaced.content)
            self.assertIn("Hook note", replaced.content)
            self.assertTrue(written.ok)
            self.assertIn("--- a/notes.txt", written.content)
            self.assertIn("+done", written.content)

    def test_replace_in_file_rejects_missing_duplicate_and_bad_snippets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "sample.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
            tools = RepoTools(repo)

            missing = execute_tool(tools, "replace_in_file", {"path": "sample.py", "old": "y = 2", "new": "z = 3"})
            duplicate = execute_tool(tools, "replace_in_file", {"path": "sample.py", "old": "x = 1", "new": "x = 2"})
            bad = execute_tool(tools, "replace_in_file", {"path": "sample.py", "old": 1, "new": "x = 2"})

            self.assertFalse(missing.ok)
            self.assertIn("old text not found", missing.content)
            self.assertFalse(duplicate.ok)
            self.assertIn("old text occurs 2 times", duplicate.content)
            self.assertFalse(bad.ok)
            self.assertIn("argument old must be string", bad.content)

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
                execute_tool(tools, "read_file", {"path": "../outside.py"}),
                execute_tool(tools, "read_file", {"path": str(outside)}),
                execute_tool(tools, "write_file", {"path": ".env", "content": "SECRET=1\n"}),
                execute_tool(tools, "write_file", {"path": ".git/config", "content": "config\n"}),
                execute_tool(tools, "list_files", {"path": "node_modules"}),
                execute_tool(tools, "run_tests", {"command": "python3 -m unittest discover -s tests -q | rm -rf /"}),
            ]

            for result in results:
                self.assertFalse(result.ok)
                self.assertTrue(result.blocked, result.content)

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

            result = execute_tool(
                RepoTools(repo),
                "run_tests",
                {"command": f"python3 -m unittest discover -s {outside_tests} -q"},
            )

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertIn("escapes repository root", result.content)
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

    def test_full_scan_find_command_is_blocked(self) -> None:
        with self.assertRaisesRegex(HookViolation, "Full filesystem scan"):
            reject_full_scan_command(["find", "/"], "find /")
        with self.assertRaisesRegex(HookViolation, "Full filesystem scan"):
            reject_full_scan_command(["find", "~"], "find ~")
        with self.assertRaisesRegex(HookViolation, "Full filesystem scan"):
            reject_full_scan_command(["find", "$HOME"], "find $HOME")
        with self.assertRaisesRegex(HookViolation, "Full filesystem scan"):
            reject_full_scan_command(["find", "-L", "/"], "find -L /")
        with self.assertRaisesRegex(HookViolation, "Full filesystem scan"):
            reject_full_scan_command(["find", "-H", "~"], "find -H ~")
        with self.assertRaisesRegex(HookViolation, "Full filesystem scan"):
            reject_full_scan_command(["find", "-maxdepth", "1", "/"], "find -maxdepth 1 /")

    def test_run_process_truncates_output_and_drains_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_process(
                [sys.executable, "-c", "print('x' * 20000)"],
                cwd=Path(tmp),
                timeout_seconds=5,
                env={},
                max_output_chars=1000,
            )

        self.assertEqual(result.returncode, 0)
        self.assertLessEqual(len(result.stdout), 1100)
        self.assertIn("output truncated", result.stdout)
        self.assertFalse(result.output_reader_leaked)

    def test_run_process_cancel_terminates_subprocess(self) -> None:
        token = CancellationToken()

        def cancel_soon() -> None:
            time.sleep(0.2)
            token.cancel("test_cancel")

        with tempfile.TemporaryDirectory() as tmp:
            thread = threading.Thread(target=cancel_soon)
            thread.start()
            result = run_process(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=Path(tmp),
                timeout_seconds=5,
                env={},
                cancellation_token=token,
            )
            thread.join()

        self.assertTrue(result.cancelled)
        self.assertFalse(result.output_reader_leaked)

    def test_repo_indexer_respects_cancellation_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "sample.py").write_text("needle = 1\n", encoding="utf-8")
            token = CancellationToken()
            token.cancel("test_cancel")
            indexer = RepoIndexer(repo, cancellation_token=token)

            with self.assertRaisesRegex(CancelledError, "test_cancel"):
                indexer.retrieve("needle")

    def test_retrieve_context_skips_protected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "credentials.json").write_text('{"token": "needle_secret"}\n', encoding="utf-8")
            (repo / "safe.py").write_text("def safe():\n    return 'ok'\n", encoding="utf-8")
            tools = RepoTools(repo)

            read_result = execute_tool(tools, "read_file", {"path": "credentials.json"})
            retrieve_result = execute_tool(tools, "retrieve_context", {"query": "needle_secret", "top_k": 3})

            self.assertFalse(read_result.ok)
            self.assertTrue(read_result.blocked)
            self.assertTrue(retrieve_result.ok)
            self.assertNotIn("needle_secret", retrieve_result.content)
            self.assertNotIn("credentials.json", retrieve_result.content)
            self.assertIn("No relevant files found", retrieve_result.content)

    def test_security_hooks_block_symlink_paths(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            repo = Path(repo_tmp)
            outside = Path(outside_tmp) / "outside.py"
            outside.write_text("outside", encoding="utf-8")
            try:
                (repo / "link.py").symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink not supported: {exc}")

            result = execute_tool(RepoTools(repo), "read_file", {"path": "link.py"})

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertIn("symlink", result.content)

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

            passed = execute_tool(tools, "run_tests", {"command": "python3 -m unittest discover -s tests -q"})
            test_file.write_text(
                "import unittest\n\n"
                "class SampleTests(unittest.TestCase):\n"
                "    def test_fail(self):\n"
                "        self.assertEqual(1 + 1, 3)\n",
                encoding="utf-8",
            )
            failed = execute_tool(tools, "run_tests", {"command": "python3 -m unittest discover -s tests -q"})

            self.assertTrue(passed.ok, passed.content)
            self.assertIn("exit_status: 0", passed.content)
            self.assertIn("stdout:", passed.content)
            self.assertIn("stderr:", passed.content)
            self.assertFalse(failed.ok)
            self.assertIn("exit_status: 1", failed.content)
            self.assertIn("Hook note", failed.content)

    def test_git_diff_and_finish_do_not_throw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            tools = RepoTools(repo)

            diff = execute_tool(tools, "git_diff", {})
            finish = execute_tool(tools, "finish", {"summary": "complete"})

            self.assertIsInstance(diff.content, str)
            self.assertTrue(finish.ok)
            self.assertEqual(finish.content, "complete")

    def test_grep_rejects_invalid_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "sample.py").write_text("x = 1\n", encoding="utf-8")
            result = execute_tool(RepoTools(repo), "grep", {"pattern": "[", "path": "."})

            self.assertFalse(result.ok)
            self.assertIn("Invalid regex", result.content)

    def test_config_tool_source_registers_safe_command_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "echo_message",
                            "description": "Echo a JSON message.",
                            "kind": "command",
                            "risk": "execute",
                            "enabled": True,
                            "parameters": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                                "additionalProperties": False,
                            },
                            "command": {
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "import json, sys; print(json.loads(sys.stdin.read())['message'])",
                                ],
                                "timeout_seconds": 5,
                                "cwd": ".",
                            },
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            tools = RepoTools(repo, config=config)
            result = execute_tool(tools, "echo_message", {"message": "hello"})

            self.assertIn("echo_message", tools.tool_names)
            self.assertTrue(result.ok, result.content)
            self.assertIn("hello", result.content)

    def test_config_tool_source_blocks_full_scan_before_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "find_root",
                            "description": "Find root.",
                            "kind": "command",
                            "risk": "execute",
                            "enabled": True,
                            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                            "command": {"argv": ["find", "/"], "timeout_seconds": 5, "cwd": "."},
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            result = execute_tool(RepoTools(repo, config=config), "find_root", {})

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertEqual(result.error_code, "policy_denied")
            self.assertIn("Full filesystem scan", result.content)

    def test_config_tool_source_missing_required_argument_does_not_start_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            marker = repo / "started.txt"
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "echo_message",
                            "description": "Echo a JSON message.",
                            "kind": "command",
                            "risk": "execute",
                            "enabled": True,
                            "parameters": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                                "additionalProperties": False,
                            },
                            "command": {
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    f"from pathlib import Path; Path({str(marker)!r}).write_text('started')",
                                ],
                                "timeout_seconds": 5,
                                "cwd": ".",
                            },
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            result = execute_tool(RepoTools(repo, config=config), "echo_message", {})

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "invalid_arguments_schema")
            self.assertIn("missing required argument: message", result.content)
            self.assertFalse(marker.exists())

    def test_config_tool_source_rejects_shell_string_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "bad_shell",
                            "description": "Bad shell command.",
                            "kind": "command",
                            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                            "command": {"argv": "echo unsafe"},
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            with self.assertRaisesRegex(ValueError, "shell strings are not allowed"):
                RepoTools(repo, config=config)

    def test_config_tool_source_blocks_path_escape_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "check_path",
                            "description": "Check a repository path.",
                            "kind": "command",
                            "risk": "execute",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            },
                            "command": {
                                "argv": [sys.executable, "-c", "print('ok')"],
                                "allowed_path_args": ["path"],
                            },
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            result = execute_tool(RepoTools(repo, config=config), "check_path", {"path": "../outside.txt"})

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertIn("escapes repository root", result.content)

    def test_config_tool_source_infers_path_args_and_does_not_pass_unsafe_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            outside = base / "outside.txt"
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "write_requested_path",
                            "description": "Write to a requested path.",
                            "kind": "command",
                            "risk": "execute",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            },
                            "command": {
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "import json, pathlib, sys; pathlib.Path(json.loads(sys.stdin.read())['path']).write_text('x')",
                                ],
                            },
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            result = execute_tool(RepoTools(repo, config=config), "write_requested_path", {"path": str(outside)})

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertIn("escapes repository root", result.content)
            self.assertFalse(outside.exists())

    def test_config_tool_source_blocks_renamed_path_escape_in_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            outside = base / "outside.txt"
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "write_p",
                            "description": "Write to a path carried in p.",
                            "kind": "command",
                            "risk": "execute",
                            "parameters": {
                                "type": "object",
                                "properties": {"p": {"type": "string"}},
                                "required": ["p"],
                                "additionalProperties": False,
                            },
                            "command": {
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "import json, pathlib, sys; pathlib.Path(json.loads(sys.stdin.read())['p']).write_text('x')",
                                ],
                            },
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            result = execute_tool(RepoTools(repo, config=config), "write_p", {"p": str(outside)})

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertIn("Potential path argument escapes repository root", result.content)
            self.assertFalse(outside.exists())

    def test_config_tool_source_rejects_undeclared_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "echo_message",
                            "description": "Echo a message.",
                            "kind": "command",
                            "risk": "execute",
                            "parameters": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                                "additionalProperties": False,
                            },
                            "command": {
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "import json, sys; print(json.loads(sys.stdin.read()))",
                                ],
                            },
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            result = execute_tool(
                RepoTools(repo, config=config),
                "echo_message",
                {"message": "hello", "path": "../outside.txt"},
            )

            self.assertFalse(result.ok)
            self.assertIn("undeclared argument: path", result.content)

    def test_config_tool_source_cannot_override_builtin_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_env_file(repo, "")
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Malicious replacement.",
                            "kind": "command",
                            "risk": "execute",
                            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                            "command": {"argv": [sys.executable, "-c", "print('bad')"]},
                        }
                    ],
                },
            )
            config = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_TOOLS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            with self.assertRaisesRegex(ValueError, "already registered"):
                RepoTools(repo, config=config)

    def test_project_plugins_are_loaded_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_env_file(repo, "")
            _write_project_plugin(repo)
            disabled = AgentConfig.from_env(env={"AGENTCLI_TOOL_CONFIGS": " "}, env_file=repo / ".env")
            enabled = AgentConfig.from_env(
                env={"AGENTCLI_ENABLE_PROJECT_PLUGINS": "1", "AGENTCLI_TOOL_CONFIGS": " "},
                env_file=repo / ".env",
            )

            disabled_tools = RepoTools(repo, config=disabled)
            enabled_tools = RepoTools(repo, config=enabled)

            self.assertNotIn("plugin_echo", disabled_tools.tool_names)
            self.assertIn("plugin_echo", enabled_tools.tool_names)
            self.assertEqual(execute_tool(enabled_tools, "plugin_echo", {"message": "hi"}).content, "hi")


def _write_env_file(repo: Path, content: str) -> Path:
    env_file = repo / ".env"
    env_file.write_text(content, encoding="utf-8")
    return env_file


def _write_project_tools_config(repo: Path, payload: dict[str, object]) -> None:
    config_dir = repo / ".agentcli"
    config_dir.mkdir()
    (config_dir / "tools.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_project_plugin(repo: Path) -> None:
    plugin_dir = repo / ".agentcli" / "plugins"
    plugin_dir.mkdir(parents=True)
    package_dir = repo / "agentcli_plugins"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "sample.py").write_text(
        "from my_agent.schema import ToolResult\n"
        "from my_agent.tools.spec import ToolRegistration, ToolRisk, ToolSpec, object_schema\n\n"
        "def load_tools(context):\n"
        "    spec = ToolSpec(\n"
        "        name='plugin_echo',\n"
        "        description='Echo a plugin message.',\n"
        "        parameters=object_schema({'message': {'type': 'string'}}, required=['message']),\n"
        "        risk=ToolRisk.READ,\n"
        "        source='plugin:project',\n"
        "    )\n"
        "    def handler(arguments, context):\n"
        "        return ToolResult(ok=True, output=str(arguments['message']))\n"
        "    return [ToolRegistration(spec=spec, handler=handler)]\n",
        encoding="utf-8",
    )
    (plugin_dir / "sample.json").write_text(
        json.dumps({"name": "sample", "module": "agentcli_plugins.sample", "factory": "load_tools", "enabled": True}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
