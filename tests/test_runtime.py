from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, MessageLike, messages_to_openai
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


def fake_config(trace_dir: Path | None = None, **overrides: object) -> AgentConfig:
    values = {
        "provider": "fake",
        "api_key": "",
        "base_url": None,
        "model": "fake",
        "temperature": 0.0,
        "max_steps": 8,
        "command_timeout": 60,
        "trace_dir": trace_dir or Path("traces"),
        "use_fake_llm": True,
    }
    values.update(overrides)
    return AgentConfig(**values)


def tool_response(name: str, arguments: dict[str, object], call_id: str | None = None) -> ChatResponse:
    raw = json.dumps(arguments, ensure_ascii=False)
    return ChatResponse(
        finish_reason="tool_calls",
        tool_calls=[LLMToolCall(id=call_id or f"call_{name}", name=name, arguments=dict(arguments), arguments_json=raw)],
    )


class RecordingFakeLLM(FakeLLM):
    def __init__(self, chat_responses: list[ChatResponse | str]):
        super().__init__(chat_responses=chat_responses)
        self.requests: list[list[dict[str, object]]] = []

    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        self.requests.append(messages_to_openai(messages))
        if tools is None:
            return ChatResponse(content="compressed summary", finish_reason="stop")
        return super().chat(messages, tools=tools)  # type: ignore[arg-type]


class RuntimeTests(unittest.TestCase):
    def test_run_agent_rejects_zero_max_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_runtime_repo(repo)

            with self.assertRaisesRegex(ValueError, "max_steps must be >= 1"):
                run_agent(
                    repo_path=repo,
                    task="Fix subtract.",
                    config=fake_config(repo / "traces"),
                    llm=FakeLLM(),
                    max_steps=0,
                    trace_dir=repo / "traces",
                )

    def test_fake_llm_repairs_sample_bug_and_writes_native_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="Fix the subtract function so it returns the first number minus the second number.",
                test_command="python -m unittest discover -s tests -q",
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                max_steps=8,
                trace_dir=base / "traces",
            )

            self.assertTrue(state.done)
            self.assertEqual(state.stop_reason, "finish_called")
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))
            self.assertTrue(any(record.call.tool == "run_tests" and record.result.ok for record in state.tool_history))
            self.assertTrue(any(record.call.tool == "finish" for record in state.tool_history))
            self.assertIn("Updated subtract", state.final_answer)
            self.assertIn("tests=passed", state.review)

            self.assertIsNotNone(state.trace_path)
            events = read_trace(state.trace_path)
            event_names = [str(event["event"]) for event in events]
            for expected in (
                "run.started",
                "tools.loaded",
                "repo.indexed",
                "llm.requested",
                "llm.completed",
                "tool.started",
                "tool.completed",
                "run.completed",
            ):
                self.assertIn(expected, event_names)
            self.assertNotIn("tool_call", event_names)
            self.assertNotIn("final_summary", event_names)
            self.assertEqual({event["run_id"] for event in events}, {state.run_id})

    def test_multi_tool_calls_execute_in_response_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    ChatResponse(
                        finish_reason="tool_calls",
                        tool_calls=[
                            LLMToolCall(
                                id="call_list",
                                name="list_files",
                                arguments={"path": "."},
                                arguments_json='{"path":"."}',
                            ),
                            LLMToolCall(
                                id="call_read",
                                name="read_file",
                                arguments={"path": "calculator.py"},
                                arguments_json='{"path":"calculator.py"}',
                            ),
                        ],
                    ),
                    ChatResponse(content="done", finish_reason="stop"),
                ]
            )

            state = run_agent(
                repo_path=repo,
                task="Inspect files.",
                config=fake_config(base / "traces"),
                llm=llm,
                max_steps=4,
                trace_dir=base / "traces",
            )

            self.assertEqual([record.call.tool for record in state.tool_history], ["list_files", "read_file"])
            self.assertEqual(state.stop_reason, "assistant_final")
            events = [event for event in read_trace(state.trace_path) if event["event"] == "tool.completed"]
            self.assertEqual([event["payload"]["name"] for event in events], ["list_files", "read_file"])

    def test_max_steps_is_enforced_inside_multi_tool_call_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    ChatResponse(
                        finish_reason="tool_calls",
                        tool_calls=[
                            LLMToolCall(
                                id="call_read",
                                name="read_file",
                                arguments={"path": "calculator.py"},
                                arguments_json='{"path":"calculator.py"}',
                            ),
                            LLMToolCall(
                                id="call_write",
                                name="write_file",
                                arguments={"path": "created.txt", "content": "should not exist\n"},
                                arguments_json='{"path":"created.txt","content":"should not exist\\n"}',
                            ),
                        ],
                    ),
                    ChatResponse(content="should not be reached", finish_reason="stop"),
                ]
            )

            state = run_agent(
                repo_path=repo,
                task="Read then write.",
                config=fake_config(base / "traces"),
                llm=llm,
                max_steps=1,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "max_steps_reached")
            self.assertEqual(state.steps, 1)
            self.assertFalse((repo / "created.txt").exists())
            completed = [event for event in read_trace(state.trace_path) if event["event"] == "tool.completed"]
            self.assertEqual([event["payload"]["name"] for event in completed], ["read_file", "write_file"])
            self.assertEqual(completed[1]["payload"]["error_code"], "max_tool_calls")

    def test_invalid_arguments_json_is_returned_as_tool_result_and_loop_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    ChatResponse(
                        finish_reason="tool_calls",
                        tool_calls=[
                            LLMToolCall(
                                id="bad_args",
                                name="read_file",
                                arguments={},
                                arguments_json="{not json",
                                arguments_error="Expecting property name",
                            )
                        ],
                    ),
                    ChatResponse(content="recovered", finish_reason="stop"),
                ]
            )

            state = run_agent(
                repo_path=repo,
                task="Read a file.",
                config=fake_config(base / "traces"),
                llm=llm,
                max_steps=4,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "assistant_final")
            self.assertEqual(state.tool_history[0].call.tool, "read_file")
            self.assertEqual(state.tool_history[0].result.reason, "invalid_arguments_json")
            self.assertEqual(state.final_answer.splitlines()[0], "recovered")

    def test_max_steps_stops_with_new_budget_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="Fix subtract.",
                test_command="python -m unittest discover -s tests -q",
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                max_steps=2,
                trace_dir=base / "traces",
            )

            self.assertTrue(state.done)
            self.assertEqual(state.stop_reason, "max_steps_reached")
            self.assertEqual(state.steps, 2)
            events = read_trace(state.trace_path)
            self.assertIn("run.completed", [event["event"] for event in events])
            self.assertIn("return a + b", (repo / "calculator.py").read_text(encoding="utf-8"))

    def test_stagnation_budget_stops_repeated_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    tool_response("read_file", {"path": "calculator.py"}, "call_1"),
                    tool_response("read_file", {"path": "calculator.py"}, "call_2"),
                    tool_response("read_file", {"path": "calculator.py"}, "call_3"),
                    ChatResponse(content="should not be reached", finish_reason="stop"),
                ]
            )

            state = run_agent(
                repo_path=repo,
                task="Loop.",
                config=fake_config(base / "traces", stagnation_window=3),
                llm=llm,
                max_steps=8,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "stagnation_detected")
            events = read_trace(state.trace_path)
            budget_events = [event for event in events if event["event"] == "budget.exceeded"]
            self.assertEqual(budget_events[-1]["payload"]["reason"], "stagnation_detected")

    def test_repeated_failure_uses_tool_name_and_arguments_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    tool_response("read_file", {"path": "missing_a.py"}, "call_1"),
                    tool_response("read_file", {"path": "missing_b.py"}, "call_2"),
                    tool_response("read_file", {"path": "missing_c.py"}, "call_3"),
                    ChatResponse(content="done after distinct failures", finish_reason="stop"),
                ]
            )

            state = run_agent(
                repo_path=repo,
                task="Read missing files.",
                config=fake_config(base / "traces", repeated_failure_window=3),
                llm=llm,
                max_steps=8,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "assistant_final")
            self.assertIn("done after distinct failures", state.final_answer)

    def test_repeated_failure_stops_same_tool_and_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    tool_response("read_file", {"path": "missing.py"}, "call_1"),
                    tool_response("read_file", {"path": "missing.py"}, "call_2"),
                    tool_response("read_file", {"path": "missing.py"}, "call_3"),
                    ChatResponse(content="should not be reached", finish_reason="stop"),
                ]
            )

            state = run_agent(
                repo_path=repo,
                task="Read missing file repeatedly.",
                config=fake_config(base / "traces", repeated_failure_window=3),
                llm=llm,
                max_steps=8,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "repeated_tool_failure")

    def test_automatic_compaction_handles_single_user_react_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            (repo / "big.txt").write_text("x" * 20000, encoding="utf-8")
            llm = RecordingFakeLLM(
                chat_responses=[
                    tool_response("read_file", {"path": "big.txt", "limit": 20000}, "call_1"),
                    tool_response("read_file", {"path": "big.txt", "limit": 20000}, "call_2"),
                    ChatResponse(content="done", finish_reason="stop"),
                ]
            )

            state = run_agent(
                repo_path=repo,
                task="Read a large file twice.",
                config=fake_config(
                    base / "traces",
                    context_window=5000,
                    response_reserve_tokens=100,
                    compression_buffer_tokens=100,
                    retain_recent_user_turns=1,
                    max_tool_result_chars=10000,
                ),
                llm=llm,
                max_steps=8,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "assistant_final")
            events = read_trace(state.trace_path)
            self.assertIn("context.compacted", [event["event"] for event in events])
            final_request = llm.requests[-1]
            self.assertTrue(any("Compressed conversation summary" in str(message.get("content", "")) for message in final_request))
            self.assert_tool_results_are_paired(final_request)

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
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                max_steps=1,
                trace_dir=base / "traces",
            )

            self.assertNotIn("needle_secret", state.repo_context)
            self.assertNotIn("credentials.json", state.repo_context)

    def assert_tool_results_are_paired(self, messages: list[dict[str, object]]) -> None:
        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            cursor = index - 1
            while cursor >= 0 and messages[cursor].get("role") == "tool":
                cursor -= 1
            self.assertGreaterEqual(cursor, 0)
            self.assertEqual(messages[cursor].get("role"), "assistant")
            self.assertTrue(messages[cursor].get("tool_calls"))


if __name__ == "__main__":
    unittest.main()
