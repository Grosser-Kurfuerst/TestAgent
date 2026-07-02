from __future__ import annotations

import json
import threading
import tempfile
import unittest
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.agent_base import AgentBase
from my_agent.agent_factory import AgentFactory
from my_agent.config import AgentConfig
from my_agent.cancellation import CancellationToken
from my_agent.hitl import ApprovalDecision, ApprovalEvent, ApprovalRequest, ApprovalResult, ApprovalScope
from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, MessageLike, messages_to_openai
from my_agent.memory import MemoryManager, MemoryScope
from my_agent.plan import AgentMode
from my_agent.runtime import CodingAgentRuntime, run_agent
from my_agent.schema import AgentState, TraceEvent
from my_agent.react import ReActAgent
from my_agent.tracing import TraceWriter


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


def write_project_tool_config(repo: Path, description: str) -> None:
    config_dir = repo / ".agentcli"
    config_dir.mkdir()
    (config_dir / "tools.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tools": [
                    {
                        "name": "oversized_project_tool",
                        "description": description,
                        "kind": "command",
                        "risk": "execute",
                        "enabled": True,
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                        "command": {
                            "argv": [
                                "python3",
                                "-c",
                                "from pathlib import Path; Path('should_not_exist.txt').write_text('bad')",
                            ],
                            "timeout_seconds": 5,
                            "cwd": ".",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def read_trace(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
        self.tool_requests: list[list[dict[str, object]]] = []

    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        self.requests.append(messages_to_openai(messages))
        self.tool_requests.append(list(tools or []))
        if tools is None:
            return ChatResponse(content="compressed summary", finish_reason="stop")
        return super().chat(messages, tools=tools)  # type: ignore[arg-type]


class NoToolChatLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.tool_chat_calls = 0

    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        if tools is not None:
            self.tool_chat_calls += 1
            raise AssertionError("tool-call LLM should not be called")
        return ChatResponse(content="[]", finish_reason="stop")


class RecordingHitlHandler:
    def __init__(self, *results: ApprovalResult, enabled: bool = True) -> None:
        self.results = list(results)
        self.enabled = enabled
        self.requests: list[ApprovalRequest] = []
        self.approved_all: set[tuple[ApprovalScope, str]] = set()

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        self.requests.append(request)
        result = self.results.pop(0) if self.results else ApprovalResult(ApprovalDecision.APPROVED)
        if result.decision == ApprovalDecision.APPROVED_ALL:
            self.approved_all.add((ApprovalScope.TOOL, request.tool_name))
        return result

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def clear_approved_all(self) -> None:
        self.approved_all.clear()

    def is_approved_all(self, *, scope: ApprovalScope, key: str) -> bool:
        return (scope, key) in self.approved_all


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

    def test_run_agent_preserves_external_cancellation_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            token = CancellationToken()

            state = run_agent(
                repo_path=repo,
                task="Return final response.",
                config=fake_config(base / "traces"),
                llm=FakeLLM(chat_responses=[ChatResponse(content="done", finish_reason="stop")]),
                trace_dir=base / "traces",
                cancellation_token=token,
            )

            self.assertIs(state.cancellation_token, token)
            self.assertEqual(state.stop_reason, "assistant_final")

    def test_run_agent_pre_cancelled_token_returns_cancelled_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            token = CancellationToken()
            token.cancel("pre_cancel")

            state = run_agent(
                repo_path=repo,
                task="Return final response.",
                config=fake_config(base / "traces"),
                llm=FakeLLM(chat_responses=[ChatResponse(content="done", finish_reason="stop")]),
                trace_dir=base / "traces",
                cancellation_token=token,
            )

            self.assertEqual(state.stop_reason, "cancelled")
            self.assertEqual(state.final_answer, "Cancelled.")
            event_names = [event["event"] for event in read_trace(state.trace_path)]
            self.assertIn("run.cancelled", event_names)

    def test_trace_writer_is_thread_safe_for_jsonl_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = TraceWriter(Path(tmp) / "trace.jsonl")

            def append_range(start: int) -> None:
                for index in range(start, start + 25):
                    writer.append(TraceEvent(event="parallel", payload={"index": index}, run_id="run"))

            threads = [threading.Thread(target=append_range, args=(offset,)) for offset in range(0, 100, 25)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            events = read_trace(writer.path)
            self.assertEqual(len(events), 100)
            self.assertEqual({event["event"] for event in events}, {"parallel"})

    def test_react_flushes_buffered_approval_events_in_tool_call_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sink_events: list[ApprovalEvent] = []
            agent = ReActAgent(
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                command_timeout=60,
                event_sink=sink_events.append,
            )
            state = AgentState.initial(base, "task")
            writer = TraceWriter(base / "trace.jsonl")
            events = [
                ApprovalEvent(event="approval.completed", payload={"tool_call_id": "b"}),
                ApprovalEvent(event="approval.completed", payload={"tool_call_id": "a"}),
            ]
            calls = [
                LLMToolCall(id="a", name="write_file", arguments={}, arguments_json="{}"),
                LLMToolCall(id="b", name="write_file", arguments={}, arguments_json="{}"),
            ]

            agent._flush_approval_events(writer, state, events, calls)

            self.assertEqual([event.payload["tool_call_id"] for event in sink_events], ["a", "b"])
            trace_events = read_trace(writer.path)
            self.assertEqual([event["payload"]["tool_call_id"] for event in trace_events], ["a", "b"])

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
                "agent.completed",
            ):
                self.assertIn(expected, event_names)
            llm_completed = [event for event in events if event["event"] == "llm.completed"]
            self.assertTrue(llm_completed)
            self.assertEqual({event["payload"].get("phase") for event in llm_completed}, {"react"})
            agent_completed = [event for event in events if event["event"] == "agent.completed"]
            self.assertEqual(agent_completed[-1]["payload"]["mode"], "react")
            self.assertEqual(agent_completed[-1]["payload"]["trace_path"], str(state.trace_path))
            self.assertNotIn("tool_call", event_names)
            self.assertNotIn("final_summary", event_names)
            self.assertEqual({event["run_id"] for event in events}, {state.run_id})

    def test_memory_disabled_uses_noop_memory_even_with_external_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            config = fake_config(base / "traces", memory_dir=base / "memory", memory_enabled=False)
            memory = MemoryManager.from_config(config=fake_config(base / "other-memory"), llm=FakeLLM(), repo_path=repo)
            memory.save_fact("This fact must not be injected.", scope=MemoryScope.PROJECT)
            llm = RecordingFakeLLM([ChatResponse(content="done", finish_reason="stop")])

            state = run_agent(
                repo_path=repo,
                task="Return final response.",
                config=config,
                llm=llm,
                trace_dir=base / "traces",
                memory_manager=memory,
            )

            events = read_trace(state.trace_path)
            event_names = [event["event"] for event in events]
            self.assertIn("memory.loaded", event_names)
            self.assertNotIn("memory.retrieved", event_names)
            first_request = json.dumps(llm.requests[0], ensure_ascii=False)
            self.assertNotIn("This fact must not be injected.", first_request)

    def test_hitl_approval_events_are_traced_and_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = FakeLLM(
                chat_responses=[
                    tool_response("write_file", {"path": "approved.txt", "content": "ok"}, "call_write"),
                    ChatResponse(content="done", finish_reason="stop"),
                ]
            )
            handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED))
            events: list[object] = []

            state = run_agent(
                repo_path=repo,
                task="Create approved file.",
                config=fake_config(base / "traces", hitl_audit_dir=base / "audit"),
                llm=llm,
                trace_dir=base / "traces",
                event_sink=events.append,
                hitl_handler=handler,
            )

            self.assertEqual(state.stop_reason, "assistant_final")
            self.assertEqual((repo / "approved.txt").read_text(encoding="utf-8"), "ok")
            self.assertEqual(len(handler.requests), 1)
            trace_events = read_trace(state.trace_path)
            requested = [event for event in trace_events if event["event"] == "approval.requested"]
            completed = [event for event in trace_events if event["event"] == "approval.completed"]
            self.assertEqual(len(requested), 1)
            self.assertEqual(len(completed), 1)
            self.assertEqual(requested[0]["payload"]["id"], completed[0]["payload"]["id"])
            self.assertEqual(requested[0]["payload"]["run_id"], state.run_id)
            event_names = [getattr(event, "event", "") for event in events]
            self.assertIn("render.flush_requested", event_names)
            self.assertIn("approval.requested", event_names)
            self.assertIn("approval.completed", event_names)

    def test_hitl_rejection_is_returned_to_model_as_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = RecordingFakeLLM(
                [
                    tool_response("write_file", {"path": "blocked.txt", "content": "x"}, "call_write"),
                    ChatResponse(content="blocked by user", finish_reason="stop"),
                ]
            )
            handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.REJECTED, reason="do not write"))

            state = run_agent(
                repo_path=repo,
                task="Create blocked file.",
                config=fake_config(base / "traces", hitl_audit_dir=base / "audit"),
                llm=llm,
                trace_dir=base / "traces",
                hitl_handler=handler,
            )

            self.assertEqual(state.stop_reason, "assistant_final")
            self.assertFalse((repo / "blocked.txt").exists())
            self.assertEqual(state.tool_history[0].result.reason, "approval_rejected")
            self.assertIn("[HITL] Operation rejected", json.dumps(llm.requests[-1], ensure_ascii=False))

    def test_run_agent_team_mode_executes_team_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="先检查 calculator.py 和测试，再修复 subtract，并运行测试",
                test_command="python -m unittest discover -s tests -q",
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                mode="team",
            )

            self.assertEqual(state.stop_reason, "team_completed")
            self.assertIn("Team succeeded", state.final_answer)
            self.assertIn("return a - b", (repo / "calculator.py").read_text(encoding="utf-8"))
            events = read_trace(state.trace_path)
            event_names = [event["event"] for event in events]
            self.assertIn("team.started", event_names)
            self.assertIn("team.step.review_completed", event_names)
            self.assertIn("team.completed", event_names)

    def test_run_agent_default_mode_uses_config_agent_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="先检查 calculator.py 和测试，再修复 subtract，并运行测试",
                test_command="python -m unittest discover -s tests -q",
                config=fake_config(base / "traces", agent_mode="team"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "team_completed")
            event_names = [event["event"] for event in read_trace(state.trace_path)]
            self.assertIn("team.started", event_names)
            self.assertNotIn("run.started", event_names)

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
            event_names = [event["event"] for event in read_trace(state.trace_path)]
            self.assertIn("tool.batch.started", event_names)
            self.assertIn("tool.batch.completed", event_names)
            completed_batches = [event for event in read_trace(state.trace_path) if event["event"] == "tool.batch.completed"]
            self.assertTrue(completed_batches[-1]["payload"]["parallel"])
            self.assertEqual(completed_batches[-1]["payload"]["groups"][0]["reason"], "read_tools")

    def test_single_tool_batch_does_not_reuse_previous_parallel_summary(self) -> None:
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
                    ChatResponse(
                        finish_reason="tool_calls",
                        tool_calls=[
                            LLMToolCall(
                                id="call_read_again",
                                name="read_file",
                                arguments={"path": "tests/test_calculator.py"},
                                arguments_json='{"path":"tests/test_calculator.py"}',
                            )
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
                max_steps=5,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "assistant_final")
            completed_batches = [event for event in read_trace(state.trace_path) if event["event"] == "tool.batch.completed"]
            self.assertEqual(len(completed_batches), 2)
            self.assertTrue(completed_batches[0]["payload"]["parallel"])
            self.assertFalse(completed_batches[1]["payload"]["parallel"])
            self.assertEqual(completed_batches[1]["payload"]["groups"][0]["ids"], ["call_read_again"])
            self.assertEqual(completed_batches[1]["payload"]["groups"][0]["reason"], "single_tool")

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

    def test_runtime_uses_memory_manager_for_messages_and_trace_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            config = fake_config(base / "traces", memory_dir=base / "memory")
            llm = RecordingFakeLLM([ChatResponse(content="done", finish_reason="stop")])
            memory = MemoryManager.from_config(config=config, llm=llm, repo_path=repo)
            memory.save_fact(
                "Project calculator.py contains calculator functions.",
                scope=MemoryScope.PROJECT,
            )

            state = run_agent(
                repo_path=repo,
                task="Inspect calculator memory.",
                config=config,
                llm=llm,
                trace_dir=base / "traces",
                memory_manager=memory,
            )

            self.assertEqual(state.stop_reason, "assistant_final")
            self.assertGreaterEqual(memory.status(include_entries=False).short_term_entries, 2)
            first_request = json.dumps(llm.requests[0], ensure_ascii=False)
            self.assertIn("Relevant long-term memory:", first_request)
            self.assertIn("calculator.py contains calculator functions", first_request)
            event_names = [event["event"] for event in read_trace(state.trace_path)]
            self.assertIn("memory.loaded", event_names)
            self.assertIn("memory.retrieved", event_names)
            self.assertIn("memory.prepared", event_names)

    def test_runtime_caps_tool_schema_and_blocks_omitted_tool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            write_project_tool_config(repo, "oversized project tool " * 5000)
            llm = RecordingFakeLLM(
                [
                    tool_response("oversized_project_tool", {}, "call_omitted"),
                    tool_response("finish", {"summary": "done"}, "call_finish"),
                ]
            )

            state = run_agent(
                repo_path=repo,
                task="Try omitted tool, then finish.",
                config=fake_config(
                    base / "traces",
                    memory_dir=base / "memory",
                    enable_project_tools=True,
                    mcp_enabled=False,
                    tool_schema_budget_tokens=500,
                    tool_schema_budget_tokens_explicit=True,
                ),
                llm=llm,
                max_steps=4,
                trace_dir=base / "traces",
            )

            self.assertFalse((repo / "should_not_exist.txt").exists())
            self.assertEqual(state.stop_reason, "finish_called")
            first_tool_names = {tool["function"]["name"] for tool in llm.tool_requests[0]}
            self.assertIn("read_file", first_tool_names)
            self.assertIn("finish", first_tool_names)
            self.assertNotIn("oversized_project_tool", first_tool_names)
            events = read_trace(state.trace_path)
            capped = [event["payload"] for event in events if event["event"] == "tools.schema_capped"][-1]
            self.assertIn("oversized_project_tool", capped["omitted"])
            blocked = [event["payload"] for event in events if event["event"] == "tool.completed" and event["payload"]["id"] == "call_omitted"][-1]
            self.assertEqual(blocked["error_code"], "tool_not_exposed")

    def test_runtime_stops_before_llm_when_context_is_over_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            llm = NoToolChatLLM()

            state = run_agent(
                repo_path=repo,
                task="Return final response.",
                config=fake_config(
                    base / "traces",
                    memory_dir=base / "memory",
                    context_window=8_000,
                    context_window_explicit=True,
                    response_reserve_tokens=7_000,
                    response_reserve_tokens_explicit=True,
                    compression_buffer_tokens=900,
                    compression_buffer_tokens_explicit=True,
                    memory_auto_extract=False,
                ),
                llm=llm,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "context_over_budget")
            self.assertEqual(llm.tool_chat_calls, 0)
            events = read_trace(state.trace_path)
            event_names = [event["event"] for event in events]
            self.assertIn("context.over_budget", event_names)
            self.assertIn("llm.skipped", event_names)
            self.assertNotIn("llm.requested", event_names)

    def test_external_memory_trace_sink_is_restored_after_react_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)
            config = fake_config(base / "traces", memory_dir=base / "memory")
            llm = RecordingFakeLLM([ChatResponse(content="done", finish_reason="stop")])
            memory = MemoryManager.from_config(config=config, llm=llm, repo_path=repo)

            state = run_agent(
                repo_path=repo,
                task="Inspect calculator memory.",
                config=config,
                llm=llm,
                trace_dir=base / "traces",
                memory_manager=memory,
            )
            before = read_trace(state.trace_path)

            memory.save_fact("用户偏好：回答中文", scope=MemoryScope.PROJECT)

            after = read_trace(state.trace_path)
            self.assertEqual(after, before)
            self.assertNotIn("memory.saved", [event["event"] for event in after])

    def test_memory_preparation_handles_single_user_react_history(self) -> None:
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
                    memory_dir=base / "memory",
                    memory_short_term_tokens=50,
                    memory_compression_trigger_ratio=0.8,
                    memory_retain_recent_turns=3,
                    memory_tool_result_chars=10000,
                ),
                llm=llm,
                max_steps=8,
                trace_dir=base / "traces",
            )

            self.assertEqual(state.stop_reason, "assistant_final")
            events = read_trace(state.trace_path)
            prepared = [event["payload"] for event in events if event["event"] == "memory.prepared"]
            self.assertTrue(prepared)
            self.assertFalse(prepared[-1]["compacted"])
            final_request = llm.requests[-1]
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

    def test_run_agent_mode_react_uses_react_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="Inspect calculator.py.",
                config=fake_config(base / "traces"),
                llm=FakeLLM(chat_responses=[ChatResponse(content="done", finish_reason="stop")]),
                trace_dir=base / "traces",
                mode="react",
            )

            events = read_trace(state.trace_path)
            event_names = [event["event"] for event in events]
            self.assertIn("run.started", event_names)
            self.assertNotIn("plan.started", event_names)

    def test_agent_factory_creates_agent_base_for_each_mode(self) -> None:
        factory = AgentFactory(
            config=fake_config(),
            llm=FakeLLM(),
            trace_dir=Path("traces"),
            command_timeout=60,
        )

        self.assertIsInstance(factory.create(AgentMode.REACT), AgentBase)
        self.assertIsInstance(factory.create(AgentMode.PLAN), AgentBase)
        self.assertIsInstance(factory.create(AgentMode.TEAM), AgentBase)

    def test_react_runner_imports_from_react_package(self) -> None:
        from my_agent.react import ReActAgent as NewReActAgent
        from my_agent.react.child_runner import ChildReActRunner as PackageChildReActRunner
        from my_agent.react.agent import ReActAgent as PackageReActAgent

        self.assertIs(NewReActAgent, PackageReActAgent)
        self.assertEqual(PackageChildReActRunner.__name__, "ChildReActRunner")

    def test_run_agent_mode_plan_uses_plan_execute_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="先检查 calculator.py，再修复 subtract，并运行测试",
                test_command="python -m unittest discover -s tests -q",
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                mode="plan",
            )

            events = read_trace(state.trace_path)
            self.assertEqual(state.stop_reason, "plan_completed")
            self.assertIn("plan.started", [event["event"] for event in events])

    def test_run_agent_mode_plan_honors_max_steps_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="先检查 calculator.py，再修复 subtract，并运行测试",
                test_command="python -m unittest discover -s tests -q",
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                max_steps=1,
                mode="plan",
            )

            self.assertEqual(state.stop_reason, "plan_failed")
            child_traces = [path for path in (base / "traces").rglob("*.jsonl") if path != state.trace_path]
            self.assertTrue(child_traces)
            child_events = read_trace(child_traces[0])
            completed = [event for event in child_events if event["event"] == "run.completed"]
            self.assertEqual(completed[-1]["payload"]["stop_reason"], "max_steps_reached")

    def test_run_agent_mode_auto_routes_complex_task_to_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="先检查 calculator.py，再修复 subtract，并运行测试",
                test_command="python -m unittest discover -s tests -q",
                config=fake_config(base / "traces"),
                llm=FakeLLM(),
                trace_dir=base / "traces",
                mode="auto",
            )

            self.assertEqual(state.stop_reason, "plan_completed")
            self.assertIn("Plan:", state.plan)

    def test_auto_route_keeps_simple_task_on_react(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            state = run_agent(
                repo_path=repo,
                task="读取 calculator.py",
                config=fake_config(base / "traces"),
                llm=FakeLLM(chat_responses=[ChatResponse(content="done", finish_reason="stop")]),
                trace_dir=base / "traces",
                mode="auto",
            )

            events = read_trace(state.trace_path)
            self.assertEqual(state.stop_reason, "assistant_final")
            self.assertNotIn("plan.started", [event["event"] for event in events])

    def test_run_agent_mode_team_requires_native_tool_support(self) -> None:
        class NoToolLLM:
            supports_tools = False

            def chat(self, messages: list[object], tools: list[dict[str, object]] | None = None) -> object:
                raise AssertionError("team runtime should fail before LLM use")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            write_runtime_repo(repo)

            with self.assertRaisesRegex(RuntimeError, "native tool-call support"):
                run_agent(
                    repo_path=repo,
                    task="Use the team orchestrator.",
                    config=fake_config(base / "traces"),
                    llm=NoToolLLM(),
                    trace_dir=base / "traces",
                    mode="team",
                )

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
