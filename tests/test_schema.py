from __future__ import annotations

import json
import unittest

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.schema import AgentState, ToolCall, ToolResult, TraceEvent


class SchemaTests(unittest.TestCase):
    def test_tool_call_parses_valid_json(self) -> None:
        call = ToolCall.from_json(
            json.dumps(
                {
                    "tool": "read_file",
                    "arguments": {"path": "calculator.py"},
                    "reason": "inspect the target file",
                }
            ),
            allowed_tools={"read_file", "finish"},
        )

        self.assertEqual(call.tool, "read_file")
        self.assertEqual(call.arguments, {"path": "calculator.py"})
        self.assertEqual(call.reason, "inspect the target file")

    def test_tool_call_rejects_non_object_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "arguments"):
            ToolCall.from_mapping({"tool": "read_file", "arguments": "calculator.py", "reason": "inspect"})

    def test_tool_call_rejects_unknown_tool(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown tool"):
            ToolCall.from_mapping({"tool": "delete", "arguments": {}, "reason": "cleanup"}, allowed_tools={"read_file"})

    def test_tool_call_requires_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "reason"):
            ToolCall.from_mapping({"tool": "read_file", "arguments": {}, "reason": ""})

    def test_tool_result_serializes_to_dict(self) -> None:
        result = ToolResult(ok=False, output="blocked", blocked=True, reason="unsafe path")

        self.assertEqual(
            result.to_dict(),
            {
                "ok": False,
                "output": "blocked",
                "blocked": True,
                "reason": "unsafe path",
            },
        )

    def test_agent_state_initializes_minimal_state(self) -> None:
        state = AgentState.initial(repo_path="examples/sample_repo", task="Fix subtract", test_command="pytest -q", run_id="run-1")

        self.assertEqual(str(state.repo_path), "examples/sample_repo")
        self.assertEqual(state.task, "Fix subtract")
        self.assertEqual(state.run_id, "run-1")
        self.assertEqual(state.test_command, "pytest -q")
        self.assertEqual(state.steps, 0)
        self.assertEqual(state.max_steps, 8)
        self.assertEqual(state.tool_history, [])

    def test_agent_state_creates_trace_event_with_shared_run_id(self) -> None:
        state = AgentState.initial(repo_path="examples/sample_repo", task="Fix subtract", run_id="run-1")

        first = state.trace_event("repo_indexed", {"repo": "examples/sample_repo"})
        second = state.trace_event("plan", {"plan": "inspect then edit"})

        self.assertEqual(first.run_id, "run-1")
        self.assertEqual(second.run_id, "run-1")

    def test_trace_event_serializes_json_line(self) -> None:
        event = TraceEvent(event="plan", payload={"plan": "inspect then edit"}, run_id="run-1", time="2026-01-01T00:00:00")

        self.assertEqual(
            json.loads(event.to_json_line()),
            {
                "time": "2026-01-01T00:00:00",
                "run_id": "run-1",
                "event": "plan",
                "payload": {"plan": "inspect then edit"},
            },
        )

    def test_trace_event_requires_explicit_run_id(self) -> None:
        with self.assertRaises(TypeError):
            TraceEvent(event="plan", payload={})  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
