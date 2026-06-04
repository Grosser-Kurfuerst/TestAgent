from __future__ import annotations

import json
import re
from typing import Protocol, Sequence

from my_agent.schema import AgentState, ToolRecord


class AgentLLM(Protocol):
    def plan(self, state: AgentState, tool_descriptions: str) -> str:
        ...

    def act(self, state: AgentState, tool_descriptions: str) -> str:
        ...

    def summarize(self, state: AgentState) -> str:
        ...


class FakeLLM:
    """Deterministic local model used by Phase 4 tests and smoke runs."""

    def __init__(self, actor_responses: Sequence[str] | None = None):
        self._actor_responses = list(actor_responses or [])

    def plan(self, state: AgentState, tool_descriptions: str) -> str:
        return (
            "1. Retrieve context for the task.\n"
            "2. Inspect the likely target file before editing.\n"
            "3. Replace the focused faulty subtract implementation.\n"
            "4. Run the configured tests and inspect the diff.\n"
            "5. Finish with a concise summary."
        )

    def act(self, state: AgentState, tool_descriptions: str) -> str:
        if self._actor_responses:
            return self._actor_responses.pop(0)

        used_tools = [record.call.tool for record in state.tool_history]
        if "retrieve_context" not in used_tools:
            return _tool_json(
                "retrieve_context",
                {"query": state.task, "top_k": 3},
                "Find the files most relevant to the task.",
            )
        if "read_file" not in used_tools:
            return _tool_json(
                "read_file",
                {"path": "calculator.py", "limit": 12000},
                "Inspect the suspected implementation file before editing.",
            )
        if "replace_in_file" not in used_tools:
            old, new = _subtract_replacement(state.tool_history)
            return _tool_json(
                "replace_in_file",
                {"path": "calculator.py", "old": old, "new": new},
                "Apply a focused replacement to fix subtract.",
            )
        if "run_tests" not in used_tools:
            return _tool_json(
                "run_tests",
                {"command": state.test_command or "python -m unittest discover -s tests -q"},
                "Run the configured tests after editing.",
            )
        if "git_diff" not in used_tools:
            return _tool_json("git_diff", {}, "Review the final patch before finishing.")
        return _tool_json(
            "finish",
            {"summary": "Updated subtract to return the first number minus the second number and ran tests."},
            "The edit and verification steps are complete.",
        )

    def summarize(self, state: AgentState) -> str:
        changed_tools = [record.call.tool for record in state.tool_history if record.call.tool in {"replace_in_file", "write_file"}]
        test_records = [record for record in state.tool_history if record.call.tool == "run_tests"]
        finish_records = [record for record in state.tool_history if record.call.tool == "finish"]

        if test_records:
            tests = "passed" if test_records[-1].result.ok else "failed"
        else:
            tests = "not run"

        lines = [
            f"Task: {state.task}",
            f"Stop reason: {state.stop_reason or 'finished'}",
            f"Steps: {state.steps}",
            f"Edits: {', '.join(changed_tools) if changed_tools else 'none'}",
            f"Tests: {tests}",
        ]
        if finish_records:
            lines.append(f"Finish summary: {finish_records[-1].result.output}")
        if state.trace_path:
            lines.append(f"Trace: {state.trace_path}")
        return "\n".join(lines)


def _tool_json(tool: str, arguments: dict[str, object], reason: str) -> str:
    return json.dumps({"tool": tool, "arguments": arguments, "reason": reason}, ensure_ascii=False)


def _subtract_replacement(history: list[ToolRecord]) -> tuple[str, str]:
    content = ""
    for record in reversed(history):
        if record.call.tool == "read_file" and record.result.ok:
            content = record.result.output
            break

    match = re.search(r"def subtract\([^\n]*\):[\s\S]*?(?=\ndef |\nasync def |\nclass |\Z)", content)
    if match:
        old = match.group(0).rstrip("\n")
        if "return a + b" in old:
            return old, old.replace("return a + b", "return a - b", 1)

    old = 'def subtract(a: int, b: int) -> int:\n    """Return a minus b."""\n    return a + b'
    new = 'def subtract(a: int, b: int) -> int:\n    """Return a minus b."""\n    return a - b'
    return old, new
