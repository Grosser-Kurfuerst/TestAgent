from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, Sequence

from my_agent.config import AgentConfig
from my_agent.llm.types import ChatResponse, ChatUsage, MessageLike, messages_to_openai
from my_agent.schema import AgentState, ToolRecord


class AgentLLM(Protocol):
    supports_tools: bool

    def chat(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        ...

    def plan(self, state: AgentState, tool_descriptions: str) -> str:
        ...

    def act(self, state: AgentState, tool_descriptions: str) -> str:
        ...

    def review(self, state: AgentState) -> str:
        ...

    def summarize(self, state: AgentState) -> str:
        ...


def build_llm(config: AgentConfig) -> AgentLLM:
    config.require_valid_provider()
    if config.use_fake_llm:
        return FakeLLM()
    if config.provider == "openai":
        config.require_api_key()
        return OpenAICompatibleLLM(config)
    raise RuntimeError(f"Unsupported LLM provider: {config.provider}")


class FakeLLM:
    """Deterministic local model used by tests and smoke runs."""

    supports_tools = False

    def __init__(
        self,
        actor_responses: Sequence[str] | None = None,
        chat_responses: Sequence[ChatResponse | str] | None = None,
    ):
        self._actor_responses = list(actor_responses or [])
        self._chat_responses = list(chat_responses or [])

    def chat(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        if self._chat_responses:
            response = self._chat_responses.pop(0)
            if isinstance(response, ChatResponse):
                return response
            return ChatResponse(content=response, finish_reason="stop")
        return ChatResponse(content="FakeLLM response.", finish_reason="stop")

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

    def review(self, state: AgentState) -> str:
        changed_tools = _changed_tools(state)
        test_status = _test_status(state)
        diff_records = [record for record in state.tool_history if record.call.tool == "git_diff" and record.result.ok]

        risk = "low" if test_status == "passed" and changed_tools else "medium"
        if test_status == "failed":
            risk = "high"
        elif test_status == "not run":
            risk = "medium"

        lines = [
            "Reviewer findings:",
            f"- Changes: {', '.join(changed_tools) if changed_tools else 'none'}",
            f"- Tests: {test_status}",
            f"- Diff reviewed: {'yes' if diff_records else 'no'}",
            f"- Risk: {risk}",
        ]
        if test_status == "failed":
            lines.append("- Action: inspect the failed test output before finishing.")
        elif test_status == "not run":
            lines.append("- Action: run the configured tests before treating the task as complete.")
        return "\n".join(lines)

    def summarize(self, state: AgentState) -> str:
        changed_tools = _changed_tools(state)
        finish_records = [record for record in state.tool_history if record.call.tool == "finish"]
        tests = _test_status(state)

        lines = [
            f"Task: {state.task}",
            f"Stop reason: {state.stop_reason or 'finished'}",
            f"Steps: {state.steps}",
            f"Edits: {', '.join(changed_tools) if changed_tools else 'none'}",
            f"Tests: {tests}",
            f"Risks: {_risk_summary(state)}",
        ]
        if state.review:
            lines.append(f"Review: {state.review}")
        if finish_records:
            lines.append(f"Finish summary: {finish_records[-1].result.output}")
        if state.trace_path:
            lines.append(f"Trace: {state.trace_path}")
        return "\n".join(lines)


class OpenAICompatibleLLM:
    """Small OpenAI-compatible chat-completions client implemented with stdlib HTTP."""

    supports_tools = True

    def __init__(self, config: AgentConfig, timeout: int = 60, max_retries: int = 2):
        config.require_api_key()
        self.api_key = config.api_key
        self.base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = config.model
        self.temperature = config.temperature
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    def plan(self, state: AgentState, tool_descriptions: str) -> str:
        return self._chat(_planner_messages(state, tool_descriptions))

    def act(self, state: AgentState, tool_descriptions: str) -> str:
        return self._chat(_actor_messages(state, tool_descriptions))

    def review(self, state: AgentState) -> str:
        return self._chat(_reviewer_messages(state))

    def summarize(self, state: AgentState) -> str:
        return self._chat(_summarizer_messages(state))

    def chat(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages_to_openai(messages),
            "temperature": self.temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        body = self._post_chat_completion(payload)
        try:
            parsed = json.loads(body)
            return ChatResponse.from_openai_payload(parsed)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM response could not be parsed: {_excerpt(body)}") from exc

    def _chat(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None) -> str:
        response = self.chat(messages, tools=tools)
        if not response.content:
            raise RuntimeError("LLM response message content was empty.")
        return response.content

    def _post_chat_completion(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        attempts = self.max_retries + 1
        last_error = ""
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = f"LLM request failed with HTTP {exc.code}: {_excerpt(error_body)}"
                if not _is_retryable_http_error(exc.code) or attempt == attempts - 1:
                    raise RuntimeError(last_error) from exc
            except urllib.error.URLError as exc:
                last_error = f"LLM request failed: {exc.reason}"
                if attempt == attempts - 1:
                    raise RuntimeError(last_error) from exc
            time.sleep(min(0.25 * (2**attempt), 1.0))
        raise RuntimeError(last_error or "LLM request failed.")


def _is_retryable_http_error(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


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


def _planner_messages(state: AgentState, tool_descriptions: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a careful AI coding planner. Produce a concise numbered plan before tools are used. "
                "Prefer small, safe, testable edits. Do not invent file contents without inspecting files first. "
                "Mention the files, symbols, or retrieval terms that should be inspected before editing. "
                "Do not call tools."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{state.task}\n\n"
                f"Repository context:\n{state.repo_context}\n\n"
                f"Project rules:\n{state.project_rules or 'No project rules found.'}\n\n"
                f"Available tools:\n{tool_descriptions}"
            ),
        },
    ]


def _actor_messages(state: AgentState, tool_descriptions: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a Claude Code style coding agent. Output exactly one JSON object and no markdown or prose. "
                "The JSON schema is {\"tool\": string, \"arguments\": object, \"reason\": string}. "
                "All three top-level keys are required. Never omit reason; reason must be a short non-empty sentence. "
                "Valid example: {\"tool\":\"read_file\",\"arguments\":{\"path\":\"solution.py\",\"limit\":12000},"
                "\"reason\":\"Inspect the current implementation before editing.\"}. "
                "The tool-specific schemas below describe only the arguments object, not the outer wrapper. "
                "Use only an available tool and choose one tool call at a time. "
                "Use retrieve_context or grep to find relevant code before editing, then read files before changing them. "
                "Prefer replace_in_file for focused edits; use write_file only when full-file replacement is safer. "
                "Run tests after code edits when a test command is available. "
                "If tests fail, inspect the failure output before another edit and do not finish early. "
                "Call finish only when the task is complete, required checks are done, or the summary clearly states missing or failed checks."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{state.task}\n\n"
                f"Plan:\n{state.plan}\n\n"
                f"Repository context:\n{state.repo_context}\n\n"
                f"Project rules:\n{state.project_rules or 'No project rules found.'}\n\n"
                f"Test command:\n{state.test_command or 'not configured'}\n\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                f"Recent tool history:\n{_history_for_prompt(state)}"
            ),
        },
    ]


def _reviewer_messages(state: AgentState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative code-review subagent. Review the completed tool history, diff output, "
                "test status, and residual risk. Identify whether the requested task is actually complete, "
                "not only whether tests passed. Return a concise review with correctness, tests, risks, "
                "and recommended follow-up. Be explicit when tests were not run or failed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{state.task}\n\n"
                f"Plan:\n{state.plan or 'No plan was recorded.'}\n\n"
                f"Stop reason: {state.stop_reason or 'unknown'}\n\n"
                f"Tool history:\n{_history_for_prompt(state, limit=12)}"
            ),
        },
    ]


def _summarizer_messages(state: AgentState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the final summarizer for a coding agent. Return a concise delivery summary with these fields: "
                "Changes, Tests, Risks, Next steps, Trace. "
                "Changes must include changed file paths and the behavioral effect of each change. "
                "Tests must include the command run and result, or explain why tests were not run or failed. "
                "Write for a developer handoff and do not hide risk."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{state.task}\n\n"
                f"Stop reason: {state.stop_reason or 'unknown'}\n"
                f"Trace path: {state.trace_path or 'not recorded'}\n\n"
                f"Reviewer output:\n{state.review or 'No review was recorded.'}\n\n"
                f"Tool history:\n{_history_for_prompt(state, limit=12)}"
            ),
        },
    ]


def _history_for_prompt(state: AgentState, limit: int = 8) -> str:
    if not state.tool_history:
        return "No tools have been called yet."
    rows = []
    for index, record in enumerate(state.tool_history[-limit:], start=max(1, len(state.tool_history) - limit + 1)):
        output = _excerpt(record.result.output, 1200)
        rows.append(
            json.dumps(
                {
                    "step": index,
                    "tool": record.call.tool,
                    "arguments": record.call.arguments,
                    "reason": record.call.reason,
                    "ok": record.result.ok,
                    "blocked": record.result.blocked,
                    "result": output,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(rows)


def _changed_tools(state: AgentState) -> list[str]:
    return [
        record.call.tool
        for record in state.tool_history
        if record.call.tool in {"replace_in_file", "write_file"} and record.result.ok
    ]


def _test_status(state: AgentState) -> str:
    test_records = [record for record in state.tool_history if record.call.tool == "run_tests"]
    if not test_records:
        return "not run"
    return "passed" if test_records[-1].result.ok else "failed"


def _risk_summary(state: AgentState) -> str:
    tests = _test_status(state)
    if tests == "failed":
        return "high; latest test run failed"
    if tests == "not run":
        return "medium; tests were not run"
    if state.stop_reason == "max_steps_reached":
        return "medium; runtime stopped at max steps"
    if state.stop_reason == "invalid_tool_call":
        return "medium; actor protocol failed"
    return "low; latest tests passed"


def _excerpt(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... truncated"


__all__ = [
    "AgentLLM",
    "ChatResponse",
    "ChatUsage",
    "FakeLLM",
    "MessageLike",
    "OpenAICompatibleLLM",
    "build_llm",
]
