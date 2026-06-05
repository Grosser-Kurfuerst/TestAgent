from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from my_agent.config import AgentConfig
from my_agent.indexer import RepoIndexer
from my_agent.llm import AgentLLM, build_llm
from my_agent.schema import AgentState, ToolCall, ToolRecord, ToolResult
from my_agent.tools import RepoTools, should_skip_path
from my_agent.tracing import TraceWriter


class CodingAgentRuntime:
    def __init__(
        self,
        config: AgentConfig | None = None,
        llm: AgentLLM | None = None,
        trace_dir: str | Path | None = None,
        command_timeout: int | None = None,
    ):
        self.config = config or AgentConfig.from_env()
        self.llm = llm or build_llm(self.config)
        self.trace_dir = Path(trace_dir) if trace_dir is not None else self.config.trace_dir
        self.command_timeout = command_timeout or self.config.command_timeout

    def run(self, state: AgentState) -> AgentState:
        state.repo_path = Path(state.repo_path).resolve()
        if state.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        writer = TraceWriter.create(self.trace_dir, state.run_id)
        state.trace_path = writer.path

        tools = RepoTools(state.repo_path, timeout=self.command_timeout)
        self._index_repo(state, writer)
        self._plan(state, tools, writer)

        while not state.done:
            if state.steps >= state.max_steps:
                state.done = True
                state.stop_reason = "max_steps_reached"
                self._trace_verify(state, writer, "Maximum step count reached before another tool call.")
                break
            self._act(state, tools, writer)
            self._verify(state, writer)

        self._review(state, writer)
        self._summarize(state, writer)
        return state

    def _index_repo(self, state: AgentState, writer: TraceWriter) -> None:
        def skip(path: Path) -> bool:
            return should_skip_path(state.repo_path, path)

        snapshot = RepoIndexer(state.repo_path, skip_predicate=skip).snapshot(query=state.task)
        state.repo_context = snapshot.as_context()
        state.project_rules = snapshot.project_rules
        writer.append(
            state.trace_event(
                "repo_indexed",
                {
                    "repo_path": str(state.repo_path),
                    "task": state.task,
                    "tree": snapshot.tree,
                    "symbols": snapshot.symbols,
                },
            )
        )

    def _plan(self, state: AgentState, tools: RepoTools, writer: TraceWriter) -> None:
        state.plan = self.llm.plan(state, tools.descriptions())
        writer.append(state.trace_event("plan", {"plan": state.plan}))

    def _act(self, state: AgentState, tools: RepoTools, writer: TraceWriter) -> None:
        raw = self.llm.act(state, tools.descriptions())
        parse_error = ""
        try:
            call = _parse_tool_call(raw, tools.tool_names)
        except ValueError as exc:
            parse_error = str(exc)
            state.invalid_tool_call_count += 1
            call = ToolCall(
                tool="invalid_tool_call",
                arguments={"raw": _raw_excerpt(raw), "error": parse_error},
                reason="Actor output was not valid JSON.",
            )
            result = ToolResult(
                ok=False,
                output=(
                    f"Invalid actor tool call ({state.invalid_tool_call_count}/2): {parse_error}. "
                    "Retry with exactly one valid JSON tool call."
                ),
                reason=parse_error,
            )
            if state.invalid_tool_call_count >= 2:
                state.done = True
                state.stop_reason = "invalid_tool_call"
            self._record_tool_event(state, writer, raw, parse_error, call, result)
            return
        else:
            state.invalid_tool_call_count = 0

        if call.tool == "run_tests" and not call.arguments.get("command"):
            call = ToolCall(
                tool=call.tool,
                arguments={**call.arguments, "command": state.test_command or "python -m unittest discover -s tests -q"},
                reason=call.reason,
            )

        result = tools.run(call.tool, call.arguments)
        self._record_tool_event(state, writer, raw, parse_error, call, result)

    def _record_tool_event(
        self,
        state: AgentState,
        writer: TraceWriter,
        raw: str,
        parse_error: str,
        call: ToolCall,
        result: ToolResult,
    ) -> None:
        record = ToolRecord(call=call, result=result)
        state.tool_history.append(record)
        state.steps += 1
        writer.append(
            state.trace_event(
                "tool_call",
                {
                    "step": state.steps,
                    "raw": raw,
                    "parse_error": parse_error,
                    "call": call.to_dict(),
                    "result": result.to_dict(),
                },
            )
        )

    def _verify(self, state: AgentState, writer: TraceWriter) -> None:
        last = state.tool_history[-1] if state.tool_history else None
        note = "continue"

        if last is None:
            note = "No tool call has been executed."
        elif last.call.tool == "invalid_tool_call":
            if state.done:
                note = "Actor returned invalid JSON twice; stopping."
            else:
                note = "Actor returned invalid JSON; retry next actor turn."
        elif last.call.tool == "finish":
            state.done = True
            if _latest_test_failed(state):
                state.stop_reason = "finished_with_failed_tests"
                note = "finish tool called after the latest test run failed."
            elif not state.stop_reason:
                state.stop_reason = "finish_called"
                note = "finish tool called."
        elif state.steps >= state.max_steps:
            state.done = True
            state.stop_reason = "max_steps_reached"
            note = "Maximum step count reached."
        elif last.result.blocked:
            note = "Tool call was blocked; continue with a safer action."
        elif last.call.tool == "run_tests" and last.result.ok:
            note = "Tests passed; inspect diff or finish next."
        elif last.call.tool == "run_tests" and not last.result.ok:
            note = "Tests failed; inspect output before another edit."
        elif last.call.tool in {"replace_in_file", "write_file"} and last.result.ok:
            note = "Edit succeeded; run tests next."

        self._trace_verify(state, writer, note)

    def _trace_verify(self, state: AgentState, writer: TraceWriter, note: str) -> None:
        last_tool = state.tool_history[-1].call.tool if state.tool_history else ""
        writer.append(
            state.trace_event(
                "verify",
                {
                    "done": state.done,
                    "note": note,
                    "last_tool": last_tool,
                    "steps": state.steps,
                    "max_steps": state.max_steps,
                    "stop_reason": state.stop_reason,
                },
            )
        )

    def _review(self, state: AgentState, writer: TraceWriter) -> None:
        state.review = self.llm.review(state).strip()
        writer.append(
            state.trace_event(
                "review",
                {
                    "review": state.review,
                    "stop_reason": state.stop_reason,
                    "steps": state.steps,
                },
            )
        )

    def _summarize(self, state: AgentState, writer: TraceWriter) -> None:
        state.final_answer = self.llm.summarize(state).strip()
        if state.trace_path and "Trace:" not in state.final_answer:
            state.final_answer += f"\nTrace: {state.trace_path}"
        writer.append(state.trace_event("final_summary", {"summary": state.final_answer, "stop_reason": state.stop_reason}))


def run_agent(
    repo_path: str | Path,
    task: str,
    test_command: str | None = None,
    config: AgentConfig | None = None,
    llm: AgentLLM | None = None,
    max_steps: int | None = None,
    trace_dir: str | Path | None = None,
) -> AgentState:
    resolved_config = config or AgentConfig.from_env()
    state = AgentState.initial(
        repo_path=Path(repo_path).resolve(),
        task=task,
        test_command=test_command,
        max_steps=resolved_config.max_steps if max_steps is None else max_steps,
    )
    runtime = CodingAgentRuntime(config=resolved_config, llm=llm, trace_dir=trace_dir)
    return runtime.run(state)


def _parse_tool_call(raw: str, allowed_tools: list[str]) -> ToolCall:
    payload = _extract_json_object(raw)
    return ToolCall.from_mapping(payload, allowed_tools=allowed_tools)


def _latest_test_failed(state: AgentState) -> bool:
    for record in reversed(state.tool_history):
        if record.call.tool == "run_tests":
            return not record.result.ok
    return False


def _raw_excerpt(raw: str, limit: int = 500) -> str:
    return raw if len(raw) <= limit else raw[:limit] + "\n... output truncated"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match is None:
        raise ValueError(f"No JSON object found in actor output: {text[:300]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Actor output JSON is not an object.")
    return parsed
