from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

from my_agent.budget import AgentBudget
from my_agent.config import AgentConfig
from my_agent.context import ContextProfile, ConversationCompactor, truncate_tool_content
from my_agent.indexer import RepoIndexer
from my_agent.llm import AgentLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message
from my_agent.schema import AgentState, ToolCall, ToolRecord, ToolResult
from my_agent.tools import RepoTools, ToolExecutionResult, ToolInvocation, should_skip_path
from my_agent.tracing import TraceWriter

EventSink = Callable[[Any], None]


class ReActRuntime:
    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: AgentLLM,
        trace_dir: str | Path,
        command_timeout: int,
        event_sink: EventSink | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.event_sink = event_sink

    def run(self, state: AgentState) -> AgentState:
        state.repo_path = Path(state.repo_path).resolve()
        if state.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        writer = TraceWriter.create(self.trace_dir, state.run_id)
        state.trace_path = writer.path
        tools = RepoTools(state.repo_path, timeout=self.command_timeout, config=self.config)
        tool_definitions = tools.tool_definitions()
        profile = ContextProfile.from_config(self.config)
        compactor = ConversationCompactor(profile, llm=self.llm)
        budget = AgentBudget.from_config(self.config, max_steps=state.max_steps)

        self._emit(
            writer,
            state.trace_event(
                "run.started",
                {"repo_path": str(state.repo_path), "task": state.task, "mode": "native_tool_calls"},
            )
        )
        self._emit(
            writer,
            state.trace_event(
                "tools.loaded",
                {
                    "count": len(tools.registry.tools),
                    "tools": [
                        {
                            "name": tool.spec.name,
                            "source": tool.spec.source,
                            "risk": tool.spec.risk.value,
                            "enabled": tool.spec.enabled,
                        }
                        for tool in tools.registry.tools
                    ],
                },
            )
        )
        self._index_repo(state, writer)
        state.plan = "Use native ReAct tool calls to inspect, edit, verify, and finish the task."

        messages = self._initial_messages(state)
        while not state.done:
            stop_reason = budget.check_before_llm()
            if stop_reason:
                self._stop_by_budget(state, writer, budget, stop_reason)
                break

            compacted = compactor.compact_if_needed(messages, tool_definitions)
            if compacted.compacted:
                self._emit(
                    writer,
                    state.trace_event(
                        "context.compacted",
                        {
                            "before_tokens": compacted.before_tokens,
                            "after_tokens": compacted.after_tokens,
                            "summary_chars": compacted.summary_chars,
                            "fallback": compacted.fallback,
                        },
                    )
                )

            budget.begin_iteration()
            self._emit(
                writer,
                state.trace_event(
                    "llm.requested",
                    {
                        "iteration": budget.iterations,
                        "message_count": len(messages),
                        "tool_count": len(tool_definitions),
                        "estimated_tokens": compactor.estimate_tokens(messages, tool_definitions),
                    },
                )
            )
            response = self._chat(messages, tool_definitions, compactor, writer, state)
            if response is None:
                break

            budget.record_usage(response.usage)
            self._emit(
                writer,
                state.trace_event(
                    "llm.completed",
                    {
                        "iteration": budget.iterations,
                        "finish_reason": response.finish_reason,
                        "content_chars": len(response.content),
                        "tool_calls": [
                            {"id": call.id, "name": call.name, "arguments": call.arguments_json}
                            for call in response.tool_calls
                        ],
                        "usage": response.usage.to_dict(),
                    },
                )
            )

            if not response.tool_calls:
                state.done = True
                state.stop_reason = "assistant_final"
                state.final_answer = response.content.strip() or "No final response returned."
                break

            budget.record_tool_calls(response.tool_calls)
            messages.append(Message(role="assistant", content=response.content or "", tool_calls=response.tool_calls))
            results = self._execute_tool_calls(state, writer, tools, response.tool_calls, profile, budget)
            budget.record_tool_results(results, response.tool_calls)
            for result in results:
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=result.id,
                        content=_tool_message_content(result, profile.max_tool_result_chars),
                    )
                )

            self._verify_after_tools(state, writer)
            stop_reason = budget.check_after_tools()
            if stop_reason and not state.done:
                self._stop_by_budget(state, writer, budget, stop_reason)

        self._finalize(state, writer, budget)
        return state

    def _index_repo(self, state: AgentState, writer: TraceWriter) -> None:
        snapshot = RepoIndexer(
            state.repo_path,
            skip_predicate=lambda path: should_skip_path(state.repo_path, path),
        ).snapshot(query=state.task)
        state.repo_context = snapshot.as_context()
        state.project_rules = snapshot.project_rules
        self._emit(
            writer,
            state.trace_event(
                "repo.indexed",
                {
                    "repo_path": str(state.repo_path),
                    "task": state.task,
                    "tree": snapshot.tree,
                    "symbols": snapshot.symbols,
                },
            )
        )

    def _initial_messages(self, state: AgentState) -> list[Message]:
        system = (
            "You are a careful coding agent. Use the provided function tools when repository inspection, edits, "
            "or verification are needed. Keep tool arguments as valid JSON objects. Inspect files before editing, "
            "run relevant tests after edits, and return a final assistant answer when the task is complete."
        )
        user = (
            f"Date: {date.today().isoformat()}\n"
            f"Repository: {state.repo_path}\n\n"
            f"Task:\n{state.task}\n\n"
            f"Repository context:\n{state.repo_context}\n\n"
            f"Project rules:\n{state.project_rules or 'No project rules found.'}\n\n"
            f"Default test command: {state.test_command or 'not configured'}"
        )
        return [Message(role="system", content=system), Message(role="user", content=user)]

    def _chat(
        self,
        messages: list[Message],
        tool_definitions: list[dict[str, Any]],
        compactor: ConversationCompactor,
        writer: TraceWriter,
        state: AgentState,
    ) -> ChatResponse | None:
        try:
            return self.llm.chat(messages, tools=tool_definitions)
        except RuntimeError as exc:
            if _looks_like_context_error(str(exc)):
                compacted = compactor.compact_now(messages, tool_definitions, focus="Retry after context length error.")
                self._emit(
                    writer,
                    state.trace_event(
                        "context.compacted",
                        {
                            "before_tokens": compacted.before_tokens,
                            "after_tokens": compacted.after_tokens,
                            "summary_chars": compacted.summary_chars,
                            "fallback": compacted.fallback,
                            "forced": True,
                        },
                    )
                )
                try:
                    return self.llm.chat(messages, tools=tool_definitions)
                except RuntimeError as retry_exc:
                    self._stop_by_llm_failure(state, writer, retry_exc)
                    return None
            self._stop_by_llm_failure(state, writer, exc)
            return None

    def _execute_tool_calls(
        self,
        state: AgentState,
        writer: TraceWriter,
        tools: RepoTools,
        tool_calls: list[LLMToolCall],
        profile: ContextProfile,
        budget: AgentBudget,
    ) -> list[ToolExecutionResult]:
        results: list[ToolExecutionResult] = []
        for call in tool_calls:
            if state.steps >= state.max_steps or budget.tool_calls + len(results) >= budget.max_tool_calls:
                result = ToolExecutionResult(
                    id=call.id,
                    name=call.name,
                    ok=False,
                    content="Tool call skipped because the maximum tool-call budget was reached.",
                    error_code="max_tool_calls",
                    retryable=False,
                )
                state.done = True
                state.stop_reason = "max_steps_reached"
                count_step = False
            elif call.arguments_error:
                result = ToolExecutionResult(
                    id=call.id,
                    name=call.name,
                    ok=False,
                    content=f"Tool arguments JSON is invalid: {call.arguments_error}",
                    error_code="invalid_arguments_json",
                    retryable=True,
                )
                count_step = True
            else:
                invocation = _invocation_from_tool_call(call, default_test_command=state.test_command)
                self._emit(
                    writer,
                    state.trace_event(
                        "tool.started",
                        {"id": call.id, "name": call.name, "arguments": invocation.arguments_json},
                    )
                )
                result = tools.execute([invocation])[0]
                count_step = True

            results.append(result)
            self._record_tool_result(state, writer, call, result, profile, count_step=count_step)
            self._emit(writer, state.trace_event("tool.completed", result.to_dict()))
        return results

    def _record_tool_result(
        self,
        state: AgentState,
        writer: TraceWriter,
        call: LLMToolCall,
        result: ToolExecutionResult,
        profile: ContextProfile,
        *,
        count_step: bool = True,
    ) -> None:
        tool_call = ToolCall(tool=call.name, arguments=_arguments_for_history(call), reason="Native LLM tool_call.")
        tool_result = ToolResult(ok=result.ok, output=result.content, blocked=result.blocked, reason=result.error_code)
        state.tool_history.append(ToolRecord(call=tool_call, result=tool_result))
        if count_step:
            state.steps += 1

    def _verify_after_tools(self, state: AgentState, writer: TraceWriter) -> None:
        last = state.tool_history[-1] if state.tool_history else None
        note = "continue"
        if last is None:
            note = "No tool calls executed."
        elif last.call.tool == "finish":
            state.done = True
            state.stop_reason = "finished_with_failed_tests" if _latest_test_failed(state) else "finish_called"
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

        self._emit(
            writer,
            state.trace_event(
                "run.status",
                {
                    "done": state.done,
                    "note": note,
                    "last_tool": last.call.tool if last else "",
                    "steps": state.steps,
                    "max_steps": state.max_steps,
                    "stop_reason": state.stop_reason,
                },
            )
        )

    def _stop_by_budget(self, state: AgentState, writer: TraceWriter, budget: AgentBudget, reason: str) -> None:
        state.done = True
        state.stop_reason = reason
        self._emit(writer, state.trace_event("budget.exceeded", {"reason": reason, "budget": budget.to_dict()}))

    def _stop_by_llm_failure(self, state: AgentState, writer: TraceWriter, exc: Exception) -> None:
        state.done = True
        state.stop_reason = "llm_failed"
        self._emit(writer, state.trace_event("llm.failed", {"error": f"{type(exc).__name__}: {exc}"}))

    def _finalize(self, state: AgentState, writer: TraceWriter, budget: AgentBudget) -> None:
        if not state.final_answer:
            state.final_answer = _deterministic_final_answer(state)
        if state.trace_path and "Trace:" not in state.final_answer:
            state.final_answer += f"\nTrace: {state.trace_path}"
        if not state.review:
            state.review = _deterministic_review(state)
        self._emit(
            writer,
            state.trace_event(
                "run.completed",
                {
                    "stop_reason": state.stop_reason,
                    "steps": state.steps,
                    "done": state.done,
                    "review": state.review,
                    "final_answer": state.final_answer,
                    "budget": budget.to_dict(),
                },
            )
        )

    def _emit(self, writer: TraceWriter, event: Any) -> None:
        writer.append(event)
        if self.event_sink is not None:
            self.event_sink(event)


def _invocation_from_tool_call(call: LLMToolCall, *, default_test_command: str | None) -> ToolInvocation:
    arguments = dict(call.arguments)
    if call.name == "run_tests" and not arguments.get("command") and default_test_command:
        arguments["command"] = default_test_command
        return ToolInvocation.from_arguments(name=call.name, arguments=arguments, invocation_id=call.id)
    return ToolInvocation(
        id=call.id,
        name=call.name,
        arguments_json=call.arguments_json,
        parsed_arguments=arguments,
    )


def _tool_message_content(result: ToolExecutionResult, limit: int) -> str:
    payload = result.to_dict()
    payload["content"] = truncate_tool_content(result.content, limit)
    if len(result.content) > limit:
        payload["original_content_chars"] = len(result.content)
    return json.dumps(payload, ensure_ascii=False)


def _arguments_for_history(call: LLMToolCall) -> dict[str, Any]:
    if call.arguments_error:
        return {"raw": call.arguments_json, "error": call.arguments_error}
    return dict(call.arguments)


def _latest_test_failed(state: AgentState) -> bool:
    for record in reversed(state.tool_history):
        if record.call.tool == "run_tests":
            return not record.result.ok
    return False


def _deterministic_review(state: AgentState) -> str:
    tests = "not run"
    for record in reversed(state.tool_history):
        if record.call.tool == "run_tests":
            tests = "passed" if record.result.ok else "failed"
            break
    return f"Native ReAct review: stop_reason={state.stop_reason or 'assistant_final'}, steps={state.steps}, tests={tests}."


def _deterministic_final_answer(state: AgentState) -> str:
    finish = next((record for record in reversed(state.tool_history) if record.call.tool == "finish"), None)
    if finish is not None:
        return finish.result.output
    return f"Stopped before an assistant final answer. Stop reason: {state.stop_reason or 'unknown'}."


def _looks_like_context_error(message: str) -> bool:
    lowered = message.lower()
    return "context" in lowered and ("length" in lowered or "window" in lowered or "maximum" in lowered)
