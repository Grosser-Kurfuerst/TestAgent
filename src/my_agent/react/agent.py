from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
from typing import Any, Callable

from my_agent.budget import AgentBudget
from my_agent.config import AgentConfig
from my_agent.hitl.handler import HitlHandler
from my_agent.hitl.types import ApprovalEvent
from my_agent.llm import AgentLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message, MessageLike, messages_to_openai
from my_agent.memory import MemoryManager
from my_agent.memory.token import estimate_tokens
from my_agent.agent_base import AgentBase
from my_agent.schema import AgentState, ToolCall, ToolRecord, ToolResult
from my_agent.tools import RepoTools, ToolExecutionResult, ToolInvocation
from my_agent.tracing import TraceWriter
from my_agent.utils.answers import append_trace_to_answer

EventSink = Callable[[Any], None]


class ReActAgent(AgentBase):
    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: AgentLLM,
        trace_dir: str | Path,
        command_timeout: int,
        event_sink: EventSink | None = None,
        memory_manager: MemoryManager | None = None,
        hitl_handler: HitlHandler | None = None,
        role_prompt: str | None = None,
        run_label: str = "native_tool_calls",
    ) -> None:
        super().__init__(
            config=config,
            llm=llm,
            trace_dir=trace_dir,
            command_timeout=command_timeout,
            event_sink=event_sink,
            memory_manager=memory_manager,
            hitl_handler=hitl_handler,
        )
        self.role_prompt = role_prompt
        self.run_label = run_label
        self._approval_event_lock = threading.Lock()
        self._approval_event_buffer: list[ApprovalEvent] | None = None

    def run(self, state: AgentState) -> AgentState:
        state.repo_path = Path(state.repo_path).resolve()
        if state.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        with self.open_run_context(state) as ctx:
            writer = ctx.writer
            memory = ctx.memory
            tools = RepoTools(
                state.repo_path,
                timeout=self.command_timeout,
                config=self.config,
                run_id=state.run_id,
                cancellation_token=state.cancellation_token,
                hitl_handler=self.hitl_handler,
                approval_observer=lambda event: self._observe_approval_event(writer, state, event),
            )
            tool_definitions = tools.tool_definitions()
            budget = AgentBudget.from_config(self.config, max_steps=state.max_steps)

            self._emit(
                writer,
                state.trace_event(
                    "run.started",
                    {"repo_path": str(state.repo_path), "task": state.task, "mode": self.run_label},
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
            snapshot = ctx.repo_snapshot
            if snapshot is None:
                raise RuntimeError("ReAct runtime requires repository context.")
            state.repo_context = snapshot.as_context()
            state.project_rules = snapshot.project_rules
            state.plan = "Use native ReAct tool calls to inspect, edit, verify, and finish the task."

            base_messages = self._initial_messages(state)
            memory.append_user_message(state.task, run_id=state.run_id)
            while not state.done:
                if _is_cancelled(state):
                    self._stop_cancelled(state, writer)
                    break
                stop_reason = budget.check_before_llm()
                if stop_reason:
                    self._stop_by_budget(state, writer, budget, stop_reason)
                    break

                messages, _, _ = memory.prepare_messages(
                    base_messages=base_messages,
                    query=state.task,
                    tools=tool_definitions,
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
                            "estimated_tokens": _estimate_prompt_tokens(messages, tool_definitions),
                        },
                    )
                )
                response = self._chat(
                    messages,
                    tool_definitions,
                    writer,
                    state,
                    memory=memory,
                    base_messages=base_messages,
                )
                if response is None:
                    break
                if _is_cancelled(state):
                    self._stop_cancelled(state, writer)
                    break
                memory.append_assistant_response(response, run_id=state.run_id)

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
                results = self._execute_tool_calls(state, writer, tools, response.tool_calls, budget)
                budget.record_tool_results(results, response.tool_calls)
                for result in results:
                    memory.append_tool_result(result, run_id=state.run_id)

                self._verify_after_tools(state, writer)
                stop_reason = budget.check_after_tools()
                if stop_reason and not state.done:
                    self._stop_by_budget(state, writer, budget, stop_reason)

            memory.extract_facts(reason="run_completed", run_id=state.run_id)
            self._finalize(state, writer, budget)
            return state

    def _initial_messages(self, state: AgentState) -> list[Message]:
        system = self.role_prompt or (
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
        messages: list[MessageLike],
        tool_definitions: list[dict[str, Any]],
        writer: TraceWriter,
        state: AgentState,
        *,
        memory: MemoryManager,
        base_messages: list[MessageLike],
    ) -> ChatResponse | None:
        try:
            return self.llm.chat(messages, tools=tool_definitions)
        except RuntimeError as exc:
            if _looks_like_context_error(str(exc)):
                retry_messages, _, _ = memory.prepare_messages(
                    base_messages=base_messages,
                    query=state.task,
                    tools=tool_definitions,
                    force_compact=True,
                    focus="Retry after context length error.",
                )
                self._emit(
                    writer,
                    state.trace_event(
                        "memory.compaction_retry",
                        {"reason": "context_length_error", "message_count": len(retry_messages)},
                    )
                )
                try:
                    return self.llm.chat(retry_messages, tools=tool_definitions)
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
        budget: AgentBudget,
    ) -> list[ToolExecutionResult]:
        self._emit_event(ApprovalEvent(event="render.flush_requested", payload={"reason": "before_tool_calls"}))
        prepared: list[tuple[LLMToolCall, ToolInvocation | ToolExecutionResult, bool]] = []
        invocations: list[ToolInvocation] = []
        slots_used = 0
        allowed = min(max(0, state.max_steps - state.steps), max(0, budget.max_tool_calls - budget.tool_calls))
        for call in tool_calls:
            if _is_cancelled(state):
                result = ToolExecutionResult(
                    id=call.id,
                    name=call.name,
                    ok=False,
                    content=f"Tool call cancelled: {state.cancellation_token.reason if state.cancellation_token else 'cancelled'}",
                    error_code="cancelled",
                    retryable=False,
                )
                prepared.append((call, result, False))
            elif slots_used >= allowed:
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
                prepared.append((call, result, False))
            elif call.arguments_error:
                result = ToolExecutionResult(
                    id=call.id,
                    name=call.name,
                    ok=False,
                    content=f"Tool arguments JSON is invalid: {call.arguments_error}",
                    error_code="invalid_arguments_json",
                    retryable=True,
                )
                prepared.append((call, result, True))
                slots_used += 1
            else:
                invocation = _invocation_from_tool_call(call, default_test_command=state.test_command)
                self._emit(
                    writer,
                    state.trace_event(
                        "tool.started",
                        {"id": call.id, "name": call.name, "arguments": invocation.arguments_json},
                    )
                )
                invocations.append(invocation)
                prepared.append((call, invocation, True))
                slots_used += 1

        executed: dict[str, ToolExecutionResult] = {}
        if invocations:
            buffered_approval_events: list[ApprovalEvent] = []
            self._emit(
                writer,
                state.trace_event(
                    "tool.batch.started",
                    {
                        "count": len(invocations),
                        "parallel": False,
                        "ids": [invocation.id for invocation in invocations],
                        "requested": True,
                    },
                ),
            )
            previous_buffer = self._set_approval_event_buffer(buffered_approval_events)
            try:
                batch_results = tools.execute_tools(invocations)
            finally:
                self._set_approval_event_buffer(previous_buffer)
            executed = {result.id: result for result in batch_results}
            self._flush_approval_events(writer, state, buffered_approval_events, tool_calls)
            batch_summary = dict(getattr(tools.registry, "last_execution_summary", {"groups": []}))
            groups = list(batch_summary.get("groups", []))
            self._emit(
                writer,
                state.trace_event(
                    "tool.batch.completed",
                    {
                        "count": len(batch_results),
                        "parallel": any(bool(group.get("parallel")) for group in groups if isinstance(group, dict)),
                        "groups": groups,
                        "timed_out": any(result.timed_out for result in batch_results),
                        "cancelled": any(result.error_code == "cancelled" for result in batch_results),
                    },
                ),
            )

        results: list[ToolExecutionResult] = []
        for call, item, count_step in prepared:
            result = executed[item.id] if isinstance(item, ToolInvocation) else item
            results.append(result)
            self._record_tool_result(state, writer, call, result, count_step=count_step)
            self._emit(writer, state.trace_event("tool.completed", result.to_dict()))
        return results

    def _record_tool_result(
        self,
        state: AgentState,
        writer: TraceWriter,
        call: LLMToolCall,
        result: ToolExecutionResult,
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

    def _stop_cancelled(self, state: AgentState, writer: TraceWriter) -> None:
        state.done = True
        state.stop_reason = "cancelled"
        state.final_answer = "Cancelled."
        self._emit(
            writer,
            state.trace_event(
                "run.cancelled",
                {"reason": state.cancellation_token.reason if state.cancellation_token else "cancelled"},
            ),
        )

    def _finalize(self, state: AgentState, writer: TraceWriter, budget: AgentBudget) -> None:
        if not state.final_answer:
            state.final_answer = _deterministic_final_answer(state)
        state.final_answer = append_trace_to_answer(state.final_answer, state.trace_path)
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
        self._emit_event(event)

    def _handle_approval_event(self, writer: TraceWriter, state: AgentState, event: ApprovalEvent) -> None:
        self._emit_trace(writer, state, event.event, dict(event.payload))
        self._emit_event(event)

    def _observe_approval_event(self, writer: TraceWriter, state: AgentState, event: ApprovalEvent) -> None:
        with self._approval_event_lock:
            if self._approval_event_buffer is not None:
                self._approval_event_buffer.append(event)
                return
        self._handle_approval_event(writer, state, event)

    def _set_approval_event_buffer(self, buffer: list[ApprovalEvent] | None) -> list[ApprovalEvent] | None:
        with self._approval_event_lock:
            previous = self._approval_event_buffer
            self._approval_event_buffer = buffer
            return previous

    def _flush_approval_events(
        self,
        writer: TraceWriter,
        state: AgentState,
        events: list[ApprovalEvent],
        tool_calls: list[LLMToolCall],
    ) -> None:
        order = {call.id: index for index, call in enumerate(tool_calls)}
        indexed = list(enumerate(events))
        indexed.sort(key=lambda item: (order.get(str(item[1].payload.get("tool_call_id", "")), len(order)), item[0]))
        for _, event in indexed:
            self._handle_approval_event(writer, state, event)


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


def _arguments_for_history(call: LLMToolCall) -> dict[str, Any]:
    if call.arguments_error:
        return {"raw": call.arguments_json, "error": call.arguments_error}
    return dict(call.arguments)


def _latest_test_failed(state: AgentState) -> bool:
    for record in reversed(state.tool_history):
        if record.call.tool == "run_tests":
            return not record.result.ok
    return False


def _is_cancelled(state: AgentState) -> bool:
    return bool(state.cancellation_token is not None and state.cancellation_token.is_cancelled())


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


def _estimate_prompt_tokens(messages: list[MessageLike], tools: list[dict[str, Any]]) -> int:
    return estimate_tokens({"messages": messages_to_openai(messages), "tools": tools or []})
