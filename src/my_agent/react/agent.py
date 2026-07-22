from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
import threading
from typing import Any, Callable

from my_agent.budget import AgentBudget
from my_agent.config import AgentConfig
from my_agent.context import AgentContextManager, ContextOverBudgetError, ToolSchemaBudget, budget_tool_definitions
from my_agent.hitl.handler import HitlHandler
from my_agent.hitl.types import ApprovalEvent
from my_agent.llm import AgentLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message, MessageLike, messages_to_openai
from my_agent.memory.api import MemoryService
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact
from my_agent.memory.evolver.writing.legacy import (
    build_write_steps_from_tool_history,
    runtime_outcome_from_tool_records,
)
from my_agent.memory.token import estimate_tokens
from my_agent.agent_base import AgentBase
from my_agent.policy.chat_template import canonicalize_messages, canonicalize_tools
from my_agent.policy.contracts import DecisionRequest
from my_agent.schema import AgentState, ToolCall, ToolRecord, ToolResult
from my_agent.tools import RepoTools, ToolExecutionResult, ToolInvocation
from my_agent.observability.tracing import TraceWriter, append_agent_completed
from my_agent.training.decision_log import (
    DecisionAttemptError,
    DecisionEventContext,
    DecisionEventRecorder,
)
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
        memory_manager: MemoryService | None = None,
        memory_embedding_retriever: Any | None = None,
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
            memory_embedding_retriever=memory_embedding_retriever,
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
            budget = AgentBudget.from_config(self.config, max_steps=state.max_steps)
            context_profile = getattr(memory, "context_profile", None) or AgentContextManager.from_config(self.config).profile
            context_manager = AgentContextManager(context_profile)
            all_tool_definitions = tools.tool_definitions()
            tool_budget = budget_tool_definitions(all_tool_definitions, context_profile)
            tool_definitions = tool_budget.definitions

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
            if tool_budget.omitted_names or tool_budget.over_budget:
                payload = {
                    "budget_tokens": tool_budget.budget_tokens,
                    "estimated_tokens": tool_budget.estimated_tokens,
                    "included_count": tool_budget.included_count,
                    "omitted_count": tool_budget.omitted_count,
                    "included": list(tool_budget.included_names),
                    "omitted": list(tool_budget.omitted_names),
                    "over_budget": tool_budget.over_budget,
                }
                self._emit(writer, state.trace_event("tools.schema_capped", payload))
            snapshot = ctx.repo_snapshot
            if snapshot is None:
                raise RuntimeError("ReAct runtime requires repository context.")
            state.repo_context = ctx.repo_context
            state.project_rules = snapshot.project_rules
            state.plan = "Use native ReAct tool calls to inspect, edit, verify, and finish the task."

            base_messages = self._initial_messages(state)
            formal_session = None
            decision_recorder = None
            formal_coordinator = memory.evolver_coordinator
            if formal_coordinator is not None:
                metadata = dict(getattr(state, "metadata", {}) or {})
                task_id = str(metadata.get("task_id") or metadata.get("source_task") or "").strip()
                task_group = str(metadata.get("task_group") or "").strip()
                stream_id = str(metadata.get("stream_id") or "").strip()
                if task_id and task_group and stream_id:
                    formal_session = memory.begin_formal_evolver_task(
                        task=state.task,
                        task_id=task_id,
                        task_group=task_group,
                        trajectory_id=state.run_id,
                        stream_id=stream_id,
                    )
                    decision_recorder = (
                        formal_coordinator.decision_recorder
                    )
                else:
                    self._emit(
                        writer,
                        state.trace_event(
                            "memory.evolver_session_skipped",
                            {
                                "reason": "missing_authoritative_task_metadata",
                                "interactive": True,
                            },
                        ),
                    )
            memory.append_task_goal(state.task, run_id=state.run_id)
            while not state.done:
                if _is_cancelled(state):
                    self._stop_cancelled(state, writer)
                    break
                stop_reason = budget.check_before_llm()
                if stop_reason:
                    self._stop_by_budget(state, writer, budget, stop_reason)
                    break

                try:
                    messages, _, _ = context_manager.prepare_messages(
                        base_messages=base_messages,
                        query=state.task,
                        tools=tool_definitions,
                        memory=memory,
                        tool_budget=tool_budget,
                    )
                except ContextOverBudgetError as exc:
                    self._stop_by_context_over_budget(state, writer, exc)
                    break

                budget.begin_iteration()
                self._emit(
                    writer,
                    state.trace_event(
                        "llm.requested",
                        {
                            "phase": "react",
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
                    context_manager=context_manager,
                    base_messages=base_messages,
                    tool_budget=tool_budget,
                    formal_session=formal_session,
                    decision_recorder=decision_recorder,
                    turn_index=budget.iterations - 1,
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
                            "phase": "react",
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
                execution_recorder = (
                    formal_coordinator.runtime_evidence_recorder
                    if formal_session is not None and formal_coordinator is not None
                    else None
                )
                results = self._execute_tool_calls(
                    state,
                    writer,
                    tools,
                    response.tool_calls,
                    budget,
                    tool_budget,
                    formal_session=formal_session,
                    execution_recorder=execution_recorder,
                    decision_id=(
                        str(response.raw.get("decision_id") or "")
                        if formal_session is not None
                        else ""
                    ),
                    decision_turn_index=(
                        int(response.raw.get("decision_turn_index", -1))
                        if formal_session is not None
                        else -1
                    ),
                    decision_step_index=(
                        int(response.raw.get("decision_step_index", -1))
                        if formal_session is not None
                        else -1
                    ),
                )
                budget.record_tool_results(results, response.tool_calls)
                for result in results:
                    memory.append_tool_result(result, run_id=state.run_id)

                self._verify_after_tools(state, writer)
                stop_reason = budget.check_after_tools()
                if stop_reason and not state.done:
                    self._stop_by_budget(state, writer, budget, stop_reason)

            tool_history = [record.to_dict() for record in state.tool_history]
            writer_metadata = dict(getattr(state, "metadata", {}) or {})
            if formal_session is not None and state.trace_path is not None:
                state.evolver_episode = AgentEpisodeArtifact(
                    session=formal_session,
                    trace_path=Path(state.trace_path),
                    stop_reason=state.stop_reason,
                    final_answer=state.final_answer,
                    tool_history=build_write_steps_from_tool_history(tool_history),
                    task=state.task,
                )
                state.evolver_coordinator = formal_coordinator
            elif formal_coordinator is None:
                memory.write_experiences_from_run(
                    task=state.task,
                    run_id=state.run_id,
                    trace_path=state.trace_path,
                    stop_reason=state.stop_reason,
                    final_answer=state.final_answer,
                    tool_history=tool_history,
                    outcome=runtime_outcome_from_tool_records(state.stop_reason, tool_history),
                    outcome_source="runtime",
                    source_task=str(writer_metadata.get("source_task") or writer_metadata.get("task_id") or ""),
                    stream_id=str(writer_metadata.get("stream_id") or ""),
                    task_type=str(writer_metadata.get("task_type") or ""),
                    memory_mode=str(writer_metadata.get("memory_mode") or ""),
                )
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
        memory: MemoryService,
        context_manager: AgentContextManager,
        base_messages: list[MessageLike],
        tool_budget: ToolSchemaBudget,
        formal_session: Any | None,
        decision_recorder: DecisionEventRecorder | None,
        turn_index: int,
    ) -> ChatResponse | None:
        if formal_session is not None:
            if decision_recorder is None:
                raise RuntimeError("formal ReAct generation requires a decision recorder")
            return self._formal_chat(
                messages,
                tool_definitions,
                writer,
                state,
                memory=memory,
                context_manager=context_manager,
                base_messages=base_messages,
                tool_budget=tool_budget,
                formal_session=formal_session,
                recorder=decision_recorder,
                turn_index=turn_index,
            )
        try:
            return self.llm.chat(messages, tools=tool_definitions)
        except RuntimeError as exc:
            if _looks_like_context_error(str(exc)):
                try:
                    retry_messages, _, _ = context_manager.prepare_messages(
                        base_messages=base_messages,
                        query=state.task,
                        tools=tool_definitions,
                        memory=memory,
                        force_compact=True,
                        focus="Retry after context length error.",
                        tool_budget=tool_budget,
                    )
                except ContextOverBudgetError as budget_exc:
                    self._stop_by_context_over_budget(state, writer, budget_exc)
                    return None
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

    def _formal_chat(
        self,
        messages: list[MessageLike],
        tool_definitions: list[dict[str, Any]],
        writer: TraceWriter,
        state: AgentState,
        *,
        memory: MemoryService,
        context_manager: AgentContextManager,
        base_messages: list[MessageLike],
        tool_budget: ToolSchemaBudget,
        formal_session: Any,
        recorder: DecisionEventRecorder,
        turn_index: int,
    ) -> ChatResponse | None:
        retry_of: str | None = None
        active_messages = messages
        for attempt in range(2):
            request = DecisionRequest(
                role="action",
                purpose="fast_loop_evidence",
                messages=canonicalize_messages(active_messages),
                tools=canonicalize_tools(tool_definitions),
                max_new_tokens=int(getattr(self.llm, "default_max_new_tokens", 1_024)),
                temperature=float(self.config.temperature),
                top_p=float(getattr(self.llm, "default_top_p", 0.95)),
            )
            context = DecisionEventContext(
                trajectory_id=formal_session.trajectory_id,
                turn_index=turn_index,
                step_index=state.steps,
                task_id=formal_session.task_id,
                task_group=formal_session.task_group,
                stream_id=formal_session.stream_id,
                memory_project_key=formal_session.memory_project_key,
                run_id=state.run_id,
                repository_revision=formal_session.repository_revision,
                candidate_snapshot_hash=formal_session.candidate_snapshot_hash,
            )
            try:
                logged = recorder.generate(request, context=context, retry_of=retry_of)
                converter = getattr(self.llm, "chat_response_from_decision", None)
                if not callable(converter):
                    raise RuntimeError(
                        "formal policy must convert its exact DecisionResponse to ChatResponse"
                    )
                response = converter(logged.response)
                return replace(response, raw={
                    **response.raw,
                    "decision_id": logged.decision_id,
                    "decision_turn_index": context.turn_index,
                    "decision_step_index": context.step_index,
                })
            except DecisionAttemptError as exc:
                if attempt == 0 and _looks_like_context_error(str(exc)):
                    retry_of = exc.decision_id
                    try:
                        active_messages, _, _ = context_manager.prepare_messages(
                            base_messages=base_messages,
                            query=state.task,
                            tools=tool_definitions,
                            memory=memory,
                            force_compact=True,
                            focus="Retry after context length error.",
                            tool_budget=tool_budget,
                        )
                    except ContextOverBudgetError as budget_exc:
                        self._stop_by_context_over_budget(state, writer, budget_exc)
                        return None
                    self._emit(
                        writer,
                        state.trace_event(
                            "memory.compaction_retry",
                            {"reason": "context_length_error", "message_count": len(active_messages)},
                        ),
                    )
                    continue
                self._stop_by_llm_failure(state, writer, exc.cause)
                return None
        return None

    def _execute_tool_calls(
        self,
        state: AgentState,
        writer: TraceWriter,
        tools: RepoTools,
        tool_calls: list[LLMToolCall],
        budget: AgentBudget,
        tool_budget: ToolSchemaBudget,
        *,
        formal_session: Any | None = None,
        execution_recorder: Any | None = None,
        decision_id: str = "",
        decision_turn_index: int = -1,
        decision_step_index: int = -1,
    ) -> list[ToolExecutionResult]:
        if formal_session is not None and (
            execution_recorder is None
            or not decision_id
            or decision_turn_index < 0
            or decision_step_index < 0
        ):
            raise RuntimeError("formal tool execution requires decision-linked evidence context")
        self._emit_event(ApprovalEvent(event="render.flush_requested", payload={"reason": "before_tool_calls"}))
        prepared: list[tuple[LLMToolCall, ToolInvocation | ToolExecutionResult, bool]] = []
        invocations: list[ToolInvocation] = []
        slots_used = 0
        allowed = min(max(0, state.max_steps - state.steps), max(0, budget.max_tool_calls - budget.tool_calls))
        exposed_tools = set(tool_budget.included_names)
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
            elif call.name not in exposed_tools:
                result = ToolExecutionResult(
                    id=call.id,
                    name=call.name,
                    ok=False,
                    content=(
                        f"Tool '{call.name}' was not exposed to the model because the tool schema budget "
                        "was exceeded. Use one of the tools available in this turn."
                    ),
                    error_code="tool_not_exposed",
                    retryable=True,
                    blocked=True,
                )
                self._emit(
                    writer,
                    state.trace_event(
                        "tool.blocked",
                        {"id": call.id, "name": call.name, "reason": "tool_not_exposed"},
                    ),
                )
                prepared.append((call, result, True))
                slots_used += 1
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
        for call_index, (call, item, count_step) in enumerate(prepared):
            result = executed[item.id] if isinstance(item, ToolInvocation) else item
            results.append(result)
            self._record_tool_result(state, writer, call, result, count_step=count_step)
            if execution_recorder is not None:
                execution_recorder.record_action_execution(
                    session=formal_session,
                    decision_id=decision_id,
                    turn_index=decision_turn_index,
                    step_index=decision_step_index,
                    call_index=call_index,
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    ok=result.ok,
                    blocked=result.blocked,
                    error_code=result.error_code,
                    output=result.content,
                )
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
        self._emit(writer, state.trace_event("llm.failed", {"phase": "react", "error": f"{type(exc).__name__}: {exc}"}))

    def _stop_by_context_over_budget(
        self,
        state: AgentState,
        writer: TraceWriter,
        exc: ContextOverBudgetError,
    ) -> None:
        state.done = True
        state.stop_reason = "context_over_budget"
        state.final_answer = str(exc)
        self._emit(
            writer,
            state.trace_event(
                "llm.skipped",
                {"phase": "react", "reason": "context_over_budget", "error": str(exc), "context": exc.payload},
            ),
        )

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
        append_agent_completed(writer, state, mode="react", run_label=self.run_label)

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
