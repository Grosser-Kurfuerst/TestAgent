from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from my_agent.config import AgentConfig
from my_agent.context import AgentContextManager, ContextOverBudgetError
from my_agent.hitl.handler import HitlHandler
from my_agent.llm import AgentLLM
from my_agent.memory import MemoryManager
from my_agent.plan.executor import PlanEvent, PlanExecutor, ReActTaskRunner
from my_agent.plan.graph import PlanValidationError
from my_agent.plan.planner import Planner
from my_agent.plan.rendering import render_plan, render_plan_final_answer, render_plan_review
from my_agent.plan.store import JsonPlanStore, PlanStore
from my_agent.plan.types import PlanState, PlanStatus, TaskStatus
from my_agent.agent_base import AgentBase
from my_agent.schema import AgentState
from my_agent.tracing import TraceWriter, append_agent_completed
from my_agent.utils.text import terminal_summary_text


class PlanReviewAction(str, Enum):
    EXECUTE = "execute"
    SUPPLEMENT = "supplement"
    CANCEL = "cancel"


@dataclass(frozen=True)
class PlanReviewDecision:
    action: PlanReviewAction
    feedback: str = ""


PlanReviewHandler = Callable[[PlanState], PlanReviewDecision]
EventSink = Callable[[object], None]


class PlanExecuteAgent(AgentBase):
    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: AgentLLM,
        trace_dir: str | Path,
        command_timeout: int,
        event_sink: EventSink | None = None,
        review_handler: PlanReviewHandler | None = None,
        state_store: PlanStore | None = None,
        require_approval: bool = False,
        memory_manager: MemoryManager | None = None,
        hitl_handler: HitlHandler | None = None,
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
        self.review_handler = review_handler or _default_review_handler
        self.state_store = state_store or JsonPlanStore(self.trace_dir / "plans")
        self.require_approval = require_approval

    def run(self, state: AgentState) -> AgentState:
        state.repo_path = Path(state.repo_path).resolve()
        repo = state.repo_path
        goal = state.task
        test_command = state.test_command
        effective_max_steps = state.max_steps
        with self.open_run_context(state) as ctx:
            writer = ctx.writer
            memory = ctx.memory
            self._emit_trace(
                writer,
                state,
                "plan.requested",
                {"repo_path": str(repo), "goal": goal, "require_approval": self.require_approval},
            )
            memory.append_task_goal(goal, run_id=state.run_id)

            try:
                if ctx.repo_snapshot is None:
                    raise RuntimeError("Plan execution requires repository context.")
                repo_context = ctx.repo_context
                planner = Planner(self.llm, max_tasks=self.config.plan_max_tasks, trace_sink=lambda event, payload: self._emit_trace(writer, state, event, payload))
                context_manager = AgentContextManager(memory.context_profile)
                planner_base_messages = planner._build_messages(goal, repo_context=repo_context, conversation=[])
                budget_plan = context_manager.budget_for_messages(base_messages=planner_base_messages, tools=[])
                memory_context = memory.build_context_for_query(goal, max_tokens=budget_plan.long_term_limit)
                planner_context = _append_memory_context(repo_context, memory_context.injected_text)
                planner_messages_with_memory = planner._build_messages(goal, repo_context=planner_context, conversation=[])
                fixed_with_memory_tokens = context_manager.estimate_tokens(planner_messages_with_memory, [])
                planner_memory_payload = {
                    "phase": "plan_planner",
                    "message_count": len(planner_messages_with_memory),
                    "memory_hits": len(memory_context.hits),
                    "memory_tokens": memory_context.estimated_tokens,
                    "estimated_prompt_tokens": fixed_with_memory_tokens,
                    "fixed_tokens": budget_plan.fixed_tokens,
                    "fixed_with_memory_tokens": fixed_with_memory_tokens,
                    "memory_budget_tokens": budget_plan.memory_budget_tokens,
                    "long_term_limit": budget_plan.long_term_limit,
                    "short_term_allowed": max(0, budget_plan.prompt_limit_tokens - fixed_with_memory_tokens),
                    "compacted": False,
                    "over_budget": fixed_with_memory_tokens >= context_manager.profile.compression_trigger_tokens,
                }
                memory.trace_context_event(
                    "memory.prepared",
                    context_manager._trace_payload(planner_memory_payload),
                )
                context_manager.raise_if_over_budget(
                    memory=memory,
                    estimated_prompt_tokens=fixed_with_memory_tokens,
                    payload=planner_memory_payload,
                )
                trace_sink = lambda event, payload: self._emit_trace(writer, state, event, payload)
                planner.trace_sink = trace_sink
                plan = planner.create_plan(
                    goal,
                    repo_context=planner_context,
                )
                plan.trace_path = str(writer.path)
                plan.status = PlanStatus.AWAITING_APPROVAL
                self.state_store.save(plan)
                self._emit_plan_state(writer, state, "plan.created", plan)
                self._emit_plan_state(writer, state, "plan.approval_requested", plan)

                decision = self.review_handler(plan) if self.require_approval else PlanReviewDecision(PlanReviewAction.EXECUTE)
                if decision.action == PlanReviewAction.CANCEL:
                    plan.status = PlanStatus.CANCELLED
                    plan.error = decision.feedback or "Plan execution was cancelled before running tasks."
                    self.state_store.save(plan)
                    self._emit_plan_state(writer, state, "plan.cancelled", plan)
                    return self._final_state(state, plan, writer)
                if decision.action == PlanReviewAction.SUPPLEMENT:
                    supplement = decision.feedback.strip()
                    if supplement:
                        plan = Planner(self.llm, max_tasks=self.config.plan_max_tasks, trace_sink=trace_sink).create_plan(
                            f"{goal}\n\nAdditional requirements:\n{supplement}",
                            repo_context=planner_context,
                        )
                        plan.trace_path = str(writer.path)
                        plan.status = PlanStatus.AWAITING_APPROVAL
                        self.state_store.save(plan)
                        self._emit_plan_state(writer, state, "plan.created", plan)

                runner = ReActTaskRunner(
                    repo_path=repo,
                    config=self.config,
                    llm=self.llm,
                    trace_dir=self.trace_dir / plan.id,
                    command_timeout=self.command_timeout,
                    test_command=test_command,
                    default_max_steps=effective_max_steps,
                    plan_task_max_steps=self.config.plan_task_max_steps,
                    event_sink=self.event_sink,
                    memory_manager=memory,
                    hitl_handler=self.hitl_handler,
                )
                executor = PlanExecutor(
                    runner,
                    store=self.state_store,
                    event_sink=lambda event: self._handle_executor_event(writer, state, event),
                    max_tasks=self.config.plan_max_tasks,
                    parallel_enabled=self.config.plan_parallel_enabled,
                    max_parallel_tasks=self.config.plan_max_parallel_tasks,
                    batch_timeout_seconds=self.config.plan_task_batch_timeout_seconds,
                    shutdown_grace_seconds=self.config.tool_shutdown_grace_seconds,
                    cancellation_token=state.cancellation_token,
                )
                completed = executor.execute(plan)
                self._record_plan_task_summaries(memory, completed, run_id=state.run_id)
                memory.extract_facts(reason="plan_completed", run_id=state.run_id)
                return self._final_state(state, completed, writer)
            except PlanValidationError as exc:
                plan = PlanState.create(goal=goal, summary="Plan validation failed.")
                plan.status = PlanStatus.FAILED
                plan.error = str(exc)
                plan.trace_path = str(writer.path)
                self.state_store.save(plan)
                self._emit_trace(
                    writer,
                    state,
                    "plan.validation_failed",
                    {"code": exc.code, "message": exc.message, "details": exc.details, "plan": plan.to_dict()},
                )
                self._emit_plan_event("plan.validation_failed", plan)
                return self._final_state(state, plan, writer, stop_reason="plan_validation_failed")
            except ContextOverBudgetError as exc:
                plan = PlanState.create(goal=goal, summary="Plan stopped because context budget was exceeded.")
                plan.status = PlanStatus.FAILED
                plan.error = str(exc)
                plan.trace_path = str(writer.path)
                self.state_store.save(plan)
                self._emit_trace(
                    writer,
                    state,
                    "plan.failed",
                    {
                        "error": str(exc),
                        "reason": "context_over_budget",
                        "context": exc.payload,
                        "plan": plan.to_dict(),
                    },
                )
                self._emit_plan_event("plan.failed", plan)
                return self._final_state(state, plan, writer, stop_reason="context_over_budget")
            except RuntimeError as exc:
                plan = PlanState.create(goal=goal, summary="Plan failed before execution.")
                plan.status = PlanStatus.FAILED
                plan.error = str(exc)
                plan.trace_path = str(writer.path)
                self.state_store.save(plan)
                self._emit_trace(writer, state, "plan.failed", {"error": str(exc), "plan": plan.to_dict()})
                self._emit_plan_event("plan.failed", plan)
                return self._final_state(state, plan, writer, stop_reason="plan_failed")

    def _record_plan_task_summaries(self, memory: MemoryManager, plan: PlanState, *, run_id: str) -> None:
        for task in plan.tasks:
            if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED}:
                continue
            details = terminal_summary_text(task.result, task.error, "No result recorded.")
            memory.append_summary(
                f"[plan task {task.id} {task.status.value}] {task.title}\n{details}",
                source="plan",
                run_id=run_id,
                metadata={
                    "plan_id": plan.id,
                    "task_id": task.id,
                    "task_status": task.status.value,
                    "task_type": task.type.value,
                },
            )

    def _handle_executor_event(self, writer: TraceWriter, state: AgentState, event: PlanEvent) -> None:
        self._emit_trace(writer, state, event.event, event.payload)
        if self.event_sink is not None:
            self.event_sink(event)

    def _emit_plan_state(self, writer: TraceWriter, state: AgentState, event: str, plan: PlanState) -> None:
        self._emit_trace(writer, state, event, {"plan_id": plan.id, "status": plan.status.value, "plan": plan.to_dict()})
        self._emit_plan_event(event, plan)

    def _emit_plan_event(self, event: str, plan: PlanState) -> None:
        if self.event_sink is not None:
            self.event_sink(PlanEvent(event=event, plan_id=plan.id, status=plan.status.value, payload={"plan": plan.to_dict()}))

    def _final_state(
        self,
        state: AgentState,
        plan: PlanState,
        writer: TraceWriter,
        *,
        stop_reason: str | None = None,
    ) -> AgentState:
        state.plan = render_plan(plan)
        state.review = render_plan_review(plan)
        state.final_answer = render_plan_final_answer(plan)
        state.done = True
        state.steps = _plan_child_steps(plan)
        state.stop_reason = stop_reason or _stop_reason_for_plan(plan)
        state.trace_path = Path(plan.trace_path) if plan.trace_path else state.trace_path
        append_agent_completed(
            writer,
            state,
            mode="plan",
            run_label="plan_execute",
            child_trace_paths=_plan_child_trace_paths(plan),
        )
        return state


def _default_review_handler(plan: PlanState) -> PlanReviewDecision:
    return PlanReviewDecision(PlanReviewAction.EXECUTE)


def _append_memory_context(repo_context: str, memory_context: str) -> str:
    if not memory_context.strip():
        return repo_context
    return f"{repo_context.rstrip()}\n\nMemory context:\n{memory_context.strip()}"


def _stop_reason_for_plan(plan: PlanState) -> str:
    if plan.status == PlanStatus.SUCCEEDED:
        return "plan_completed"
    if plan.status == PlanStatus.CANCELLED:
        return "plan_cancelled"
    if any(task.status == TaskStatus.FAILED for task in plan.tasks):
        return "plan_failed"
    if plan.status == PlanStatus.FAILED:
        return "plan_failed"
    return f"plan_{plan.status.value}"


def _plan_child_trace_paths(plan: PlanState) -> list[str]:
    return [task.trace_path for task in plan.tasks if task.trace_path]


def _plan_child_steps(plan: PlanState) -> int:
    return sum(_trace_steps(Path(trace_path)) for trace_path in _plan_child_trace_paths(plan))


def _trace_steps(trace_path: Path) -> int:
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    last_steps = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("event") not in {"run.completed", "agent.completed"}:
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        value = payload.get("steps")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            last_steps = value
    return last_steps
