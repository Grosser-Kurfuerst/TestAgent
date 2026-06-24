from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from my_agent.config import AgentConfig
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
from my_agent.tracing import TraceWriter
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
            memory.append_user_message(goal, run_id=state.run_id)

            try:
                if ctx.repo_snapshot is None:
                    raise RuntimeError("Plan execution requires repository context.")
                repo_context = ctx.repo_snapshot.as_context()
                planner_context = _append_memory_context(repo_context, memory.build_context_for_query(goal).injected_text)
                plan = Planner(self.llm, max_tasks=self.config.plan_max_tasks).create_plan(
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
                    return self._final_state(state, plan)
                if decision.action == PlanReviewAction.SUPPLEMENT:
                    supplement = decision.feedback.strip()
                    if supplement:
                        plan = Planner(self.llm, max_tasks=self.config.plan_max_tasks).create_plan(
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
                )
                completed = executor.execute(plan)
                self._record_plan_task_summaries(memory, completed, run_id=state.run_id)
                memory.extract_facts(reason="plan_completed", run_id=state.run_id)
                return self._final_state(state, completed)
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
                return self._final_state(state, plan, stop_reason="plan_validation_failed")
            except RuntimeError as exc:
                plan = PlanState.create(goal=goal, summary="Plan failed before execution.")
                plan.status = PlanStatus.FAILED
                plan.error = str(exc)
                plan.trace_path = str(writer.path)
                self.state_store.save(plan)
                self._emit_trace(writer, state, "plan.failed", {"error": str(exc), "plan": plan.to_dict()})
                self._emit_plan_event("plan.failed", plan)
                return self._final_state(state, plan, stop_reason="plan_failed")

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
        *,
        stop_reason: str | None = None,
    ) -> AgentState:
        state.plan = render_plan(plan)
        state.review = render_plan_review(plan)
        state.final_answer = render_plan_final_answer(plan)
        state.done = True
        state.stop_reason = stop_reason or _stop_reason_for_plan(plan)
        state.trace_path = Path(plan.trace_path) if plan.trace_path else state.trace_path
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
