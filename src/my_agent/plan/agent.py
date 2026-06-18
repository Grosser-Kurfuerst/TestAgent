from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from my_agent.config import AgentConfig
from my_agent.indexer import RepoIndexer
from my_agent.llm import AgentLLM
from my_agent.plan.executor import JsonPlanStore, PlanEvent, PlanExecutor, PlanStore, ReActTaskRunner
from my_agent.plan.graph import PlanValidationError
from my_agent.plan.planner import Planner
from my_agent.plan.types import PlanState, PlanStatus, PlanTask, TaskStatus
from my_agent.schema import AgentState, TraceEvent
from my_agent.tools import should_skip_path
from my_agent.tracing import TraceWriter


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


class PlanExecuteAgent:
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
    ) -> None:
        self.config = config
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.event_sink = event_sink
        self.review_handler = review_handler or _default_review_handler
        self.state_store = state_store or JsonPlanStore(self.trace_dir / "plans")

    def run(
        self,
        *,
        repo_path: str | Path,
        goal: str,
        test_command: str | None = None,
        max_steps: int | None = None,
        require_approval: bool = False,
    ) -> AgentState:
        repo = Path(repo_path).resolve()
        effective_max_steps = max_steps if max_steps is not None else self.config.max_steps
        state = AgentState.initial(
            repo_path=repo,
            task=goal,
            test_command=test_command,
            max_steps=effective_max_steps,
        )
        writer = TraceWriter.create(self.trace_dir, state.run_id)
        state.trace_path = writer.path
        self._emit_trace(
            writer,
            state,
            "plan.requested",
            {"repo_path": str(repo), "goal": goal, "require_approval": require_approval},
        )

        try:
            repo_context = self._repo_context(repo, goal, writer, state)
            plan = Planner(self.llm, max_tasks=self.config.plan_max_tasks).create_plan(goal, repo_context=repo_context)
            plan.trace_path = str(writer.path)
            plan.status = PlanStatus.AWAITING_APPROVAL
            self.state_store.save(plan)
            self._emit_plan_state(writer, state, "plan.created", plan)
            self._emit_plan_state(writer, state, "plan.approval_requested", plan)

            decision = self.review_handler(plan) if require_approval else PlanReviewDecision(PlanReviewAction.EXECUTE)
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
                        repo_context=repo_context,
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
            )
            executor = PlanExecutor(
                runner,
                store=self.state_store,
                event_sink=lambda event: self._handle_executor_event(writer, state, event),
                max_tasks=self.config.plan_max_tasks,
            )
            completed = executor.execute(plan)
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
            self._emit_event("plan.validation_failed", plan)
            return self._final_state(state, plan, stop_reason="plan_validation_failed")
        except RuntimeError as exc:
            plan = PlanState.create(goal=goal, summary="Plan failed before execution.")
            plan.status = PlanStatus.FAILED
            plan.error = str(exc)
            plan.trace_path = str(writer.path)
            self.state_store.save(plan)
            self._emit_trace(writer, state, "plan.failed", {"error": str(exc), "plan": plan.to_dict()})
            self._emit_event("plan.failed", plan)
            return self._final_state(state, plan, stop_reason="plan_failed")

    def _repo_context(self, repo: Path, goal: str, writer: TraceWriter, state: AgentState) -> str:
        snapshot = RepoIndexer(repo, skip_predicate=lambda path: should_skip_path(repo, path)).snapshot(query=goal)
        context = snapshot.as_context()
        self._emit_trace(
            writer,
            state,
            "repo.indexed",
            {"repo_path": str(repo), "task": goal, "tree": snapshot.tree, "symbols": snapshot.symbols},
        )
        return context

    def _handle_executor_event(self, writer: TraceWriter, state: AgentState, event: PlanEvent) -> None:
        self._emit_trace(writer, state, event.event, event.payload)
        if self.event_sink is not None:
            self.event_sink(event)

    def _emit_plan_state(self, writer: TraceWriter, state: AgentState, event: str, plan: PlanState) -> None:
        self._emit_trace(writer, state, event, {"plan_id": plan.id, "status": plan.status.value, "plan": plan.to_dict()})
        self._emit_event(event, plan)

    def _emit_trace(self, writer: TraceWriter, state: AgentState, event: str, payload: dict[str, object]) -> None:
        writer.append(TraceEvent(event=event, payload=payload, run_id=state.run_id))

    def _emit_event(self, event: str, plan: PlanState) -> None:
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


def render_plan(plan: PlanState) -> str:
    lines = [
        f"Plan: {plan.id}",
        f"Status: {plan.status.value}",
        f"Summary: {plan.summary or 'No summary provided.'}",
        "Tasks:",
    ]
    for task in plan.tasks:
        lines.append(f"- {task.id} [{task.status.value}] {task.title}")
    return "\n".join(lines)


def render_plan_review(plan: PlanState) -> str:
    counts = _task_counts(plan.tasks)
    parts = [f"{status}={count}" for status, count in sorted(counts.items())]
    details = ", ".join(parts) if parts else "no tasks"
    return f"Plan review: status={plan.status.value}, {details}, trace={plan.trace_path or 'none'}."


def render_plan_final_answer(plan: PlanState) -> str:
    lines = [
        f"Plan {plan.status.value}: {plan.summary or plan.goal}",
        "",
        "Tasks:",
    ]
    for task in plan.tasks:
        line = f"- {task.id} {task.status.value}: {task.title}"
        if task.error:
            line += f" ({task.error})"
        lines.append(line)
    if plan.result:
        lines.extend(["", "Result:", plan.result])
    if plan.error:
        lines.extend(["", "Error:", plan.error])
    if plan.trace_path:
        lines.extend(["", f"Trace: {plan.trace_path}"])
    return "\n".join(lines)


def _default_review_handler(plan: PlanState) -> PlanReviewDecision:
    return PlanReviewDecision(PlanReviewAction.EXECUTE)


def _task_counts(tasks: list[PlanTask]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.status.value] = counts.get(task.status.value, 0) + 1
    return counts


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
