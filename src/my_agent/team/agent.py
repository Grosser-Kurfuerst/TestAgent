from __future__ import annotations

import threading
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from datetime import datetime
import inspect
from pathlib import Path
from queue import Queue
from typing import Any, Callable

from my_agent.config import AgentConfig
from my_agent.context import AgentContextManager, ContextOverBudgetError
from my_agent.cancellation import CancellationToken
from my_agent.events import BufferedEventSink
from my_agent.hitl.handler import HitlHandler
from my_agent.llm import AgentLLM
from my_agent.memory import MemoryManager
from my_agent.plan import PlanValidationError, TaskResult
from my_agent.agent_base import AgentBase
from my_agent.schema import AgentState
from my_agent.parallel import create_bounded_executor, shutdown_executor
from my_agent.team.graph import execution_batches, get_executable_steps, validate_team_graph
from my_agent.team.planner import TeamPlanner
from my_agent.team.prompts import build_team_planner_messages
from my_agent.team.rendering import render_team_final_answer, render_team_plan, render_team_review
from my_agent.team.store import JsonTeamStore, TeamStore
from my_agent.team.sub_agent import SubAgent
from my_agent.team.types import AgentRole, ExecutionStep, ReviewDecision, StepStatus, TeamState, TeamStatus
from my_agent.observability.tracing import TraceWriter, append_agent_completed
from my_agent.utils.numbers import positive_or_default
from my_agent.utils.text import single_line, terminal_summary_text

EventSink = Callable[[object], None]


@dataclass(frozen=True)
class TeamEvent:
    event: str
    team_id: str
    status: str
    step_id: str = ""
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event,
            "team_id": self.team_id,
            "status": self.status,
            "step_id": self.step_id,
            "payload": dict(self.payload),
        }


class TeamAgent(AgentBase):
    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: AgentLLM,
        trace_dir: str | Path,
        command_timeout: int,
        event_sink: EventSink | None = None,
        planner: TeamPlanner | None = None,
        state_store: TeamStore | None = None,
        worker_factory: Callable[[int], Any] | None = None,
        reviewer_factory: Callable[[str], Any] | None = None,
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
        self.planner = planner or TeamPlanner(llm, max_steps=config.team_max_steps)
        self.state_store = state_store or JsonTeamStore(self.trace_dir / "teams")
        self.worker_factory = worker_factory
        self.reviewer_factory = reviewer_factory
        self._repo_path = Path(".").resolve()
        self._test_command: str | None = None
        self._memory: MemoryManager | None = None
        self._step_max_steps = config.team_step_max_steps
        self._state_lock = threading.RLock()
        self._event_local = threading.local()

    def run(self, state: AgentState) -> AgentState:
        state.repo_path = Path(state.repo_path).resolve()
        repo = state.repo_path
        goal = state.task
        test_command = state.test_command
        self._repo_path = repo
        self._test_command = test_command
        self._step_max_steps = self.config.team_step_max_steps
        state.max_steps = positive_or_default(state.max_steps, self.config.team_max_steps)
        with self.open_run_context(state) as ctx:
            writer = ctx.writer
            memory = ctx.memory
            self._memory = memory
            self._emit_trace(writer, state, "team.requested", {"repo_path": str(repo), "goal": goal})
            memory.append_task_goal(goal, run_id=state.run_id)

            try:
                if ctx.repo_snapshot is None:
                    raise RuntimeError("Team execution requires repository context.")
                repo_context = ctx.repo_context
                context_manager = AgentContextManager(memory.context_profile)
                planner_base_messages = build_team_planner_messages(
                    goal,
                    repo_context=repo_context,
                    memory_context="",
                    conversation=[],
                )
                budget_plan = context_manager.budget_for_messages(base_messages=planner_base_messages, tools=[])
                retrieved_memory_context = memory.build_context_for_query(goal, max_tokens=budget_plan.long_term_limit)
                memory_context = retrieved_memory_context.injected_text
                planner_messages_with_memory = build_team_planner_messages(
                    goal,
                    repo_context=repo_context,
                    memory_context=memory_context,
                    conversation=[],
                )
                fixed_with_memory_tokens = context_manager.estimate_tokens(planner_messages_with_memory, [])
                planner_memory_payload = {
                    "phase": "team_planner",
                    "message_count": len(planner_messages_with_memory),
                    "memory_hits": len(retrieved_memory_context.hits),
                    "memory_tokens": retrieved_memory_context.estimated_tokens,
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
                planner_trace_snapshot = _set_trace_sink(
                    self.planner,
                    lambda event, payload: self._emit_trace(writer, state, event, payload),
                )
                try:
                    try:
                        team = self.planner.create_team_plan(
                            goal,
                            repo_context=repo_context,
                            memory_context=memory_context,
                        )
                    finally:
                        _restore_trace_sink(self.planner, planner_trace_snapshot)
                except RuntimeError as exc:
                    team = TeamState.create(goal=goal, summary="Team planning failed.")
                    team.status = TeamStatus.FAILED
                    team.error = str(exc)
                    team.trace_path = str(writer.path)
                    self._save_and_emit(
                        writer,
                        state,
                        team,
                        "team.validation_failed",
                        extra={"code": "team_planner_failed", "message": str(exc)},
                    )
                    return self._final_state(state, team, writer, stop_reason="team_planner_failed")
                team.trace_path = str(writer.path)
                team.status = TeamStatus.RUNNING
                team.started_at = team.started_at or _now()
                validate_team_graph(team.steps, max_steps=state.max_steps)
                team.execution_order = _flatten_batches(execution_batches(team.steps, max_steps=state.max_steps))
                self._save_and_emit(writer, state, team, "team.plan.created")
                self._emit_trace(
                    writer,
                    state,
                    "team.graph.validated",
                    {
                        "team_id": team.id,
                        "batches": execution_batches(team.steps, max_steps=state.max_steps),
                    },
                )
                self._save_and_emit(
                    writer,
                    state,
                    team,
                    "team.started",
                    extra={
                        "worker_count": self.config.team_worker_count,
                        "parallel_enabled": self.config.team_parallel_enabled,
                    },
                )
                completed = self._execute_team(team, writer, state)
                self._record_team_step_summaries(memory, completed, run_id=state.run_id)
                memory.extract_facts(reason="team_completed", run_id=state.run_id)
                return self._final_state(state, completed, writer)
            except PlanValidationError as exc:
                team = TeamState.create(goal=goal, summary="Team plan validation failed.")
                team.status = TeamStatus.FAILED
                team.error = str(exc)
                team.trace_path = str(writer.path)
                self._save_and_emit(
                    writer,
                    state,
                    team,
                    "team.validation_failed",
                    extra={"code": exc.code, "message": exc.message, "details": exc.details},
                )
                stop_reason = (
                    "team_planner_failed"
                    if exc.code == "team_planner_llm_failed"
                    else "team_validation_failed"
                )
                return self._final_state(state, team, writer, stop_reason=stop_reason)
            except ContextOverBudgetError as exc:
                team = TeamState.create(goal=goal, summary="Team planning stopped because context budget was exceeded.")
                team.status = TeamStatus.FAILED
                team.error = str(exc)
                team.trace_path = str(writer.path)
                self._save_and_emit(
                    writer,
                    state,
                    team,
                    "team.failed",
                    extra={"reason": "context_over_budget", "context": exc.payload},
                )
                return self._final_state(state, team, writer, stop_reason="context_over_budget")
            except RuntimeError as exc:
                team = TeamState.create(goal=goal, summary="Team execution failed before completion.")
                team.status = TeamStatus.FAILED
                team.error = str(exc)
                team.trace_path = str(writer.path)
                self._save_and_emit(writer, state, team, "team.failed")
                return self._final_state(state, team, writer, stop_reason="team_failed")

    def run_step(
        self,
        team: TeamState,
        step: ExecutionStep,
        *,
        worker: Any | None = None,
        reviewer: Any | None = None,
        context: str | None = None,
        writer: TraceWriter | None = None,
        state: AgentState | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        feedback = ""
        dependency_context = context if context is not None else self.build_dependency_context(team, step)
        active_worker = worker or self._make_worker(1)
        active_reviewer = reviewer or self._make_reviewer(step.id)
        reviewer_trace_snapshot = _set_trace_sink(
            active_reviewer,
            (lambda event, payload: self._emit_trace(writer, state, event, payload))
            if writer is not None and state is not None
            else None,
        )
        max_attempts = self.config.team_max_retries + 1
        last_output = ""

        try:
            for attempt in range(1, max_attempts + 1):
                if _token_cancelled(cancellation_token):
                    self._mark_cancelled(
                        team,
                        step,
                        f"Step was cancelled: {cancellation_token.reason or 'cancelled'}",
                        writer,
                        state,
                    )
                    return
                with self._state_lock:
                    step.attempts = attempt
                    step.status = StepStatus.RUNNING
                    step.started_at = step.started_at or _now()
                    step.worker_name = getattr(active_worker, "name", "")
                    self._save_and_emit(writer, state, team, "team.step.started", step)

                try:
                    result = _execute_worker_step(
                        active_worker,
                        team,
                        step,
                        dependency_context,
                        feedback=feedback,
                        cancellation_token=cancellation_token,
                    )
                except Exception as exc:  # noqa: BLE001 - a crashing worker must fail only this step.
                    self._mark_failed(
                        team,
                        step,
                        f"Worker crashed: {type(exc).__name__}: {exc}",
                        writer,
                        state,
                    )
                    return
                with self._state_lock:
                    step.trace_path = result.trace_path
                if result.stop_reason == "cancelled" or _token_cancelled(cancellation_token):
                    self._mark_cancelled(
                        team,
                        step,
                        result.error
                        or f"Step was cancelled: {cancellation_token.reason if cancellation_token else 'cancelled'}",
                        writer,
                        state,
                    )
                    return
                if not result.ok:
                    self._mark_failed(team, step, result.error or "Worker failed.", writer, state, output=result.output)
                    return
                last_output = result.output
                if _token_cancelled(cancellation_token):
                    self._mark_cancelled(
                        team,
                        step,
                        f"Step was cancelled: {cancellation_token.reason or 'cancelled'}",
                        writer,
                        state,
                    )
                    return
                with self._state_lock:
                    self._save_and_emit(
                        writer,
                        state,
                        team,
                        "team.step.worker_completed",
                        step,
                        extra={"attempt": attempt, "worker_result": result.to_dict()},
                    )

                if _token_cancelled(cancellation_token):
                    self._mark_cancelled(
                        team,
                        step,
                        f"Step was cancelled: {cancellation_token.reason or 'cancelled'}",
                        writer,
                        state,
                    )
                    return
                with self._state_lock:
                    step.status = StepStatus.REVIEWING
                    self._save_and_emit(writer, state, team, "team.step.review_started", step)
                try:
                    decision: ReviewDecision = active_reviewer.review_step(
                        team.goal,
                        step,
                        dependency_context,
                        result.output,
                    )
                except Exception as exc:  # noqa: BLE001 - reviewer failures must not corrupt team state.
                    self._mark_failed(
                        team,
                        step,
                        f"Reviewer crashed: {type(exc).__name__}: {exc}",
                        writer,
                        state,
                        output=result.output,
                    )
                    return
                with self._state_lock:
                    step.review_summary = decision.summary
                    step.review_issues = list(decision.issues)
                    step.review_suggestions = list(decision.suggestions)
                    self._save_and_emit(
                        writer,
                        state,
                        team,
                        "team.step.review_completed",
                        step,
                        extra={"attempt": attempt, "review": _review_payload(decision)},
                    )
                if hasattr(active_reviewer, "clear_history"):
                    active_reviewer.clear_history()

                if _token_cancelled(cancellation_token):
                    self._mark_cancelled(
                        team,
                        step,
                        f"Step was cancelled: {cancellation_token.reason or 'cancelled'}",
                        writer,
                        state,
                    )
                    return

                if decision.approved:
                    self._mark_completed(team, step, result, writer, state)
                    return

                feedback = decision.feedback_text or "Reviewer rejected the result without additional detail."
                self._save_and_emit(
                    writer,
                    state,
                    team,
                    "team.step.review_rejected",
                    step,
                    extra={"attempt": attempt, "review": _review_payload(decision)},
                )
                if attempt < max_attempts:
                    self._save_and_emit(
                        writer,
                        state,
                        team,
                        "team.step.retry_started",
                        step,
                        extra={"attempt": attempt + 1, "feedback": feedback},
                    )
        finally:
            _restore_trace_sink(active_reviewer, reviewer_trace_snapshot)

        if self.config.team_allow_unapproved_results:
            self._mark_completed(
                team,
                step,
                TaskResult.success(step.id, last_output, trace_path=step.trace_path, stop_reason="review_rejected"),
                writer,
                state,
            )
            return
        self._mark_failed(team, step, "Reviewer rejected result after max retries.", writer, state, output=last_output)

    def build_dependency_context(self, team: TeamState, step: ExecutionStep) -> str:
        step_map = team.step_by_id()
        chunks: list[str] = []
        limit = max(1, self.config.team_dependency_context_chars)
        for dependency_id in step.dependencies:
            dependency = step_map.get(dependency_id)
            if dependency is None:
                continue
            result = dependency.result.strip() or dependency.error.strip() or "No result recorded."
            review = dependency.review_summary.strip() or "No review summary recorded."
            chunks.append(
                "\n".join(
                    [
                        f"- {dependency.id} {dependency.status.value}: {dependency.title}",
                        f"  review: {single_line(review, 500)}",
                        f"  result: {single_line(result, limit)}",
                    ]
                )
            )
        return "\n".join(chunks) if chunks else "No completed dependencies."

    def _execute_team(
        self,
        team: TeamState,
        writer: TraceWriter,
        state: AgentState,
        *,
        parallel_enabled: bool | None = None,
    ) -> TeamState:
        use_parallel = self.config.team_parallel_enabled if parallel_enabled is None else parallel_enabled
        worker_pool = self._make_worker_pool()
        while True:
            if _state_cancelled(state):
                self._cancel_unfinished(team, writer, state, "Cancelled because team execution was cancelled.")
                break
            executable = get_executable_steps(team.steps)
            if not executable:
                break
            for step in executable:
                with self._state_lock:
                    step.status = StepStatus.READY
                    self._save_and_emit(writer, state, team, "team.step.ready", step)

            self._save_and_emit(
                writer,
                state,
                team,
                "team.batch.started",
                extra={
                    "batch": [step.id for step in executable],
                    "worker_count": min(len(executable), max(1, self.config.team_worker_count)),
                    "parallel": use_parallel and len(executable) > 1,
                },
            )

            if use_parallel and len(executable) > 1:
                self.run_batch_parallel(team, executable, worker_pool=worker_pool, writer=writer, state=state)
            else:
                for step in executable:
                    if _state_cancelled(state):
                        self._cancel_unfinished(team, writer, state, "Cancelled because team execution was cancelled.")
                        break
                    self._run_with_worker_from_pool(team, step, worker_pool, writer, state, _child_token(state))
            if _state_cancelled(state):
                self._cancel_unfinished(team, writer, state, "Cancelled because team execution was cancelled.")
                break

        self._skip_residual_steps(team, writer, state)
        self._finish_team(team, writer, state)
        return team

    def _execute_serial(self, team: TeamState, writer: TraceWriter, state: AgentState) -> TeamState:
        return self._execute_team(team, writer, state, parallel_enabled=False)

    def run_batch_parallel(
        self,
        team: TeamState,
        batch: list[ExecutionStep],
        *,
        worker_pool: Queue[Any],
        writer: TraceWriter,
        state: AgentState,
    ) -> None:
        max_workers = min(len(batch), max(1, self.config.team_worker_count))
        buffers = BufferedEventSink(self.event_sink)
        step_tokens = {step.id: _child_token(state) for step in batch}
        executor = create_bounded_executor(max_workers=max_workers, thread_name_prefix="agentcli-team")
        futures: dict[Future[None], ExecutionStep] = {}
        try:
            for step in batch:
                if _state_cancelled(state):
                    break
                futures[
                    executor.submit(
                        self._run_with_worker_from_pool,
                        team,
                        step,
                        worker_pool,
                        writer,
                        state,
                        step_tokens[step.id],
                        buffers.buffer_for(step.id).append,
                    )
                ] = step

            done, not_done = wait(set(futures), timeout=self.config.team_step_batch_timeout_seconds)
            timed_out = set(not_done)
            still_running = set()
            if not_done:
                reason = "cancelled" if _state_cancelled(state) else "batch_timeout"
                for future, step in futures.items():
                    if future in not_done and step_tokens[step.id] is not None:
                        step_tokens[step.id].cancel(reason)
                done_after_grace, still_running = wait(not_done, timeout=self.config.tool_shutdown_grace_seconds)
                done = done.union(done_after_grace)

            for future, step in futures.items():
                if future in timed_out and not _state_cancelled(state):
                    self._mark_failed(
                        team,
                        step,
                        f"Step batch timed out after {self.config.team_step_batch_timeout_seconds}s.",
                        writer,
                        state,
                        force=True,
                    )
                elif future in still_running:
                    if _state_cancelled(state):
                        self._mark_cancelled(
                            team,
                            step,
                            "Cancelled because team execution was cancelled.",
                            writer,
                            state,
                        )
                    else:
                        self._mark_failed(
                            team,
                            step,
                            f"Step batch timed out after {self.config.team_step_batch_timeout_seconds}s.",
                            writer,
                            state,
                        )
                elif future in done and not future.cancelled():
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - one crashed future must not stop the batch.
                        self._mark_failed(
                            team,
                            step,
                            f"Parallel step crashed: {type(exc).__name__}: {exc}",
                            writer,
                            state,
                        )
            buffers.flush_in_order([step.id for step in batch])
            if timed_out and not _state_cancelled(state):
                self._save_and_emit(
                    writer,
                    state,
                    team,
                    "team.batch.timeout",
                    extra={
                        "batch": [step.id for step in batch],
                        "timeout_seconds": self.config.team_step_batch_timeout_seconds,
                    },
                )
        finally:
            shutdown_executor(executor)

    def _make_worker_pool(self) -> Queue[Any]:
        pool: Queue[Any] = Queue()
        for index in range(1, max(1, self.config.team_worker_count) + 1):
            pool.put(self._make_worker(index))
        return pool

    def _run_with_worker_from_pool(
        self,
        team: TeamState,
        step: ExecutionStep,
        worker_pool: Queue[Any],
        writer: TraceWriter,
        state: AgentState,
        cancellation_token: CancellationToken | None = None,
        event_sink: Callable[[object], None] | None = None,
    ) -> None:
        worker = worker_pool.get()
        previous_sink = getattr(self._event_local, "sink", None)
        if event_sink is not None:
            self._event_local.sink = event_sink
        try:
            reviewer = self._make_reviewer(step.id)
            context = self.build_dependency_context(team, step)
            self.run_step(
                team,
                step,
                worker=worker,
                reviewer=reviewer,
                context=context,
                writer=writer,
                state=state,
                cancellation_token=cancellation_token,
            )
        finally:
            if event_sink is not None:
                if previous_sink is None:
                    try:
                        del self._event_local.sink
                    except AttributeError:
                        pass
                else:
                    self._event_local.sink = previous_sink
            if hasattr(worker, "clear_history"):
                worker.clear_history()
            worker_pool.put(worker)

    def _skip_residual_steps(self, team: TeamState, writer: TraceWriter, state: AgentState) -> None:
        step_map = team.step_by_id()
        for step in team.steps:
            if step.status not in {StepStatus.PENDING, StepStatus.READY}:
                continue
            blockers = [
                dependency
                for dependency in step.dependencies
                if step_map[dependency].status != StepStatus.COMPLETED
            ]
            reason = "Skipped because dependencies did not complete"
            if blockers:
                reason += f": {', '.join(blockers)}"
            with self._state_lock:
                step.status = StepStatus.SKIPPED
                step.error = reason
                step.ended_at = _now()
                self._save_and_emit(writer, state, team, "team.step.skipped", step)

    def _cancel_unfinished(self, team: TeamState, writer: TraceWriter, state: AgentState, reason: str) -> None:
        for step in team.steps:
            if step.status in _STEP_TERMINAL_STATUSES:
                continue
            self._mark_cancelled(team, step, reason, writer, state)

    def _finish_team(self, team: TeamState, writer: TraceWriter, state: AgentState) -> None:
        failed = [step for step in team.steps if step.status == StepStatus.FAILED]
        skipped = [step for step in team.steps if step.status == StepStatus.SKIPPED]
        cancelled = [step for step in team.steps if step.status == StepStatus.CANCELLED]
        with self._state_lock:
            if cancelled:
                team.status = TeamStatus.CANCELLED
                team.error = _summarize_step_errors(cancelled)
                event = "team.cancelled"
            elif failed or skipped:
                team.status = TeamStatus.FAILED
                team.error = _summarize_step_errors(failed + skipped)
                event = "team.failed"
            else:
                team.status = TeamStatus.SUCCEEDED
                team.result = _summarize_step_results(team.steps)
                event = "team.completed"
            team.ended_at = _now()
            self._save_and_emit(writer, state, team, event)

    def _mark_completed(
        self,
        team: TeamState,
        step: ExecutionStep,
        result: TaskResult,
        writer: TraceWriter | None,
        state: AgentState | None,
    ) -> None:
        with self._state_lock:
            if step.status in _STEP_TERMINAL_STATUSES and step.ended_at:
                return
            step.status = StepStatus.COMPLETED
            step.result = result.output
            step.error = ""
            step.trace_path = result.trace_path
            step.ended_at = _now()
            self._save_and_emit(writer, state, team, "team.step.completed", step)

    def _mark_failed(
        self,
        team: TeamState,
        step: ExecutionStep,
        error: str,
        writer: TraceWriter | None,
        state: AgentState | None,
        *,
        output: str = "",
        force: bool = False,
    ) -> None:
        with self._state_lock:
            if not force and step.status in _STEP_TERMINAL_STATUSES and step.ended_at:
                return
            step.status = StepStatus.FAILED
            step.result = output
            step.error = error or "Step failed."
            step.ended_at = _now()
            self._save_and_emit(writer, state, team, "team.step.failed", step)

    def _mark_cancelled(
        self,
        team: TeamState,
        step: ExecutionStep,
        reason: str,
        writer: TraceWriter | None,
        state: AgentState | None,
    ) -> None:
        with self._state_lock:
            if step.status in _STEP_TERMINAL_STATUSES and step.ended_at:
                return
            step.status = StepStatus.CANCELLED
            step.error = reason or "Step was cancelled."
            step.ended_at = _now()
            self._save_and_emit(writer, state, team, "team.step.cancelled", step)

    def _make_worker(self, index: int) -> Any:
        if self.worker_factory is not None:
            return self.worker_factory(index)
        return SubAgent(
            name=f"worker-{index}",
            role=AgentRole.WORKER,
            config=self.config,
            llm=self.llm,
            repo_path=self._repo_path,
            trace_dir=self.trace_dir,
            command_timeout=self.command_timeout,
            memory_manager=self._memory,
            event_sink=self._forward_worker_event,
            hitl_handler=self.hitl_handler,
            test_command=self._test_command,
            step_max_steps=self._step_max_steps,
        )

    def _make_reviewer(self, step_id: str) -> Any:
        if self.reviewer_factory is not None:
            return self.reviewer_factory(step_id)
        return SubAgent(
            name=f"reviewer-{step_id}",
            role=AgentRole.REVIEWER,
            config=self.config,
            llm=self.llm,
            repo_path=self._repo_path,
            trace_dir=self.trace_dir,
            command_timeout=self.command_timeout,
            memory_manager=self._memory,
            event_sink=self._forward_worker_event,
            hitl_handler=self.hitl_handler,
            test_command=self._test_command,
        )

    def _record_team_step_summaries(self, memory: MemoryManager, team: TeamState, *, run_id: str) -> None:
        for step in team.steps:
            if step.status not in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.CANCELLED}:
                continue
            details = terminal_summary_text(step.result, step.error, "No result recorded.")
            memory.append_summary(
                f"[team step {step.id} {step.status.value}] {step.title}\n{details}",
                source="team",
                run_id=run_id,
                metadata={
                    "team_id": team.id,
                    "step_id": step.id,
                    "step_status": step.status.value,
                    "step_type": step.type.value,
                },
            )

    def _save_and_emit(
        self,
        writer: TraceWriter | None,
        state: AgentState | None,
        team: TeamState,
        event: str,
        step: ExecutionStep | None = None,
        *,
        extra: dict[str, object] | None = None,
    ) -> None:
        with self._state_lock:
            self.state_store.save(team)
            payload = _event_payload(team, step)
            if extra:
                payload.update(extra)
            if writer is not None and state is not None:
                self._emit_trace(writer, state, event, payload)
            event_sink = getattr(self._event_local, "sink", None) or self.event_sink
            if event_sink is not None:
                event_sink(
                    TeamEvent(
                        event=event,
                        team_id=team.id,
                        status=step.status.value if step is not None else team.status.value,
                        step_id=step.id if step is not None else "",
                        payload=payload,
                    )
                )

    def _forward_worker_event(self, event: object) -> None:
        event_sink = getattr(self._event_local, "sink", None) or self.event_sink
        if event_sink is None:
            return
        with self._state_lock:
            event_sink(event)

    def _final_state(
        self,
        state: AgentState,
        team: TeamState,
        writer: TraceWriter,
        *,
        stop_reason: str | None = None,
    ) -> AgentState:
        state.plan = render_team_plan(team)
        state.review = render_team_review(team)
        state.final_answer = render_team_final_answer(team)
        state.done = True
        state.steps = sum(step.attempts for step in team.steps)
        state.stop_reason = stop_reason or _stop_reason_for_team(team)
        state.trace_path = Path(team.trace_path) if team.trace_path else state.trace_path
        append_agent_completed(
            writer,
            state,
            mode="team",
            run_label="team",
            child_trace_paths=_team_child_trace_paths(team),
        )
        return state


def _event_payload(team: TeamState, step: ExecutionStep | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "team_id": team.id,
        "goal": team.goal,
        "status": step.status.value if step is not None else team.status.value,
        "team": team.to_dict(),
    }
    if step is not None:
        payload.update(
            {
                "step_id": step.id,
                "title": step.title,
                "type": step.type.value,
                "step": step.to_dict(),
            }
        )
    return payload


def _review_payload(decision: ReviewDecision) -> dict[str, object]:
    return {
        "approved": decision.approved,
        "summary": decision.summary,
        "issues": list(decision.issues),
        "suggestions": list(decision.suggestions),
        "parse_error": decision.parse_error,
    }


def _flatten_batches(batches: list[list[str]]) -> list[str]:
    return [step_id for batch in batches for step_id in batch]


def _summarize_step_results(steps: list[ExecutionStep]) -> str:
    return "\n".join(step.result.strip() for step in steps if step.result.strip())


def _summarize_step_errors(steps: list[ExecutionStep]) -> str:
    return "\n".join(f"{step.id}: {step.error}" for step in steps if step.error)


def _stop_reason_for_team(team: TeamState) -> str:
    if team.status == TeamStatus.SUCCEEDED:
        return "team_completed"
    if team.status == TeamStatus.CANCELLED:
        return "team_cancelled"
    if team.status == TeamStatus.FAILED:
        return "team_failed"
    return f"team_{team.status.value}"


def _team_child_trace_paths(team: TeamState) -> list[str]:
    return [step.trace_path for step in team.steps if step.trace_path]


_NO_TRACE_SINK_SETTER = object()


def _set_trace_sink(target: Any, trace_sink: Callable[[str, dict[str, object]], None] | None) -> Any:
    setter = getattr(target, "set_trace_sink", None)
    if not callable(setter):
        return _NO_TRACE_SINK_SETTER
    return setter(trace_sink)


def _restore_trace_sink(target: Any, snapshot: Any) -> None:
    if snapshot is _NO_TRACE_SINK_SETTER:
        return
    setter = getattr(target, "set_trace_sink", None)
    if callable(setter):
        setter(snapshot)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_STEP_TERMINAL_STATUSES = {
    StepStatus.COMPLETED,
    StepStatus.FAILED,
    StepStatus.SKIPPED,
    StepStatus.CANCELLED,
}


def _state_cancelled(state: AgentState) -> bool:
    return bool(state.cancellation_token is not None and state.cancellation_token.is_cancelled())


def _token_cancelled(token: CancellationToken | None) -> bool:
    return bool(token is not None and token.is_cancelled())


def _child_token(state: AgentState) -> CancellationToken | None:
    if state.cancellation_token is None:
        return None
    return state.cancellation_token.child()


def _execute_worker_step(
    worker: Any,
    team: TeamState,
    step: ExecutionStep,
    context: str,
    *,
    feedback: str,
    cancellation_token: CancellationToken | None,
) -> TaskResult:
    method = worker.execute_step
    if _accepts_keyword(method, "cancellation_token"):
        return method(team, step, context, feedback=feedback, cancellation_token=cancellation_token)
    return method(team, step, context, feedback=feedback)


def _accepts_keyword(callable_obj: Callable[..., object], name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in signature.parameters.values()
    )
