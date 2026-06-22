from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Protocol

from my_agent.config import AgentConfig
from my_agent.indexer import RepoIndexer
from my_agent.llm import AgentLLM
from my_agent.memory import MemoryManager
from my_agent.plan import PlanValidationError, TaskResult
from my_agent.schema import AgentState, TraceEvent
from my_agent.team.graph import execution_batches, get_executable_steps, validate_team_graph
from my_agent.team.planner import TeamPlanner
from my_agent.team.sub_agent import SubAgent
from my_agent.team.types import AgentRole, ExecutionStep, ReviewDecision, StepStatus, TeamState, TeamStatus
from my_agent.text_safety import sanitize_json_value
from my_agent.tools import should_skip_path
from my_agent.tracing import TraceWriter

EventSink = Callable[[object], None]


class TeamStore(Protocol):
    def save(self, team: TeamState) -> None:
        ...

    def get(self, team_id: str) -> TeamState | None:
        ...


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


class InMemoryTeamStore:
    def __init__(self) -> None:
        self._teams: dict[str, TeamState] = {}

    def save(self, team: TeamState) -> None:
        self._teams[team.id] = TeamState.from_dict(team.to_dict())

    def get(self, team_id: str) -> TeamState | None:
        team = self._teams.get(team_id)
        return TeamState.from_dict(team.to_dict()) if team is not None else None


class JsonTeamStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, team: TeamState) -> None:
        path = self._path_for(team.id)
        path.write_text(json.dumps(sanitize_json_value(team.to_dict()), ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, team_id: str) -> TeamState | None:
        path = self._path_for(team_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return TeamState.from_dict(payload)

    def _path_for(self, team_id: str) -> Path:
        safe_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in team_id)
        return self.directory / f"{safe_id}.json"


class TeamOrchestrator:
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
    ) -> None:
        self.config = config
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.event_sink = event_sink
        self.planner = planner or TeamPlanner(llm, max_steps=config.team_max_steps)
        self.state_store = state_store or JsonTeamStore(self.trace_dir / "teams")
        self.worker_factory = worker_factory
        self.reviewer_factory = reviewer_factory
        self._repo_path = Path(".").resolve()
        self._test_command: str | None = None
        self._memory: MemoryManager | None = None
        self._step_max_steps = config.team_step_max_steps
        self._state_lock = threading.RLock()

    def run(
        self,
        *,
        repo_path: str | Path,
        goal: str,
        test_command: str | None = None,
        max_steps: int | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> AgentState:
        repo = Path(repo_path).resolve()
        self._repo_path = repo
        self._test_command = test_command
        self._step_max_steps = self.config.team_step_max_steps
        state = AgentState.initial(
            repo_path=repo,
            task=goal,
            test_command=test_command,
            max_steps=_positive_or_default(max_steps, self.config.team_max_steps),
        )
        writer = TraceWriter.create(self.trace_dir, state.run_id)
        state.trace_path = writer.path
        memory, memory_trace_snapshot = self._memory_for_run(repo, state, writer, memory_manager)
        self._memory = memory

        try:
            self._emit_trace(writer, state, "team.requested", {"repo_path": str(repo), "goal": goal})
            self._emit_memory_loaded(writer, state, memory)
            memory.append_user_message(goal, run_id=state.run_id)

            try:
                repo_context = self._repo_context(repo, goal, writer, state)
                memory_context = memory.build_context_for_query(goal).injected_text
                try:
                    team = self.planner.create_team_plan(
                        goal,
                        repo_context=repo_context,
                        memory_context=memory_context,
                    )
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
                    return self._final_state(state, team, stop_reason="team_planner_failed")
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
                return self._final_state(state, completed)
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
                stop_reason = "team_planner_failed" if exc.code == "team_planner_llm_failed" else "team_validation_failed"
                return self._final_state(state, team, stop_reason=stop_reason)
            except RuntimeError as exc:
                team = TeamState.create(goal=goal, summary="Team execution failed before completion.")
                team.status = TeamStatus.FAILED
                team.error = str(exc)
                team.trace_path = str(writer.path)
                self._save_and_emit(writer, state, team, "team.failed")
                return self._final_state(state, team, stop_reason="team_failed")
        finally:
            if memory_trace_snapshot is not None:
                memory.restore_trace_sink(memory_trace_snapshot)

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
    ) -> None:
        feedback = ""
        dependency_context = context if context is not None else self.build_dependency_context(team, step)
        active_worker = worker or self._make_worker(1)
        active_reviewer = reviewer or self._make_reviewer(step.id)
        max_attempts = self.config.team_max_retries + 1
        last_output = ""

        for attempt in range(1, max_attempts + 1):
            with self._state_lock:
                step.attempts = attempt
                step.status = StepStatus.RUNNING
                step.started_at = step.started_at or _now()
                step.worker_name = getattr(active_worker, "name", "")
                self._save_and_emit(writer, state, team, "team.step.started", step)

            try:
                result: TaskResult = active_worker.execute_step(team, step, dependency_context, feedback=feedback)
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
            if not result.ok:
                self._mark_failed(team, step, result.error or "Worker failed.", writer, state, output=result.output)
                return
            last_output = result.output
            with self._state_lock:
                self._save_and_emit(
                    writer,
                    state,
                    team,
                    "team.step.worker_completed",
                    step,
                    extra={"attempt": attempt, "worker_result": result.to_dict()},
                )

            with self._state_lock:
                step.status = StepStatus.REVIEWING
                self._save_and_emit(writer, state, team, "team.step.review_started", step)
            try:
                decision: ReviewDecision = active_reviewer.review_step(team.goal, step, dependency_context, result.output)
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
                        f"  review: {_single_line(review, 500)}",
                        f"  result: {_single_line(result, limit)}",
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
                    self._run_with_worker_from_pool(team, step, worker_pool, writer, state)

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
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agentcli-team") as executor:
            futures = {
                executor.submit(self._run_with_worker_from_pool, team, step, worker_pool, writer, state): step
                for step in batch
            }
            for future in as_completed(futures):
                step = futures[future]
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
    ) -> None:
        worker = worker_pool.get()
        try:
            reviewer = self._make_reviewer(step.id)
            context = self.build_dependency_context(team, step)
            self.run_step(team, step, worker=worker, reviewer=reviewer, context=context, writer=writer, state=state)
        finally:
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
    ) -> None:
        with self._state_lock:
            step.status = StepStatus.FAILED
            step.result = output
            step.error = error or "Step failed."
            step.ended_at = _now()
            self._save_and_emit(writer, state, team, "team.step.failed", step)

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
            test_command=self._test_command,
        )

    def _memory_for_run(
        self,
        repo: Path,
        state: AgentState,
        writer: TraceWriter,
        memory_manager: MemoryManager | None,
    ) -> tuple[MemoryManager, tuple[object | None, object | None] | None]:
        trace_sink = lambda event, payload: self._emit_trace(writer, state, event, payload)
        if memory_manager is not None:
            snapshot = memory_manager.set_trace_sink(trace_sink)
            return memory_manager, snapshot
        return (
            MemoryManager.from_config(
                config=self.config,
                llm=self.llm,
                repo_path=repo,
                session_id=state.run_id,
                trace_sink=trace_sink,
            ),
            None,
        )

    def _repo_context(self, repo: Path, goal: str, writer: TraceWriter, state: AgentState) -> str:
        snapshot = RepoIndexer(repo, skip_predicate=lambda path: should_skip_path(repo, path)).snapshot(query=goal)
        self._emit_trace(
            writer,
            state,
            "repo.indexed",
            {"repo_path": str(repo), "task": goal, "tree": snapshot.tree, "symbols": snapshot.symbols},
        )
        return snapshot.as_context()

    def _emit_memory_loaded(self, writer: TraceWriter, state: AgentState, memory: MemoryManager) -> None:
        status = memory.status(include_entries=False)
        self._emit_trace(
            writer,
            state,
            "memory.loaded",
            {
                "storage_path": status.storage_path,
                "short_term_entries": status.short_term_entries,
                "long_term_entries": status.long_term_entries,
                "long_term_tokens": status.long_term_tokens,
            },
        )

    def _record_team_step_summaries(self, memory: MemoryManager, team: TeamState, *, run_id: str) -> None:
        for step in team.steps:
            if step.status not in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.CANCELLED}:
                continue
            details = step.result.strip() or step.error.strip() or "No result recorded."
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
            if self.event_sink is not None:
                self.event_sink(
                    TeamEvent(
                        event=event,
                        team_id=team.id,
                        status=step.status.value if step is not None else team.status.value,
                        step_id=step.id if step is not None else "",
                        payload=payload,
                    )
                )

    def _emit_trace(self, writer: TraceWriter, state: AgentState, event: str, payload: dict[str, object]) -> None:
        writer.append(TraceEvent(event=event, payload=payload, run_id=state.run_id))

    def _forward_worker_event(self, event: object) -> None:
        if self.event_sink is None:
            return
        with self._state_lock:
            self.event_sink(event)

    def _final_state(
        self,
        state: AgentState,
        team: TeamState,
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
        return state


def render_team_plan(team: TeamState) -> str:
    lines = [
        f"Team plan: {team.id}",
        f"Status: {team.status.value}",
        f"Summary: {team.summary or 'No summary provided.'}",
        "Steps:",
    ]
    for step in team.steps:
        deps = ", ".join(step.dependencies) if step.dependencies else "none"
        lines.append(f"- {step.id} [{step.status.value}] {step.type.value} {step.title} (deps: {deps})")
    return "\n".join(lines)


def render_team_review(team: TeamState) -> str:
    counts = _step_counts(team.steps)
    parts = [f"{status}={count}" for status, count in sorted(counts.items())]
    details = ", ".join(parts) if parts else "no steps"
    return f"Team review: status={team.status.value}, {details}, trace={team.trace_path or 'none'}."


def render_team_final_answer(team: TeamState) -> str:
    lines = [
        f"Team {team.status.value}: {team.summary or team.goal}",
        "",
        "Steps:",
    ]
    for step in team.steps:
        line = f"- {step.id} {step.status.value}: {step.title}"
        if step.error:
            line += f" ({step.error})"
        lines.append(line)
    if team.result:
        lines.extend(["", "Result:", team.result])
    if team.error:
        lines.extend(["", "Error:", team.error])
    if team.trace_path:
        lines.extend(["", f"Trace: {team.trace_path}"])
    return "\n".join(lines)


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


def _step_counts(steps: list[ExecutionStep]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.status.value] = counts.get(step.status.value, 0) + 1
    return counts


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


def _single_line(text: str, limit: int) -> str:
    normalized = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _positive_or_default(value: int | None, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return max(1, default)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
