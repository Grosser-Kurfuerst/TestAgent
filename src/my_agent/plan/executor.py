from __future__ import annotations

from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from my_agent.cancellation import CancellationToken
from my_agent.config import AgentConfig
from my_agent.events import BufferedEventSink
from my_agent.hitl.handler import HitlHandler
from my_agent.memory import MemoryService
from my_agent.parallel import create_bounded_executor, shutdown_executor
from my_agent.plan.graph import TaskGraph
from my_agent.plan.store import InMemoryPlanStore, PlanStore
from my_agent.plan.types import PlanState, PlanStatus, PlanTask, TaskResult, TaskStatus, TaskType
from my_agent.react.child_runner import ChildReActRequest, ChildReActRunner
from my_agent.utils.numbers import positive_or_default
from my_agent.utils.text import single_line


class TaskRunner(Protocol):
    def run_task(self, plan: PlanState, task: PlanTask) -> TaskResult:
        ...


class PlanCancelled(Exception):
    """Raised by task runners when a plan should stop as cancelled."""


@dataclass(frozen=True)
class PlanEvent:
    event: str
    plan_id: str
    status: str
    task_id: str = ""
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event,
            "plan_id": self.plan_id,
            "status": self.status,
            "task_id": self.task_id,
            "payload": dict(self.payload),
        }


class PlanExecutor:
    def __init__(
        self,
        runner: TaskRunner,
        *,
        store: PlanStore | None = None,
        event_sink: Callable[[PlanEvent], None] | None = None,
        max_tasks: int = 12,
        parallel_enabled: bool = False,
        max_parallel_tasks: int = 4,
        batch_timeout_seconds: int = 1_800,
        shutdown_grace_seconds: int = 2,
        cancellation_token: CancellationToken | None = None,
    ):
        self.runner = runner
        self.store = store or InMemoryPlanStore()
        self.event_sink = event_sink
        self.max_tasks = max_tasks
        self.parallel_enabled = parallel_enabled
        self.max_parallel_tasks = max(1, max_parallel_tasks)
        self.batch_timeout_seconds = max(0.001, float(batch_timeout_seconds))
        self.shutdown_grace_seconds = max(0, shutdown_grace_seconds)
        self.cancellation_token = cancellation_token

    def execute(self, plan: PlanState) -> PlanState:
        graph = TaskGraph(plan.tasks, max_tasks=self.max_tasks)
        graph.validate()
        batches = graph.execution_batches()
        plan.execution_order = [task_id for batch in batches for task_id in batch]
        plan.status = PlanStatus.RUNNING
        plan.started_at = plan.started_at or _now()
        plan.error = ""
        self._save_and_emit(plan, "plan.started")

        for batch_ids in batches:
            if self._is_cancelled():
                self._cancel_remaining(plan, exclude=set())
                break

            executable: list[PlanTask] = []
            for task_id in batch_ids:
                task = plan.get_task(task_id)
                if task is None or task.status in _TERMINAL_STATUSES:
                    continue
                blocked_by = _blocked_dependencies(plan, task)
                if blocked_by:
                    self._mark_skipped(
                        plan,
                        task,
                        f"Skipped because dependencies did not succeed: {', '.join(blocked_by)}",
                    )
                    continue
                self._mark_ready(plan, task)
                executable.append(task)

            if not executable:
                continue

            cancelled = False
            applied_task_ids: set[str] = set()
            for task, result in self._execute_task_batch(plan, executable):
                applied_task_ids.add(task.id)
                if result.ok:
                    self._mark_succeeded(plan, task, result)
                elif result.stop_reason == "cancelled":
                    self._mark_cancelled(plan, task, result.error or "Task was cancelled.")
                    cancelled = True
                else:
                    self._mark_failed(plan, task, result)

            if cancelled or self._is_cancelled():
                self._cancel_remaining(plan, exclude=applied_task_ids)
                break

        self._finish_plan(plan)
        return plan

    def _execute_task_batch(self, plan: PlanState, tasks: list[PlanTask]) -> list[tuple[PlanTask, TaskResult]]:
        if len(tasks) == 1 or not self.parallel_enabled:
            results: list[tuple[PlanTask, TaskResult]] = []
            for task in tasks:
                result = self._run_task(plan, task)
                results.append((task, result))
                if result.stop_reason == "cancelled":
                    break
            return results
        return self._execute_task_batch_parallel(plan, tasks)

    def _execute_task_batch_parallel(self, plan: PlanState, tasks: list[PlanTask]) -> list[tuple[PlanTask, TaskResult]]:
        buffers = BufferedEventSink(self.event_sink)
        task_tokens = {task.id: self._child_token() for task in tasks}
        max_workers = min(len(tasks), self.max_parallel_tasks, 4)
        executor = create_bounded_executor(max_workers=max_workers, thread_name_prefix="agentcli-plan")
        futures: dict[str, Future[TaskResult]] = {}
        try:
            for task in tasks:
                if self._is_cancelled():
                    break
                self._mark_started(plan, task)
                runner = _runner_with_event_sink(self.runner, buffers.buffer_for(task.id).append)
                worker_plan, worker_task = _snapshot_plan_task(plan, task)
                futures[task.id] = executor.submit(
                    _run_task_worker,
                    runner,
                    worker_plan,
                    worker_task,
                    task_tokens[task.id],
                )

            done, not_done = wait(set(futures.values()), timeout=self.batch_timeout_seconds)
            timed_out = set(not_done)
            still_running = set()
            if not_done:
                for task_id, future in futures.items():
                    if future in not_done and task_tokens[task_id] is not None:
                        task_tokens[task_id].cancel("batch_timeout")
                done_after_grace, still_running = wait(not_done, timeout=self.shutdown_grace_seconds)
                done = done.union(done_after_grace)

            results: list[tuple[PlanTask, TaskResult]] = []
            for task in tasks:
                future = futures.get(task.id)
                if future is None:
                    results.append(
                        (
                            task,
                            TaskResult.failure(
                                task.id,
                                "Task was cancelled before it was scheduled.",
                                stop_reason="cancelled",
                            ),
                        )
                    )
                elif future in timed_out and not self._is_cancelled():
                    results.append((task, _batch_timeout_result(task.id, self.batch_timeout_seconds)))
                elif future in done and not future.cancelled():
                    results.append((task, _future_result(task.id, future)))
                else:
                    results.append((task, _batch_timeout_result(task.id, self.batch_timeout_seconds)))
            buffers.flush_in_order([task.id for task in tasks])
            return results
        finally:
            shutdown_executor(executor)

    def _run_task(self, plan: PlanState, task: PlanTask) -> TaskResult:
        if self._is_cancelled():
            return TaskResult.failure(task.id, "Task was cancelled before it started.", stop_reason="cancelled")
        self._mark_started(plan, task)
        return _run_task_worker(self.runner, plan, task, self._child_token())

    def _mark_started(self, plan: PlanState, task: PlanTask) -> None:
        task.status = TaskStatus.RUNNING
        task.started_at = task.started_at or _now()
        plan.current_task_id = task.id
        self._save_and_emit(plan, "plan.task.started", task)

    def _mark_ready(self, plan: PlanState, task: PlanTask) -> None:
        task.status = TaskStatus.READY
        self._save_and_emit(plan, "plan.task.ready", task)

    def _mark_succeeded(self, plan: PlanState, task: PlanTask, result: TaskResult) -> None:
        task.status = TaskStatus.SUCCEEDED
        task.result = result.output
        task.error = ""
        task.trace_path = result.trace_path
        task.ended_at = _now()
        self._save_and_emit(plan, "plan.task.completed", task)

    def _mark_failed(self, plan: PlanState, task: PlanTask, result: TaskResult) -> None:
        task.status = TaskStatus.FAILED
        task.result = result.output
        task.error = result.error or "Task failed."
        task.trace_path = result.trace_path
        task.ended_at = _now()
        self._save_and_emit(plan, "plan.task.failed", task)

    def _mark_cancelled(self, plan: PlanState, task: PlanTask, reason: str) -> None:
        task.status = TaskStatus.CANCELLED
        task.error = reason
        task.ended_at = _now()
        self._save_and_emit(plan, "plan.task.cancelled", task)

    def _cancel_remaining(self, plan: PlanState, *, exclude: set[str]) -> None:
        for task in plan.tasks:
            if task.id in exclude or task.status in _TERMINAL_STATUSES:
                continue
            task.status = TaskStatus.CANCELLED
            task.error = "Cancelled because plan execution was cancelled."
            task.ended_at = _now()
            self._save_and_emit(plan, "plan.task.cancelled", task)

    def _mark_skipped(self, plan: PlanState, task: PlanTask, reason: str) -> None:
        task.status = TaskStatus.SKIPPED
        task.error = reason
        task.ended_at = _now()
        self._save_and_emit(plan, "plan.task.skipped", task)

    def _finish_plan(self, plan: PlanState) -> None:
        plan.current_task_id = ""
        failed = [task for task in plan.tasks if task.status == TaskStatus.FAILED]
        skipped = [task for task in plan.tasks if task.status == TaskStatus.SKIPPED]
        cancelled = [task for task in plan.tasks if task.status == TaskStatus.CANCELLED]

        if cancelled:
            plan.status = PlanStatus.CANCELLED
            plan.error = _summarize_errors(cancelled)
            event = "plan.cancelled"
        elif failed or skipped:
            plan.status = PlanStatus.FAILED
            plan.error = _summarize_errors(failed + skipped)
            event = "plan.failed"
        else:
            plan.status = PlanStatus.SUCCEEDED
            plan.result = _summarize_results(plan.tasks)
            event = "plan.completed"
        plan.ended_at = _now()
        self._save_and_emit(plan, event)

    def _save_and_emit(self, plan: PlanState, event: str, task: PlanTask | None = None) -> None:
        self.store.save(plan)
        if self.event_sink is None:
            return
        self.event_sink(
            PlanEvent(
                event=event,
                plan_id=plan.id,
                status=task.status.value if task is not None else plan.status.value,
                task_id=task.id if task is not None else "",
                payload=_event_payload(plan, task),
            )
        )

    def _child_token(self) -> CancellationToken | None:
        if self.cancellation_token is None:
            return None
        return self.cancellation_token.child()

    def _is_cancelled(self) -> bool:
        return bool(self.cancellation_token is not None and self.cancellation_token.is_cancelled())


_TERMINAL_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.SKIPPED,
    TaskStatus.CANCELLED,
}


def _run_task_worker(
    runner: TaskRunner,
    plan: PlanState,
    task: PlanTask,
    cancellation_token: CancellationToken | None,
) -> TaskResult:
    try:
        run_with_token = getattr(runner, "run_task_with_token", None)
        if callable(run_with_token):
            return run_with_token(plan, task, cancellation_token=cancellation_token)
        return runner.run_task(plan, task)
    except PlanCancelled as exc:
        return TaskResult.failure(task.id, str(exc) or "Task was cancelled.", stop_reason="cancelled")
    except Exception as exc:
        return TaskResult.failure(task.id, str(exc), stop_reason="runner_exception")


def _snapshot_plan_task(plan: PlanState, task: PlanTask) -> tuple[PlanState, PlanTask]:
    snapshot = PlanState.from_dict(plan.to_dict())
    snapshot_task = snapshot.get_task(task.id)
    if snapshot_task is None:
        return snapshot, PlanTask.from_dict(task.to_dict())
    return snapshot, snapshot_task


def _runner_with_event_sink(runner: TaskRunner, event_sink: Callable[[object], None]) -> TaskRunner:
    with_event_sink = getattr(runner, "with_event_sink", None)
    if callable(with_event_sink):
        return with_event_sink(event_sink)
    return runner


def _future_result(task_id: str, future: Future[TaskResult]) -> TaskResult:
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 - task boundary converts worker failures.
        return TaskResult.failure(
            task_id,
            f"Task crashed: {type(exc).__name__}: {exc}",
            stop_reason="runner_exception",
        )


def _batch_timeout_result(task_id: str, timeout_seconds: int) -> TaskResult:
    return TaskResult.failure(
        task_id,
        f"Task batch timed out after {timeout_seconds}s.",
        stop_reason="batch_timeout",
    )


def _blocked_dependencies(plan: PlanState, task: PlanTask) -> list[str]:
    task_map = plan.task_by_id()
    return [dependency for dependency in task.depends_on if task_map[dependency].status != TaskStatus.SUCCEEDED]


def _event_payload(plan: PlanState, task: PlanTask | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "plan_id": plan.id,
        "goal": plan.goal,
        "status": task.status.value if task is not None else plan.status.value,
        "plan": plan.to_dict(),
    }
    if task is not None:
        payload.update(
            {
                "task_id": task.id,
                "title": task.title,
                "type": task.type.value,
                "task": task.to_dict(),
            }
        )
    return payload


def _summarize_results(tasks: list[PlanTask]) -> str:
    outputs = [task.result.strip() for task in tasks if task.result.strip()]
    return "\n".join(outputs)


def _summarize_errors(tasks: list[PlanTask]) -> str:
    errors = [f"{task.id}: {task.error}" for task in tasks if task.error]
    return "\n".join(errors)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ReActTaskRunner:
    def __init__(
        self,
        *,
        repo_path: str | Path,
        config: AgentConfig,
        llm: object,
        trace_dir: str | Path,
        command_timeout: int,
        test_command: str | None = None,
        default_max_steps: int | None = None,
        plan_task_max_steps: int | None = None,
        event_sink: Callable[[object], None] | None = None,
        memory_manager: MemoryService | None = None,
        hitl_handler: HitlHandler | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.test_command = test_command
        self.default_max_steps = positive_or_default(default_max_steps, config.max_steps)
        self.plan_task_max_steps = positive_or_default(plan_task_max_steps, getattr(config, "plan_task_max_steps", 6))
        self.event_sink = event_sink
        self.memory_manager = memory_manager
        self.hitl_handler = hitl_handler
        self.child_runner = ChildReActRunner(
            config=config,
            llm=llm,
            command_timeout=command_timeout,
            event_sink=event_sink,
            memory_manager=memory_manager,
            hitl_handler=hitl_handler,
        )

    def with_event_sink(self, event_sink: Callable[[object], None]) -> "ReActTaskRunner":
        return ReActTaskRunner(
            repo_path=self.repo_path,
            config=self.config,
            llm=self.llm,
            trace_dir=self.trace_dir,
            command_timeout=self.command_timeout,
            test_command=self.test_command,
            default_max_steps=self.default_max_steps,
            plan_task_max_steps=self.plan_task_max_steps,
            event_sink=event_sink,
            memory_manager=self.memory_manager,
            hitl_handler=self.hitl_handler,
        )

    def run_task(self, plan: PlanState, task: PlanTask) -> TaskResult:
        return self.run_task_with_token(plan, task, cancellation_token=None)

    def run_task_with_token(
        self,
        plan: PlanState,
        task: PlanTask,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> TaskResult:
        run_id = f"{task.id}_{plan.id}"
        try:
            return self.child_runner.run(
                ChildReActRequest(
                    task_id=task.id,
                    repo_path=self.repo_path,
                    task=self.build_task_prompt(plan, task),
                    test_command=self.test_command,
                    run_id=run_id,
                    trace_dir=self.trace_dir,
                    max_steps=self.max_steps_for_task(task),
                    memory_session_id=f"{plan.id}:{task.id}",
                    cancellation_token=cancellation_token,
                )
            )
        except KeyboardInterrupt as exc:
            raise PlanCancelled("Task execution was interrupted.") from exc

    def build_task_prompt(self, plan: PlanState, task: PlanTask) -> str:
        dependency_results = []
        task_map = plan.task_by_id()
        for dependency_id in task.depends_on:
            dependency = task_map.get(dependency_id)
            if dependency is None:
                continue
            summary = dependency.result.strip() or dependency.error.strip() or "No result recorded."
            dependency_results.append(f"- {dependency.id} ({dependency.status.value}): {single_line(summary, 500)}")

        lines = [
            f"Overall goal:\n{plan.goal}",
            f"Current plan summary:\n{plan.summary or 'No summary provided.'}",
            f"Current task id: {task.id}",
            f"Current task type: {task.type.value}",
            f"Current task title: {task.title}",
            f"Current task description:\n{task.description}",
            f"Acceptance criteria:\n{task.acceptance or 'Complete this task in a verifiable way.'}",
            "Execution boundary:\nOnly complete the current task. Use dependency results as context, but do not skip ahead to later plan tasks unless required to make the current task verifiable.",
            "Dependency results:",
            "\n".join(dependency_results) if dependency_results else "No completed dependencies.",
        ]
        return "\n\n".join(lines)

    def max_steps_for_task(self, task: PlanTask) -> int:
        candidates = [self.default_max_steps, self.plan_task_max_steps]
        if task.max_steps is not None:
            candidates.append(task.max_steps)
        else:
            candidates.append(_DEFAULT_TASK_MAX_STEPS.get(task.type, self.plan_task_max_steps))
        return max(1, min(value for value in candidates if value >= 1))


_DEFAULT_TASK_MAX_STEPS = {
    TaskType.INSPECT: 4,
    TaskType.EDIT: 6,
    TaskType.TEST: 3,
    TaskType.VERIFY: 4,
    TaskType.ANALYSIS: 3,
    TaskType.DOCUMENTATION: 5,
}
