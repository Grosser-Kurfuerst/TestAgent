from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from my_agent.config import AgentConfig
from my_agent.plan.graph import TaskGraph
from my_agent.plan.types import PlanState, PlanStatus, PlanTask, TaskResult, TaskStatus, TaskType
from my_agent.react_runtime import ReActRuntime
from my_agent.schema import AgentState
from my_agent.text_safety import sanitize_json_value


class PlanStore(Protocol):
    def save(self, plan: PlanState) -> None:
        ...

    def get(self, plan_id: str) -> PlanState | None:
        ...


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


class InMemoryPlanStore:
    def __init__(self) -> None:
        self._plans: dict[str, PlanState] = {}

    def save(self, plan: PlanState) -> None:
        self._plans[plan.id] = PlanState.from_dict(plan.to_dict())

    def get(self, plan_id: str) -> PlanState | None:
        plan = self._plans.get(plan_id)
        return PlanState.from_dict(plan.to_dict()) if plan is not None else None


class JsonPlanStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, plan: PlanState) -> None:
        path = self._path_for(plan.id)
        path.write_text(json.dumps(sanitize_json_value(plan.to_dict()), ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, plan_id: str) -> PlanState | None:
        path = self._path_for(plan_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return PlanState.from_dict(payload)

    def _path_for(self, plan_id: str) -> Path:
        safe_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in plan_id)
        return self.directory / f"{safe_id}.json"


class PlanExecutor:
    def __init__(
        self,
        runner: TaskRunner,
        *,
        store: PlanStore | None = None,
        event_sink: Callable[[PlanEvent], None] | None = None,
        max_tasks: int = 12,
    ):
        self.runner = runner
        self.store = store or InMemoryPlanStore()
        self.event_sink = event_sink
        self.max_tasks = max_tasks

    def execute(self, plan: PlanState) -> PlanState:
        graph = TaskGraph(plan.tasks, max_tasks=self.max_tasks)
        graph.validate()
        plan.execution_order = graph.topological_order()
        plan.status = PlanStatus.RUNNING
        plan.started_at = plan.started_at or _now()
        plan.error = ""
        self._save_and_emit(plan, "plan.started")

        for task_id in plan.execution_order:
            task = plan.get_task(task_id)
            if task is None or task.status in _TERMINAL_STATUSES:
                continue

            task_map = plan.task_by_id()
            blocked_by = [
                dependency for dependency in task.depends_on if task_map[dependency].status != TaskStatus.SUCCEEDED
            ]
            if blocked_by:
                self._mark_skipped(plan, task, f"Skipped because dependencies did not succeed: {', '.join(blocked_by)}")
                continue

            self._mark_ready(plan, task)
            result = self._run_task(plan, task)
            if result.ok:
                self._mark_succeeded(plan, task, result)
            elif result.stop_reason == "cancelled":
                self._mark_cancelled(plan, task, result.error or "Task was cancelled.")
                self._cancel_remaining(plan, exclude={task.id})
                break
            else:
                self._mark_failed(plan, task, result)

        self._finish_plan(plan)
        return plan

    def _run_task(self, plan: PlanState, task: PlanTask) -> TaskResult:
        task.status = TaskStatus.RUNNING
        task.started_at = task.started_at or _now()
        plan.current_task_id = task.id
        self._save_and_emit(plan, "plan.task.started", task)
        try:
            return self.runner.run_task(plan, task)
        except PlanCancelled as exc:
            return TaskResult.failure(task.id, str(exc) or "Task was cancelled.", stop_reason="cancelled")
        except Exception as exc:
            return TaskResult.failure(task.id, str(exc), stop_reason="runner_exception")

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


_TERMINAL_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.SKIPPED,
    TaskStatus.CANCELLED,
}


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
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.test_command = test_command
        self.default_max_steps = _positive_or_default(default_max_steps, config.max_steps)
        self.plan_task_max_steps = _positive_or_default(plan_task_max_steps, getattr(config, "plan_task_max_steps", 6))
        self.event_sink = event_sink

    def run_task(self, plan: PlanState, task: PlanTask) -> TaskResult:
        state = AgentState.initial(
            repo_path=self.repo_path,
            task=self.build_task_prompt(plan, task),
            test_command=self.test_command,
            max_steps=self.max_steps_for_task(task),
            run_id=f"{task.id}_{plan.id}",
        )
        try:
            final_state = ReActRuntime(
                config=self.config,
                llm=self.llm,  # type: ignore[arg-type]
                trace_dir=self.trace_dir,
                command_timeout=self.command_timeout,
                event_sink=self.event_sink,
            ).run(state)
        except KeyboardInterrupt as exc:
            raise PlanCancelled("Task execution was interrupted.") from exc

        output = final_state.final_answer or final_state.review
        trace_path = str(final_state.trace_path or "")
        if _react_state_succeeded(final_state):
            return TaskResult.success(
                task.id,
                output,
                trace_path=trace_path,
                stop_reason=final_state.stop_reason,
            )
        return TaskResult.failure(
            task.id,
            _react_failure_message(final_state),
            output=output,
            trace_path=trace_path,
            stop_reason=final_state.stop_reason,
        )

    def build_task_prompt(self, plan: PlanState, task: PlanTask) -> str:
        dependency_results = []
        task_map = plan.task_by_id()
        for dependency_id in task.depends_on:
            dependency = task_map.get(dependency_id)
            if dependency is None:
                continue
            summary = dependency.result.strip() or dependency.error.strip() or "No result recorded."
            dependency_results.append(f"- {dependency.id} ({dependency.status.value}): {_single_line(summary, 500)}")

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


def _react_state_succeeded(state: AgentState) -> bool:
    return state.stop_reason in {"finish_called", "assistant_final"}


def _react_failure_message(state: AgentState) -> str:
    if state.review:
        return state.review
    return f"ReAct task failed with stop_reason={state.stop_reason or 'unknown'}."


def _single_line(text: str, limit: int) -> str:
    normalized = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


_DEFAULT_TASK_MAX_STEPS = {
    TaskType.INSPECT: 4,
    TaskType.EDIT: 6,
    TaskType.TEST: 3,
    TaskType.VERIFY: 4,
    TaskType.ANALYSIS: 3,
    TaskType.DOCUMENTATION: 5,
}


def _positive_or_default(value: int | None, default: int) -> int:
    if value is None or isinstance(value, bool) or value < 1:
        return max(1, default)
    return value
