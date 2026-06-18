from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from my_agent.text_safety import repair_surrogates


class TaskType(str, Enum):
    INSPECT = "inspect"
    EDIT = "edit"
    TEST = "test"
    VERIFY = "verify"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"

    @classmethod
    def from_value(cls, value: object) -> "TaskType":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for item in cls:
                if item.value == normalized:
                    return item
        return cls.ANALYSIS


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

    @classmethod
    def from_value(cls, value: object) -> "TaskStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for item in cls:
                if item.value == normalized:
                    return item
        return cls.PENDING


class PlanStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_value(cls, value: object) -> "PlanStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for item in cls:
                if item.value == normalized:
                    return item
        return cls.CREATED


@dataclass
class TaskResult:
    task_id: str
    ok: bool
    output: str = ""
    error: str = ""
    trace_path: str = ""
    stop_reason: str = ""

    @classmethod
    def success(
        cls,
        task_id: str,
        output: str = "",
        *,
        trace_path: str = "",
        stop_reason: str = "",
    ) -> "TaskResult":
        return cls(task_id=task_id, ok=True, output=output, trace_path=trace_path, stop_reason=stop_reason)

    @classmethod
    def failure(
        cls,
        task_id: str,
        error: str,
        *,
        output: str = "",
        trace_path: str = "",
        stop_reason: str = "",
    ) -> "TaskResult":
        return cls(
            task_id=task_id,
            ok=False,
            output=output,
            error=error,
            trace_path=trace_path,
            stop_reason=stop_reason,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskResult":
        return cls(
            task_id=_string(payload.get("task_id")),
            ok=bool(payload.get("ok")),
            output=_string(payload.get("output")),
            error=_string(payload.get("error")),
            trace_path=_string(payload.get("trace_path")),
            stop_reason=_string(payload.get("stop_reason")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "trace_path": self.trace_path,
            "stop_reason": self.stop_reason,
        }


@dataclass
class PlanTask:
    id: str
    title: str
    description: str
    type: TaskType = TaskType.ANALYSIS
    depends_on: list[str] = field(default_factory=list)
    acceptance: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    trace_path: str = ""
    max_steps: int | None = None
    started_at: str = ""
    ended_at: str = ""

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        self.title = _single_line(self.title)
        self.description = _string(self.description).strip()
        self.type = TaskType.from_value(self.type)
        self.depends_on = [_string(dep).strip() for dep in self.depends_on if _string(dep).strip()]
        self.acceptance = _string(self.acceptance).strip()
        self.status = TaskStatus.from_value(self.status)
        self.result = _string(self.result)
        self.error = _string(self.error)
        self.trace_path = _string(self.trace_path)
        self.started_at = _string(self.started_at)
        self.ended_at = _string(self.ended_at)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanTask":
        depends = payload.get("depends_on", payload.get("dependencies", []))
        if not isinstance(depends, list):
            depends = []
        raw_max_steps = payload.get("max_steps")
        max_steps = raw_max_steps if isinstance(raw_max_steps, int) and not isinstance(raw_max_steps, bool) else None
        return cls(
            id=_string(payload.get("id")),
            title=_string(payload.get("title")) or _default_title(payload),
            description=_string(payload.get("description")),
            type=TaskType.from_value(payload.get("type")),
            depends_on=[_string(dep) for dep in depends],
            acceptance=_string(payload.get("acceptance")),
            status=TaskStatus.from_value(payload.get("status")),
            result=_string(payload.get("result")),
            error=_string(payload.get("error")),
            trace_path=_string(payload.get("trace_path")),
            max_steps=max_steps,
            started_at=_string(payload.get("started_at")),
            ended_at=_string(payload.get("ended_at")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.type.value,
            "depends_on": list(self.depends_on),
            "acceptance": self.acceptance,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "trace_path": self.trace_path,
            "max_steps": self.max_steps,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class PlanState:
    id: str
    goal: str
    summary: str = ""
    tasks: list[PlanTask] = field(default_factory=list)
    status: PlanStatus = PlanStatus.CREATED
    execution_order: list[str] = field(default_factory=list)
    current_task_id: str = ""
    result: str = ""
    error: str = ""
    trace_path: str = ""
    created_at: str = ""
    started_at: str = ""
    ended_at: str = ""

    def __post_init__(self) -> None:
        self.id = self.id.strip() or new_plan_id()
        self.goal = _string(self.goal).strip()
        self.summary = _string(self.summary).strip()
        self.tasks = [task if isinstance(task, PlanTask) else PlanTask.from_dict(task) for task in self.tasks]
        self.status = PlanStatus.from_value(self.status)
        self.execution_order = [_string(task_id).strip() for task_id in self.execution_order if _string(task_id).strip()]
        self.current_task_id = _string(self.current_task_id).strip()
        self.result = _string(self.result)
        self.error = _string(self.error)
        self.trace_path = _string(self.trace_path)
        self.created_at = _string(self.created_at)
        self.started_at = _string(self.started_at)
        self.ended_at = _string(self.ended_at)

    @classmethod
    def create(cls, goal: str, *, summary: str = "", tasks: list[PlanTask] | None = None) -> "PlanState":
        return cls(id=new_plan_id(), goal=goal, summary=summary, tasks=list(tasks or []), created_at=_now())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanState":
        raw_tasks = payload.get("tasks", [])
        tasks = raw_tasks if isinstance(raw_tasks, list) else []
        raw_order = payload.get("execution_order", [])
        execution_order = raw_order if isinstance(raw_order, list) else []
        return cls(
            id=_string(payload.get("id")),
            goal=_string(payload.get("goal")),
            summary=_string(payload.get("summary")),
            tasks=[PlanTask.from_dict(task) for task in tasks if isinstance(task, Mapping)],
            status=PlanStatus.from_value(payload.get("status")),
            execution_order=[_string(task_id) for task_id in execution_order],
            current_task_id=_string(payload.get("current_task_id")),
            result=_string(payload.get("result")),
            error=_string(payload.get("error")),
            trace_path=_string(payload.get("trace_path")),
            created_at=_string(payload.get("created_at")),
            started_at=_string(payload.get("started_at")),
            ended_at=_string(payload.get("ended_at")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "summary": self.summary,
            "tasks": [task.to_dict() for task in self.tasks],
            "status": self.status.value,
            "execution_order": list(self.execution_order),
            "current_task_id": self.current_task_id,
            "result": self.result,
            "error": self.error,
            "trace_path": self.trace_path,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def task_by_id(self) -> dict[str, PlanTask]:
        return {task.id: task for task in self.tasks}

    def get_task(self, task_id: str) -> PlanTask | None:
        return self.task_by_id().get(task_id)


def new_plan_id() -> str:
    return f"plan_{uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_title(payload: Mapping[str, Any]) -> str:
    description = _string(payload.get("description")).strip()
    if description:
        return _single_line(description)
    return _string(payload.get("id"))


def _single_line(value: object, limit: int = 120) -> str:
    text = _string(value).replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _string(value: object) -> str:
    return repair_surrogates(value) if isinstance(value, str) else ""
