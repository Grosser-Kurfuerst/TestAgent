from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from my_agent.plan import TaskType
from my_agent.text_safety import repair_surrogates


class AgentRole(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    REVIEWER = "reviewer"


class TeamStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    REVIEWING = "reviewing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_value(cls, value: object) -> "TeamStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for item in cls:
                if item.value == normalized:
                    return item
        return cls.CREATED


class StepStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

    @classmethod
    def from_value(cls, value: object) -> "StepStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for item in cls:
                if item.value == normalized:
                    return item
        return cls.PENDING


@dataclass
class ExecutionStep:
    id: str
    title: str
    description: str
    type: TaskType = TaskType.ANALYSIS
    dependencies: list[str] = field(default_factory=list)
    acceptance: str = ""
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    error: str = ""
    review_summary: str = ""
    review_issues: list[str] = field(default_factory=list)
    review_suggestions: list[str] = field(default_factory=list)
    attempts: int = 0
    worker_name: str = ""
    trace_path: str = ""
    started_at: str = ""
    ended_at: str = ""

    def __post_init__(self) -> None:
        self.id = _string(self.id).strip()
        self.title = _single_line(self.title)
        self.description = _string(self.description).strip()
        self.type = _task_type_from_value(self.type)
        self.dependencies = [_string(dep).strip() for dep in self.dependencies if _string(dep).strip()]
        self.acceptance = _string(self.acceptance).strip()
        self.status = StepStatus.from_value(self.status)
        self.result = _string(self.result)
        self.error = _string(self.error)
        self.review_summary = _string(self.review_summary)
        self.review_issues = [_string(issue).strip() for issue in self.review_issues if _string(issue).strip()]
        self.review_suggestions = [
            _string(suggestion).strip() for suggestion in self.review_suggestions if _string(suggestion).strip()
        ]
        self.attempts = max(0, int(self.attempts)) if isinstance(self.attempts, int) else 0
        self.worker_name = _string(self.worker_name)
        self.trace_path = _string(self.trace_path)
        self.started_at = _string(self.started_at)
        self.ended_at = _string(self.ended_at)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionStep":
        dependencies = _raw_string_list(payload.get("dependencies", payload.get("depends_on", [])))
        return cls(
            id=_string(payload.get("id")),
            title=_string(payload.get("title")) or _default_title(payload),
            description=_string(payload.get("description")),
            type=_task_type_from_value(payload.get("type")),
            dependencies=dependencies,
            acceptance=_string(payload.get("acceptance")),
            status=StepStatus.from_value(payload.get("status")),
            result=_string(payload.get("result")),
            error=_string(payload.get("error")),
            review_summary=_string(payload.get("review_summary")),
            review_issues=_raw_string_list(payload.get("review_issues", [])),
            review_suggestions=_raw_string_list(payload.get("review_suggestions", [])),
            attempts=payload.get("attempts") if isinstance(payload.get("attempts"), int) else 0,
            worker_name=_string(payload.get("worker_name")),
            trace_path=_string(payload.get("trace_path")),
            started_at=_string(payload.get("started_at")),
            ended_at=_string(payload.get("ended_at")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.type.value,
            "dependencies": list(self.dependencies),
            "acceptance": self.acceptance,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "review_summary": self.review_summary,
            "review_issues": list(self.review_issues),
            "review_suggestions": list(self.review_suggestions),
            "attempts": self.attempts,
            "worker_name": self.worker_name,
            "trace_path": self.trace_path,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass(frozen=True)
class ReviewDecision:
    approved: bool
    summary: str = ""
    issues: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    raw: str = ""
    parse_error: str = ""

    @property
    def feedback_text(self) -> str:
        parts: list[str] = []
        if self.parse_error:
            parts.append(f"Review parse error: {self.parse_error}")
        if self.summary:
            parts.append(f"Review summary: {self.summary}")
        if self.issues:
            parts.append("Issues:\n" + "\n".join(f"- {issue}" for issue in self.issues))
        if self.suggestions:
            parts.append("Suggestions:\n" + "\n".join(f"- {suggestion}" for suggestion in self.suggestions))
        if not parts and self.raw:
            parts.append(self.raw)
        return "\n\n".join(parts)


@dataclass
class TeamState:
    id: str
    goal: str
    summary: str = ""
    steps: list[ExecutionStep] = field(default_factory=list)
    status: TeamStatus = TeamStatus.CREATED
    execution_order: list[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    trace_path: str = ""
    created_at: str = ""
    started_at: str = ""
    ended_at: str = ""

    def __post_init__(self) -> None:
        self.id = _string(self.id).strip() or new_team_id()
        self.goal = _string(self.goal).strip()
        self.summary = _string(self.summary).strip()
        self.steps = [step if isinstance(step, ExecutionStep) else ExecutionStep.from_dict(step) for step in self.steps]
        self.status = TeamStatus.from_value(self.status)
        self.execution_order = [_string(step_id).strip() for step_id in self.execution_order if _string(step_id).strip()]
        self.result = _string(self.result)
        self.error = _string(self.error)
        self.trace_path = _string(self.trace_path)
        self.created_at = _string(self.created_at)
        self.started_at = _string(self.started_at)
        self.ended_at = _string(self.ended_at)

    @classmethod
    def create(
        cls,
        goal: str,
        *,
        summary: str = "",
        steps: list[ExecutionStep] | None = None,
    ) -> "TeamState":
        return cls(id=new_team_id(), goal=goal, summary=summary, steps=list(steps or []), created_at=_now())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeamState":
        raw_steps = payload.get("steps", [])
        steps = raw_steps if isinstance(raw_steps, list) else []
        raw_order = payload.get("execution_order", [])
        execution_order = raw_order if isinstance(raw_order, list) else []
        return cls(
            id=_string(payload.get("id")),
            goal=_string(payload.get("goal")),
            summary=_string(payload.get("summary")),
            steps=[ExecutionStep.from_dict(step) for step in steps if isinstance(step, Mapping)],
            status=TeamStatus.from_value(payload.get("status")),
            execution_order=[_string(step_id) for step_id in execution_order],
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
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status.value,
            "execution_order": list(self.execution_order),
            "result": self.result,
            "error": self.error,
            "trace_path": self.trace_path,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def step_by_id(self) -> dict[str, ExecutionStep]:
        return {step.id: step for step in self.steps}

    def get_step(self, step_id: str) -> ExecutionStep | None:
        return self.step_by_id().get(step_id)


def new_team_id() -> str:
    return f"team_{uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _task_type_from_value(value: object) -> TaskType:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "file_read": TaskType.INSPECT,
            "read": TaskType.INSPECT,
            "file_write": TaskType.EDIT,
            "write": TaskType.EDIT,
            "command": TaskType.TEST,
            "verification": TaskType.VERIFY,
        }
        if normalized in aliases:
            return aliases[normalized]
    return TaskType.from_value(value)


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


def _raw_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_string(item).strip() for item in value if _string(item).strip()]


def _string(value: object) -> str:
    return repair_surrogates(value) if isinstance(value, str) else ""
