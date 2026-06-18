from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from my_agent.plan.types import PlanTask


_TASK_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass
class PlanValidationError(ValueError):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class TaskGraph:
    def __init__(self, tasks: Iterable[PlanTask], *, max_tasks: int = 12):
        self.tasks = list(tasks)
        self.max_tasks = max_tasks

    def validate(self) -> None:
        if not self.tasks:
            raise PlanValidationError("empty_plan", "Plan must contain at least one task.")
        if len(self.tasks) > self.max_tasks:
            raise PlanValidationError(
                "too_many_tasks",
                f"Plan contains {len(self.tasks)} tasks, exceeding the limit of {self.max_tasks}.",
                {"max_tasks": self.max_tasks, "task_count": len(self.tasks)},
            )

        seen: set[str] = set()
        for task in self.tasks:
            if not task.id:
                raise PlanValidationError("invalid_task_id", "Task id must be non-empty.", {"task": task.to_dict()})
            if not _TASK_ID_RE.fullmatch(task.id):
                raise PlanValidationError("invalid_task_id", f"Task id is invalid: {task.id}", {"task_id": task.id})
            if task.id in seen:
                raise PlanValidationError("duplicate_task_id", f"Duplicate task id: {task.id}", {"task_id": task.id})
            seen.add(task.id)

        task_ids = {task.id for task in self.tasks}
        for task in self.tasks:
            for dependency in task.depends_on:
                if dependency == task.id:
                    raise PlanValidationError(
                        "self_dependency",
                        f"Task {task.id} cannot depend on itself.",
                        {"task_id": task.id},
                    )
                if dependency not in task_ids:
                    raise PlanValidationError(
                        "missing_dependency",
                        f"Task {task.id} depends on unknown task {dependency}.",
                        {"task_id": task.id, "dependency": dependency},
                    )

        self._topological_order_checked()

    def topological_order(self) -> list[str]:
        self.validate()
        return self._topological_order_checked()

    def execution_batches(self) -> list[list[str]]:
        self.validate()
        remaining = {task.id: task for task in self.tasks}
        completed: set[str] = set()
        batches: list[list[str]] = []

        while remaining:
            batch = [
                task.id
                for task in self.tasks
                if task.id in remaining and all(dependency in completed for dependency in task.depends_on)
            ]
            if not batch:
                raise self._cycle_error(remaining)
            batches.append(batch)
            for task_id in batch:
                completed.add(task_id)
                remaining.pop(task_id, None)

        return batches

    def _topological_order_checked(self) -> list[str]:
        ordered: list[str] = []
        task_map = {task.id: task for task in self.tasks}
        indegree = {task.id: 0 for task in self.tasks}
        dependents: dict[str, list[str]] = {task.id: [] for task in self.tasks}

        for task in self.tasks:
            for dependency in task.depends_on:
                indegree[task.id] += 1
                dependents.setdefault(dependency, []).append(task.id)

        ready = [task.id for task in self.tasks if indegree[task.id] == 0]
        while ready:
            task_id = ready.pop(0)
            ordered.append(task_id)
            for dependent_id in dependents.get(task_id, []):
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    ready.append(dependent_id)

        if len(ordered) != len(self.tasks):
            remaining = {task_id: task_map[task_id] for task_id, degree in indegree.items() if degree > 0}
            raise self._cycle_error(remaining)
        return ordered

    def _cycle_error(self, remaining: dict[str, PlanTask]) -> PlanValidationError:
        candidates = [task.id for task in self.tasks if task.id in remaining]
        return PlanValidationError(
            "cycle_detected",
            "Plan contains a dependency cycle.",
            {"cycle_candidates": candidates},
        )
