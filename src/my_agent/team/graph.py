from __future__ import annotations

from my_agent.plan import PlanTask, TaskGraph
from my_agent.team.types import ExecutionStep, StepStatus


def validate_team_graph(steps: list[ExecutionStep], *, max_steps: int = 12) -> None:
    TaskGraph(_steps_to_plan_tasks(steps), max_tasks=max_steps).validate()


def topological_order(steps: list[ExecutionStep], *, max_steps: int = 12) -> list[str]:
    return TaskGraph(_steps_to_plan_tasks(steps), max_tasks=max_steps).topological_order()


def execution_batches(steps: list[ExecutionStep], *, max_steps: int = 12) -> list[list[str]]:
    return TaskGraph(_steps_to_plan_tasks(steps), max_tasks=max_steps).execution_batches()


def get_executable_steps(steps: list[ExecutionStep]) -> list[ExecutionStep]:
    statuses = {step.id: step.status for step in steps}
    return [
        step
        for step in steps
        if step.status == StepStatus.PENDING
        and all(statuses.get(dependency) == StepStatus.COMPLETED for dependency in step.dependencies)
    ]


def _steps_to_plan_tasks(steps: list[ExecutionStep]) -> list[PlanTask]:
    return [
        PlanTask(
            id=step.id,
            title=step.title,
            description=step.description,
            type=step.type,
            depends_on=list(step.dependencies),
            acceptance=step.acceptance,
        )
        for step in steps
    ]
