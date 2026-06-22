from __future__ import annotations

from my_agent.plan.types import PlanState, PlanTask


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


def _task_counts(tasks: list[PlanTask]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.status.value] = counts.get(task.status.value, 0) + 1
    return counts

