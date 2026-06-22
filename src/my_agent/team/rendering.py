from __future__ import annotations

from my_agent.team.types import ExecutionStep, TeamState


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


def _step_counts(steps: list[ExecutionStep]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.status.value] = counts.get(step.status.value, 0) + 1
    return counts

