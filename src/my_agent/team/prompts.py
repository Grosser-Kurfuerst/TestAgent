from __future__ import annotations

from typing import Sequence

from my_agent.llm.types import MessageLike


TEAM_PLANNER_SYSTEM_PROMPT = """\
You are the planner in a Multi-Agent coding team.

Analyze the user request and return a JSON execution plan for an orchestrator.
The orchestrator is the only coordinator. Sub-agents do not talk to each other.

Return only JSON with this shape:
{
  "summary": "short task summary",
  "steps": [
    {
      "id": "step_1",
      "title": "short title",
      "description": "specific executable step",
      "type": "inspect|edit|test|verify|analysis|documentation",
      "dependencies": [],
      "acceptance": "how the reviewer can approve this step"
    }
  ]
}

Rules:
- Use unique step ids.
- Keep independent steps dependency-free so the orchestrator can run them in parallel later.
- Add a dependency only when a step needs the previous step's result.
- Simple tasks can use 1-3 steps; complex tasks can use 5-10 steps.
- Do not include markdown or commentary.
"""


TEAM_WORKER_SYSTEM_PROMPT = """\
You are a worker sub-agent in a Multi-Agent coding team.

The orchestrator assigns exactly one execution step at a time. Complete only
the current step, use dependency context when relevant, and do not skip ahead
to later steps. Use repository tools when inspection, edits, or verification
are needed. Inspect files before editing, run relevant tests after edits, and
finish with a concise result that the reviewer can evaluate.
"""


TEAM_REVIEWER_SYSTEM_PROMPT = """\
You are a reviewer sub-agent in a Multi-Agent coding team.

Review only the worker result for the current step. Do not call tools and do
not modify files. Return only JSON with this shape:
{
  "approved": true,
  "summary": "short review summary",
  "issues": [],
  "suggestions": []
}

Set approved to false when the result is incomplete, unverifiable, violates the
step boundary, or misses the acceptance criteria.
"""


def build_team_planner_messages(
    goal: str,
    *,
    repo_context: str = "",
    memory_context: str = "",
    conversation: Sequence[MessageLike] | None = None,
) -> list[MessageLike]:
    return [
        {"role": "system", "content": TEAM_PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Goal:\n{goal.strip()}\n\n"
                f"Repository context:\n{repo_context.strip() or 'No repo context was provided.'}\n\n"
                f"Relevant memory:\n{memory_context.strip() or 'No memory context was provided.'}\n\n"
                f"Recent conversation:\n{_render_conversation(conversation or [])}"
            ),
        },
    ]


def build_worker_prompt(
    goal: str,
    step_id: str,
    step_type: str,
    title: str,
    description: str,
    acceptance: str,
    *,
    dependency_context: str = "",
    feedback: str = "",
    test_command: str | None = None,
) -> str:
    lines = [
        f"Overall goal:\n{goal.strip()}",
        "Current step:",
        f"- id: {step_id}",
        f"- type: {step_type}",
        f"- title: {title.strip() or step_id}",
        f"- description:\n{description.strip()}",
        f"- acceptance:\n{acceptance.strip() or 'Complete this step in a verifiable way.'}",
        "Execution boundary:\nOnly complete this step. Do not perform later dependent steps unless required to verify the current step.",
        "Completed dependency context:",
        dependency_context.strip() or "No completed dependencies.",
        f"Default test command: {test_command or 'not configured'}",
    ]
    if feedback.strip():
        lines.extend(["Reviewer feedback from previous attempt:", feedback.strip()])
    return "\n\n".join(lines)


def build_reviewer_prompt(
    goal: str,
    step_id: str,
    step_type: str,
    title: str,
    description: str,
    acceptance: str,
    *,
    dependency_context: str = "",
    result: str = "",
) -> str:
    return "\n\n".join(
        [
            f"Overall goal:\n{goal.strip()}",
            "Step under review:",
            f"- id: {step_id}",
            f"- type: {step_type}",
            f"- title: {title.strip() or step_id}",
            f"- description:\n{description.strip()}",
            f"- acceptance:\n{acceptance.strip() or 'Complete this step in a verifiable way.'}",
            "Completed dependency context:",
            dependency_context.strip() or "No completed dependencies.",
            "Worker result:",
            result.strip() or "No worker result was provided.",
            "Return only the reviewer JSON object.",
        ]
    )


def _render_conversation(conversation: Sequence[MessageLike], limit: int = 6) -> str:
    rendered: list[str] = []
    for message in list(conversation)[-limit:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "unknown")).strip() or "unknown"
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            rendered.append(f"{role}: {content.strip()}")
    return "\n".join(rendered) if rendered else "No recent conversation."
