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
