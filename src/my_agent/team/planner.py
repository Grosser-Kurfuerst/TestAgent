from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from my_agent.llm import AgentLLM
from my_agent.llm.types import MessageLike
from my_agent.plan import PlanValidationError
from my_agent.team.graph import topological_order, validate_team_graph
from my_agent.team.prompts import build_team_planner_messages
from my_agent.team.types import ExecutionStep, TeamState


class TeamPlanner:
    def __init__(self, llm: AgentLLM, *, max_steps: int = 12):
        self.llm = llm
        self.max_steps = max_steps

    def create_team_plan(
        self,
        goal: str,
        *,
        repo_context: str = "",
        memory_context: str = "",
        conversation: Sequence[MessageLike] | None = None,
    ) -> TeamState:
        messages = build_team_planner_messages(
            goal,
            repo_context=repo_context,
            memory_context=memory_context,
            conversation=conversation or [],
        )
        try:
            response = self.llm.chat(messages, tools=None)
        except Exception as exc:
            raise PlanValidationError(
                "team_planner_llm_failed",
                f"Team planner LLM call failed: {exc}",
                {"error": str(exc)},
            ) from exc
        return self.parse_team_plan(goal, response.content)

    def parse_team_plan(self, goal: str, raw_text: str) -> TeamState:
        try:
            payload = json.loads(_strip_json_code_fence(raw_text))
        except json.JSONDecodeError as exc:
            raise PlanValidationError(
                "invalid_team_plan_json",
                f"Team planner response is not valid JSON: {exc}",
                {"error": str(exc)},
            ) from exc

        if not isinstance(payload, Mapping):
            raise PlanValidationError("invalid_team_plan_json", "Team planner response must be a JSON object.")

        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raw_steps = payload.get("tasks")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanValidationError(
                "empty_plan",
                "Team planner response must contain a non-empty steps or tasks array.",
            )

        original_ids = [_string(step.get("id")).strip() if isinstance(step, Mapping) else "" for step in raw_steps]
        non_empty_originals = [step_id for step_id in original_ids if step_id]
        if len(set(non_empty_originals)) != len(non_empty_originals):
            raise PlanValidationError(
                "duplicate_task_id",
                "Team planner response contains duplicate original step ids.",
            )

        id_mapping = {original_id: f"step_{index}" for index, original_id in enumerate(original_ids, start=1) if original_id}
        normalized_ids = [f"step_{index}" for index in range(1, len(raw_steps) + 1)]

        steps: list[ExecutionStep] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise PlanValidationError(
                    "invalid_team_plan_json",
                    "Each team step must be a JSON object.",
                    {"step_index": index},
                )
            step_id = normalized_ids[index]
            description = _string(raw_step.get("description")).strip()
            title = _string(raw_step.get("title")).strip() or _single_line(description) or step_id
            dependencies = [_map_dependency(dep, id_mapping, normalized_ids) for dep in _raw_dependencies(raw_step, index)]
            steps.append(
                ExecutionStep(
                    id=step_id,
                    title=title,
                    description=description,
                    type=raw_step.get("type"),  # type: ignore[arg-type]
                    dependencies=dependencies,
                    acceptance=_string(raw_step.get("acceptance")).strip(),
                )
            )

        validate_team_graph(steps, max_steps=self.max_steps)
        team = TeamState.create(goal=goal, summary=_string(payload.get("summary")).strip(), steps=steps)
        team.execution_order = topological_order(steps, max_steps=self.max_steps)
        return team


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


_MISSING = object()


def _raw_dependencies(raw_step: Mapping[str, Any], step_index: int) -> list[str]:
    raw = raw_step.get("dependencies", _MISSING)
    if raw is _MISSING:
        raw = raw_step.get("depends_on", _MISSING)
    if raw is _MISSING or raw is None:
        return []
    if not isinstance(raw, list):
        raise PlanValidationError(
            "invalid_team_plan_json",
            "Step dependencies must be an array of non-empty strings.",
            {"step_index": step_index},
        )

    dependencies: list[str] = []
    for dep_index, dependency in enumerate(raw):
        if not isinstance(dependency, str) or not dependency.strip():
            raise PlanValidationError(
                "invalid_team_plan_json",
                "Step dependencies must be non-empty strings.",
                {"step_index": step_index, "dependency_index": dep_index},
            )
        dependencies.append(dependency.strip())
    return dependencies


def _map_dependency(dependency: object, id_mapping: dict[str, str], normalized_ids: list[str]) -> str:
    raw = _string(dependency).strip()
    if raw in id_mapping:
        return id_mapping[raw]
    if raw in normalized_ids:
        return raw
    return raw


def _single_line(value: str, limit: int = 120) -> str:
    text = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""
