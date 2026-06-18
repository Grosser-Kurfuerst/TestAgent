from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from my_agent.llm import AgentLLM
from my_agent.llm.types import MessageLike
from my_agent.plan.graph import PlanValidationError, TaskGraph
from my_agent.plan.types import PlanState, PlanTask, TaskType


class Planner:
    def __init__(self, llm: AgentLLM, *, max_tasks: int = 12):
        self.llm = llm
        self.max_tasks = max_tasks

    def create_plan(
        self,
        goal: str,
        *,
        repo_context: str = "",
        conversation: Sequence[MessageLike] | None = None,
    ) -> PlanState:
        messages = self._build_messages(goal, repo_context=repo_context, conversation=conversation or [])
        try:
            response = self.llm.chat(messages, tools=None)
        except Exception as exc:
            raise PlanValidationError(
                "planner_llm_failed",
                f"Planner LLM call failed: {exc}",
                {"error": str(exc)},
            ) from exc
        return self.parse_plan(goal, response.content)

    def parse_plan(self, goal: str, raw_text: str) -> PlanState:
        try:
            payload = json.loads(_strip_json_code_fence(raw_text))
        except json.JSONDecodeError as exc:
            raise PlanValidationError(
                "invalid_plan_json",
                f"Planner response is not valid JSON: {exc}",
                {"error": str(exc)},
            ) from exc

        if not isinstance(payload, Mapping):
            raise PlanValidationError("invalid_plan_json", "Planner response must be a JSON object.")

        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise PlanValidationError("empty_plan", "Planner response must contain a non-empty tasks array.")

        original_ids = [_string(task.get("id")).strip() if isinstance(task, Mapping) else "" for task in raw_tasks]
        non_empty_originals = [task_id for task_id in original_ids if task_id]
        if len(set(non_empty_originals)) != len(non_empty_originals):
            raise PlanValidationError("duplicate_task_id", "Planner response contains duplicate original task ids.")

        id_mapping = {original_id: f"task_{index}" for index, original_id in enumerate(original_ids, start=1) if original_id}
        normalized_ids = [f"task_{index}" for index in range(1, len(raw_tasks) + 1)]

        tasks: list[PlanTask] = []
        for index, raw_task in enumerate(raw_tasks):
            if not isinstance(raw_task, Mapping):
                raise PlanValidationError(
                    "invalid_plan_json",
                    "Each planner task must be a JSON object.",
                    {"task_index": index},
                )
            task_id = normalized_ids[index]
            description = _string(raw_task.get("description")).strip()
            title = _string(raw_task.get("title")).strip() or _single_line(description) or task_id
            depends_on = [_map_dependency(dep, id_mapping, normalized_ids) for dep in _raw_dependencies(raw_task, index)]
            tasks.append(
                PlanTask(
                    id=task_id,
                    title=title,
                    description=description,
                    type=TaskType.from_value(raw_task.get("type")),
                    depends_on=depends_on,
                    acceptance=_string(raw_task.get("acceptance")).strip(),
                    max_steps=_int_or_none(raw_task.get("max_steps")),
                )
            )

        graph = TaskGraph(tasks, max_tasks=self.max_tasks)
        graph.validate()
        summary = _string(payload.get("summary")).strip()
        plan = PlanState.create(goal=goal, summary=summary, tasks=tasks)
        plan.execution_order = graph.topological_order()
        return plan

    def _build_messages(
        self,
        goal: str,
        *,
        repo_context: str,
        conversation: Sequence[MessageLike],
    ) -> list[MessageLike]:
        schema = (
            "Return only JSON with this shape: "
            '{"summary": "...", "tasks": [{"id": "task_1", "title": "...", '
            '"description": "...", "type": "inspect|edit|test|verify|analysis|documentation", '
            '"depends_on": [], "acceptance": "..."}]}. '
            "Use at most the configured task limit. Do not include markdown."
        )
        context = repo_context.strip() or "No repo context was provided."
        recent = _render_recent_conversation(conversation)
        return [
            {"role": "system", "content": "You are a planner for a coding agent. " + schema},
            {
                "role": "user",
                "content": (
                    f"Goal:\n{goal.strip()}\n\n"
                    f"Repository context:\n{context}\n\n"
                    f"Recent conversation:\n{recent}"
                ),
            },
        ]


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


_MISSING = object()


def _raw_dependencies(raw_task: Mapping[str, Any], task_index: int) -> list[str]:
    raw = raw_task.get("depends_on", _MISSING)
    if raw is _MISSING:
        raw = raw_task.get("dependencies", _MISSING)
    if raw is _MISSING or raw is None:
        return []
    if not isinstance(raw, list):
        raise PlanValidationError(
            "invalid_plan_json",
            "Task depends_on must be an array of non-empty strings.",
            {"task_index": task_index},
        )

    dependencies: list[str] = []
    for dep_index, dependency in enumerate(raw):
        if not isinstance(dependency, str) or not dependency.strip():
            raise PlanValidationError(
                "invalid_plan_json",
                "Task dependencies must be non-empty strings.",
                {"task_index": task_index, "dependency_index": dep_index},
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


def _render_recent_conversation(conversation: Sequence[MessageLike], limit: int = 6) -> str:
    rendered: list[str] = []
    for message in list(conversation)[-limit:]:
        if isinstance(message, Mapping):
            role = _string(message.get("role")) or "unknown"
            content = _string(message.get("content")).strip()
            if content:
                rendered.append(f"{role}: {content}")
    return "\n".join(rendered) if rendered else "No recent conversation."


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _single_line(value: str, limit: int = 120) -> str:
    text = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""
