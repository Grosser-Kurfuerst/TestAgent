from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from my_agent.context.profile import ContextProfile
from my_agent.context.tokens import estimate_tokens

CORE_TOOL_NAMES = {
    "list_files",
    "read_file",
    "grep",
    "retrieve_context",
    "replace_in_file",
    "write_file",
    "run_tests",
    "git_diff",
    "finish",
}


@dataclass(frozen=True)
class ToolSchemaBudget:
    definitions: list[dict[str, Any]]
    included_names: tuple[str, ...]
    omitted_names: tuple[str, ...]
    budget_tokens: int
    estimated_tokens: int
    over_budget: bool = False

    @property
    def omitted_count(self) -> int:
        return len(self.omitted_names)

    @property
    def included_count(self) -> int:
        return len(self.included_names)


def budget_tool_definitions(tools: list[dict[str, Any]], profile: ContextProfile) -> ToolSchemaBudget:
    budget = max(1, profile.tool_schema_budget_tokens)
    enabled_tools = list(tools)
    core = [tool for tool in enabled_tools if tool_name(tool) in CORE_TOOL_NAMES]
    non_core = [tool for tool in enabled_tools if tool_name(tool) not in CORE_TOOL_NAMES]
    ordered = [*core, *non_core]

    included: list[dict[str, Any]] = []
    omitted: list[str] = []
    over_budget = False
    included_ids: set[int] = set()
    for tool in ordered:
        name = tool_name(tool)
        is_core = name in CORE_TOOL_NAMES
        candidate_tokens = estimate_tokens([*included, tool])
        if not is_core and candidate_tokens > budget:
            if name:
                omitted.append(name)
            continue
        if is_core and candidate_tokens > budget:
            over_budget = True
        included.append(tool)
        included_ids.add(id(tool))

    for tool in enabled_tools:
        if id(tool) not in included_ids:
            name = tool_name(tool)
            if name and name not in omitted:
                omitted.append(name)

    return ToolSchemaBudget(
        definitions=included,
        included_names=tuple(name for name in (tool_name(tool) for tool in included) if name),
        omitted_names=tuple(omitted),
        budget_tokens=budget,
        estimated_tokens=estimate_tokens(included),
        over_budget=over_budget or estimate_tokens(included) > budget,
    )


def tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return name if isinstance(name, str) else ""
