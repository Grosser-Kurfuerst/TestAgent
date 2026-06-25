from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from my_agent.tools.execution import ToolInvocation
from my_agent.tools.spec import ToolContext, ToolRisk

if TYPE_CHECKING:
    from my_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolBatchGroup:
    invocations: list[ToolInvocation]
    parallel: bool
    side_effect_free: bool
    reason: str


def build_tool_batch_groups(
    invocations: list[ToolInvocation],
    *,
    registry: "ToolRegistry",
    parsed_arguments: dict[str, dict[str, object]],
    context: ToolContext,
) -> list[ToolBatchGroup]:
    groups: list[ToolBatchGroup] = []
    current_reads: list[ToolInvocation] = []
    current_writes: list[ToolInvocation] = []
    current_write_resources: set[str] = set()

    def flush_reads() -> None:
        nonlocal current_reads
        if current_reads:
            groups.append(
                ToolBatchGroup(
                    invocations=current_reads,
                    parallel=len(current_reads) > 1,
                    side_effect_free=True,
                    reason="read_tools",
                )
            )
            current_reads = []

    def flush_writes() -> None:
        nonlocal current_writes, current_write_resources
        if current_writes:
            groups.append(
                ToolBatchGroup(
                    invocations=current_writes,
                    parallel=len(current_writes) > 1,
                    side_effect_free=False,
                    reason="independent_write_tools",
                )
            )
            current_writes = []
            current_write_resources = set()

    for invocation in invocations:
        tool = registry.get_registered(invocation.name)
        arguments = parsed_arguments.get(invocation.id)
        if tool is None or arguments is None:
            flush_reads()
            flush_writes()
            groups.append(ToolBatchGroup([invocation], parallel=False, side_effect_free=False, reason="unresolved"))
            continue

        if tool.spec.risk == ToolRisk.READ and _read_tool_can_parallel(tool):
            flush_writes()
            current_reads.append(invocation)
            continue

        if tool.spec.risk == ToolRisk.WRITE and tool.parallel_side_effect_safe and tool.resource_resolver is not None:
            resources = _tool_resources(tool.resource_resolver, arguments, context)
            if resources:
                flush_reads()
                if current_write_resources.intersection(resources):
                    flush_writes()
                current_writes.append(invocation)
                current_write_resources.update(resources)
                continue

        flush_reads()
        flush_writes()
        groups.append(ToolBatchGroup([invocation], parallel=False, side_effect_free=False, reason="side_effect_barrier"))

    flush_reads()
    flush_writes()
    return groups


def _tool_resources(resolver: object, arguments: dict[str, object], context: ToolContext) -> set[str]:
    try:
        resources = resolver(arguments, context)  # type: ignore[misc]
    except Exception:
        return set()
    return {str(resource) for resource in resources if str(resource)}


def _read_tool_can_parallel(tool: object) -> bool:
    if not bool(getattr(tool, "cancellation_safe", False)):
        return False
    spec = getattr(tool, "spec", None)
    source = getattr(spec, "source", "") if spec is not None else ""
    return source == "builtin" or getattr(tool, "resource_resolver", None) is not None
