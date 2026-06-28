from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_agent.mcp.manager import McpServerManager, McpServerManagerPool
from my_agent.schema import ToolResult
from my_agent.tools.spec import ToolContext, ToolRegistration, ToolRisk, ToolSource, ToolSpec


@dataclass(frozen=True)
class McpToolSource(ToolSource):
    repo_root: Path
    config: Any | None = None
    manager: McpServerManager | None = None
    name: str = "mcp"

    def load(self, context: ToolContext) -> list[ToolRegistration]:
        if self.config is None or not bool(getattr(self.config, "mcp_enabled", True)):
            return []
        if self.manager is not None:
            manager = self.manager
            manager.start_all(max_wait_seconds=_startup_wait_seconds(self.config))
        else:
            manager = McpServerManagerPool.get(self.repo_root, self.config)
        registrations: list[ToolRegistration] = []
        for descriptor in manager.tool_descriptors():
            registrations.append(self._registration_for(manager, descriptor))
        return registrations

    def _registration_for(self, manager: McpServerManager, descriptor) -> ToolRegistration:
        tool_name = descriptor.namespaced_name

        def handler(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
            if context.cancellation_token is not None:
                context.cancellation_token.raise_if_cancelled()
            return manager.call_tool(tool_name, arguments)

        def preflight(arguments: dict[str, Any], context: ToolContext) -> None:
            return None

        return ToolRegistration(
            spec=ToolSpec(
                name=tool_name,
                description=(
                    f"{descriptor.description}\n\n"
                    f"MCP server: {descriptor.server_name}. Original MCP tool: {descriptor.name}. "
                    "This is an external MCP tool and may access files, network, or third-party services depending on the server."
                ),
                parameters=descriptor.input_schema,
                risk=ToolRisk.EXTERNAL,
                source=f"mcp:{descriptor.server_name}",
                timeout_seconds=int(getattr(self.config, "mcp_call_timeout_seconds", 60) or 60),
            ),
            handler=handler,
            preflight=preflight,
            parallel_side_effect_safe=False,
            cancellation_safe=False,
        )


def _startup_wait_seconds(config: Any | None) -> int:
    value = getattr(config, "mcp_startup_wait_seconds", 8)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 8
