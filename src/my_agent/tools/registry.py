from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from my_agent.schema import ToolResult
from my_agent.tools.hooks import HookViolation


ToolHandler = Callable[[dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def register(self, name: str, description: str, handler: ToolHandler) -> None:
        if not name.strip():
            raise ValueError("Tool name must be non-empty.")
        if not description.strip():
            raise ValueError("Tool description must be non-empty.")
        self._tools[name] = RegisteredTool(name=name, description=description.strip(), handler=handler)

    def descriptions(self) -> str:
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output=f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            return ToolResult(ok=False, output="Tool arguments must be a JSON object.")

        try:
            result = tool.handler(dict(arguments))
        except HookViolation as exc:
            return ToolResult(ok=False, output=str(exc), blocked=True, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - tool boundary must convert runtime errors into observations
            return ToolResult(ok=False, output=f"Tool error: {type(exc).__name__}: {exc}")

        if not isinstance(result, ToolResult):
            return ToolResult(ok=False, output=f"Tool error: {name} returned an invalid result.")
        return result
