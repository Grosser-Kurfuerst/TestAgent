from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from my_agent.mcp.schema import sanitize_schema
from my_agent.schema import ToolResult


PROTOCOL_VERSION = "2024-11-05"


@dataclass(frozen=True)
class McpToolDescriptor:
    server_name: str
    name: str
    namespaced_name: str
    description: str
    input_schema: dict[str, Any]


def namespaced_tool_name(server_name: str, tool_name: str) -> str:
    return f"mcp__{server_name}__{tool_name}"


def initialize_params() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "agentcli", "version": "0.1.0"},
    }


def parse_tool_descriptors(server_name: str, result: object) -> list[McpToolDescriptor]:
    if not isinstance(result, dict):
        return []
    tools = result.get("tools")
    if not isinstance(tools, list):
        return []
    descriptors: list[McpToolDescriptor] = []
    for raw_tool in tools:
        if not isinstance(raw_tool, dict):
            continue
        name = raw_tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        description = raw_tool.get("description")
        descriptors.append(
            McpToolDescriptor(
                server_name=server_name,
                name=name.strip(),
                namespaced_name=namespaced_tool_name(server_name, name.strip()),
                description=str(description).strip() if isinstance(description, str) and description.strip() else "MCP server provided external tool.",
                input_schema=sanitize_schema(raw_tool.get("inputSchema")),
            )
        )
    return descriptors


def call_tool_params(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": tool_name, "arguments": dict(arguments)}


def parse_call_tool_result(result: object) -> ToolResult:
    if not isinstance(result, dict):
        return ToolResult(ok=False, output="MCP tool returned an invalid result.")
    is_error = result.get("isError") is True
    content = result.get("content")
    text = _content_to_text(content)
    if is_error:
        return ToolResult(ok=False, output=f"MCP tool returned error: {text or 'without content.'}")
    return ToolResult(ok=True, output=text)


def _content_to_text(content: object) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif isinstance(item_type, str):
            parts.append(f"[MCP tool returned {item_type} content omitted]")
    return "\n".join(parts)
