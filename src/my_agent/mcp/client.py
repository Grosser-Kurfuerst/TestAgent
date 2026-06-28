from __future__ import annotations

from typing import Any

from my_agent.mcp.jsonrpc import JsonRpcClient
from my_agent.mcp.protocol import (
    McpToolDescriptor,
    call_tool_params,
    initialize_params,
    parse_call_tool_result,
    parse_tool_descriptors,
)
from my_agent.mcp.transport import McpTransport
from my_agent.schema import ToolResult


class McpClient:
    def __init__(
        self,
        server_name: str,
        transport: McpTransport,
        *,
        initialize_timeout_seconds: int = 60,
        call_timeout_seconds: int = 60,
    ) -> None:
        self.server_name = server_name
        self.transport = transport
        self.rpc = JsonRpcClient(transport)
        self.initialize_timeout_seconds = initialize_timeout_seconds
        self.call_timeout_seconds = call_timeout_seconds
        self.server_capabilities: dict[str, Any] = {}

    def initialize(self) -> None:
        result = self.rpc.request(
            "initialize",
            initialize_params(),
            timeout_seconds=self.initialize_timeout_seconds,
        )
        if isinstance(result, dict) and isinstance(result.get("capabilities"), dict):
            self.server_capabilities = dict(result["capabilities"])
        self.rpc.notify("notifications/initialized", {})

    def list_tools(self) -> list[McpToolDescriptor]:
        result = self.rpc.request("tools/list", {}, timeout_seconds=30)
        return parse_tool_descriptors(self.server_name, result)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        result = self.rpc.request(
            "tools/call",
            call_tool_params(tool_name, arguments),
            timeout_seconds=self.call_timeout_seconds,
        )
        return parse_call_tool_result(result)

    def on_notification(self, callback) -> None:
        self.rpc.on_notification(callback)

    def stderr_lines(self) -> list[str]:
        return self.transport.stderr_lines()

    def process_id(self) -> int | None:
        return self.transport.process_id()

    def transport_name(self) -> str:
        return self.transport.transport_name()

    def close(self) -> None:
        self.rpc.close()
