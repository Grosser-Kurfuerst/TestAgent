from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from my_agent.mcp.client import McpClient
from my_agent.mcp.config import McpServerConfig
from my_agent.mcp.protocol import McpToolDescriptor


class McpServerStatus(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class McpServer:
    name: str
    config: McpServerConfig
    status: McpServerStatus = McpServerStatus.DISABLED
    client: McpClient | None = None
    tools: list[McpToolDescriptor] = field(default_factory=list)
    error: str = ""
    started_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.config.disabled:
            self.status = McpServerStatus.DISABLED

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.client = None
        self.tools = []

    def logs(self) -> list[str]:
        if self.client is None:
            return []
        return self.client.stderr_lines()

    def process_id(self) -> int | None:
        if self.client is None:
            return None
        return self.client.process_id()

    def transport_name(self) -> str:
        if self.client is not None:
            return self.client.transport_name()
        return self.config.transport_name
