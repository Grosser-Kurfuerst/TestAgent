from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from my_agent.schema import ToolResult


class ToolRisk(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ToolContext:
    repo_root: Path
    timeout_seconds: int = 60
    config: Any | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: ToolRisk = ToolRisk.READ
    source: str = "builtin"
    timeout_seconds: int | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ToolSpec.name must be non-empty.")
        if not self.description.strip():
            raise ValueError("ToolSpec.description must be non-empty.")
        if not isinstance(self.parameters, dict) or self.parameters.get("type") != "object":
            raise ValueError("ToolSpec.parameters must be an object JSON schema.")

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolHandler(Protocol):
    def __call__(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


@dataclass(frozen=True)
class ToolRegistration:
    spec: ToolSpec
    handler: ToolHandler


class ToolSource(Protocol):
    name: str

    def load(self, context: ToolContext) -> list[ToolRegistration]:
        ...


def object_schema(
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties or {}),
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = list(required)
    return schema
