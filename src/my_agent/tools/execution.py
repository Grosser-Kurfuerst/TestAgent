from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from my_agent.schema import ToolResult


@dataclass(frozen=True)
class ToolInvocation:
    id: str
    name: str
    arguments_json: str
    parsed_arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arguments(cls, name: str, arguments: dict[str, Any], invocation_id: str | None = None) -> "ToolInvocation":
        return cls(
            id=invocation_id or f"tool_{uuid4().hex}",
            name=name,
            arguments_json=json.dumps(arguments, ensure_ascii=False),
            parsed_arguments=dict(arguments),
        )

    def arguments(self) -> dict[str, Any]:
        if self.parsed_arguments:
            return dict(self.parsed_arguments)
        try:
            payload = json.loads(self.arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool arguments JSON is invalid: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return dict(payload)


@dataclass(frozen=True)
class ToolExecutionResult:
    id: str
    name: str
    ok: bool
    content: str
    elapsed_ms: int = 0
    error_code: str = ""
    retryable: bool = False
    blocked: bool = False
    timed_out: bool = False

    @classmethod
    def from_tool_result(
        cls,
        invocation: ToolInvocation,
        result: ToolResult,
        elapsed_ms: int = 0,
        error_code: str = "",
        retryable: bool = False,
        timed_out: bool = False,
    ) -> "ToolExecutionResult":
        code = error_code
        timed_out = timed_out or result.reason == "timeout"
        if not code and timed_out:
            code = "tool_timeout"
        elif not code and result.reason == "cancelled":
            code = "cancelled"
        elif not code and result.blocked:
            code = "blocked"
        elif not code and not result.ok:
            code = "tool_failed"
        return cls(
            id=invocation.id,
            name=invocation.name,
            ok=result.ok,
            content=result.output,
            elapsed_ms=elapsed_ms,
            error_code=code,
            retryable=retryable,
            blocked=result.blocked,
            timed_out=timed_out,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "ok": self.ok,
            "content": self.content,
            "elapsed_ms": self.elapsed_ms,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "blocked": self.blocked,
            "timed_out": self.timed_out,
        }
