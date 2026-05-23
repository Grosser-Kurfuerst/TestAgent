from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @classmethod
    def from_json(cls, text: str, allowed_tools: Iterable[str] | None = None) -> "ToolCall":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ToolCall JSON is invalid: {exc}") from exc
        return cls.from_mapping(payload, allowed_tools=allowed_tools)

    @classmethod
    def from_mapping(cls, payload: MappingLike, allowed_tools: Iterable[str] | None = None) -> "ToolCall":
        if not isinstance(payload, dict):
            raise ValueError("ToolCall must be a JSON object.")

        tool = payload.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("ToolCall.tool must be a non-empty string.")
        tool = tool.strip()

        if allowed_tools is not None and tool not in set(allowed_tools):
            raise ValueError(f"Unknown tool: {tool}")

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("ToolCall.arguments must be a JSON object.")

        reason = payload.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("ToolCall.reason must be a non-empty string.")

        return cls(tool=tool, arguments=dict(arguments), reason=reason.strip())

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": self.arguments, "reason": self.reason}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    # 安全拦截
    blocked: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "blocked": self.blocked,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ToolRecord:
    call: ToolCall
    result: ToolResult

    def to_dict(self) -> dict[str, Any]:
        return {"call": self.call.to_dict(), "result": self.result.to_dict()}


@dataclass
class AgentState:
    repo_path: Path
    task: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    test_command: str | None = None
    repo_context: str = ""
    project_rules: str = ""
    plan: str = ""
    tool_history: list[ToolRecord] = field(default_factory=list)
    steps: int = 0
    max_steps: int = 8
    trace_path: Path | None = None
    review: str = ""
    final_answer: str = ""
    done: bool = False

    @classmethod
    def initial(
        cls,
        repo_path: str | Path,
        task: str,
        test_command: str | None = None,
        max_steps: int = 8,
        run_id: str | None = None,
    ) -> "AgentState":
        if not str(task).strip():
            raise ValueError("AgentState.task must be non-empty.")
        return cls(
            repo_path=Path(repo_path),
            task=str(task).strip(),
            run_id=run_id or str(uuid4()),
            test_command=test_command,
            max_steps=max_steps,
        )

    def trace_event(self, event: str, payload: dict[str, Any]) -> "TraceEvent":
        return TraceEvent(event=event, payload=payload, run_id=self.run_id)


@dataclass(frozen=True)
class TraceEvent:
    event: str
    payload: dict[str, Any]
    run_id: str
    time: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if not self.event.strip():
            raise ValueError("TraceEvent.event must be non-empty.")
        if not self.run_id.strip():
            raise ValueError("TraceEvent.run_id must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "run_id": self.run_id,
            "event": self.event,
            "payload": self.payload,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


MappingLike = dict[str, Any]
