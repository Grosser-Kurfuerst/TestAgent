from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class RiskLevel(str, Enum):
    SAFE = "safe"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    APPROVED_ALL = "approved_all"
    MODIFIED = "modified"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class ApprovalScope(str, Enum):
    TOOL = "tool"
    SERVER = "server"


@dataclass(frozen=True)
class ApprovalRequest:
    request_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments_json: str
    risk_level: RiskLevel
    risk_description: str
    suggestion: str = ""
    caller_context: str = ""
    server_name: str = ""
    sensitive_notice: str = ""
    force_per_call: bool = False

    def to_display_text(self) -> str:
        parts = [
            f"Request: {self.request_id}",
            f"Tool: {self.tool_name}",
            f"Risk: {self.risk_level.value} - {self.risk_description}",
            f"Arguments: {self.arguments_json}",
        ]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        if self.caller_context:
            parts.append(f"Context: {self.caller_context}")
        if self.sensitive_notice:
            parts.append(f"Notice: {self.sensitive_notice}")
        return "\n".join(parts)


@dataclass(frozen=True)
class ApprovalResult:
    decision: ApprovalDecision
    modified_arguments_json: str | None = None
    reason: str = ""
    scope: ApprovalScope = ApprovalScope.TOOL

    @property
    def approved(self) -> bool:
        return self.decision in {
            ApprovalDecision.APPROVED,
            ApprovalDecision.APPROVED_ALL,
            ApprovalDecision.MODIFIED,
        }

    def effective_arguments(self, original: str) -> str:
        if self.decision == ApprovalDecision.MODIFIED and self.modified_arguments_json is not None:
            return self.modified_arguments_json
        return original


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    risk_level: RiskLevel
    reason: str = ""
    description: str = ""


@dataclass(frozen=True)
class ApprovalEvent:
    event: str
    payload: dict[str, Any] = field(default_factory=dict)


def make_approval_request_id(*, run_id: str, tool_call_id: str) -> str:
    if run_id and tool_call_id:
        return f"{run_id}:{tool_call_id}"
    return f"approval_{uuid4().hex[:12]}"
