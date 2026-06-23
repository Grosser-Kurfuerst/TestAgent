from __future__ import annotations

from my_agent.hitl.audit import AuditEntry, AuditLog, AuditRecordResult
from my_agent.hitl.handler import HitlHandler, NonInteractiveHitlHandler, SwitchableHitlHandler, TerminalHitlHandler
from my_agent.hitl.policy import ApprovalPolicy, NoopRiskJudge, RiskJudge, StaticApprovalPolicy
from my_agent.hitl.registry import HitlToolRegistry
from my_agent.hitl.types import (
    ApprovalDecision,
    ApprovalEvent,
    ApprovalRequest,
    ApprovalResult,
    ApprovalScope,
    PolicyDecision,
    RiskLevel,
    make_approval_request_id,
)

__all__ = [
    "ApprovalDecision",
    "ApprovalEvent",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalResult",
    "ApprovalScope",
    "AuditEntry",
    "AuditLog",
    "AuditRecordResult",
    "HitlHandler",
    "HitlToolRegistry",
    "NoopRiskJudge",
    "NonInteractiveHitlHandler",
    "PolicyDecision",
    "RiskJudge",
    "RiskLevel",
    "StaticApprovalPolicy",
    "SwitchableHitlHandler",
    "TerminalHitlHandler",
    "make_approval_request_id",
]
