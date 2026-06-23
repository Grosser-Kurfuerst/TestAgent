from __future__ import annotations

from typing import Protocol

from my_agent.hitl.types import PolicyDecision, RiskLevel
from my_agent.tools.hooks import HookViolation, validate_tool_call_preflight
from my_agent.tools.registry import RegisteredTool
from my_agent.tools.spec import ToolContext, ToolRisk


class ApprovalPolicy(Protocol):
    def evaluate(
        self,
        registered_tool: RegisteredTool,
        arguments: dict[str, object],
        context: ToolContext,
    ) -> PolicyDecision:
        ...


class RiskJudge(Protocol):
    def judge(
        self,
        registered_tool: RegisteredTool,
        arguments: dict[str, object],
        static: PolicyDecision,
    ) -> PolicyDecision:
        ...


class NoopRiskJudge:
    def judge(
        self,
        registered_tool: RegisteredTool,
        arguments: dict[str, object],
        static: PolicyDecision,
    ) -> PolicyDecision:
        return static


class StaticApprovalPolicy:
    def __init__(self, *, medium_risk_mode: str = "ask") -> None:
        self.medium_risk_mode = medium_risk_mode

    def evaluate(
        self,
        registered_tool: RegisteredTool,
        arguments: dict[str, object],
        context: ToolContext,
    ) -> PolicyDecision:
        try:
            if registered_tool.preflight is not None:
                registered_tool.preflight(dict(arguments), context)
            else:
                validate_tool_call_preflight(
                    tool_name=registered_tool.spec.name,
                    arguments=arguments,
                    repo_root=context.repo_root,
                )
        except (HookViolation, ValueError) as exc:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk_level=RiskLevel.HIGH,
                reason=str(exc),
                description="Static policy denied the tool call before approval.",
            )

        risk = registered_tool.spec.risk
        if risk == ToolRisk.READ:
            return PolicyDecision(
                allowed=True,
                requires_approval=False,
                risk_level=RiskLevel.SAFE,
                description="Read-only tool does not require approval.",
            )
        if risk == ToolRisk.EXECUTE:
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                risk_level=RiskLevel.HIGH,
                reason="Tool risk EXECUTE requires approval.",
                description="Command execution requires human approval.",
            )
        if self.medium_risk_mode == "deny":
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk_level=RiskLevel.MEDIUM,
                reason=f"Tool risk {risk.value.upper()} denied by medium risk policy.",
                description="Medium risk operation denied by policy.",
            )
        if self.medium_risk_mode == "allow":
            return PolicyDecision(
                allowed=True,
                requires_approval=False,
                risk_level=RiskLevel.MEDIUM,
                description="Medium risk operation allowed by policy.",
            )
        return PolicyDecision(
            allowed=True,
            requires_approval=True,
            risk_level=RiskLevel.MEDIUM,
            reason=f"Tool risk {risk.value.upper()} requires approval.",
            description="Side-effecting tool requires human approval.",
        )
