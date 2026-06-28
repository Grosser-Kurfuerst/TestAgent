from __future__ import annotations

from typing import Protocol

from my_agent.hitl.types import PolicyDecision, RiskLevel
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
    def __init__(
        self,
        *,
        medium_risk_mode: str = "ask",
        judge: RiskJudge | None = None,
        judge_enabled: bool = False,
    ) -> None:
        self.medium_risk_mode = medium_risk_mode
        self.judge = judge or NoopRiskJudge()
        self.judge_enabled = judge_enabled

    def evaluate(
        self,
        registered_tool: RegisteredTool,
        arguments: dict[str, object],
        context: ToolContext,
    ) -> PolicyDecision:
        if _is_mcp_tool(registered_tool):
            if not bool(getattr(context.config, "mcp_require_approval", True)):
                return PolicyDecision(
                    allowed=True,
                    requires_approval=False,
                    risk_level=RiskLevel.MEDIUM,
                    description="MCP tool approval disabled by configuration.",
                )
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                risk_level=RiskLevel.MEDIUM,
                reason="MCP tools require approval by default.",
                description="External MCP tool may access files, network, or third-party services.",
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
            return self._maybe_judge_medium(
                registered_tool,
                arguments,
                PolicyDecision(
                    allowed=True,
                    requires_approval=False,
                    risk_level=RiskLevel.MEDIUM,
                    description="Medium risk operation allowed by policy.",
                ),
            )
        return self._maybe_judge_medium(
            registered_tool,
            arguments,
            PolicyDecision(
                allowed=True,
                requires_approval=True,
                risk_level=RiskLevel.MEDIUM,
                reason=f"Tool risk {risk.value.upper()} requires approval.",
                description="Side-effecting tool requires human approval.",
            ),
        )

    def _maybe_judge_medium(
        self,
        registered_tool: RegisteredTool,
        arguments: dict[str, object],
        static: PolicyDecision,
    ) -> PolicyDecision:
        if not self.judge_enabled:
            return static
        return self.judge.judge(registered_tool, arguments, static)


def _is_mcp_tool(registered_tool: RegisteredTool) -> bool:
    source = registered_tool.spec.source
    return registered_tool.spec.name.startswith("mcp__") or source.startswith("mcp:")
