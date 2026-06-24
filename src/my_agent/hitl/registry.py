from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from my_agent.hitl.audit import AuditEntry, AuditLog, utc_timestamp
from my_agent.hitl.display import summarize_arguments
from my_agent.hitl.handler import HitlHandler
from my_agent.hitl.policy import ApprovalPolicy, StaticApprovalPolicy
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
from my_agent.tools.execution import ToolExecutionResult, ToolInvocation
from my_agent.tools.registry import RegisteredTool, ToolRegistry
from my_agent.tools.spec import ToolContext

ApprovalObserver = Callable[[ApprovalEvent], None]


@dataclass(frozen=True)
class _HitlCallContext:
    invocation: ToolInvocation
    tool: RegisteredTool
    arguments: dict[str, object]
    request_id: str
    started: float


class HitlToolRegistry(ToolRegistry):
    def __init__(
        self,
        *,
        context: ToolContext,
        handler: HitlHandler,
        policy: ApprovalPolicy | None = None,
        audit_log: AuditLog | None = None,
        observer: ApprovalObserver | None = None,
        run_id: str = "",
    ) -> None:
        super().__init__(context=context)
        self.handler = handler
        self.policy = policy or StaticApprovalPolicy()
        self.audit_log = audit_log or AuditLog.from_config(context.config)
        self.observer = observer
        self.run_id = run_id or context.run_id

    def _execute_one(self, invocation: ToolInvocation) -> ToolExecutionResult:
        prepared = self._prepare_call(invocation)
        if isinstance(prepared, ToolExecutionResult):
            return prepared
        ctx = prepared

        preflight_denied = self._deny_if_preflight_failed(ctx)
        if preflight_denied is not None:
            return preflight_denied

        decision = self.policy.evaluate(ctx.tool, ctx.arguments, self.context)
        if not decision.allowed:
            return self._handle_policy_denied(ctx, decision)

        if not self.handler.is_enabled():
            return self._execute_and_audit(
                ctx,
                policy=decision,
                approval_decision="disabled",
                reason=decision.reason,
            )

        if not decision.requires_approval:
            return self._execute_and_audit(
                ctx,
                policy=decision,
                approval_decision="none",
                reason=decision.reason,
            )

        if self.handler.is_approved_all(scope=ApprovalScope.TOOL, key=ctx.tool.spec.name):
            return self._execute_and_audit(
                ctx,
                policy=decision,
                approval_decision=ApprovalDecision.APPROVED_ALL.value,
                reason="approved_all cache",
                audit_safe=True,
            )

        approval = self._request_approval(ctx, decision)
        return self._handle_approval(ctx, decision, approval)

    def _prepare_call(self, invocation: ToolInvocation) -> _HitlCallContext | ToolExecutionResult:
        resolved = self._resolve_tool(invocation)
        if isinstance(resolved, ToolExecutionResult):
            return resolved
        tool = resolved

        parsed = self._parse_arguments(invocation)
        if isinstance(parsed, ToolExecutionResult):
            return parsed
        arguments = parsed

        schema_result = self._validate_schema(invocation, tool, arguments)
        if schema_result is not None:
            return schema_result

        return _HitlCallContext(
            invocation=invocation,
            tool=tool,
            arguments=arguments,
            request_id=make_approval_request_id(run_id=self.run_id, tool_call_id=invocation.id),
            started=time.monotonic(),
        )

    def _deny_if_preflight_failed(
        self,
        ctx: _HitlCallContext,
        *,
        invocation: ToolInvocation | None = None,
        arguments: dict[str, object] | None = None,
        approval_decision: str = "none",
    ) -> ToolExecutionResult | None:
        effective_invocation = ctx.invocation if invocation is None else invocation
        effective_arguments = ctx.arguments if arguments is None else arguments
        preflight_result = self._preflight_policy(effective_invocation, ctx.tool, effective_arguments)
        if preflight_result is not None:
            reason = _policy_reason_from_result(preflight_result)
            return self._audit_and_return(
                ctx,
                policy=PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    risk_level=RiskLevel.HIGH,
                    reason=reason,
                    description="Static policy denied the tool call before approval.",
                ),
                approval_decision=approval_decision,
                outcome="policy_denied",
                reason=reason,
                result=preflight_result,
                invocation=effective_invocation,
                arguments=effective_arguments,
            )
        return None

    def _handle_policy_denied(
        self,
        ctx: _HitlCallContext,
        decision: PolicyDecision,
    ) -> ToolExecutionResult:
        result = _policy_denied_result(ctx.invocation, decision)
        return self._audit_and_return(
            ctx,
            policy=decision,
            approval_decision="none",
            outcome="policy_denied",
            reason=decision.reason,
            result=result,
        )

    def _request_approval(self, ctx: _HitlCallContext, decision: PolicyDecision) -> ApprovalResult:
        request = ApprovalRequest(
            request_id=ctx.request_id,
            run_id=self.run_id,
            tool_call_id=ctx.invocation.id,
            tool_name=ctx.tool.spec.name,
            arguments_json=json.dumps(ctx.arguments, ensure_ascii=False),
            risk_level=decision.risk_level,
            risk_description=decision.description or decision.reason,
            server_name="",
        )
        self._emit(
            "approval.requested",
            {
                "id": request.request_id,
                "run_id": request.run_id,
                "tool_call_id": request.tool_call_id,
                "tool_name": request.tool_name,
                "risk_level": request.risk_level.value,
                "reason": decision.reason,
                "arguments_summary": summarize_arguments(ctx.arguments),
            },
        )
        approval = self.handler.request_approval(request)
        self._emit(
            "approval.completed",
            {
                "id": request.request_id,
                "run_id": request.run_id,
                "tool_call_id": request.tool_call_id,
                "tool_name": request.tool_name,
                "decision": approval.decision.value,
                "scope": approval.scope.value,
                "modified": approval.decision == ApprovalDecision.MODIFIED,
                "elapsed_ms": _elapsed_ms(ctx.started),
                "reason": approval.reason,
            },
        )
        return approval

    def _handle_approval(
        self,
        ctx: _HitlCallContext,
        decision: PolicyDecision,
        approval: ApprovalResult,
    ) -> ToolExecutionResult:
        if approval.decision == ApprovalDecision.REJECTED:
            return self._handle_rejected(ctx, decision, approval)
        if approval.decision == ApprovalDecision.SKIPPED:
            return self._handle_skipped(ctx, decision, approval)
        if approval.decision == ApprovalDecision.MODIFIED:
            return self._handle_modified(ctx, decision, approval)
        return self._execute_and_audit(
            ctx,
            policy=decision,
            approval_decision=approval.decision.value,
            reason=approval.reason,
            audit_safe=True,
        )

    def _handle_rejected(
        self,
        ctx: _HitlCallContext,
        decision: PolicyDecision,
        approval: ApprovalResult,
    ) -> ToolExecutionResult:
        result = ToolExecutionResult(
            id=ctx.invocation.id,
            name=ctx.invocation.name,
            ok=False,
            content=f"[HITL] Operation rejected: {approval.reason or 'No reason provided.'}\nChoose a safer alternative or explain the blocker.",
            error_code="approval_rejected",
            retryable=True,
            blocked=True,
        )
        return self._audit_and_return(
            ctx,
            policy=decision,
            approval_decision=approval.decision.value,
            outcome="approval_rejected",
            reason=approval.reason,
            result=result,
        )

    def _handle_skipped(
        self,
        ctx: _HitlCallContext,
        decision: PolicyDecision,
        approval: ApprovalResult,
    ) -> ToolExecutionResult:
        result = ToolExecutionResult(
            id=ctx.invocation.id,
            name=ctx.invocation.name,
            ok=False,
            content="[HITL] Operation skipped by user.\nDo not retry the same tool call unless the user explicitly asks.",
            error_code="approval_skipped",
            retryable=False,
            blocked=True,
        )
        return self._audit_and_return(
            ctx,
            policy=decision,
            approval_decision=approval.decision.value,
            outcome="approval_skipped",
            reason=approval.reason,
            result=result,
        )

    def _handle_modified(
        self,
        ctx: _HitlCallContext,
        decision: PolicyDecision,
        approval: ApprovalResult,
    ) -> ToolExecutionResult:
        modified = ToolInvocation(
            id=ctx.invocation.id,
            name=ctx.invocation.name,
            arguments_json=approval.effective_arguments(ctx.invocation.arguments_json),
        )
        modified_parsed = self._parse_arguments(modified)
        if isinstance(modified_parsed, ToolExecutionResult):
            return self._audit_and_return(
                ctx,
                policy=decision,
                approval_decision=approval.decision.value,
                outcome=modified_parsed.error_code or "invalid_arguments",
                reason=modified_parsed.content,
                result=modified_parsed,
                invocation=modified,
            )
        modified_schema = self._validate_schema(modified, ctx.tool, modified_parsed)
        if modified_schema is not None:
            return self._audit_and_return(
                ctx,
                policy=decision,
                approval_decision=approval.decision.value,
                outcome=modified_schema.error_code or "invalid_arguments_schema",
                reason=modified_schema.content,
                result=modified_schema,
                invocation=modified,
                arguments=modified_parsed,
            )
        modified_preflight_denied = self._deny_if_preflight_failed(
            ctx,
            invocation=modified,
            arguments=modified_parsed,
            approval_decision=approval.decision.value,
        )
        if modified_preflight_denied is not None:
            return modified_preflight_denied
        modified_policy = self.policy.evaluate(ctx.tool, modified_parsed, self.context)
        if not modified_policy.allowed:
            result = _policy_denied_result(modified, modified_policy)
            return self._audit_and_return(
                ctx,
                policy=modified_policy,
                approval_decision=approval.decision.value,
                outcome="policy_denied",
                reason=modified_policy.reason,
                result=result,
                invocation=modified,
                arguments=modified_parsed,
            )
        return self._execute_and_audit(
            ctx,
            policy=modified_policy,
            approval_decision=approval.decision.value,
            reason=approval.reason,
            invocation=modified,
            arguments=modified_parsed,
            audit_safe=True,
        )

    def _execute_and_audit(
        self,
        ctx: _HitlCallContext,
        *,
        policy: PolicyDecision,
        approval_decision: str,
        reason: str,
        invocation: ToolInvocation | None = None,
        arguments: dict[str, object] | None = None,
        audit_safe: bool = False,
    ) -> ToolExecutionResult:
        effective_invocation = ctx.invocation if invocation is None else invocation
        effective_arguments = ctx.arguments if arguments is None else arguments
        result = self._execute_registered(effective_invocation, ctx.tool, effective_arguments)
        if audit_safe or policy.risk_level != RiskLevel.SAFE:
            self._record_audit(
                request_id=ctx.request_id,
                invocation=effective_invocation,
                tool=ctx.tool,
                arguments=effective_arguments,
                policy=policy,
                approval_decision=approval_decision,
                outcome=_outcome_from_result(result),
                reason=reason,
                started=ctx.started,
            )
        return result

    def _audit_and_return(
        self,
        ctx: _HitlCallContext,
        *,
        policy: PolicyDecision,
        approval_decision: str,
        outcome: str,
        reason: str,
        result: ToolExecutionResult,
        invocation: ToolInvocation | None = None,
        arguments: dict[str, object] | None = None,
    ) -> ToolExecutionResult:
        self._record_audit(
            request_id=ctx.request_id,
            invocation=ctx.invocation if invocation is None else invocation,
            tool=ctx.tool,
            arguments=ctx.arguments if arguments is None else arguments,
            policy=policy,
            approval_decision=approval_decision,
            outcome=outcome,
            reason=reason,
            started=ctx.started,
        )
        return result

    def _record_audit(
        self,
        *,
        request_id: str,
        invocation: ToolInvocation,
        tool: RegisteredTool,
        arguments: dict[str, object],
        policy: PolicyDecision,
        approval_decision: str,
        outcome: str,
        reason: str,
        started: float,
    ) -> None:
        result = self.audit_log.record(
            AuditEntry(
                timestamp=utc_timestamp(),
                request_id=request_id,
                run_id=self.run_id,
                tool_call_id=invocation.id,
                tool_name=tool.spec.name,
                arguments_summary=summarize_arguments(arguments),
                risk_level=policy.risk_level.value,
                policy_decision="allow" if policy.allowed else "deny",
                approval_decision=approval_decision,
                approver="hitl" if approval_decision and approval_decision != "none" else "policy",
                outcome=outcome,
                reason=reason,
                elapsed_ms=_elapsed_ms(started),
            )
        )
        if not result.ok:
            self._emit(
                "approval.audit_failed",
                {
                    "id": request_id,
                    "run_id": self.run_id,
                    "tool_call_id": invocation.id,
                    "tool_name": tool.spec.name,
                    "error": result.error,
                },
            )

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        if self.observer is not None:
            self.observer(ApprovalEvent(event=event, payload=payload))


def _policy_denied_result(invocation: ToolInvocation, decision: PolicyDecision) -> ToolExecutionResult:
    return ToolExecutionResult(
        id=invocation.id,
        name=invocation.name,
        ok=False,
        content=f"[POLICY] Operation denied before approval: {decision.reason}",
        error_code="policy_denied",
        retryable=False,
        blocked=True,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _outcome_from_result(result: ToolExecutionResult) -> str:
    return "tool_ok" if result.ok else result.error_code or "tool_failed"


def _policy_reason_from_result(result: ToolExecutionResult) -> str:
    prefix = "[POLICY] Operation denied before approval: "
    if result.content.startswith(prefix):
        return result.content[len(prefix) :]
    return result.content
