from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.hitl import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResult,
    ApprovalScope,
    AuditLog,
    HitlToolRegistry,
    NonInteractiveHitlHandler,
    StaticApprovalPolicy,
)
from my_agent.schema import ToolResult
from my_agent.tools import ToolContext, ToolInvocation, ToolRegistration, ToolRisk, ToolSpec
from my_agent.tools.spec import object_schema


class RecordingHitlHandler:
    def __init__(self, *results: ApprovalResult, enabled: bool = True) -> None:
        self.results = list(results)
        self.enabled = enabled
        self.requests: list[ApprovalRequest] = []
        self.approved_all: set[tuple[ApprovalScope, str]] = set()

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        self.requests.append(request)
        result = self.results.pop(0) if self.results else ApprovalResult(ApprovalDecision.APPROVED)
        if result.decision == ApprovalDecision.APPROVED_ALL:
            key = request.tool_name if result.scope == ApprovalScope.TOOL else request.server_name
            self.approved_all.add((result.scope, key))
        return result

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def clear_approved_all(self) -> None:
        self.approved_all.clear()

    def is_approved_all(self, *, scope: ApprovalScope, key: str) -> bool:
        return (scope, key) in self.approved_all


class McpHitlTests(unittest.TestCase):
    def test_mcp_tool_requires_approval_even_when_medium_risk_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            called = 0
            events = []
            handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED))

            def real_handler(_args, _ctx):
                nonlocal called
                called += 1
                return ToolResult(ok=True, output="approved")

            registry = _registry(
                repo,
                handler=handler,
                audit_dir=repo / "audit",
                policy=StaticApprovalPolicy(medium_risk_mode="allow"),
                observer=events.append,
            )
            registry.register(_mcp_registration(real_handler))

            result = _execute(registry, {"message": "hello"})

            self.assertTrue(result.ok, result.content)
            self.assertEqual(called, 1)
            self.assertEqual(len(handler.requests), 1)
            self.assertEqual(handler.requests[0].server_name, "fake")
            requested = next(event for event in events if event.event == "approval.requested")
            completed = next(event for event in events if event.event == "approval.completed")
            self.assertEqual(requested.payload["server_name"], "fake")
            self.assertEqual(completed.payload["server_name"], "fake")
            audit_lines = _read_audit_lines(repo / "audit")
            self.assertEqual(audit_lines[0]["tool_name"], "mcp__fake__echo")
            self.assertEqual(audit_lines[0]["server_name"], "fake")
            self.assertEqual(audit_lines[0]["approval_decision"], "approved")

    def test_mcp_require_approval_false_allows_without_prompt_but_audits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.REJECTED))
            registry = _registry(
                repo,
                handler=handler,
                audit_dir=repo / "audit",
                config=SimpleNamespace(mcp_require_approval=False),
            )
            registry.register(_mcp_registration(lambda _args, _ctx: ToolResult(ok=True, output="no prompt")))

            result = _execute(registry, {"message": "hello"})

            self.assertTrue(result.ok, result.content)
            self.assertEqual(handler.requests, [])
            audit_lines = _read_audit_lines(repo / "audit")
            self.assertEqual(len(audit_lines), 1)
            self.assertEqual(audit_lines[0]["server_name"], "fake")
            self.assertEqual(audit_lines[0]["approval_decision"], "none")

    def test_non_interactive_hitl_rejects_mcp_without_calling_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            called = False

            def handler(_args, _ctx):
                nonlocal called
                called = True
                return ToolResult(ok=True, output="should not run")

            registry = _registry(
                repo,
                handler=NonInteractiveHitlHandler(enabled=True),
                audit_dir=repo / "audit",
            )
            registry.register(_mcp_registration(handler))

            result = _execute(registry, {"message": "hello"})

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "approval_rejected")
            self.assertFalse(called)
            audit_lines = _read_audit_lines(repo / "audit")
            self.assertEqual(audit_lines[0]["server_name"], "fake")
            self.assertEqual(audit_lines[0]["outcome"], "approval_rejected")

    def test_rejected_and_skipped_mcp_do_not_call_handler(self) -> None:
        for decision, error_code in (
            (ApprovalDecision.REJECTED, "approval_rejected"),
            (ApprovalDecision.SKIPPED, "approval_skipped"),
        ):
            with self.subTest(decision=decision):
                with tempfile.TemporaryDirectory() as tmp:
                    repo = Path(tmp)
                    called = False

                    def handler(_args, _ctx):
                        nonlocal called
                        called = True
                        return ToolResult(ok=True, output="should not run")

                    registry = _registry(
                        repo,
                        handler=RecordingHitlHandler(ApprovalResult(decision)),
                        audit_dir=repo / "audit",
                    )
                    registry.register(_mcp_registration(handler))

                    result = _execute(registry, {"message": "hello"})

                    self.assertFalse(result.ok)
                    self.assertEqual(result.error_code, error_code)
                    self.assertFalse(called)


def _registry(
    repo: Path,
    *,
    handler,
    audit_dir: Path,
    config: object | None = None,
    policy: StaticApprovalPolicy | None = None,
    observer=None,
) -> HitlToolRegistry:
    if config is None:
        config = SimpleNamespace(mcp_require_approval=True)
    if not hasattr(config, "hitl_audit_dir"):
        config.hitl_audit_dir = audit_dir
    return HitlToolRegistry(
        context=ToolContext(repo_root=repo, config=config, run_id="run-mcp"),
        handler=handler,
        policy=policy or StaticApprovalPolicy(),
        audit_log=AuditLog(audit_dir),
        observer=observer,
        run_id="run-mcp",
    )


def _mcp_registration(handler) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            name="mcp__fake__echo",
            description="Echo via MCP.",
            parameters=object_schema(
                {"message": {"type": "string"}},
                required=["message"],
                additional_properties=False,
            ),
            risk=ToolRisk.EXTERNAL,
            source="mcp:fake",
        ),
        handler=handler,
        preflight=lambda _args, _context: None,
    )


def _execute(registry: HitlToolRegistry, arguments: dict[str, object]):
    return registry.execute([ToolInvocation.from_arguments("mcp__fake__echo", arguments, invocation_id="call-1")])[0]


def _read_audit_lines(audit_dir: Path) -> list[dict[str, object]]:
    files = sorted(audit_dir.glob("*.jsonl"))
    lines: list[dict[str, object]] = []
    for file in files:
        lines.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
    return lines


if __name__ == "__main__":
    unittest.main()
