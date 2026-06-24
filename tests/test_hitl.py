from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.hitl import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResult,
    ApprovalScope,
    AuditLog,
    HitlToolRegistry,
    NonInteractiveHitlHandler,
    PolicyDecision,
    RiskLevel,
    StaticApprovalPolicy,
    TerminalHitlHandler,
)
from my_agent.hitl.display import display_width, format_approval_box
from my_agent.schema import ToolResult
from my_agent.tools import HookViolation, RepoTools, ToolContext, ToolInvocation, ToolRegistration, ToolRisk, ToolSpec
from my_agent.tools.registry import ToolRegistry
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
            self.approved_all.add((result.scope, request.tool_name if result.scope == ApprovalScope.TOOL else request.server_name))
        return result

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def clear_approved_all(self) -> None:
        self.approved_all.clear()

    def is_approved_all(self, *, scope: ApprovalScope, key: str) -> bool:
        return (scope, key) in self.approved_all


def _request(tool_name: str = "write_file") -> ApprovalRequest:
    return ApprovalRequest(
        request_id="run-1:call-1",
        run_id="run-1",
        tool_call_id="call-1",
        tool_name=tool_name,
        arguments_json='{"path": "中文🙂.txt", "content": "hello"}',
        risk_level=RiskLevel.MEDIUM,
        risk_description="Side-effecting tool requires approval.",
    )


def _terminal(stdin: str, stdout: io.StringIO | None = None) -> TerminalHitlHandler:
    return TerminalHitlHandler(enabled=True, stdin=io.StringIO(stdin), stdout=stdout or io.StringIO(), require_tty=False)


def _execute(executor: ToolRegistry | RepoTools, name: str, arguments: dict[str, object] | None = None):
    return executor.execute([ToolInvocation.from_arguments(name=name, arguments=arguments or {}, invocation_id="call-1")])[0]


class HitlDisplayTests(unittest.TestCase):
    def test_display_width_counts_cjk_and_emoji_as_wide(self) -> None:
        self.assertEqual(display_width("中文🙂a"), 7)

    def test_approval_box_keeps_right_border_aligned(self) -> None:
        box = format_approval_box(_request(), width=64)
        widths = {display_width(line) for line in box.splitlines()}
        self.assertEqual(len(widths), 1, box)


class HitlPolicyTests(unittest.TestCase):
    def test_static_policy_maps_risk_to_approval_need(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(repo_root=Path(tmp))
            policy = StaticApprovalPolicy()
            expectations = {
                ToolRisk.READ: (RiskLevel.SAFE, False),
                ToolRisk.WRITE: (RiskLevel.MEDIUM, True),
                ToolRisk.NETWORK: (RiskLevel.MEDIUM, True),
                ToolRisk.EXTERNAL: (RiskLevel.MEDIUM, True),
                ToolRisk.EXECUTE: (RiskLevel.HIGH, True),
            }
            for risk, expected in expectations.items():
                registry = ToolRegistry(context=context)
                registry.register(_registration("tool_" + risk.value, risk=risk))
                decision = policy.evaluate(registry.tools[0], {}, context)
                self.assertTrue(decision.allowed)
                self.assertEqual((decision.risk_level, decision.requires_approval), expected)

    def test_static_policy_denies_before_handler_for_unsafe_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            context = ToolContext(repo_root=repo)
            registry = ToolRegistry(context=context)
            called = False

            def handler(_: dict[str, object], __: ToolContext) -> ToolResult:
                nonlocal called
                called = True
                return ToolResult(ok=True, output="ran")

            registry.register(_registration("write_file", risk=ToolRisk.WRITE, handler=handler))
            result = _execute(registry, "write_file", {"path": "../outside.txt", "content": "x"})

            self.assertFalse(result.ok)
            self.assertTrue(result.blocked)
            self.assertEqual(result.error_code, "policy_denied")
            self.assertFalse(called)

    def test_static_policy_does_not_run_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(repo_root=Path(tmp))
            registry = ToolRegistry(context=context)
            called = False

            def preflight(_: dict[str, object], __: ToolContext) -> None:
                nonlocal called
                called = True
                raise HookViolation("registry owns hard preflight")

            registry.register(
                ToolRegistration(
                    spec=ToolSpec(
                        name="custom_write",
                        description="custom write",
                        parameters=object_schema({"path": {"type": "string"}}, required=["path"]),
                        risk=ToolRisk.WRITE,
                        source="test",
                    ),
                    handler=lambda _args, _context: ToolResult(ok=True, output="ok"),
                    preflight=preflight,
                )
            )
            decision = StaticApprovalPolicy().evaluate(registry.tools[0], {"path": "../outside.txt"}, context)

            self.assertFalse(called)
            self.assertTrue(decision.allowed)
            self.assertEqual(decision.risk_level, RiskLevel.MEDIUM)
            self.assertTrue(decision.requires_approval)

    def test_risk_judge_can_refine_medium_risk_but_not_high(self) -> None:
        class FakeJudge:
            def __init__(self) -> None:
                self.calls = 0

            def judge(self, registered_tool, arguments, static):  # type: ignore[no-untyped-def]
                self.calls += 1
                return PolicyDecision(
                    allowed=True,
                    requires_approval=False,
                    risk_level=static.risk_level,
                    reason="judge allow",
                    description="fake judge allowed medium risk",
                )

        with tempfile.TemporaryDirectory() as tmp:
            context = ToolContext(repo_root=Path(tmp))
            registry = ToolRegistry(context=context)
            registry.register(_registration("custom_write", risk=ToolRisk.WRITE))
            registry.register(_registration("custom_exec", risk=ToolRisk.EXECUTE))
            judge = FakeJudge()
            policy = StaticApprovalPolicy(judge=judge, judge_enabled=True)

            medium = policy.evaluate(registry.tools[0], {"path": "a.txt", "content": "x"}, context)
            high = policy.evaluate(registry.tools[1], {"path": "a.txt", "content": "x"}, context)

        self.assertTrue(medium.allowed)
        self.assertFalse(medium.requires_approval)
        self.assertEqual(medium.reason, "judge allow")
        self.assertTrue(high.allowed)
        self.assertTrue(high.requires_approval)
        self.assertEqual(judge.calls, 1)


class HitlHandlerTests(unittest.TestCase):
    def test_terminal_handler_accepts_approve_variants(self) -> None:
        self.assertEqual(
            _terminal("\n").request_approval(_request()).decision,
            ApprovalDecision.APPROVED,
        )
        self.assertEqual(
            _terminal("y\n").request_approval(_request()).decision,
            ApprovalDecision.APPROVED,
        )

    def test_terminal_handler_approve_all_caches_tool_scope(self) -> None:
        handler = _terminal("a\n")
        result = handler.request_approval(_request("write_file"))

        self.assertEqual(result.decision, ApprovalDecision.APPROVED_ALL)
        self.assertTrue(handler.is_approved_all(scope=ApprovalScope.TOOL, key="write_file"))

    def test_terminal_handler_approve_all_ignores_server_scope_in_stage6(self) -> None:
        request = ApprovalRequest(
            request_id="run-1:call-1",
            run_id="run-1",
            tool_call_id="call-1",
            tool_name="mcp__srv__write",
            arguments_json="{}",
            risk_level=RiskLevel.MEDIUM,
            risk_description="external write",
            server_name="srv",
        )
        handler = _terminal("a\n")
        result = handler.request_approval(request)

        self.assertEqual(result.decision, ApprovalDecision.APPROVED_ALL)
        self.assertEqual(result.scope, ApprovalScope.TOOL)
        self.assertTrue(handler.is_approved_all(scope=ApprovalScope.TOOL, key="mcp__srv__write"))
        self.assertFalse(handler.is_approved_all(scope=ApprovalScope.SERVER, key="srv"))

    def test_terminal_handler_modify_reject_skip_and_invalid_json_retry(self) -> None:
        modified = _terminal("m\n[]\nm\n{\"path\":\"b.txt\"}\n").request_approval(_request())
        rejected = _terminal("n\nno config edits\n").request_approval(_request())
        skipped = _terminal("s\n").request_approval(_request())

        self.assertEqual(modified.decision, ApprovalDecision.MODIFIED)
        self.assertEqual(json.loads(modified.modified_arguments_json or "{}"), {"path": "b.txt"})
        self.assertEqual(rejected.decision, ApprovalDecision.REJECTED)
        self.assertEqual(rejected.reason.strip(), "no config edits")
        self.assertEqual(skipped.decision, ApprovalDecision.SKIPPED)

    def test_terminal_handler_modify_eof_rejects_and_empty_reprompts(self) -> None:
        eof = _terminal("m\n").request_approval(_request())
        empty_then_reject = _terminal("m\n\nn\nno args\n").request_approval(_request())

        self.assertEqual(eof.decision, ApprovalDecision.REJECTED)
        self.assertEqual(empty_then_reject.decision, ApprovalDecision.REJECTED)
        self.assertEqual(empty_then_reject.reason.strip(), "no args")

    def test_terminal_handler_eof_and_invalid_input_fail_safe_reject(self) -> None:
        eof = _terminal("").request_approval(_request())
        invalid = _terminal("x\nx\nx\nx\nx\n").request_approval(_request())

        self.assertEqual(eof.decision, ApprovalDecision.REJECTED)
        self.assertEqual(invalid.decision, ApprovalDecision.REJECTED)

    def test_terminal_handler_rejects_non_tty_without_reading(self) -> None:
        result = TerminalHitlHandler(enabled=True, stdin=io.StringIO("y\n"), stdout=io.StringIO()).request_approval(_request())

        self.assertEqual(result.decision, ApprovalDecision.REJECTED)
        self.assertIn("not interactive", result.reason)

    def test_non_interactive_handler_rejects_without_blocking(self) -> None:
        result = NonInteractiveHitlHandler(enabled=True).request_approval(_request())

        self.assertEqual(result.decision, ApprovalDecision.REJECTED)
        self.assertIn("no interactive input", result.reason)

    def test_terminal_handler_serializes_concurrent_prompts(self) -> None:
        stdin = io.StringIO("y\ny\ny\n")
        stdout = io.StringIO()
        handler = TerminalHitlHandler(enabled=True, stdin=stdin, stdout=stdout, require_tty=False)
        results: list[ApprovalDecision] = []

        threads = [threading.Thread(target=lambda: results.append(handler.request_approval(_request()).decision)) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results, [ApprovalDecision.APPROVED] * 3)
        self.assertEqual(stdout.getvalue().count("HITL approval"), 3)


class HitlRegistryTests(unittest.TestCase):
    def test_hitl_disabled_matches_regular_registry_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            audit_dir = repo / "audit"
            config = SimpleNamespace(hitl_audit_dir=audit_dir, hitl_medium_risk_mode="ask")
            handler = RecordingHitlHandler(enabled=False)
            tools = RepoTools(repo, config=config, hitl_handler=handler, run_id="run-disabled")

            result = _execute(tools, "write_file", {"path": "a.txt", "content": "x"})

            self.assertTrue(result.ok, result.content)
            self.assertEqual(handler.requests, [])
            self.assertTrue((repo / "a.txt").exists())
            audit_lines = _read_audit_lines(audit_dir)
            self.assertEqual(len(audit_lines), 1)
            self.assertEqual(audit_lines[0]["run_id"], "run-disabled")
            self.assertEqual(audit_lines[0]["approval_decision"], "disabled")
            self.assertEqual(audit_lines[0]["outcome"], "tool_ok")

    def test_hitl_disabled_still_runs_registry_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            called = False

            def handler(_: dict[str, object], __: ToolContext) -> ToolResult:
                nonlocal called
                called = True
                return ToolResult(ok=True, output="should not run")

            def preflight(_: dict[str, object], __: ToolContext) -> None:
                raise HookViolation("blocked by preflight")

            hitl = HitlToolRegistry(
                context=ToolContext(repo_root=repo),
                handler=RecordingHitlHandler(enabled=False),
                audit_log=AuditLog(repo / "audit"),
            )
            regular = ToolRegistry(context=ToolContext(repo_root=repo))
            registration = ToolRegistration(
                spec=ToolSpec(
                    name="custom_write",
                    description="custom write",
                    parameters=object_schema({"path": {"type": "string"}}, required=["path"]),
                    risk=ToolRisk.WRITE,
                    source="test",
                ),
                handler=handler,
                preflight=preflight,
            )
            hitl.register(registration)
            regular.register(registration)

            hitl_result = _execute(hitl, "custom_write", {"path": "a.txt"})
            regular_result = _execute(regular, "custom_write", {"path": "a.txt"})

            self.assertEqual(hitl_result.error_code, "policy_denied")
            self.assertEqual(regular_result.error_code, "policy_denied")
            self.assertFalse(called)

    def test_hitl_registry_runs_preflight_once_before_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            preflight_calls = 0

            def handler(_: dict[str, object], __: ToolContext) -> ToolResult:
                return ToolResult(ok=True, output="ran")

            def preflight(_: dict[str, object], __: ToolContext) -> None:
                nonlocal preflight_calls
                preflight_calls += 1

            hitl = HitlToolRegistry(
                context=ToolContext(repo_root=repo),
                handler=RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED)),
                audit_log=AuditLog(repo / "audit"),
                run_id="run-preflight-once",
            )
            hitl.register(
                ToolRegistration(
                    spec=ToolSpec(
                        name="custom_write",
                        description="custom write",
                        parameters=object_schema({"path": {"type": "string"}}, required=["path"]),
                        risk=ToolRisk.WRITE,
                        source="test",
                    ),
                    handler=handler,
                    preflight=preflight,
                )
            )

            result = _execute(hitl, "custom_write", {"path": "a.txt"})

            self.assertTrue(result.ok, result.content)
            self.assertEqual(preflight_calls, 1)

    def test_read_file_does_not_trigger_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "a.txt").write_text("x", encoding="utf-8")
            handler = RecordingHitlHandler()
            tools = RepoTools(repo, hitl_handler=handler)

            result = _execute(tools, "read_file", {"path": "a.txt"})

            self.assertTrue(result.ok)
            self.assertEqual(handler.requests, [])

    def test_policy_denied_path_does_not_call_hitl_or_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            audit_dir = repo / "audit"
            config = SimpleNamespace(hitl_audit_dir=audit_dir, hitl_medium_risk_mode="ask")
            handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED))
            tools = RepoTools(repo, config=config, hitl_handler=handler, run_id="run-deny")

            result = _execute(tools, "write_file", {"path": "../outside.txt", "content": "x"})

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "policy_denied")
            self.assertEqual(handler.requests, [])
            self.assertFalse((Path(tmp) / "outside.txt").exists())
            audit_lines = _read_audit_lines(audit_dir)
            self.assertEqual(len(audit_lines), 1)
            self.assertEqual(audit_lines[0]["run_id"], "run-deny")
            self.assertEqual(audit_lines[0]["outcome"], "policy_denied")

    def test_reject_skip_approve_modified_and_approve_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            audit_dir = repo / "audit"
            config = SimpleNamespace(hitl_audit_dir=audit_dir, hitl_medium_risk_mode="ask")

            reject_handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.REJECTED, reason="not now"))
            rejected = _execute(RepoTools(repo, config=config, hitl_handler=reject_handler, run_id="run-1"), "write_file", {"path": "r.txt", "content": "x"})
            skip_handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.SKIPPED))
            skipped = _execute(RepoTools(repo, config=config, hitl_handler=skip_handler, run_id="run-1"), "write_file", {"path": "s.txt", "content": "x"})
            approve_handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED))
            approved = _execute(RepoTools(repo, config=config, hitl_handler=approve_handler, run_id="run-1"), "write_file", {"path": "a.txt", "content": "x"})
            modified_handler = RecordingHitlHandler(
                ApprovalResult(
                    ApprovalDecision.MODIFIED,
                    modified_arguments_json=json.dumps({"path": "m.txt", "content": "changed"}),
                )
            )
            modified = _execute(RepoTools(repo, config=config, hitl_handler=modified_handler, run_id="run-1"), "write_file", {"path": "wrong.txt", "content": "x"})
            approve_all_handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED_ALL))
            approve_all_tools = RepoTools(repo, config=config, hitl_handler=approve_all_handler, run_id="run-1")
            first_all = _execute(approve_all_tools, "write_file", {"path": "all1.txt", "content": "x"})
            second_all = _execute(approve_all_tools, "write_file", {"path": "all2.txt", "content": "x"})

            self.assertEqual(rejected.error_code, "approval_rejected")
            self.assertTrue(rejected.blocked)
            self.assertEqual(skipped.error_code, "approval_skipped")
            self.assertTrue(approved.ok, approved.content)
            self.assertTrue(modified.ok, modified.content)
            self.assertFalse((repo / "wrong.txt").exists())
            self.assertEqual((repo / "m.txt").read_text(encoding="utf-8"), "changed")
            self.assertTrue(first_all.ok)
            self.assertTrue(second_all.ok)
            self.assertEqual(len(approve_all_handler.requests), 1)

            audit_lines = _read_audit_lines(audit_dir)
            outcomes = {line["outcome"] for line in audit_lines}
            self.assertIn("approval_rejected", outcomes)
            self.assertIn("approval_skipped", outcomes)
            self.assertIn("tool_ok", outcomes)
            self.assertTrue(all(line["run_id"] == "run-1" for line in audit_lines))

    def test_modified_arguments_are_revalidated_by_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            handler = RecordingHitlHandler(
                ApprovalResult(
                    ApprovalDecision.MODIFIED,
                    modified_arguments_json=json.dumps({"path": "../outside.txt", "content": "x"}),
                )
            )
            result = _execute(RepoTools(repo, hitl_handler=handler, run_id="run-1"), "write_file", {"path": "safe.txt", "content": "x"})

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "policy_denied")
            self.assertFalse((Path(tmp) / "outside.txt").exists())

    def test_modified_arguments_schema_failure_writes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            audit_dir = repo / "audit"
            config = SimpleNamespace(hitl_audit_dir=audit_dir, hitl_medium_risk_mode="ask")
            handler = RecordingHitlHandler(
                ApprovalResult(
                    ApprovalDecision.MODIFIED,
                    modified_arguments_json=json.dumps({"path": "missing-content.txt"}),
                )
            )

            result = _execute(
                RepoTools(repo, config=config, hitl_handler=handler, run_id="run-modified-schema"),
                "write_file",
                {"path": "original.txt", "content": "x"},
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "invalid_arguments_schema")
            self.assertFalse((repo / "missing-content.txt").exists())
            audit_lines = _read_audit_lines(audit_dir)
            self.assertEqual(len(audit_lines), 1)
            self.assertEqual(audit_lines[0]["run_id"], "run-modified-schema")
            self.assertEqual(audit_lines[0]["approval_decision"], "modified")
            self.assertEqual(audit_lines[0]["outcome"], "invalid_arguments_schema")

    def test_modified_arguments_parse_failure_writes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            audit_dir = repo / "audit"
            config = SimpleNamespace(hitl_audit_dir=audit_dir, hitl_medium_risk_mode="ask")
            handler = RecordingHitlHandler(
                ApprovalResult(ApprovalDecision.MODIFIED, modified_arguments_json="[]")
            )

            result = _execute(
                RepoTools(repo, config=config, hitl_handler=handler, run_id="run-modified-parse"),
                "write_file",
                {"path": "original.txt", "content": "x"},
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "invalid_arguments")
            audit_lines = _read_audit_lines(audit_dir)
            self.assertEqual(len(audit_lines), 1)
            self.assertEqual(audit_lines[0]["run_id"], "run-modified-parse")
            self.assertEqual(audit_lines[0]["approval_decision"], "modified")
            self.assertEqual(audit_lines[0]["outcome"], "invalid_arguments")

    def test_medium_risk_allow_path_still_writes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            audit_dir = repo / "audit"
            config = SimpleNamespace(hitl_audit_dir=audit_dir, hitl_medium_risk_mode="allow")
            handler = RecordingHitlHandler()

            result = _execute(
                RepoTools(repo, config=config, hitl_handler=handler, run_id="run-allow"),
                "write_file",
                {"path": "allowed.txt", "content": "x"},
            )

            self.assertTrue(result.ok, result.content)
            self.assertEqual(handler.requests, [])
            audit_lines = _read_audit_lines(audit_dir)
            self.assertEqual(len(audit_lines), 1)
            self.assertEqual(audit_lines[0]["run_id"], "run-allow")
            self.assertEqual(audit_lines[0]["approval_decision"], "none")
            self.assertEqual(audit_lines[0]["outcome"], "tool_ok")

    def test_dynamic_config_tool_preflight_blocks_before_hitl_and_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            repo.mkdir()
            marker = base / "started.txt"
            outside = base / "outside.txt"
            _write_env_file(repo)
            _write_project_tools_config(
                repo,
                {
                    "version": 1,
                    "tools": [
                        {
                            "name": "write_requested_path",
                            "description": "Write requested path.",
                            "kind": "command",
                            "risk": "execute",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            },
                            "command": {
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    (
                                        "import pathlib, sys; "
                                        f"pathlib.Path({str(marker)!r}).write_text('started'); "
                                        "pathlib.Path(sys.argv[1]).write_text('x')"
                                    ),
                                    "{path}",
                                ],
                            },
                        }
                    ],
                },
            )
            config = SimpleNamespace(
                enable_project_tools=True,
                enable_project_plugins=False,
                tool_config_paths=(),
                hitl_audit_dir=repo / "audit",
                hitl_medium_risk_mode="ask",
            )
            handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED))
            tools = RepoTools(repo, config=config, hitl_handler=handler, run_id="run-1")

            result = _execute(tools, "write_requested_path", {"path": str(outside)})

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "policy_denied")
            self.assertEqual(handler.requests, [])
            self.assertFalse(marker.exists())
            self.assertFalse(outside.exists())
            audit_lines = _read_audit_lines(repo / "audit")
            self.assertEqual(len(audit_lines), 1)
            self.assertEqual(audit_lines[0]["outcome"], "policy_denied")

    def test_observer_gets_approval_events_and_audit_failures_do_not_override_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            audit_file = repo / "audit-is-file"
            audit_file.write_text("not a dir", encoding="utf-8")
            config = SimpleNamespace(hitl_audit_dir=audit_file, hitl_medium_risk_mode="ask")
            events = []
            handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED))

            result = _execute(
                RepoTools(repo, config=config, hitl_handler=handler, approval_observer=events.append, run_id="run-42"),
                "write_file",
                {"path": "a.txt", "content": "x"},
            )

            self.assertTrue(result.ok, result.content)
            event_names = [event.event for event in events]
            self.assertIn("approval.requested", event_names)
            self.assertIn("approval.completed", event_names)
            self.assertIn("approval.audit_failed", event_names)
            requested = next(event for event in events if event.event == "approval.requested")
            completed = next(event for event in events if event.event == "approval.completed")
            self.assertEqual(requested.payload["id"], completed.payload["id"])
            self.assertEqual(requested.payload["run_id"], "run-42")

    def test_policy_denied_audit_failure_emits_observer_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            audit_file = repo / "audit-is-file"
            audit_file.write_text("not a dir", encoding="utf-8")
            config = SimpleNamespace(hitl_audit_dir=audit_file, hitl_medium_risk_mode="ask")
            events = []
            handler = RecordingHitlHandler(ApprovalResult(ApprovalDecision.APPROVED))

            result = _execute(
                RepoTools(repo, config=config, hitl_handler=handler, approval_observer=events.append, run_id="run-deny"),
                "write_file",
                {"path": "../outside.txt", "content": "x"},
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, "policy_denied")
            self.assertIn("approval.audit_failed", [event.event for event in events])


def _registration(
    name: str,
    *,
    risk: ToolRisk,
    handler=None,
) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            name=name,
            description="test tool",
            parameters=object_schema(
                {"path": {"type": "string"}, "content": {"type": "string"}},
                required=["path", "content"] if name == "write_file" else None,
                additional_properties=False,
            ),
            risk=risk,
            source="test",
        ),
        handler=handler or (lambda _args, _context: ToolResult(ok=True, output="ok")),
    )


def _write_env_file(repo: Path) -> None:
    (repo / ".env").write_text("", encoding="utf-8")


def _write_project_tools_config(repo: Path, payload: dict[str, object]) -> None:
    config_dir = repo / ".agentcli"
    config_dir.mkdir()
    (config_dir / "tools.json").write_text(json.dumps(payload), encoding="utf-8")


def _read_audit_lines(audit_dir: Path) -> list[dict[str, object]]:
    files = sorted(audit_dir.glob("*.jsonl"))
    lines: list[dict[str, object]] = []
    for file in files:
        lines.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
    return lines


if __name__ == "__main__":
    unittest.main()
