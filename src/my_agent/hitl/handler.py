from __future__ import annotations

import json
import sys
import threading
from typing import Callable, TextIO, Protocol

from my_agent.hitl.display import format_approval_box
from my_agent.hitl.types import ApprovalDecision, ApprovalRequest, ApprovalResult, ApprovalScope


class HitlHandler(Protocol):
    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        ...

    def is_enabled(self) -> bool:
        ...

    def set_enabled(self, enabled: bool) -> None:
        ...

    def clear_approved_all(self) -> None:
        ...

    def is_approved_all(self, *, scope: ApprovalScope, key: str) -> bool:
        ...


class TerminalHitlHandler:
    def __init__(
        self,
        *,
        enabled: bool = False,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        before_prompt: Callable[[], None] | None = None,
        require_tty: bool = True,
    ) -> None:
        self._enabled = enabled
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._before_prompt = before_prompt
        self._require_tty = require_tty
        self._lock = threading.RLock()
        self._approved_all: set[tuple[ApprovalScope, str]] = set()

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def clear_approved_all(self) -> None:
        with self._lock:
            self._approved_all.clear()

    def is_approved_all(self, *, scope: ApprovalScope, key: str) -> bool:
        return (scope, key) in self._approved_all

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        with self._lock:
            if not self._enabled:
                return ApprovalResult(decision=ApprovalDecision.APPROVED)
            if self._require_tty and not _is_interactive(self._stdin):
                return ApprovalResult(
                    decision=ApprovalDecision.REJECTED,
                    reason="HITL approval required but stdin is not interactive.",
                )
            if not request.force_per_call and self.is_approved_all(scope=ApprovalScope.TOOL, key=request.tool_name):
                return ApprovalResult(decision=ApprovalDecision.APPROVED_ALL, scope=ApprovalScope.TOOL)
            if self._before_prompt is not None:
                self._before_prompt()
            self._stdout.write("\n" + format_approval_box(request) + "\n")
            self._stdout.flush()
            return self._prompt_until_decision(request)

    def _prompt_until_decision(self, request: ApprovalRequest) -> ApprovalResult:
        for _ in range(5):
            self._stdout.write("Choose [Enter/y] approve, [a] approve all, [m] modify, [n] reject, [s] skip: ")
            self._stdout.flush()
            try:
                raw = self._stdin.readline()
            except Exception as exc:  # noqa: BLE001 - fail safe at terminal boundary
                return ApprovalResult(decision=ApprovalDecision.REJECTED, reason=f"读取输入失败: {exc}")
            if raw == "":
                return ApprovalResult(decision=ApprovalDecision.REJECTED, reason="输入流已关闭")
            choice = raw.strip().lower()
            if choice in {"", "y"}:
                return ApprovalResult(decision=ApprovalDecision.APPROVED)
            if choice == "a":
                if request.force_per_call:
                    self._stdout.write("This request requires per-call approval.\n")
                    continue
                self._approved_all.add((ApprovalScope.TOOL, request.tool_name))
                return ApprovalResult(decision=ApprovalDecision.APPROVED_ALL, scope=ApprovalScope.TOOL)
            if choice == "m":
                modified = self._prompt_modified_arguments()
                if modified is not None:
                    return modified
                continue
            if choice == "n":
                reason = self._read_optional_line("Reason: ")
                return ApprovalResult(decision=ApprovalDecision.REJECTED, reason=reason.strip())
            if choice == "s":
                return ApprovalResult(decision=ApprovalDecision.SKIPPED)
            self._stdout.write(f"Unrecognized option: {raw.strip()}\n")
        return ApprovalResult(decision=ApprovalDecision.REJECTED, reason="连续多次无效输入")

    def _prompt_modified_arguments(self) -> ApprovalResult | None:
        raw = self._read_line("Modified arguments JSON: ")
        if raw is None:
            return ApprovalResult(decision=ApprovalDecision.REJECTED, reason="输入流已关闭")
        if not raw.strip():
            self._stdout.write("Modified arguments cannot be empty.\n")
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._stdout.write(f"Invalid JSON: {exc}\n")
            return None
        if not isinstance(payload, dict):
            self._stdout.write("Modified arguments must be a JSON object.\n")
            return None
        return ApprovalResult(
            decision=ApprovalDecision.MODIFIED,
            modified_arguments_json=json.dumps(payload, ensure_ascii=False),
        )

    def _read_optional_line(self, prompt: str) -> str:
        line = self._read_line(prompt)
        return "" if line is None else line

    def _read_line(self, prompt: str) -> str | None:
        self._stdout.write(prompt)
        self._stdout.flush()
        try:
            line = self._stdin.readline()
        except Exception:  # noqa: BLE001
            return None
        return None if line == "" else line


class NonInteractiveHitlHandler:
    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._approved_all: set[tuple[ApprovalScope, str]] = set()

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def clear_approved_all(self) -> None:
        self._approved_all.clear()

    def is_approved_all(self, *, scope: ApprovalScope, key: str) -> bool:
        return (scope, key) in self._approved_all

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        if not self._enabled:
            return ApprovalResult(decision=ApprovalDecision.APPROVED)
        return ApprovalResult(
            decision=ApprovalDecision.REJECTED,
            reason="HITL approval required but no interactive input is available.",
        )


class SwitchableHitlHandler:
    def __init__(self, delegate: HitlHandler) -> None:
        self._delegate = delegate

    def set_delegate(self, delegate: HitlHandler) -> None:
        self._delegate = delegate

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        return self._delegate.request_approval(request)

    def is_enabled(self) -> bool:
        return self._delegate.is_enabled()

    def set_enabled(self, enabled: bool) -> None:
        self._delegate.set_enabled(enabled)

    def clear_approved_all(self) -> None:
        self._delegate.clear_approved_all()

    def is_approved_all(self, *, scope: ApprovalScope, key: str) -> bool:
        return self._delegate.is_approved_all(scope=scope, key=key)


def _is_interactive(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty is not None and isatty())
