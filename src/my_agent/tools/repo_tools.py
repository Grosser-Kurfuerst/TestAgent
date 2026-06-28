from __future__ import annotations

import difflib
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from my_agent.cancellation import CancellationToken
from my_agent.indexer import RepoIndexer, TEXT_EXTENSIONS
from my_agent.schema import ToolResult
from my_agent.tools.execution import ToolExecutionResult, ToolInvocation
from my_agent.tools.hooks import (
    post_tool_check,
    should_skip_path,
    validate_read_path,
    validate_test_command,
    validate_write_path,
)
from my_agent.tools.builtin import BuiltinToolSource
from my_agent.tools.process import run_process
from my_agent.tools.registry import ToolRegistry
from my_agent.tools.spec import ToolContext

if TYPE_CHECKING:
    from my_agent.hitl.handler import HitlHandler
    from my_agent.hitl.types import ApprovalEvent


class RepoTools:
    def __init__(
        self,
        repo_path: str | Path,
        timeout: int = 60,
        config: Any | None = None,
        include_dynamic: bool = True,
        *,
        run_id: str = "",
        cancellation_token: CancellationToken | None = None,
        hitl_handler: HitlHandler | None = None,
        approval_observer: Callable[[ApprovalEvent], None] | None = None,
    ):
        self.repo_root = Path(repo_path).resolve()
        if not self.repo_root.exists() or not self.repo_root.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {self.repo_root}")
        self.timeout = timeout
        self.config = config
        self.context = ToolContext(
            repo_root=self.repo_root,
            timeout_seconds=timeout,
            config=config,
            run_id=run_id,
            cancellation_token=cancellation_token,
            max_parallel_tools=int(getattr(config, "max_parallel_tools", 4) or 4),
            tool_batch_timeout_seconds=int(getattr(config, "tool_batch_timeout_seconds", 60) or 60),
            tool_shutdown_grace_seconds=int(getattr(config, "tool_shutdown_grace_seconds", 2) or 2),
            max_process_output_chars=int(getattr(config, "max_process_output_chars", 8_000) or 8_000),
        )
        if hitl_handler is None:
            self.registry = ToolRegistry(context=self.context)
        else:
            from my_agent.hitl.audit import AuditLog
            from my_agent.hitl.policy import StaticApprovalPolicy
            from my_agent.hitl.registry import HitlToolRegistry

            self.registry = HitlToolRegistry(
                context=self.context,
                handler=hitl_handler,
                policy=StaticApprovalPolicy(
                    medium_risk_mode=str(getattr(config, "hitl_medium_risk_mode", "ask") or "ask"),
                    judge_enabled=bool(getattr(config, "hitl_llm_judge_enabled", False)),
                ),
                audit_log=AuditLog.from_config(config),
                observer=approval_observer,
                run_id=run_id,
            )
        self._register_defaults()
        if include_dynamic:
            self._register_dynamic_sources()

    @property
    def tool_names(self) -> list[str]:
        return self.registry.tool_names

    def tool_definitions(self) -> list[dict[str, Any]]:
        return self.registry.tool_definitions()

    def execute(self, invocations: list[ToolInvocation]) -> list[ToolExecutionResult]:
        return self.execute_tools(invocations)

    def execute_tools(self, invocations: list[ToolInvocation]) -> list[ToolExecutionResult]:
        return [self._with_post_tool_note(result) for result in self.registry.execute_tools(invocations)]

    def _with_post_tool_note(self, result: ToolExecutionResult) -> ToolExecutionResult:
        if result.blocked:
            return result
        note = post_tool_check(result.name, result.ok, result.content)
        if not note:
            return result
        return ToolExecutionResult(
            id=result.id,
            name=result.name,
            ok=result.ok,
            content=f"{result.content}\n\nHook note: {note}",
            elapsed_ms=result.elapsed_ms,
            error_code=result.error_code,
            retryable=result.retryable,
            blocked=result.blocked,
            timed_out=result.timed_out,
        )

    def _register_defaults(self) -> None:
        self.registry.load_source(BuiltinToolSource(self), self.context)

    def _register_dynamic_sources(self) -> None:
        from my_agent.mcp.source import McpToolSource
        from my_agent.tools.config_source import ConfigToolSource
        from my_agent.tools.plugin_source import PluginToolSource

        for source in ConfigToolSource.sources_for(self.repo_root, self.config):
            self.registry.load_source(source, self.context)
        self.registry.load_source(PluginToolSource(self.repo_root, self.config), self.context)
        self.registry.load_source(McpToolSource(self.repo_root, self.config), self.context)

    def _list_files(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        context = context or self.context
        base = validate_read_path(self.repo_root, arguments.get("path", "."))
        if not base.exists():
            return ToolResult(ok=False, output=f"Path does not exist: {arguments.get('path', '.')}")

        files: list[str] = []
        targets = [base] if base.is_file() else base.rglob("*")
        for path in targets:
            _raise_if_cancelled(context)
            if should_skip_path(self.repo_root, path) or not path.is_file():
                continue
            files.append(path.relative_to(self.repo_root).as_posix())
            if len(files) >= 120:
                files.append("... output truncated")
                break
        return ToolResult(ok=True, output="\n".join(files) or "No files found.")

    def _read_file(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        context = context or self.context
        path = validate_read_path(self.repo_root, arguments["path"])
        if not path.exists() or not path.is_file():
            return ToolResult(ok=False, output=f"File not found: {arguments['path']}")

        limit = arguments.get("limit", 12000)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return ToolResult(ok=False, output="read_file limit must be a positive integer.")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(ok=False, output=f"File could not be read: {exc}")

        offset = arguments.get("offset")
        if offset is not None:
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
                return ToolResult(ok=False, output="read_file offset must be a positive integer.")
            lines = text.splitlines()
            selected = lines[offset - 1 : offset - 1 + limit]
            suffix = "\n... file truncated" if offset - 1 + limit < len(lines) else ""
            return ToolResult(ok=True, output="\n".join(selected) + suffix)

        if len(text) > limit:
            text = text[:limit] + "\n... file truncated"
        return ToolResult(ok=True, output=text)

    def _grep(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        context = context or self.context
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return ToolResult(ok=False, output="grep requires a non-empty pattern.")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(ok=False, output=f"Invalid regex: {exc}")

        base = validate_read_path(self.repo_root, arguments.get("path", "."))
        if not base.exists():
            return ToolResult(ok=False, output=f"Path does not exist: {arguments.get('path', '.')}")

        matches: list[str] = []
        targets = [base] if base.is_file() else base.rglob("*")
        for path in targets:
            _raise_if_cancelled(context)
            if should_skip_path(self.repo_root, path) or not path.is_file() or not _is_text_file(path):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                _raise_if_cancelled(context)
                if regex.search(line):
                    rel = path.relative_to(self.repo_root).as_posix()
                    matches.append(f"{rel}:{line_number}: {line}")
                    if len(matches) >= 80:
                        return ToolResult(ok=True, output="\n".join(matches) + "\n... matches truncated")
        return ToolResult(ok=True, output="\n".join(matches) or "No matches found.")

    def _retrieve_context(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        context = context or self.context
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(ok=False, output="retrieve_context requires a non-empty query.")
        top_k = arguments.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            return ToolResult(ok=False, output="top_k must be >= 1.")
        indexer = RepoIndexer(
            self.repo_root,
            skip_predicate=lambda path: should_skip_path(self.repo_root, path),
            cancellation_token=context.cancellation_token,
        )
        return ToolResult(ok=True, output=indexer.retrieve(query=query, top_k=top_k))

    def _replace_in_file(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        context = context or self.context
        path = validate_write_path(self.repo_root, arguments["path"])
        old = arguments.get("old")
        new = arguments.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return ToolResult(ok=False, output="replace_in_file requires string old and new fields.")
        if not old:
            return ToolResult(ok=False, output="replace_in_file requires a non-empty old snippet.")
        if not path.exists() or not path.is_file():
            return ToolResult(ok=False, output=f"File not found: {arguments['path']}")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(ok=False, output=f"File could not be read: {exc}")
        count = content.count(old)
        if count == 0:
            return ToolResult(ok=False, output="old text not found; inspect the file before retrying.")
        if count > 1:
            return ToolResult(ok=False, output=f"old text occurs {count} times; provide a more specific snippet.")

        _raise_if_cancelled(context)
        updated = content.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        rel = path.relative_to(self.repo_root).as_posix()
        return ToolResult(ok=True, output=_unified_diff(rel, content, updated) or f"Updated {rel} with no textual diff.")

    def _write_file(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        context = context or self.context
        path = validate_write_path(self.repo_root, arguments["path"])
        content = arguments.get("content")
        if not isinstance(content, str):
            return ToolResult(ok=False, output="write_file requires string content.")
        if path.exists() and not path.is_file():
            return ToolResult(ok=False, output=f"Path is not a file: {arguments['path']}")

        _raise_if_cancelled(context)
        old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        _raise_if_cancelled(context)
        path.write_text(content, encoding="utf-8")
        rel = path.relative_to(self.repo_root).as_posix()
        return ToolResult(ok=True, output=_unified_diff(rel, old, content) or f"Wrote {rel} with no textual diff.")

    def _run_tests(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        context = context or self.context
        command = str(arguments.get("command") or "pytest -q")
        parts = _subprocess_command(validate_test_command(command, repo_root=self.repo_root))
        result = run_process(
            parts,
            cwd=self.repo_root,
            timeout_seconds=self.timeout,
            env=_test_env(self.repo_root),
            cancellation_token=context.cancellation_token,
            max_output_chars=context.max_process_output_chars,
        )
        if result.start_failed:
            return ToolResult(ok=False, output=f"test command failed to start: {result.start_failed}", reason="start_failed")
        if result.timed_out:
            output = _format_test_output("timeout", result.stdout, result.stderr + f"\nCommand timed out after {self.timeout}s.")
            return ToolResult(ok=False, output=output, reason="timeout")
        if result.cancelled:
            output = _format_test_output("cancelled", result.stdout, result.stderr + "\nCommand cancelled.")
            return ToolResult(ok=False, output=output, reason="cancelled")

        output = _format_test_output(result.returncode or 0, result.stdout, result.stderr)
        return ToolResult(ok=result.returncode == 0, output=output)

    def _git_diff(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        context = context or self.context
        result = run_process(
            ["git", "diff", "--"],
            cwd=self.repo_root,
            timeout_seconds=self.timeout,
            env=dict(os.environ),
            cancellation_token=context.cancellation_token,
            max_output_chars=context.max_process_output_chars,
        )
        if result.start_failed:
            return ToolResult(ok=False, output=f"git diff failed: {result.start_failed}")
        if result.timed_out:
            return ToolResult(ok=False, output=f"git diff timed out after {self.timeout}s.", reason="timeout")
        if result.cancelled:
            return ToolResult(ok=False, output="git diff cancelled.", reason="cancelled")
        if result.returncode != 0:
            output = result.stderr.strip() or result.stdout.strip() or "git diff failed; repository may not be a git repo."
            return ToolResult(ok=False, output=output)
        return ToolResult(ok=True, output=result.stdout.strip() or "No git diff.")

    def _finish(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        summary = arguments.get("summary", "Finished.")
        return ToolResult(ok=True, output=str(summary or "Finished."))

    def _resource_path(self, candidate: object, context: ToolContext, *, write: bool = False) -> str:
        validator = validate_write_path if write else validate_read_path
        path = validator(context.repo_root, candidate)
        return path.relative_to(context.repo_root).as_posix()


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def _unified_diff(rel_path: str, old: str, new: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
    )


def _subprocess_command(parts: list[str]) -> list[str]:
    command = Path(parts[0]).name.lower()
    if command == "pytest":
        return [sys.executable, "-m", "pytest", *parts[1:]]
    if command.startswith("python") and len(parts) >= 3 and parts[1] == "-m":
        return [sys.executable, *parts[1:]]
    return parts


def _test_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    return env


def _format_test_output(exit_status: int | str, stdout: str, stderr: str) -> str:
    return f"exit_status: {exit_status}\nstdout:\n{stdout.strip()}\nstderr:\n{stderr.strip()}"


def _raise_if_cancelled(context: ToolContext) -> None:
    if context.cancellation_token is not None:
        context.cancellation_token.raise_if_cancelled()


def file_resource(arguments: dict[str, Any], context: ToolContext) -> set[str]:
    path = validate_write_path(context.repo_root, arguments["path"])
    return {f"file:{path.relative_to(context.repo_root).as_posix()}"}


def read_file_resource(arguments: dict[str, Any], context: ToolContext) -> set[str]:
    path = validate_read_path(context.repo_root, arguments.get("path", "."))
    return {f"file:{path.relative_to(context.repo_root).as_posix()}"}
