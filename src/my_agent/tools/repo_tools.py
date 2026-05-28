from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from my_agent.indexer import RepoIndexer, TEXT_EXTENSIONS
from my_agent.schema import ToolResult
from my_agent.tools.hooks import (
    HookViolation,
    post_tool_check,
    should_skip_path,
    validate_read_path,
    validate_test_command,
    validate_tool_call,
    validate_write_path,
)
from my_agent.tools.registry import ToolRegistry


class RepoTools:
    def __init__(self, repo_path: str | Path, timeout: int = 60):
        self.repo_root = Path(repo_path).resolve()
        if not self.repo_root.exists() or not self.repo_root.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {self.repo_root}")
        self.timeout = timeout
        self.registry = ToolRegistry()
        self._register_defaults()

    @property
    def tool_names(self) -> list[str]:
        return self.registry.tool_names

    def descriptions(self) -> str:
        return self.registry.descriptions()

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        if arguments is None:
            args: dict[str, Any] = {}
        elif not isinstance(arguments, dict):
            return ToolResult(ok=False, output="Tool arguments must be a JSON object.")
        else:
            args = arguments
        try:
            validate_tool_call(self.repo_root, name, args)
        except HookViolation as exc:
            return ToolResult(ok=False, output=str(exc), blocked=True, reason=str(exc))

        result = self.registry.run(name, args)
        if result.blocked:
            return result
        note = post_tool_check(name, result.ok, result.output)
        if note:
            return ToolResult(
                ok=result.ok,
                output=f"{result.output}\n\nHook note: {note}",
                blocked=result.blocked,
                reason=result.reason,
            )
        return result

    def _register_defaults(self) -> None:
        self.registry.register("list_files", '{"path": "."} list repository files under path.', self._list_files)
        self.registry.register("read_file", '{"path": "relative.py", "limit": 12000} read a text file.', self._read_file)
        self.registry.register("grep", '{"pattern": "regex", "path": "."} search text files with a regex.', self._grep)
        self.registry.register(
            "retrieve_context",
            '{"query": "symbol or task", "top_k": 5} retrieve relevant code snippets.',
            self._retrieve_context,
        )
        self.registry.register(
            "replace_in_file",
            '{"path": "file.py", "old": "exact text", "new": "replacement"} make one exact replacement.',
            self._replace_in_file,
        )
        self.registry.register("write_file", '{"path": "file.py", "content": "full content"} overwrite a file.', self._write_file)
        self.registry.register("run_tests", '{"command": "pytest -q"} run an allowlisted test command.', self._run_tests)
        self.registry.register("git_diff", "{} show git diff.", self._git_diff)
        self.registry.register("finish", '{"summary": "final answer"} finish the task.', self._finish)

    def _list_files(self, arguments: dict[str, Any]) -> ToolResult:
        base = validate_read_path(self.repo_root, arguments.get("path", "."))
        if not base.exists():
            return ToolResult(ok=False, output=f"Path does not exist: {arguments.get('path', '.')}")

        files: list[str] = []
        targets = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in targets:
            if should_skip_path(self.repo_root, path) or not path.is_file():
                continue
            files.append(path.relative_to(self.repo_root).as_posix())
            if len(files) >= 120:
                files.append("... output truncated")
                break
        return ToolResult(ok=True, output="\n".join(files) or "No files found.")

    def _read_file(self, arguments: dict[str, Any]) -> ToolResult:
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

        if len(text) > limit:
            text = text[:limit] + "\n... file truncated"
        return ToolResult(ok=True, output=text)

    def _grep(self, arguments: dict[str, Any]) -> ToolResult:
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
        targets = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in targets:
            if should_skip_path(self.repo_root, path) or not path.is_file() or not _is_text_file(path):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                if regex.search(line):
                    rel = path.relative_to(self.repo_root).as_posix()
                    matches.append(f"{rel}:{line_number}: {line}")
                    if len(matches) >= 80:
                        return ToolResult(ok=True, output="\n".join(matches) + "\n... matches truncated")
        return ToolResult(ok=True, output="\n".join(matches) or "No matches found.")

    def _retrieve_context(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(ok=False, output="retrieve_context requires a non-empty query.")
        top_k = arguments.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            return ToolResult(ok=False, output="top_k must be >= 1.")
        indexer = RepoIndexer(self.repo_root, skip_predicate=lambda path: should_skip_path(self.repo_root, path))
        return ToolResult(ok=True, output=indexer.retrieve(query=query, top_k=top_k))

    def _replace_in_file(self, arguments: dict[str, Any]) -> ToolResult:
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

        updated = content.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        rel = path.relative_to(self.repo_root).as_posix()
        return ToolResult(ok=True, output=_unified_diff(rel, content, updated) or f"Updated {rel} with no textual diff.")

    def _write_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = validate_write_path(self.repo_root, arguments["path"])
        content = arguments.get("content")
        if not isinstance(content, str):
            return ToolResult(ok=False, output="write_file requires string content.")
        if path.exists() and not path.is_file():
            return ToolResult(ok=False, output=f"Path is not a file: {arguments['path']}")

        old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rel = path.relative_to(self.repo_root).as_posix()
        return ToolResult(ok=True, output=_unified_diff(rel, old, content) or f"Wrote {rel} with no textual diff.")

    def _run_tests(self, arguments: dict[str, Any]) -> ToolResult:
        command = str(arguments.get("command") or "pytest -q")
        parts = _subprocess_command(validate_test_command(command, repo_root=self.repo_root))
        try:
            completed = subprocess.run(
                parts,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                env=_test_env(self.repo_root),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            output = _format_test_output("timeout", stdout, stderr + f"\nCommand timed out after {self.timeout}s.")
            return ToolResult(ok=False, output=output, reason="timeout")

        output = _format_test_output(completed.returncode, completed.stdout, completed.stderr)
        return ToolResult(ok=completed.returncode == 0, output=output)

    def _git_diff(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            completed = subprocess.run(
                ["git", "diff", "--"],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(ok=False, output=f"git diff failed: {exc}")
        if completed.returncode != 0:
            output = completed.stderr.strip() or completed.stdout.strip() or "git diff failed; repository may not be a git repo."
            return ToolResult(ok=False, output=output)
        return ToolResult(ok=True, output=completed.stdout.strip() or "No git diff.")

    def _finish(self, arguments: dict[str, Any]) -> ToolResult:
        summary = arguments.get("summary", "Finished.")
        return ToolResult(ok=True, output=str(summary or "Finished."))


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
