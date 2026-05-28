from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any


IGNORED_PATH_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}

PROTECTED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}

PROTECTED_SUFFIXES = {
    ".cer",
    ".crt",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
}

DANGEROUS_COMMAND_PARTS = (
    "rm -rf",
    "sudo",
    "mkfs",
    "dd ",
    ":(){",
    "chmod -r 777",
    "chown -r",
    "curl ",
    "wget ",
    "killall",
    "pkill",
)

SHELL_CONTROL_TOKENS = ("|", ">", "<", ";", "&&", "||", "`", "$(")


class HookViolation(RuntimeError):
    pass


def ensure_inside_repo(repo_root: Path, candidate: str | Path) -> Path:
    root = repo_root.resolve(strict=True)
    if not isinstance(candidate, (str, Path)):
        raise HookViolation("Path must be a string.")
    if isinstance(candidate, str) and not candidate.strip():
        raise HookViolation("Path must be non-empty.")

    candidate_path = Path(candidate)
    raw_path = candidate_path if candidate_path.is_absolute() else root / candidate_path
    # 防止通过链接绕过
    if _has_symlink_component(root, raw_path):
        raise HookViolation(f"Refusing to follow symlink path: {candidate}")
    try:
        resolved = raw_path.resolve(strict=False)
    except OSError as exc:
        raise HookViolation(f"Path cannot be resolved safely: {candidate}") from exc

    if not resolved.is_relative_to(root):
        raise HookViolation(f"Path escapes repository root: {candidate}")
    return resolved


def validate_read_path(repo_root: Path, candidate: str | Path) -> Path:
    path = ensure_inside_repo(repo_root, candidate)
    _reject_ignored_path(repo_root, path)
    _reject_protected_file(path)
    return path


def validate_write_path(repo_root: Path, candidate: str | Path) -> Path:
    path = ensure_inside_repo(repo_root, candidate)
    _reject_ignored_path(repo_root, path)
    _reject_protected_file(path)
    return path


def validate_tool_call(repo_root: Path, tool_name: str, arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise HookViolation("Tool arguments must be a JSON object.")

    if tool_name == "read_file":
        path = arguments.get("path")
        if not path:
            raise HookViolation("read_file requires a path.")
        validate_read_path(repo_root, path)
    elif tool_name in {"list_files", "grep"}:
        validate_read_path(repo_root, arguments.get("path", "."))
    elif tool_name in {"write_file", "replace_in_file"}:
        path = arguments.get("path")
        if not path:
            raise HookViolation(f"{tool_name} requires a path.")
        validate_write_path(repo_root, path)
    elif tool_name == "run_tests":
        validate_test_command(str(arguments.get("command") or "pytest -q"), repo_root=repo_root)


def validate_test_command(command: str, repo_root: Path | None = None) -> list[str]:
    normalized = command.strip() or "pytest -q"
    lowered = normalized.lower()
    if any(token in normalized for token in SHELL_CONTROL_TOKENS):
        raise HookViolation(f"Dangerous shell syntax blocked: {command}")
    if any(part in lowered for part in DANGEROUS_COMMAND_PARTS):
        raise HookViolation(f"Dangerous command blocked: {command}")

    try:
        parts = shlex.split(normalized)
    except ValueError as exc:
        raise HookViolation(f"Test command cannot be parsed: {command}") from exc
    if not parts:
        parts = ["pytest", "-q"]
    if not _is_allowed_test_command(parts):
        raise HookViolation(
            "Test command is not in allowlist. Use pytest, python -m pytest, "
            "python -m unittest, npm test, npm run test, pnpm test, or yarn test."
        )
    if repo_root is not None:
        _validate_test_path_arguments(repo_root, parts)
    return parts


def should_skip_path(repo_root: Path, path: Path) -> bool:
    try:
        path.relative_to(repo_root)
    except ValueError:
        return True
    return path.is_symlink() or _has_ignored_part(repo_root, path) or _is_protected_file(path)


def post_tool_check(tool_name: str, ok: bool, output: str) -> str | None:
    if tool_name in {"write_file", "replace_in_file"} and ok:
        return "File edited. Inspect git_diff or run tests before finishing."
    if tool_name == "run_tests" and not ok:
        return "Tests failed. Inspect stdout and stderr before editing again."
    return None


def _reject_ignored_path(repo_root: Path, path: Path) -> None:
    if _has_ignored_part(repo_root, path):
        raise HookViolation(f"Refusing to access generated or internal path: {path}")


def _reject_protected_file(path: Path) -> None:
    if _is_protected_file(path):
        raise HookViolation(f"Refusing to access protected file: {path.name}")


def _has_ignored_part(repo_root: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return True
    return any(part in IGNORED_PATH_NAMES for part in rel.parts)


def _is_protected_file(path: Path) -> bool:
    name = path.name.lower()
    return name in PROTECTED_FILE_NAMES or path.suffix.lower() in PROTECTED_SUFFIXES


def _has_symlink_component(root: Path, raw_path: Path) -> bool:
    try:
        rel = raw_path.relative_to(root)
    except ValueError:
        return False

    current = root
    for part in rel.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        if current.is_symlink():
            return True
    return False


def _is_allowed_test_command(parts: list[str]) -> bool:
    command = Path(parts[0]).name.lower()
    if command == "pytest":
        return True
    if _is_python_command(command) and len(parts) >= 3 and parts[1] == "-m" and parts[2] in {"pytest", "unittest"}:
        return True
    if command == "npm" and len(parts) >= 2:
        return parts[1] == "test" or (len(parts) >= 3 and parts[1] == "run" and parts[2] == "test")
    if command in {"pnpm", "yarn"} and len(parts) >= 2:
        return parts[1] == "test"
    return False


def _is_python_command(command: str) -> bool:
    return command.startswith("python") or command == Path(sys.executable).name.lower()


def _validate_test_path_arguments(repo_root: Path, parts: list[str]) -> None:
    command = Path(parts[0]).name.lower()
    args = parts[1:]
    if _is_python_command(command) and len(parts) >= 3 and parts[1] == "-m":
        module = parts[2]
        args = parts[3:]
        if module == "unittest":
            _validate_unittest_args(repo_root, args)
            return
        if module == "pytest":
            _validate_pytest_args(repo_root, args)
            return
    if command == "pytest":
        _validate_pytest_args(repo_root, args)
        return
    _validate_path_like_tokens(repo_root, args)


def _validate_pytest_args(repo_root: Path, args: list[str]) -> None:
    if "--pyargs" in args:
        raise HookViolation("pytest --pyargs is not allowed for repository-scoped test runs.")
    _validate_path_like_tokens(
        repo_root,
        args,
        path_options={
            "--basetemp",
            "--cache-dir",
            "--confcutdir",
            "--cov-config",
            "--cov-report",
            "--html",
            "--json-report-file",
            "--junit-xml",
            "--junitxml",
            "--log-file",
            "--rootdir",
            "--template",
        },
        validate_all_positionals=False,
    )


def _validate_unittest_args(repo_root: Path, args: list[str]) -> None:
    if not args:
        return
    non_option_args = [arg for arg in args if not arg.startswith("-")]
    if non_option_args and non_option_args[0] != "discover":
        raise HookViolation("unittest commands must use repository-scoped discover.")
    _validate_path_like_tokens(
        repo_root,
        args,
        path_options={"-s", "--start-directory", "-t", "--top-level-directory"},
        validate_all_positionals=False,
    )


def _validate_path_like_tokens(
    repo_root: Path,
    args: list[str],
    path_options: set[str] | None = None,
    validate_all_positionals: bool = False,
) -> None:
    path_options = path_options or set()
    index = 0
    while index < len(args):
        token = args[index]
        if token in path_options:
            if index + 1 >= len(args):
                raise HookViolation(f"Test command option requires a path: {token}")
            _validate_test_path(repo_root, args[index + 1])
            index += 2
            continue
        matched_long_option = False
        for option in path_options:
            prefix = f"{option}="
            if token.startswith(prefix):
                _validate_test_path(repo_root, token[len(prefix) :])
                matched_long_option = True
                break
        if matched_long_option:
            index += 1
            continue
        if token.startswith("--") and "=" in token:
            _, value = token.split("=", 1)
            if _looks_like_path_argument(value):
                _validate_test_path(repo_root, value)
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        if validate_all_positionals or _looks_like_path_argument(token):
            _validate_test_path(repo_root, token)
        index += 1


def _looks_like_path_argument(value: str) -> bool:
    path_part = value.split("::", 1)[0]
    return (
        Path(path_part).is_absolute()
        or path_part.startswith((".", ".."))
        or "/" in path_part
        or "\\" in path_part
    )


def _validate_test_path(repo_root: Path, value: str) -> None:
    path_part = _extract_test_path_value(value)
    if not path_part:
        raise HookViolation("Test command path must be non-empty.")
    path = validate_read_path(repo_root, path_part)
    if path.exists() and path.is_file() and _is_protected_file(path):
        raise HookViolation(f"Refusing to run tests from protected file: {path.name}")


def _extract_test_path_value(value: str) -> str:
    path_part = value.split("::", 1)[0]
    if ":" in path_part:
        prefix, suffix = path_part.split(":", 1)
        if prefix in {"annotate", "html", "json", "lcov", "term", "term-missing", "xml"} and suffix:
            return suffix
    return path_part
