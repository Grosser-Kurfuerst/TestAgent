from __future__ import annotations

import os
from pathlib import Path

from my_agent.tools.hooks import HookViolation


_FIND_OPTIONS_WITH_VALUE = {
    "-amin",
    "-anewer",
    "-atime",
    "-cmin",
    "-cnewer",
    "-context",
    "-ctime",
    "-exec",
    "-execdir",
    "-fls",
    "-fprint",
    "-fprint0",
    "-fprintf",
    "-fstype",
    "-gid",
    "-group",
    "-ilname",
    "-iname",
    "-inum",
    "-ipath",
    "-iregex",
    "-iwholename",
    "-links",
    "-lname",
    "-maxdepth",
    "-mindepth",
    "-mmin",
    "-mtime",
    "-name",
    "-newer",
    "-path",
    "-perm",
    "-printf",
    "-regex",
    "-samefile",
    "-size",
    "-type",
    "-uid",
    "-user",
    "-wholename",
}


def reject_full_scan_command(argv: list[str], raw_command: str | None = None, *, cwd: Path | None = None) -> None:
    if not argv:
        return
    if Path(argv[0]).name.lower() != "find":
        return
    if len(argv) < 2:
        return

    for target in _find_path_operands(argv[1:]):
        if _is_full_scan_target(target):
            raise HookViolation(
                "Full filesystem scan is blocked. Use list_files, read_file, or grep within the repository instead."
            )


def _expand_find_target(target: str) -> Path:
    if target == "$HOME" or target == "${HOME}":
        return Path(os.environ.get("HOME", str(Path.home()))).expanduser().resolve()
    return Path(target).expanduser().resolve()


def _find_path_operands(args: list[str]) -> list[str]:
    operands: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            operands.extend(args[index + 1 :])
            break
        if token in {"-H", "-L", "-P", "-O0", "-O1", "-O2", "-O3", "-D"}:
            index += 2 if token == "-D" else 1
            continue
        if token in _FIND_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        operands.append(token)
        index += 1
    return operands


def _is_full_scan_target(target: str) -> bool:
    expanded = _expand_find_target(target)
    home = Path.home().resolve()
    return target in {"~", "$HOME", "${HOME}"} or expanded == Path("/").resolve() or expanded == home
