from __future__ import annotations

"""Helpers for creating small generated coding-task repositories."""

import re
from pathlib import Path


def write_python_task_repo(repo_path: Path, skeleton: str, tests: list[str]) -> None:
    test_lines = "from solution import *\n\n\ndef test_mbpp_generated() -> None:\n"
    for test in tests:
        test_lines += f"    {test}\n"
    write_repo(repo_path, {"solution.py": skeleton, "tests/test_solution.py": test_lines})


def write_repo(repo_path: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        target = repo_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (repo_path / "AGENT.md").write_text(
        "# Generated Coding Task Repo\n\n- Make minimal edits.\n- Run `pytest -q` after edits.\n",
        encoding="utf-8",
    )


def python_skeleton_from_solution(code: str) -> str:
    match = re.search(r"^def\s+\w+\s*\([^\n]*\)\s*(?:->\s*[^:]+)?:", code, flags=re.MULTILINE)
    if not match:
        return "# TODO: implement solution\n"
    signature = match.group(0)
    return f"{signature}\n    pass\n"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


__all__ = [
    "python_skeleton_from_solution",
    "safe_id",
    "write_python_task_repo",
    "write_repo",
]
