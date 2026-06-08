from __future__ import annotations

"""Typed SFT sample schema and sample factories."""

from pathlib import Path
from typing import Any, Iterable, TypedDict


class ToolCallOutput(TypedDict):
    tool: str
    arguments: dict[str, Any]
    reason: str


class RequiredSftFields(TypedDict):
    instruction: str
    input: dict[str, Any]
    output: dict[str, Any]


class SftSample(RequiredSftFields, total=False):
    metadata: dict[str, Any]


def make_tool_call_sample(
    *,
    task: str,
    plan: str,
    history: Iterable[dict[str, Any]],
    tool: str,
    arguments: dict[str, Any],
    reason: str,
    metadata: dict[str, Any],
) -> SftSample:
    return {
        "instruction": "根据用户任务、计划和已有工具轨迹，选择下一步工具调用。",
        "input": {
            "task": task,
            "plan": plan,
            "history": list(history),
        },
        "output": {
            "tool": tool,
            "arguments": arguments,
            "reason": reason,
        },
        "metadata": metadata,
    }


def make_write_file_sample(
    *,
    task: str,
    repo_context: str,
    path: str,
    content: str,
    reason: str,
    metadata: dict[str, Any],
) -> SftSample:
    return {
        "instruction": "根据用户任务和仓库上下文，选择下一步工具调用。",
        "input": {"task": task, "repo_context": repo_context, "history": []},
        "output": {
            "tool": "write_file",
            "arguments": {"path": path, "content": content},
            "reason": reason,
        },
        "metadata": metadata,
    }


def make_strategy_sample(
    *,
    repo: str,
    task: str,
    test_command: str,
    metadata: dict[str, Any],
) -> SftSample:
    return {
        "instruction": "根据本地代码任务，制定并执行最小工具调用策略。",
        "input": {
            "repo": repo,
            "task": task,
            "test_command": test_command,
        },
        "output": {
            "strategy": [
                {"tool": "retrieve_context", "purpose": "定位与任务相关的代码和测试文件"},
                {"tool": "read_file", "purpose": "读取需要修改的目标文件"},
                {"tool": "replace_in_file 或 write_file", "purpose": "进行最小、安全的代码修改"},
                {"tool": "run_tests", "purpose": "运行测试验证修改"},
            ]
        },
        "metadata": metadata,
    }


def make_repair_plan_sample(
    *,
    repo_name: Any,
    base_commit: Any,
    problem_statement: Any,
    plan: str,
    validation: str,
    metadata: dict[str, Any],
) -> SftSample:
    return {
        "instruction": "根据真实 GitHub issue 描述，制定代码仓库修复计划。",
        "input": {
            "repo_name": repo_name,
            "base_commit": base_commit,
            "problem_statement": problem_statement,
        },
        "output": {
            "plan": plan,
            "validation": validation,
        },
        "metadata": metadata,
    }


def validate_sft_sample(record: dict[str, Any], *, path: Path, line_num: int) -> None:
    missing = [key for key in ("instruction", "input", "output") if key not in record]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"{path}:{line_num}: missing required SFT field(s): {fields}")
    instruction = record["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{path}:{line_num}: instruction must be a non-empty string")
    if not isinstance(record["input"], dict):
        raise ValueError(f"{path}:{line_num}: input must be a JSON object")
    if not isinstance(record["output"], dict):
        raise ValueError(f"{path}:{line_num}: output must be a JSON object")


__all__ = [
    "SftSample",
    "ToolCallOutput",
    "make_repair_plan_sample",
    "make_strategy_sample",
    "make_tool_call_sample",
    "make_write_file_sample",
    "validate_sft_sample",
]
