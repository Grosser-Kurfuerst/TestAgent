from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from my_agent.context import budget_tool_definitions
from my_agent.tools import RepoTools


def _last_memory_prepared_from_trace(trace_path: Path | None) -> dict[str, object] | None:
    if trace_path is None:
        return None
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    latest: dict[str, object] | None = None
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("event") != "memory.prepared":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            latest = dict(payload)
    return latest


def _payload_value(payload: dict[str, object] | None, key: str) -> str:
    if payload is None:
        return "not available"
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        return "not available"
    return str(value)


def _discover_test_command(repo_path: Path) -> str | None:
    for filename in ("AGENT.md", "AGENTS.md"):
        path = repo_path / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        command = _first_backticked_test_command(text)
        if command:
            return command
    tests_dir = repo_path / "tests"
    if tests_dir.exists() and any(tests_dir.glob("test*.py")):
        return "python -m unittest discover -s tests -q"
    return None


def _first_backticked_test_command(text: str) -> str | None:
    parts = text.split("`")
    for index in range(1, len(parts), 2):
        command = parts[index].strip()
        if _looks_like_test_command(command):
            return command
    return None


def _looks_like_test_command(command: str) -> bool:
    normalized = " ".join(command.split())
    return (
        normalized.startswith("pytest")
        or normalized.startswith("python -m pytest")
        or normalized.startswith("python -m unittest")
        or normalized in {"npm test", "pnpm test", "yarn test"}
        or normalized.startswith("npm run test")
    )


def _tool_summary(tools: RepoTools) -> str:
    counts = Counter(tool.spec.source for tool in tools.registry.tools if tool.spec.enabled)
    if not counts:
        return "0 enabled"
    return ", ".join(f"{count} {source}" for source, count in sorted(counts.items()))


def format_tools_text(
    tools: RepoTools,
    approval_label: Callable[..., str],
) -> str:
    lines = ["name\tsource\trisk\tapproval\tdescription"]
    for tool in tools.registry.tools:
        if not tool.spec.enabled:
            continue
        lines.append(
            "\t".join(
                [
                    tool.spec.name,
                    tool.spec.source,
                    tool.spec.risk.value,
                    approval_label(source=tool.spec.source, risk=tool.spec.risk.value),
                    tool.spec.description,
                ]
            )
        )
    return "\n".join(lines)


def format_context_text(
    *,
    memory: object,
    profile: object,
    tools: RepoTools,
    latest_trace: Path | None,
    last_memory_prepared: dict[str, object] | None,
    mcp_summary: str,
    test_command: str | None,
) -> str:
    status = memory.status(include_entries=False)
    tool_budget = budget_tool_definitions(tools.tool_definitions(), profile)
    prepared = last_memory_prepared or _last_memory_prepared_from_trace(latest_trace)
    return "\n".join(
        [
            f"system/project: rebuilt per run",
            f"context window: {profile.max_context_tokens} ({profile.dynamic_profile_source})",
            f"prompt limit: {profile.compression_trigger_tokens}",
            f"repo index budget: {profile.repo_context_budget_tokens}",
            f"tool schema budget: {profile.tool_schema_budget_tokens}",
            f"last memory budget: {_payload_value(prepared, 'memory_budget_tokens')}",
            f"last long-term limit: {_payload_value(prepared, 'long_term_limit')}",
            f"last short-term allowed: {_payload_value(prepared, 'short_term_allowed')}",
            f"short-term: {status.short_term_entries} entries, {status.short_term_tokens} tokens",
            f"short-term storage cap: {status.short_term_storage_token_limit}",
            f"long-term: {status.long_term_entries} entries, {status.long_term_tokens} tokens",
            f"tools: {tool_budget.included_count} exposed, {tool_budget.omitted_count} omitted",
            f"mcp: {mcp_summary}",
            f"default test command: {test_command or 'not configured'}",
            f"compression trigger: {profile.compression_trigger_tokens}",
            f"retain recent turns: {status.retain_recent_turns}",
            f"max tool result chars: {profile.tool_result_char_limit}",
        ]
    )


def format_memory_text(memory: object) -> str:
    status = memory.status(include_entries=True)
    lines = [
        "Memory",
        f"project: {status.project_key}",
        f"storage: {status.storage_path}",
        (
            f"short-term: {status.short_term_entries} entries, {status.short_term_tokens} tokens, "
            f"storage cap {status.short_term_storage_token_limit}"
        ),
        f"long-term: {status.long_term_entries} entries, {status.long_term_tokens} tokens",
        (
            f"compression: trigger {int(status.compression_trigger_ratio * 100)}%, "
            f"retain {status.retain_recent_turns} turns, map chunk {status.map_chunk_size}"
        ),
        "",
        "Long-term entries:",
    ]
    if not status.long_term_entries_detail:
        lines.append("- none")
        return "\n".join(lines)
    for entry in status.long_term_entries_detail:
        timestamp = entry.created_at.isoformat()
        lines.append(f"- {entry.id} [{entry.type.value} {entry.scope.value} {entry.source} {timestamp}] {entry.content}")
    return "\n".join(lines)
