from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from my_agent.context import budget_tool_definitions
from my_agent.tools import RepoTools


def _last_memory_prepared_from_trace(trace_path: Path | None) -> dict[str, object] | None:
    return _last_trace_payload(trace_path, "memory.prepared")


def _last_evolver_candidates_from_trace(trace_path: Path | None) -> dict[str, object] | None:
    return _last_trace_payload(trace_path, "memory.evolver_candidates")


def _last_evolver_selected_from_trace(trace_path: Path | None) -> dict[str, object] | None:
    return _last_trace_payload(trace_path, "memory.evolver_selected")


def _last_trace_payload(trace_path: Path | None, event_name: str) -> dict[str, object] | None:
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
        if not isinstance(event, dict) or event.get("event") != event_name:
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


def _payload_count(payload: dict[str, object] | None, key: str, fallback_key: str = "") -> str:
    if payload is None:
        return "not available"
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    if fallback_key:
        fallback = payload.get(fallback_key)
        if isinstance(fallback, list):
            return str(len(fallback))
    return "not available"


def _payload_tier_distribution(payload: dict[str, object] | None) -> str:
    if payload is None:
        return "not available"
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        return "none"
    parts: list[str] = []
    for tier, count in sorted(tiers.items()):
        if isinstance(tier, str) and tier and isinstance(count, int) and not isinstance(count, bool):
            parts.append(f"{tier}:{count}")
    return ", ".join(parts) if parts else "none"


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
    last_evolver_candidates: dict[str, object] | None = None,
    last_evolver_selected: dict[str, object] | None = None,
) -> str:
    status = memory.status(include_entries=False)
    tool_budget = budget_tool_definitions(tools.tool_definitions(), profile)
    prepared = last_memory_prepared or _last_memory_prepared_from_trace(latest_trace)
    evolver_candidates = last_evolver_candidates or _last_evolver_candidates_from_trace(latest_trace)
    evolver_selected = last_evolver_selected or _last_evolver_selected_from_trace(latest_trace)
    evolver_lines = _format_evolver_context_lines(memory, evolver_candidates, evolver_selected, prepared)
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
            *evolver_lines,
            f"tools: {tool_budget.included_count} exposed, {tool_budget.omitted_count} omitted",
            f"mcp: {mcp_summary}",
            f"default test command: {test_command or 'not configured'}",
            f"compression trigger: {profile.compression_trigger_tokens}",
            f"retain recent turns: {status.retain_recent_turns}",
            f"max tool result chars: {profile.tool_result_char_limit}",
        ]
    )


def _format_evolver_context_lines(
    memory: object,
    candidates_payload: dict[str, object] | None,
    selected_payload: dict[str, object] | None,
    prepared_payload: dict[str, object] | None,
) -> list[str]:
    config = getattr(memory, "config", None)
    mode = str(getattr(config, "memory_evolver_mode", "off") or "off")
    if mode not in {"retrieve_select", "full"}:
        return []

    candidates_payload = candidates_payload or _last_evolver_candidates_from_memory(memory)
    selected_payload = selected_payload or _last_evolver_selected_from_memory(memory)
    selected_payload = selected_payload or _evolver_selected_from_prepared(prepared_payload)
    lines = [f"evolver selector: enabled ({mode})"]
    if candidates_payload is None and selected_payload is None:
        lines.append("evolver selection: No evolver selection has been prepared in this session.")
        return lines
    lines.extend(
        [
            (
                "evolver selection: "
                f"candidates={_payload_count(candidates_payload, 'candidate_count', 'candidate_ids')}, "
                f"selected={_payload_count(selected_payload, 'selected_count', 'selected_ids')}"
            ),
            f"evolver selected tiers: {_payload_tier_distribution(selected_payload)}",
            f"evolver selection policy: {_payload_value(selected_payload or candidates_payload, 'selection_policy')}",
        ]
    )
    return lines


def _evolver_selected_from_prepared(payload: dict[str, object] | None) -> dict[str, object] | None:
    if payload is None:
        return None
    memory_hits = payload.get("memory_hits")
    if not isinstance(memory_hits, int) or isinstance(memory_hits, bool) or memory_hits < 0:
        return None
    if memory_hits == 0:
        return None
    return {"selected_count": memory_hits}


def _last_evolver_candidates_from_memory(memory: object) -> dict[str, object] | None:
    selection = getattr(memory, "last_evolver_selection", None)
    candidates = getattr(selection, "candidates", None)
    if candidates is None:
        return None
    return {
        "candidate_count": len(candidates),
        "selection_policy": str(getattr(selection, "policy", "") or ""),
    }


def _last_evolver_selected_from_memory(memory: object) -> dict[str, object] | None:
    selection = getattr(memory, "last_evolver_selection", None)
    selected = getattr(selection, "selected", None)
    if selected is None:
        return None
    tiers: Counter[str] = Counter()
    selected_ids: list[str] = []
    for item in selected:
        candidate = getattr(item, "candidate", None)
        if candidate is None:
            continue
        selected_ids.append(str(getattr(candidate, "id", "") or ""))
        tier = getattr(candidate, "tier", None)
        tier_value = str(getattr(tier, "value", "") or "")
        if tier_value:
            tiers[tier_value] += 1
    return {
        "selected_count": len(selected),
        "selected_ids": selected_ids,
        "tiers": dict(sorted(tiers.items())),
        "selection_policy": str(getattr(selection, "policy", "") or ""),
    }


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
