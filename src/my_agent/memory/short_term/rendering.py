"""Short-term message rendering and group-preserving token budgeting."""

from __future__ import annotations

from typing import Any

from my_agent.llm.types import LLMToolCall, Message, MessageLike, messages_to_openai
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryType


def render_short_term_messages(
    entries: list[MemoryEntry],
    *,
    max_tokens: int | None = None,
) -> list[MessageLike]:
    selected = entries
    if max_tokens is not None:
        selected = entries_within_token_budget(entries, max_tokens)
    return render_short_term_entries(selected)


def render_short_term_entries(entries: list[MemoryEntry]) -> list[MessageLike]:
    rendered: list[MessageLike] = []
    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        if entry.type == MemoryType.SUMMARY:
            rendered.append(Message(role="user", content=entry.content))
            idx += 1
        elif entry.source == "task_goal":
            rendered.append(Message(role="user", content=f"[Task goal]\n{entry.content}"))
            idx += 1
        elif entry.source == "user":
            rendered.append(Message(role="user", content=entry.content))
            idx += 1
        elif entry.source == "assistant":
            tool_calls = _tool_calls_from_metadata(entry.metadata)
            if not tool_calls:
                rendered.append(Message(role="assistant", content=entry.content or ""))
                idx += 1
                continue

            tool_entries, next_idx = _contiguous_tool_entries(entries, idx + 1)
            if _tool_entries_match_calls(tool_entries, tool_calls):
                rendered.append(
                    Message(
                        role="assistant",
                        content=entry.content or "",
                        tool_calls=tool_calls,
                    )
                )
                rendered.extend(
                    _tool_message_from_entry(tool_entry)
                    for tool_entry in tool_entries
                )
            else:
                rendered.append(Message(
                    role="user",
                    content=_incomplete_tool_call_memory(entry, tool_entries),
                ))
            idx = next_idx
        elif entry.source.startswith("tool:"):
            rendered.append(Message(
                role="user",
                content=f"[Tool result memory]\n{entry.content}",
            ))
            idx += 1
        else:
            rendered.append(Message(role="user", content=entry.content))
            idx += 1
    return rendered


def entries_within_token_budget(
    entries: list[MemoryEntry],
    max_tokens: int,
) -> list[MemoryEntry]:
    if max_tokens <= 0 or not entries:
        return []
    groups = _entry_groups(entries)
    selected: list[list[MemoryEntry]] = []
    start_idx = 0
    if groups and groups[0] and groups[0][0].source == "task_goal":
        if _rendered_groups_tokens([groups[0]]) <= max_tokens:
            selected.append(groups[0])
        start_idx = 1
    tail: list[list[MemoryEntry]] = []
    for group in reversed(groups[start_idx:]):
        candidate_tail = [group, *tail]
        if _rendered_groups_tokens([*selected, *candidate_tail]) > max_tokens:
            continue
        tail = candidate_tail
    selected.extend(tail)
    return [entry for group in selected for entry in group]


def _entry_groups(entries: list[MemoryEntry]) -> list[list[MemoryEntry]]:
    groups: list[list[MemoryEntry]] = []
    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        if entry.source == "assistant":
            tool_entries, next_idx = _contiguous_tool_entries(entries, idx + 1)
            groups.append([entry, *tool_entries])
            idx = next_idx
            continue
        groups.append([entry])
        idx += 1
    return groups


def _rendered_groups_tokens(groups: list[list[MemoryEntry]]) -> int:
    entries = [entry for group in groups for entry in group]
    if not entries:
        return 0
    return estimate_tokens(messages_to_openai(render_short_term_entries(entries)))


def _contiguous_tool_entries(
    entries: list[MemoryEntry],
    start: int,
) -> tuple[list[MemoryEntry], int]:
    tool_entries: list[MemoryEntry] = []
    idx = start
    while idx < len(entries) and entries[idx].source.startswith("tool:"):
        tool_entries.append(entries[idx])
        idx += 1
    return tool_entries, idx


def _tool_entries_match_calls(
    tool_entries: list[MemoryEntry],
    tool_calls: list[LLMToolCall],
) -> bool:
    if len(tool_entries) != len(tool_calls):
        return False
    entry_ids = [
        str(entry.metadata.get("tool_call_id") or entry.id)
        for entry in tool_entries
    ]
    call_ids = [call.id for call in tool_calls]
    return entry_ids == call_ids


def _tool_message_from_entry(entry: MemoryEntry) -> Message:
    return Message(
        role="tool",
        content=entry.content,
        tool_call_id=str(entry.metadata.get("tool_call_id") or entry.id),
        name=str(entry.metadata.get("tool_name") or entry.source.removeprefix("tool:")),
    )


def _incomplete_tool_call_memory(
    assistant: MemoryEntry,
    tool_entries: list[MemoryEntry],
) -> str:
    parts = ["[Incomplete tool-call memory]"]
    if assistant.content:
        parts.append(f"assistant: {assistant.content}")
    tool_names = assistant.metadata.get("tool_calls")
    if tool_names:
        parts.append(f"tool_calls: {tool_names}")
    for tool_entry in tool_entries:
        parts.append(f"{tool_entry.source}: {tool_entry.content}")
    return "\n".join(parts)


def _tool_calls_from_metadata(metadata: dict[str, Any]) -> list[LLMToolCall]:
    raw_calls = metadata.get("tool_calls_payload")
    if not isinstance(raw_calls, list):
        return []
    calls: list[LLMToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        try:
            calls.append(LLMToolCall.from_openai(raw))
        except ValueError:
            continue
    return calls


__all__ = [
    "entries_within_token_budget",
    "render_short_term_entries",
    "render_short_term_messages",
]
