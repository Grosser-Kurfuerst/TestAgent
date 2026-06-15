from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from my_agent.llm.types import Message, MessageLike, messages_to_openai


@dataclass(frozen=True)
class ContextProfile:
    max_context_tokens: int = 128_000
    response_reserve_tokens: int = 8_000
    compression_buffer_tokens: int = 8_000
    retain_recent_user_turns: int = 3
    max_tool_result_chars: int = 12_000
    max_summary_input_chars: int = 60_000

    @property
    def compression_trigger_tokens(self) -> int:
        return max(1, self.max_context_tokens - self.response_reserve_tokens - self.compression_buffer_tokens)

    @classmethod
    def from_config(cls, config: Any) -> "ContextProfile":
        return cls(
            max_context_tokens=getattr(config, "context_window", 128_000),
            response_reserve_tokens=getattr(config, "response_reserve_tokens", 8_000),
            compression_buffer_tokens=getattr(config, "compression_buffer_tokens", 8_000),
            retain_recent_user_turns=getattr(config, "retain_recent_user_turns", 3),
            max_tool_result_chars=getattr(config, "max_tool_result_chars", 12_000),
            max_summary_input_chars=getattr(config, "max_summary_input_chars", 60_000),
        )


@dataclass(frozen=True)
class CompactResult:
    compacted: bool
    before_tokens: int
    after_tokens: int
    summary_chars: int = 0
    fallback: bool = False


class ConversationCompactor:
    def __init__(self, profile: ContextProfile, llm: Any | None = None) -> None:
        self.profile = profile
        self.llm = llm

    def estimate_tokens(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> int:
        return estimate_tokens({"messages": messages_to_openai(messages), "tools": tools or []})

    def compact_if_needed(
        self,
        messages: list[MessageLike],
        tools: list[dict[str, Any]] | None = None,
        *,
        force: bool = False,
        focus: str = "",
    ) -> CompactResult:
        before = self.estimate_tokens(messages, tools)
        if not force and before < self.profile.compression_trigger_tokens:
            return CompactResult(False, before, before)

        system_end = 1 if messages and _role(messages[0]) == "system" else 0
        prefix_end, split_idx = self._split_indices(messages, system_end)
        if split_idx <= prefix_end:
            return CompactResult(False, before, before)

        old_messages = messages[prefix_end:split_idx]
        if not old_messages:
            return CompactResult(False, before, before)

        summary, fallback = self._summarize(old_messages, focus=focus)
        rebuilt: list[MessageLike] = []
        rebuilt.extend(messages[:prefix_end])
        rebuilt.append(Message(role="user", content="[Compressed conversation summary]\n" + summary.strip()))
        rebuilt.append(Message(role="assistant", content="Understood. Continue from this compressed state."))
        rebuilt.extend(messages[split_idx:])

        after = self.estimate_tokens(rebuilt, tools)
        messages.clear()
        messages.extend(rebuilt)
        return CompactResult(True, before, after, summary_chars=len(summary), fallback=fallback)

    def _split_indices(self, messages: list[MessageLike], system_end: int) -> tuple[int, int]:
        user_indices = [idx for idx in range(system_end, len(messages)) if _role(messages[idx]) == "user"]
        retain = max(1, self.profile.retain_recent_user_turns)
        if len(user_indices) > retain:
            return system_end, user_indices[-retain]

        prefix_end = system_end
        if len(messages) > system_end and _role(messages[system_end]) == "user":
            prefix_end = system_end + 1
        if len(messages) - prefix_end <= 2:
            return prefix_end, prefix_end

        tail_messages = max(2, retain * 2)
        split_idx = max(prefix_end, len(messages) - tail_messages)
        while split_idx > prefix_end and _role(messages[split_idx]) == "tool":
            split_idx -= 1
        return prefix_end, split_idx

    def compact_now(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None, *, focus: str = "") -> CompactResult:
        return self.compact_if_needed(messages, tools, force=True, focus=focus)

    def _summarize(self, messages: list[MessageLike], *, focus: str = "") -> tuple[str, bool]:
        if self.llm is not None:
            prompt = _summary_prompt(messages, self.profile.max_summary_input_chars, focus=focus)
            try:
                response = self.llm.chat(
                    [
                        Message(role="system", content="Summarize conversation history for a coding agent. Output only the summary."),
                        Message(role="user", content=prompt),
                    ],
                    tools=None,
                )
                if response.content.strip():
                    return response.content.strip(), False
            except Exception:
                pass
        return deterministic_summary(messages, self.profile.max_summary_input_chars), True


def truncate_tool_content(content: str, limit: int) -> str:
    if limit < 1 or len(content) <= limit:
        return content
    head_len = max(1, limit // 2)
    tail_len = max(1, limit - head_len)
    omitted = len(content) - head_len - tail_len
    return f"{content[:head_len]}\n[... truncated {omitted} chars ...]\n{content[-tail_len:]}"


def estimate_tokens(value: Any) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def deterministic_summary(messages: list[MessageLike], limit: int) -> str:
    lines = ["Previous conversation was compacted deterministically."]
    for message in messages:
        role = _role(message).upper() or "MESSAGE"
        content = _content(message)
        if content:
            lines.append(f"{role}: {_single_line(content, 500)}")
        tool_calls = _tool_calls(message)
        for tool_name, args in tool_calls:
            lines.append(f"ASSISTANT TOOL_CALL {tool_name}: {_single_line(args, 500)}")
        if sum(len(line) for line in lines) >= limit:
            lines.append("Summary input exceeded the configured limit.")
            break
    return "\n".join(lines)


def _summary_prompt(messages: list[MessageLike], limit: int, *, focus: str) -> str:
    body = deterministic_summary(messages, limit)
    focus_text = f"\nFocus: {focus.strip()}\n" if focus.strip() else ""
    return (
        "Compress this coding-agent conversation. Preserve user goals, important constraints, "
        "tool calls/results, edited files, test results, and unresolved issues."
        f"{focus_text}\n=== Conversation ===\n{body}\n=== End ==="
    )


def _role(message: MessageLike) -> str:
    if isinstance(message, Message):
        return message.role
    value = message.get("role", "") if isinstance(message, dict) else ""
    return value if isinstance(value, str) else ""


def _content(message: MessageLike) -> str:
    if isinstance(message, Message):
        return message.content or ""
    value = message.get("content", "") if isinstance(message, dict) else ""
    return value if isinstance(value, str) else ""


def _tool_calls(message: MessageLike) -> list[tuple[str, str]]:
    if isinstance(message, Message):
        return [(call.name, call.arguments_json) for call in message.tool_calls]
    raw_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    calls: list[tuple[str, str]] = []
    if not isinstance(raw_calls, list):
        return calls
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function", {})
        if not isinstance(function, dict):
            continue
        name = function.get("name", "")
        args = function.get("arguments", "")
        if isinstance(name, str):
            calls.append((name, args if isinstance(args, str) else ""))
    return calls


def _single_line(text: str, limit: int) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... truncated"
