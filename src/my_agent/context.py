from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from my_agent.llm.types import Message, MessageLike, messages_to_openai

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_SHORT_TERM_TOKENS = 24_000
DEFAULT_MEMORY_CONTEXT_TOKENS = 2_000
DEFAULT_TOOL_RESULT_CHARS = 500


@dataclass(frozen=True)
class ContextProfile:
    max_context_tokens: int = DEFAULT_CONTEXT_WINDOW
    response_reserve_tokens: int = 8_000
    compression_buffer_tokens: int = 8_000
    retain_recent_user_turns: int = 3
    max_tool_result_chars: int = 12_000
    max_summary_input_chars: int = 60_000
    short_term_token_limit: int = DEFAULT_SHORT_TERM_TOKENS
    memory_context_tokens: int = DEFAULT_MEMORY_CONTEXT_TOKENS
    tool_result_char_limit: int = DEFAULT_TOOL_RESULT_CHARS
    compression_trigger_ratio: float = 0.8
    dynamic_profile_source: str = "default"

    @property
    def compression_trigger_tokens(self) -> int:
        return _auto_compact_trigger_tokens(
            self.max_context_tokens,
            self.response_reserve_tokens,
            self.compression_buffer_tokens,
        )

    @classmethod
    def from_config(cls, config: Any) -> "ContextProfile":
        return cls.resolve(config, getattr(config, "model", ""))

    @classmethod
    def resolve(cls, config: Any, model: str | None = None) -> "ContextProfile":
        configured_window = _positive_int(getattr(config, "context_window", DEFAULT_CONTEXT_WINDOW), DEFAULT_CONTEXT_WINDOW)
        model_window = _infer_model_context_window(model or getattr(config, "model", ""))
        if _is_explicit(config, "context_window", DEFAULT_CONTEXT_WINDOW, "context_window_explicit"):
            window = configured_window
            source = "config"
        elif model_window is not None:
            window = model_window
            source = "model"
        else:
            window = DEFAULT_CONTEXT_WINDOW
            source = "default"

        short_term_limit = (
            _positive_int(getattr(config, "memory_short_term_tokens", DEFAULT_SHORT_TERM_TOKENS), DEFAULT_SHORT_TERM_TOKENS)
            if _is_explicit(
                config,
                "memory_short_term_tokens",
                DEFAULT_SHORT_TERM_TOKENS,
                "memory_short_term_tokens_explicit",
            )
            else _short_term_budget(window)
        )
        memory_context_tokens = (
            _positive_int(getattr(config, "memory_context_tokens", DEFAULT_MEMORY_CONTEXT_TOKENS), DEFAULT_MEMORY_CONTEXT_TOKENS)
            if _is_explicit(
                config,
                "memory_context_tokens",
                DEFAULT_MEMORY_CONTEXT_TOKENS,
                "memory_context_tokens_explicit",
            )
            else _memory_context_tokens(window)
        )
        tool_result_limit = (
            _positive_int(getattr(config, "memory_tool_result_chars", DEFAULT_TOOL_RESULT_CHARS), DEFAULT_TOOL_RESULT_CHARS)
            if _is_explicit(
                config,
                "memory_tool_result_chars",
                DEFAULT_TOOL_RESULT_CHARS,
                "memory_tool_result_chars_explicit",
            )
            else _tool_result_char_limit(window)
        )
        response_reserve = (
            _positive_int(getattr(config, "response_reserve_tokens", 8_000), 8_000)
            if _is_explicit(
                config,
                "response_reserve_tokens",
                8_000,
                "response_reserve_tokens_explicit",
            )
            else _response_reserve_tokens(window)
        )
        compression_buffer = (
            _positive_int(getattr(config, "compression_buffer_tokens", 8_000), 8_000)
            if _is_explicit(
                config,
                "compression_buffer_tokens",
                8_000,
                "compression_buffer_tokens_explicit",
            )
            else _compression_buffer_tokens(window)
        )
        return cls(
            max_context_tokens=window,
            response_reserve_tokens=response_reserve,
            compression_buffer_tokens=compression_buffer,
            retain_recent_user_turns=getattr(config, "retain_recent_user_turns", 3),
            max_tool_result_chars=getattr(config, "max_tool_result_chars", 12_000),
            max_summary_input_chars=getattr(config, "max_summary_input_chars", 60_000),
            short_term_token_limit=short_term_limit,
            memory_context_tokens=memory_context_tokens,
            tool_result_char_limit=tool_result_limit,
            compression_trigger_ratio=getattr(config, "memory_compression_trigger_ratio", 0.8),
            dynamic_profile_source=source,
        )


class AgentContextManager:
    """Build LLM context from memory and own context-window decisions."""

    def __init__(self, profile: ContextProfile) -> None:
        self.profile = profile

    @classmethod
    def from_config(cls, config: Any, model: str | None = None) -> "AgentContextManager":
        return cls(ContextProfile.resolve(config, model or getattr(config, "model", "")))

    def estimate_tokens(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> int:
        return estimate_tokens({"messages": messages_to_openai(messages), "tools": tools or []})

    def prepare_messages(
        self,
        *,
        base_messages: list[MessageLike],
        query: str,
        tools: list[dict[str, Any]],
        memory: Any,
        force_compact: bool = False,
        focus: str = "",
    ) -> tuple[list[MessageLike], Any, Any | None]:
        memory_context = memory.build_context_for_query(query, max_tokens=self.profile.memory_context_tokens)
        messages = _inject_memory_context(list(base_messages), memory_context)
        messages.extend(memory.render_short_term_messages())
        estimated_prompt_tokens = self.estimate_tokens(messages, tools)

        compaction = None
        if force_compact or estimated_prompt_tokens >= self.profile.compression_trigger_tokens:
            compaction = memory.compact_short_term(
                tools=tools,
                force=True,
                focus=focus,
                before_tokens=estimated_prompt_tokens,
                trace_completed=False,
                trigger_tokens=self.profile.compression_trigger_tokens,
            )
            if compaction and compaction.compacted:
                memory_context = memory.build_context_for_query(query, max_tokens=self.profile.memory_context_tokens)
                messages = _inject_memory_context(list(base_messages), memory_context)
                messages.extend(memory.render_short_term_messages())
                after_prompt_tokens = self.estimate_tokens(messages, tools)
                memory.trace_context_event(
                    "memory.compacted",
                    self._trace_payload(
                        {
                            "before_tokens": compaction.before_tokens,
                            "after_tokens": after_prompt_tokens,
                            "map_count": compaction.map_count,
                            "reduce_used": compaction.reduce_used,
                            "extracted_facts": compaction.extracted_facts,
                            "fallback": compaction.fallback,
                            "estimated_prompt_tokens": after_prompt_tokens,
                        }
                    ),
                )
                compaction = compaction.__class__(
                    compacted=compaction.compacted,
                    before_tokens=compaction.before_tokens,
                    after_tokens=after_prompt_tokens,
                    map_count=compaction.map_count,
                    reduce_used=compaction.reduce_used,
                    extracted_facts=compaction.extracted_facts,
                    fallback=compaction.fallback,
                )
        memory.trace_context_event(
            "memory.prepared",
            self._trace_payload(
                {
                    "message_count": len(messages),
                    "memory_hits": len(memory_context.hits),
                    "memory_tokens": memory_context.estimated_tokens,
                    "estimated_prompt_tokens": self.estimate_tokens(messages, tools),
                    "compacted": bool(compaction and compaction.compacted),
                }
            ),
        )
        return messages, memory_context, compaction

    def _trace_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched.update(
            {
                "context_window": self.profile.max_context_tokens,
                "compression_trigger_tokens": self.profile.compression_trigger_tokens,
                "short_term_token_limit": self.profile.short_term_token_limit,
                "tool_result_char_limit": self.profile.tool_result_char_limit,
                "dynamic_profile_source": self.profile.dynamic_profile_source,
            }
        )
        return enriched


def estimate_tokens(value: Any) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def _inject_memory_context(base_messages: list[MessageLike], context: Any) -> list[MessageLike]:
    injected_text = getattr(context, "injected_text", "")
    if not injected_text:
        return base_messages
    memory_message = Message(role="system", content=injected_text)
    if base_messages and _role(base_messages[0]) == "system":
        return [base_messages[0], memory_message, *base_messages[1:]]
    return [memory_message, *base_messages]


def _is_explicit(config: Any, field_name: str, default: int, explicit_attr: str) -> bool:
    explicit = getattr(config, explicit_attr, False)
    if isinstance(explicit, bool) and explicit:
        return True
    value = getattr(config, field_name, default)
    return isinstance(value, int) and not isinstance(value, bool) and value != default


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def _short_term_budget(window: int) -> int:
    return max(4_000, int(window * 0.45))


def _memory_context_tokens(window: int) -> int:
    return _clamp(int(max(1, window) * 0.01), 1_000, 20_000)


def _tool_result_char_limit(window: int) -> int:
    return max(4_000, min(60_000, window // 4))


def _response_reserve_tokens(window: int) -> int:
    return _clamp(int(max(1, window) * 0.10), 4_000, 100_000)


def _compression_buffer_tokens(window: int) -> int:
    return _clamp(int(max(1, window) * 0.05), 2_000, 50_000)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _auto_compact_trigger_tokens(
    window: int,
    response_reserve_tokens: int,
    compression_buffer_tokens: int,
) -> int:
    safe_window = max(1, window)
    summary_reserve = min(
        safe_window - 1,
        _positive_int(response_reserve_tokens, _response_reserve_tokens(safe_window)),
    )
    buffer = min(
        safe_window - 1,
        _positive_int(compression_buffer_tokens, _compression_buffer_tokens(safe_window)),
    )
    trigger = safe_window - summary_reserve - buffer
    return max(1, min(safe_window - 1, trigger))


def _infer_model_context_window(model: str) -> int | None:
    normalized = (model or "").strip().lower()
    if not normalized:
        return None
    for marker, value in (
        ("1000k", 1_000_000),
        ("1m", 1_000_000),
        ("256k", 256_000),
        ("200k", 200_000),
        ("128k", 128_000),
        ("64k", 64_000),
        ("32k", 32_000),
        ("16k", 16_000),
        ("8k", 8_000),
    ):
        if marker in normalized:
            return value
    if any(marker in normalized for marker in ("gpt-4.1", "deepseek-v4")):
        return 1_000_000
    if any(marker in normalized for marker in ("o3", "o4-mini")):
        return 200_000
    if any(marker in normalized for marker in ("kimi-k2.6", "stepfun")):
        return 256_000
    if any(marker in normalized for marker in ("gpt-4o", "gpt-4-turbo", "claude")):
        return 128_000
    if normalized == "fake":
        return DEFAULT_CONTEXT_WINDOW
    return None


def _role(message: MessageLike) -> str:
    if isinstance(message, Message):
        return message.role
    value = message.get("role", "") if isinstance(message, dict) else ""
    return value if isinstance(value, str) else ""
