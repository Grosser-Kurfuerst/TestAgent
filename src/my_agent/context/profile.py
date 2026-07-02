from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_SHORT_TERM_TOKENS = 24_000
DEFAULT_SHORT_TERM_STORAGE_TOKENS = 108_800
DEFAULT_MEMORY_CONTEXT_TOKENS = 2_000
DEFAULT_TOOL_RESULT_CHARS = 500
DEFAULT_REPO_CONTEXT_BUDGET_TOKENS = 15_360
DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS = 10_240


@dataclass(frozen=True)
class ContextProfile:
    max_context_tokens: int = DEFAULT_CONTEXT_WINDOW
    response_reserve_tokens: int = 8_000
    compression_buffer_tokens: int = 8_000
    retain_recent_user_turns: int = 3
    max_tool_result_chars: int = 12_000
    max_summary_input_chars: int = 60_000
    short_term_storage_token_limit: int = DEFAULT_SHORT_TERM_STORAGE_TOKENS
    memory_context_tokens: int = DEFAULT_MEMORY_CONTEXT_TOKENS
    tool_result_char_limit: int = DEFAULT_TOOL_RESULT_CHARS
    repo_context_budget_tokens: int = DEFAULT_REPO_CONTEXT_BUDGET_TOKENS
    tool_schema_budget_tokens: int = DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS
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
        dynamic_short_term_storage_limit = _short_term_storage_budget(
            window,
            response_reserve,
            compression_buffer,
        )
        short_term_storage_limit = (
            _positive_int(
                getattr(config, "memory_short_term_tokens", dynamic_short_term_storage_limit),
                dynamic_short_term_storage_limit,
            )
            if _is_explicit(
                config,
                "memory_short_term_tokens",
                DEFAULT_SHORT_TERM_TOKENS,
                "memory_short_term_tokens_explicit",
            )
            else dynamic_short_term_storage_limit
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
        repo_context_budget = (
            _positive_int(
                getattr(config, "repo_context_budget_tokens", DEFAULT_REPO_CONTEXT_BUDGET_TOKENS),
                DEFAULT_REPO_CONTEXT_BUDGET_TOKENS,
            )
            if _is_explicit(
                config,
                "repo_context_budget_tokens",
                DEFAULT_REPO_CONTEXT_BUDGET_TOKENS,
                "repo_context_budget_tokens_explicit",
            )
            else _repo_context_budget_tokens(window)
        )
        tool_schema_budget = (
            _positive_int(
                getattr(config, "tool_schema_budget_tokens", DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS),
                DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS,
            )
            if _is_explicit(
                config,
                "tool_schema_budget_tokens",
                DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS,
                "tool_schema_budget_tokens_explicit",
            )
            else _tool_schema_budget_tokens(window)
        )
        return cls(
            max_context_tokens=window,
            response_reserve_tokens=response_reserve,
            compression_buffer_tokens=compression_buffer,
            retain_recent_user_turns=getattr(config, "retain_recent_user_turns", 3),
            max_tool_result_chars=getattr(config, "max_tool_result_chars", 12_000),
            max_summary_input_chars=getattr(config, "max_summary_input_chars", 60_000),
            short_term_storage_token_limit=short_term_storage_limit,
            memory_context_tokens=memory_context_tokens,
            tool_result_char_limit=tool_result_limit,
            repo_context_budget_tokens=repo_context_budget,
            tool_schema_budget_tokens=tool_schema_budget,
            compression_trigger_ratio=getattr(config, "memory_compression_trigger_ratio", 0.8),
            dynamic_profile_source=source,
        )


def long_term_budget_tokens(memory_budget_tokens: int, profile_limit: int) -> int:
    if memory_budget_tokens <= 0:
        return 0
    dynamic_limit = max(500, int(memory_budget_tokens * 0.03))
    return max(0, min(profile_limit, dynamic_limit, memory_budget_tokens))


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


def _short_term_storage_budget(
    window: int,
    response_reserve_tokens: int,
    compression_buffer_tokens: int,
) -> int:
    return _auto_compact_trigger_tokens(window, response_reserve_tokens, compression_buffer_tokens)


def _memory_context_tokens(window: int) -> int:
    return _clamp(int(max(1, window) * 0.01), 1_000, 20_000)


def _tool_result_char_limit(window: int) -> int:
    return max(4_000, min(60_000, window // 4))


def _repo_context_budget_tokens(window: int) -> int:
    return _clamp(int(max(1, window) * 0.12), 12_000, 120_000)


def _tool_schema_budget_tokens(window: int) -> int:
    return _clamp(int(max(1, window) * 0.08), 8_000, 80_000)


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
