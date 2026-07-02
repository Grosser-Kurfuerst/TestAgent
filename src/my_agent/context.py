from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from my_agent.llm.types import Message, MessageLike, messages_to_openai

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_SHORT_TERM_TOKENS = 24_000
DEFAULT_SHORT_TERM_STORAGE_TOKENS = 108_800
DEFAULT_MEMORY_CONTEXT_TOKENS = 2_000
DEFAULT_TOOL_RESULT_CHARS = 500
DEFAULT_REPO_CONTEXT_BUDGET_TOKENS = 15_360
DEFAULT_TOOL_SCHEMA_BUDGET_TOKENS = 10_240
CORE_TOOL_NAMES = {
    "list_files",
    "read_file",
    "grep",
    "retrieve_context",
    "replace_in_file",
    "write_file",
    "run_tests",
    "git_diff",
    "finish",
}


@dataclass(frozen=True)
class ToolSchemaBudget:
    definitions: list[dict[str, Any]]
    included_names: tuple[str, ...]
    omitted_names: tuple[str, ...]
    budget_tokens: int
    estimated_tokens: int
    over_budget: bool = False

    @property
    def omitted_count(self) -> int:
        return len(self.omitted_names)

    @property
    def included_count(self) -> int:
        return len(self.included_names)


@dataclass(frozen=True)
class ContextBudgetPlan:
    prompt_limit_tokens: int
    fixed_tokens: int
    memory_budget_tokens: int
    long_term_limit: int
    short_term_allowed: int

    def to_trace_payload(self) -> dict[str, int]:
        return {
            "prompt_limit_tokens": self.prompt_limit_tokens,
            "fixed_tokens": self.fixed_tokens,
            "memory_budget_tokens": self.memory_budget_tokens,
            "long_term_limit": self.long_term_limit,
            "short_term_allowed": self.short_term_allowed,
        }


class ContextOverBudgetError(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = dict(payload)
        estimated = self.payload.get("estimated_prompt_tokens", "unknown")
        limit = self.payload.get("compression_trigger_tokens", "unknown")
        super().__init__(
            f"Context prompt exceeds budget: estimated {estimated} tokens >= prompt limit {limit} tokens. "
            "LLM request was not sent."
        )


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


class AgentContextManager:
    """Build LLM context from memory and own context-window decisions."""

    def __init__(self, profile: ContextProfile) -> None:
        self.profile = profile

    @classmethod
    def from_config(cls, config: Any, model: str | None = None) -> "AgentContextManager":
        return cls(ContextProfile.resolve(config, model or getattr(config, "model", "")))

    def estimate_tokens(self, messages: list[MessageLike], tools: list[dict[str, Any]] | None = None) -> int:
        return estimate_tokens({"messages": messages_to_openai(messages), "tools": tools or []})

    def budget_for_messages(
        self,
        *,
        base_messages: list[MessageLike],
        tools: list[dict[str, Any]] | None = None,
    ) -> ContextBudgetPlan:
        prompt_limit_tokens = self.profile.compression_trigger_tokens
        fixed_tokens = self.estimate_tokens(list(base_messages), tools or [])
        memory_budget_tokens = max(0, prompt_limit_tokens - fixed_tokens)
        long_term_limit = _long_term_budget_tokens(memory_budget_tokens, self.profile.memory_context_tokens)
        return ContextBudgetPlan(
            prompt_limit_tokens=prompt_limit_tokens,
            fixed_tokens=fixed_tokens,
            memory_budget_tokens=memory_budget_tokens,
            long_term_limit=long_term_limit,
            short_term_allowed=max(0, memory_budget_tokens - long_term_limit),
        )

    def raise_if_over_budget(
        self,
        *,
        memory: Any,
        estimated_prompt_tokens: int,
        payload: dict[str, Any],
    ) -> None:
        if estimated_prompt_tokens < self.profile.compression_trigger_tokens:
            return
        over_budget_payload = dict(payload)
        over_budget_payload["estimated_prompt_tokens"] = estimated_prompt_tokens
        enriched = self.trace_over_budget(memory=memory, payload=over_budget_payload)
        raise ContextOverBudgetError(enriched)

    def trace_over_budget(self, *, memory: Any, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = self._trace_payload(payload)
        memory.trace_context_event("context.over_budget", enriched)
        return enriched

    def prepare_messages(
        self,
        *,
        base_messages: list[MessageLike],
        query: str,
        tools: list[dict[str, Any]],
        memory: Any,
        force_compact: bool = False,
        focus: str = "",
        tool_budget: ToolSchemaBudget | None = None,
    ) -> tuple[list[MessageLike], Any, Any | None]:
        budget_plan = self.budget_for_messages(base_messages=base_messages, tools=tools)
        prompt_limit_tokens = budget_plan.prompt_limit_tokens
        fixed_tokens = budget_plan.fixed_tokens
        memory_budget_tokens = budget_plan.memory_budget_tokens
        long_term_limit = budget_plan.long_term_limit
        memory_context = memory.build_context_for_query(query, max_tokens=long_term_limit)
        messages_without_short = _inject_memory_context(list(base_messages), memory_context)
        fixed_with_memory_tokens = self.estimate_tokens(messages_without_short, tools)
        short_term_allowed = max(0, prompt_limit_tokens - fixed_with_memory_tokens)
        messages = list(messages_without_short)
        messages.extend(memory.render_short_term_messages())
        estimated_prompt_tokens = self.estimate_tokens(messages, tools)

        compaction = None
        short_term_budgeted = False
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
                memory_context = memory.build_context_for_query(query, max_tokens=long_term_limit)
                messages_without_short = _inject_memory_context(list(base_messages), memory_context)
                fixed_with_memory_tokens = self.estimate_tokens(messages_without_short, tools)
                short_term_allowed = max(0, prompt_limit_tokens - fixed_with_memory_tokens)
                messages = list(messages_without_short)
                messages.extend(memory.render_short_term_messages(max_tokens=short_term_allowed))
                short_term_budgeted = True
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
                            "fixed_tokens": fixed_tokens,
                            "fixed_with_memory_tokens": fixed_with_memory_tokens,
                            "memory_budget_tokens": memory_budget_tokens,
                            "long_term_limit": long_term_limit,
                            "short_term_allowed": short_term_allowed,
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
            elif estimated_prompt_tokens >= self.profile.compression_trigger_tokens:
                messages = list(messages_without_short)
                messages.extend(memory.render_short_term_messages(max_tokens=short_term_allowed))
                short_term_budgeted = True
                estimated_prompt_tokens = self.estimate_tokens(messages, tools)
        final_estimated_tokens = self.estimate_tokens(messages, tools)
        over_budget_payload = None
        if final_estimated_tokens >= self.profile.compression_trigger_tokens:
            over_budget_payload = self.trace_over_budget(
                memory=memory,
                payload={
                    "estimated_prompt_tokens": final_estimated_tokens,
                    "fixed_tokens": fixed_tokens,
                    "fixed_with_memory_tokens": fixed_with_memory_tokens,
                    "memory_budget_tokens": memory_budget_tokens,
                    "long_term_limit": long_term_limit,
                    "short_term_allowed": short_term_allowed,
                },
            )
        memory.trace_context_event(
            "memory.prepared",
            self._trace_payload(
                {
                    "message_count": len(messages),
                    "memory_hits": len(memory_context.hits),
                    "memory_tokens": memory_context.estimated_tokens,
                    "estimated_prompt_tokens": final_estimated_tokens,
                    "fixed_tokens": fixed_tokens,
                    "fixed_with_memory_tokens": fixed_with_memory_tokens,
                    "memory_budget_tokens": memory_budget_tokens,
                    "long_term_limit": long_term_limit,
                    "short_term_allowed": short_term_allowed,
                    "short_term_budgeted": short_term_budgeted,
                    "tool_schema_tokens": _tool_schema_tokens(tools, tool_budget),
                    "tool_schema_budget_tokens": self.profile.tool_schema_budget_tokens,
                    "tools_included": _tool_names(tools, tool_budget),
                    "tools_omitted": list(tool_budget.omitted_names) if tool_budget is not None else [],
                    "tool_schema_over_budget": bool(tool_budget and tool_budget.over_budget),
                    "repo_context_budget_tokens": self.profile.repo_context_budget_tokens,
                    "compacted": bool(compaction and compaction.compacted),
                    "over_budget": over_budget_payload is not None,
                }
            ),
        )
        if over_budget_payload is not None:
            raise ContextOverBudgetError(over_budget_payload)
        return messages, memory_context, compaction

    def _trace_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched.update(
            {
                "context_window": self.profile.max_context_tokens,
                "compression_trigger_tokens": self.profile.compression_trigger_tokens,
                "short_term_storage_token_limit": self.profile.short_term_storage_token_limit,
                "repo_context_budget_tokens": self.profile.repo_context_budget_tokens,
                "tool_schema_budget_tokens": self.profile.tool_schema_budget_tokens,
                "tool_result_char_limit": self.profile.tool_result_char_limit,
                "dynamic_profile_source": self.profile.dynamic_profile_source,
            }
        )
        return enriched


def budget_tool_definitions(tools: list[dict[str, Any]], profile: ContextProfile) -> ToolSchemaBudget:
    budget = max(1, profile.tool_schema_budget_tokens)
    enabled_tools = list(tools)
    core = [tool for tool in enabled_tools if _tool_name(tool) in CORE_TOOL_NAMES]
    non_core = [tool for tool in enabled_tools if _tool_name(tool) not in CORE_TOOL_NAMES]
    ordered = [*core, *non_core]

    included: list[dict[str, Any]] = []
    omitted: list[str] = []
    over_budget = False
    included_ids: set[int] = set()
    for tool in ordered:
        name = _tool_name(tool)
        is_core = name in CORE_TOOL_NAMES
        candidate_tokens = estimate_tokens([*included, tool])
        if not is_core and candidate_tokens > budget:
            if name:
                omitted.append(name)
            continue
        if is_core and candidate_tokens > budget:
            over_budget = True
        included.append(tool)
        included_ids.add(id(tool))

    for tool in enabled_tools:
        if id(tool) not in included_ids:
            name = _tool_name(tool)
            if name and name not in omitted:
                omitted.append(name)

    return ToolSchemaBudget(
        definitions=included,
        included_names=tuple(name for name in (_tool_name(tool) for tool in included) if name),
        omitted_names=tuple(omitted),
        budget_tokens=budget,
        estimated_tokens=estimate_tokens(included),
        over_budget=over_budget or estimate_tokens(included) > budget,
    )


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


def _long_term_budget_tokens(memory_budget_tokens: int, profile_limit: int) -> int:
    if memory_budget_tokens <= 0:
        return 0
    dynamic_limit = max(500, int(memory_budget_tokens * 0.03))
    return max(0, min(profile_limit, dynamic_limit, memory_budget_tokens))


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


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return name if isinstance(name, str) else ""


def _tool_names(tools: list[dict[str, Any]], tool_budget: ToolSchemaBudget | None = None) -> list[str]:
    if tool_budget is not None:
        return list(tool_budget.included_names)
    return [name for name in (_tool_name(tool) for tool in tools) if name]


def _tool_schema_tokens(tools: list[dict[str, Any]], tool_budget: ToolSchemaBudget | None = None) -> int:
    if tool_budget is not None:
        return tool_budget.estimated_tokens
    return estimate_tokens(tools)
