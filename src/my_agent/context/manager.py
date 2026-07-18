from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from my_agent.context.errors import ContextOverBudgetError
from my_agent.context.profile import ContextProfile, long_term_budget_tokens
from my_agent.context.tokens import estimate_tokens
from my_agent.context.tool_budget import ToolSchemaBudget, tool_name
from my_agent.llm.types import Message, MessageLike, messages_to_openai
from my_agent.memory.api import MemoryService


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
        long_term_limit = long_term_budget_tokens(memory_budget_tokens, self.profile.memory_context_tokens)
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
        memory: MemoryService,
        estimated_prompt_tokens: int,
        payload: dict[str, Any],
    ) -> None:
        if estimated_prompt_tokens < self.profile.compression_trigger_tokens:
            return
        over_budget_payload = dict(payload)
        over_budget_payload["estimated_prompt_tokens"] = estimated_prompt_tokens
        enriched = self.trace_over_budget(memory=memory, payload=over_budget_payload)
        raise ContextOverBudgetError(enriched)

    def trace_over_budget(
        self,
        *,
        memory: MemoryService,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        enriched = self._trace_payload(payload)
        memory.trace_context_event("context.over_budget", enriched)
        return enriched

    def prepare_messages(
        self,
        *,
        base_messages: list[MessageLike],
        query: str,
        tools: list[dict[str, Any]],
        memory: MemoryService,
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


def _inject_memory_context(base_messages: list[MessageLike], context: Any) -> list[MessageLike]:
    injected_text = getattr(context, "injected_text", "")
    if not injected_text:
        return base_messages
    memory_message = Message(role="system", content=injected_text)
    if base_messages and _role(base_messages[0]) == "system":
        return [base_messages[0], memory_message, *base_messages[1:]]
    return [memory_message, *base_messages]


def _role(message: MessageLike) -> str:
    if isinstance(message, Message):
        return message.role
    value = message.get("role", "") if isinstance(message, dict) else ""
    return value if isinstance(value, str) else ""


def _tool_names(tools: list[dict[str, Any]], tool_budget: ToolSchemaBudget | None = None) -> list[str]:
    if tool_budget is not None:
        return list(tool_budget.included_names)
    return [name for name in (tool_name(tool) for tool in tools) if name]


def _tool_schema_tokens(tools: list[dict[str, Any]], tool_budget: ToolSchemaBudget | None = None) -> int:
    if tool_budget is not None:
        return tool_budget.estimated_tokens
    return estimate_tokens(tools)
