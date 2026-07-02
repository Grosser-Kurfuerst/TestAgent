from __future__ import annotations

from typing import Any

__all__ = [
    "AgentBase",
    "AgentBudget",
    "AgentFactory",
    "BufferedEventSink",
    "CancelledError",
    "CancellationToken",
    "CodingAgentRuntime",
    "EventBuffer",
    "EventSink",
    "MemoryTraceSnapshot",
    "RunContext",
    "run_agent",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentBase", "EventSink", "MemoryTraceSnapshot", "RunContext"}:
        from my_agent.runtime import base

        return getattr(base, name)
    if name == "AgentBudget":
        from my_agent.runtime.budget import AgentBudget

        return AgentBudget
    if name in {"CancelledError", "CancellationToken"}:
        from my_agent.runtime import cancellation

        return getattr(cancellation, name)
    if name in {"BufferedEventSink", "EventBuffer"}:
        from my_agent.runtime import events

        return getattr(events, name)
    if name == "AgentFactory":
        from my_agent.runtime.factory import AgentFactory

        return AgentFactory
    if name in {"CodingAgentRuntime", "run_agent"}:
        from my_agent.runtime import runner

        return getattr(runner, name)
    raise AttributeError(f"module 'my_agent.runtime' has no attribute {name!r}")
