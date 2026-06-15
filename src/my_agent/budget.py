from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from my_agent.llm.types import ChatUsage, LLMToolCall
from my_agent.tools.execution import ToolExecutionResult


@dataclass
class AgentBudget:
    max_iterations: int = 50
    max_tool_calls: int = 200
    max_elapsed_seconds: int = 1800
    token_budget: int | None = None
    stagnation_window: int = 3
    repeated_failure_window: int = 3
    started_at: float = field(default_factory=time.monotonic)
    iterations: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _recent_signatures: deque[str] = field(default_factory=deque)
    _recent_failures: deque[str] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1.")
        if self.max_elapsed_seconds < 1:
            raise ValueError("max_elapsed_seconds must be >= 1.")
        if self.stagnation_window < 2:
            raise ValueError("stagnation_window must be >= 2.")
        if self.repeated_failure_window < 2:
            raise ValueError("repeated_failure_window must be >= 2.")

    @classmethod
    def from_config(cls, config: Any, *, max_steps: int) -> "AgentBudget":
        return cls(
            max_iterations=getattr(config, "max_iterations", max_steps),
            max_tool_calls=min(getattr(config, "max_tool_calls", max_steps), max_steps),
            max_elapsed_seconds=getattr(config, "max_elapsed_seconds", 1800),
            token_budget=getattr(config, "token_budget", None),
            stagnation_window=getattr(config, "stagnation_window", 3),
            repeated_failure_window=getattr(config, "repeated_failure_window", 3),
        )

    def begin_iteration(self) -> None:
        self.iterations += 1

    def record_usage(self, usage: ChatUsage) -> None:
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens

    def record_tool_calls(self, tool_calls: list[LLMToolCall]) -> None:
        if not tool_calls:
            self._recent_signatures.clear()
            return
        signature = ";".join(_signature(call.name, call.arguments_json) for call in tool_calls)
        self._recent_signatures.append(signature)
        while len(self._recent_signatures) > self.stagnation_window:
            self._recent_signatures.popleft()

    def record_tool_results(self, results: list[ToolExecutionResult], tool_calls: list[LLMToolCall] | None = None) -> None:
        self.tool_calls += len(results)
        signatures = {call.id: _signature(call.name, call.arguments_json) for call in tool_calls or []}
        for result in results:
            if result.ok:
                continue
            self._recent_failures.append(signatures.get(result.id, result.name))
            while len(self._recent_failures) > self.repeated_failure_window:
                self._recent_failures.popleft()

    def check_before_llm(self) -> str | None:
        return self._check(include_tool_count=False)

    def check_after_tools(self) -> str | None:
        return self._check(include_tool_count=True)

    def _check(self, *, include_tool_count: bool) -> str | None:
        if time.monotonic() - self.started_at >= self.max_elapsed_seconds:
            return "max_elapsed_seconds"
        if self.iterations >= self.max_iterations:
            return "max_iterations"
        if include_tool_count and self.tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if self.token_budget is not None and self.input_tokens + self.output_tokens >= self.token_budget:
            return "token_budget_exceeded"
        if self._repeated_failure():
            return "repeated_tool_failure"
        if self._stagnated():
            return "stagnation_detected"
        return None

    def _stagnated(self) -> bool:
        if len(self._recent_signatures) < self.stagnation_window:
            return False
        first = self._recent_signatures[0]
        return all(signature == first for signature in self._recent_signatures)

    def _repeated_failure(self) -> bool:
        if len(self._recent_failures) < self.repeated_failure_window:
            return False
        first = self._recent_failures[0]
        return all(signature == first for signature in self._recent_failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_elapsed_seconds": self.max_elapsed_seconds,
            "token_budget": self.token_budget,
            "stagnation_window": self.stagnation_window,
            "repeated_failure_window": self.repeated_failure_window,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def _signature(name: str, arguments_json: str) -> str:
    try:
        parsed = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        parsed = arguments_json
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{name}:{canonical}"
