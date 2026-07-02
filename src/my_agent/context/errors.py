from __future__ import annotations

from typing import Any


class ContextOverBudgetError(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = dict(payload)
        estimated = self.payload.get("estimated_prompt_tokens", "unknown")
        limit = self.payload.get("compression_trigger_tokens", "unknown")
        super().__init__(
            f"Context prompt exceeds budget: estimated {estimated} tokens >= prompt limit {limit} tokens. "
            "LLM request was not sent."
        )
