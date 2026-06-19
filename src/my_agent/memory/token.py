from __future__ import annotations

import json
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Rough token estimate for memory accounting.

    Strings are estimated with a mixed Chinese/English heuristic (~1.5
    characters per token for CJK, ~4 characters per token otherwise) matching
    the paicli reference. Arbitrary objects are serialized to JSON first and
    then estimated at 4 characters per token, mirroring the existing
    :func:`my_agent.context.estimate_tokens` fallback.
    """
    if isinstance(value, str):
        return _estimate_text_tokens(value)
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, default=str)
        return max(1, len(text) // 4)
    return max(1, len(str(value)) // 4) if value is not None else 0


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    chinese = sum(1 for char in text if "一" <= char <= "鿿")
    other = len(text) - chinese
    # ceil(chinese/1.5 + other/4) without importing math.ceil to stay tiny.
    raw = chinese / 1.5 + other / 4.0
    tokens = int(raw) + (1 if raw > int(raw) else 0)
    return max(1, tokens)


def entry_token_count(content: str) -> int:
    """Convenience estimator used when building memory entries."""
    return estimate_tokens(content)


__all__ = ["entry_token_count", "estimate_tokens"]
