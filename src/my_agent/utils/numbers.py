from __future__ import annotations


def positive_or_default(value: int | None, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return max(1, default)

