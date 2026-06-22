from __future__ import annotations


def single_line(text: str, limit: int) -> str:
    normalized = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def terminal_summary_text(result: str, error: str, fallback: str) -> str:
    return result.strip() or error.strip() or fallback

