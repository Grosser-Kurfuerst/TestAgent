from __future__ import annotations

import json
import re
from typing import Any, Mapping

from my_agent.team.types import ReviewDecision


def parse_review_decision(raw_text: str) -> ReviewDecision:
    raw = str(raw_text or "")
    stripped = _strip_json_code_fence(raw)
    if not stripped:
        return ReviewDecision(approved=False, raw=raw, parse_error="empty_review")

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return ReviewDecision(approved=False, raw=raw, parse_error=f"invalid_json: {exc}")

    if not isinstance(payload, Mapping):
        return ReviewDecision(approved=False, raw=raw, parse_error="review_must_be_object")

    approved = payload.get("approved")
    if not isinstance(approved, bool):
        return ReviewDecision(
            approved=False,
            summary=_string(payload.get("summary")),
            issues=tuple(_string_list(payload.get("issues"))),
            suggestions=tuple(_string_list(payload.get("suggestions"))),
            raw=raw,
            parse_error="missing_approved",
        )

    return ReviewDecision(
        approved=approved,
        summary=_string(payload.get("summary")),
        issues=tuple(_string_list(payload.get("issues"))),
        suggestions=tuple(_string_list(payload.get("suggestions"))),
        raw=raw,
    )


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result
