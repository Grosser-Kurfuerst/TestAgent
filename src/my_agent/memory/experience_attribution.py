"""Neutral persistence helpers for typed experience attribution fields."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from math import isfinite
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from my_agent.memory.evolver.types import ExperienceMemory


ATTRIBUTION_DECIMAL_PLACES = 6


class AttributionRecordLike(Protocol):
    value: float
    confidence: float
    candidate_count: int
    selected_count: int
    not_selected_count: int
    success_when_selected: float | None
    success_when_candidate_not_selected: float | None
    reward_when_selected: float | None
    reward_when_candidate_not_selected: float | None
    last_used: str


def canonical_attribution_float(value: float, *, field_name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return round(numeric, ATTRIBUTION_DECIMAL_PLACES)


def canonical_optional_attribution_float(
    value: float | None,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return canonical_attribution_float(value, field_name=field_name)


def replace_experience_attribution(
    target: "ExperienceMemory",
    record: AttributionRecordLike,
    *,
    updated_at: datetime | str | None,
) -> "ExperienceMemory":
    """Build and schema-validate the canonical persisted replacement."""
    return replace(
        target,
        attribution_value=canonical_attribution_float(record.value, field_name="value"),
        attribution_confidence=canonical_attribution_float(
            record.confidence,
            field_name="confidence",
        ),
        candidate_count=record.candidate_count,
        selected_count=record.selected_count,
        not_selected_count=record.not_selected_count,
        success_when_selected=canonical_optional_attribution_float(
            record.success_when_selected,
            field_name="success_when_selected",
        ),
        success_when_candidate_not_selected=canonical_optional_attribution_float(
            record.success_when_candidate_not_selected,
            field_name="success_when_candidate_not_selected",
        ),
        reward_when_selected=canonical_optional_attribution_float(
            record.reward_when_selected,
            field_name="reward_when_selected",
        ),
        reward_when_candidate_not_selected=canonical_optional_attribution_float(
            record.reward_when_candidate_not_selected,
            field_name="reward_when_candidate_not_selected",
        ),
        last_used=_attribution_datetime(record.last_used, field_name="last_used"),
        attribution_updated_at=(
            datetime.now(timezone.utc)
            if updated_at in (None, "")
            else _attribution_datetime(updated_at, field_name="attribution_updated_at")
        ),
    )


def _attribution_datetime(value: object, *, field_name: str) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"attribution {field_name} must be an ISO datetime") from exc
    else:
        raise ValueError(f"attribution {field_name} must be an ISO datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"attribution {field_name} must be timezone-aware")
    return parsed


__all__ = [
    "ATTRIBUTION_DECIMAL_PLACES",
    "AttributionRecordLike",
    "canonical_attribution_float",
    "canonical_optional_attribution_float",
    "replace_experience_attribution",
]
