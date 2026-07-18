"""Validation and parsing helpers for maintenance contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any
import math

from my_agent.memory.evolver.maintenance.contracts import MaintenancePlanError
from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.experience.serialization import experience_from_dict


def _require_aware_datetime(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value


def _parse_datetime(value: str) -> datetime | None:
    if not str(value or ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _validate_range(name: str, value: Any, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not minimum <= float(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _validate_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_nonnegative_int_mapping(
    payload: Mapping[str, Any],
    *,
    expected_fields: frozenset[str],
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise MaintenancePlanError(f"{name} must be an object")
    actual_fields = frozenset(payload)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise MaintenancePlanError(
            f"{name} fields mismatch: missing={missing}, extra={extra}"
        )
    for field_name in sorted(expected_fields):
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MaintenancePlanError(
                f"{name}.{field_name} must be a non-negative integer"
            )


def _validate_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_evidence_values(
    *,
    value: Any,
    confidence: Any,
    candidate_count: Any,
    selected_count: Any,
    not_selected_count: Any,
    writer_confidence: Any,
    has_attribution: Any,
    last_used: str,
) -> None:
    if not isinstance(has_attribution, bool):
        raise MaintenancePlanError("maintenance evidence has_attribution must be boolean")
    for name, number, minimum, maximum in (
        ("value", value, -1.0, 1.0),
        ("confidence", confidence, 0.0, 1.0),
        ("writer_confidence", writer_confidence, 0.0, 1.0),
    ):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or not minimum <= float(number) <= maximum
        ):
            raise MaintenancePlanError(
                f"maintenance evidence {name} must be finite and between {minimum} and {maximum}"
            )
    for name, count in (
        ("candidate_count", candidate_count),
        ("selected_count", selected_count),
        ("not_selected_count", not_selected_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise MaintenancePlanError(
                f"maintenance evidence {name} must be a non-negative integer"
            )
    if candidate_count != selected_count + not_selected_count:
        raise MaintenancePlanError("maintenance evidence candidate counts are inconsistent")
    if not has_attribution and any(
        (
            float(value) != 0.0,
            float(confidence) != 0.0,
            candidate_count != 0,
            selected_count != 0,
            not_selected_count != 0,
        )
    ):
        raise MaintenancePlanError(
            "maintenance evidence without attribution must use neutral attribution values"
        )
    if last_used and _parse_datetime(last_used) is None:
        raise MaintenancePlanError(
            "maintenance evidence last_used must be a timezone-aware timestamp"
        )


def _evidence_float(value: Any, name: str) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaintenancePlanError(f"maintenance evidence {name} must be numeric")
    return float(value)


def _evidence_int(value: Any, name: str) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise MaintenancePlanError(f"maintenance evidence {name} must be an integer")
    return value


def _evidence_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise MaintenancePlanError("maintenance evidence has_attribution must be boolean")
    return value


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaintenancePlanError("operation redundancy_score must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MaintenancePlanError("operation redundancy_score must be finite")
    return result


def _required_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MaintenancePlanError(f"{name} must be an integer")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise MaintenancePlanError("operation string arrays must be arrays")
    if any(not isinstance(item, str) for item in value):
        raise MaintenancePlanError("operation string arrays must contain strings")
    return tuple(value)


def _mapping_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise MaintenancePlanError("operation mapping fields must be arrays")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise MaintenancePlanError("operation mapping arrays must contain objects")
        result.append(dict(item))
    return tuple(result)


def _experience_tuple(value: Any, name: str) -> tuple[ExperienceMemory, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise MaintenancePlanError(f"operation {name} must be an array")
    result: list[ExperienceMemory] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise MaintenancePlanError(f"operation {name} must contain objects")
        try:
            result.append(experience_from_dict(item))
        except (TypeError, ValueError) as exc:
            raise MaintenancePlanError(
                f"operation {name} contains an invalid experience"
            ) from exc
    return tuple(result)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaintenancePlanError(f"{name} must be an object")
    return value


def _valid_tier(value: str) -> ExperienceTier | None:
    try:
        return ExperienceTier(str(value))
    except ValueError:
        return None


__all__: list[str] = []
