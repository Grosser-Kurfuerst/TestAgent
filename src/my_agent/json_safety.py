"""Strict JSON decoding helpers for persisted trust-boundary artifacts."""

from __future__ import annotations

from typing import Any, NoReturn
import json


def loads_json_strict(text: str) -> Any:
    """Decode standard JSON while rejecting duplicate object keys."""
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonstandard_constant,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


__all__ = ["loads_json_strict"]
