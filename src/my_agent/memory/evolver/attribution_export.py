"""Strict JSONL I/O for paper attribution evidence and events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable, TypeVar
import json

from my_agent.memory.evolver.attribution_schema import CandidateExposure, PaperAttributionRecord
from my_agent.policy.identity import canonical_json_bytes


T = TypeVar("T")


def write_candidate_exposures(
    exposures: Iterable[CandidateExposure],
    path: str | Path,
) -> Path:
    return _write_jsonl(
        path,
        (exposure.to_dict() for exposure in exposures),
        sort_key=lambda payload: (
            int(payload["collection_round"]),
            int(payload["task_ordinal"]),
            str(payload["task_id"]),
            str(payload["memory_id"]),
        ),
    )


def load_candidate_exposures(path: str | Path) -> tuple[CandidateExposure, ...]:
    return _load_jsonl(path, CandidateExposure.from_dict)


def write_attribution_events(
    records: Iterable[PaperAttributionRecord],
    path: str | Path,
) -> Path:
    return _write_jsonl(
        path,
        (record.to_dict() for record in records),
        sort_key=lambda payload: (str(payload["memory_project_key"]), str(payload["memory_id"])),
    )


def load_attribution_events(path: str | Path) -> tuple[PaperAttributionRecord, ...]:
    return _load_jsonl(path, PaperAttributionRecord.from_dict)


def _write_jsonl(
    path: str | Path,
    payloads: Iterable[Mapping[str, Any]],
    *,
    sort_key: Callable[[Mapping[str, Any]], Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted((dict(payload) for payload in payloads), key=sort_key)
    with output.open("w", encoding="utf-8") as handle:
        for payload in ordered:
            handle.write(canonical_json_bytes(payload).decode("utf-8") + "\n")
    return output


def _load_jsonl(path: str | Path, loader: Callable[[Mapping[str, Any]], T]) -> tuple[T, ...]:
    records: list[T] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid attribution JSON at line {line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"attribution JSON line {line_number} must be an object")
            records.append(loader(payload))
    return tuple(records)


__all__ = [
    "load_attribution_events",
    "load_candidate_exposures",
    "write_attribution_events",
    "write_candidate_exposures",
]
