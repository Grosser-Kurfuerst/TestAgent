from __future__ import annotations

"""Shared JSONL reading with strict and report-friendly modes."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonlRecord = tuple[int, dict[str, Any]]


@dataclass(frozen=True)
class JsonlReadResult:
    records: list[JsonlRecord]
    errors: tuple[str, ...] = ()

    @property
    def skipped(self) -> int:
        return len(self.errors)


def read_jsonl(
    path: str | Path,
    *,
    strict: bool = True,
    allow_single: bool = True,
) -> JsonlReadResult:
    """Read JSON object records from a JSONL file."""

    source = Path(path)
    errors: list[str] = []
    if not source.exists():
        message = f"JSONL input file not found: {source}"
        if strict:
            raise FileNotFoundError(message)
        return JsonlReadResult(records=[], errors=(message,))

    raw = source.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return JsonlReadResult(records=[])

    if allow_single:
        parsed = _parse_single_json(raw)
        if parsed is not _NOT_SINGLE_JSON:
            records = _records_from_single_json(parsed, source, strict=strict, errors=errors)
            return JsonlReadResult(records=records, errors=tuple(errors))

    records: list[JsonlRecord] = []
    for line_num, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            _handle_error(f"{source}:{line_num}: invalid JSONL record", strict=strict, errors=errors, cause=exc)
            continue
        if not isinstance(item, dict):
            _handle_error(f"{source}:{line_num}: expected a JSON object", strict=strict, errors=errors)
            continue
        records.append((line_num, item))
    return JsonlReadResult(records=records, errors=tuple(errors))


_NOT_SINGLE_JSON = object()


def _parse_single_json(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _NOT_SINGLE_JSON


def _records_from_single_json(
    parsed: object,
    path: Path,
    *,
    strict: bool,
    errors: list[str],
) -> list[JsonlRecord]:
    if isinstance(parsed, dict):
        return [(1, parsed)]
    if isinstance(parsed, list):
        records: list[JsonlRecord] = []
        for idx, item in enumerate(parsed, start=1):
            if isinstance(item, dict):
                records.append((idx, item))
            else:
                _handle_error(f"{path}:{idx}: expected a JSON object", strict=strict, errors=errors)
        return records
    _handle_error(f"{path}: expected a JSON object or JSONL records", strict=strict, errors=errors)
    return []


def _handle_error(
    message: str,
    *,
    strict: bool,
    errors: list[str],
    cause: Exception | None = None,
) -> None:
    if strict:
        raise ValueError(message) from cause
    errors.append(message)


__all__ = ["JsonlReadResult", "JsonlRecord", "read_jsonl"]
