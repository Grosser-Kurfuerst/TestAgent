"""Formal maintenance intent/completion journal schema and IO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os

from filelock import FileLock

from my_agent.json_safety import loads_json_strict
from my_agent.memory.evolver.maintenance.history_io import history_lock_path
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256, require_sha256

FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION = "opd-formal-maintenance-history-v1"
MAINTENANCE_HISTORY_FILENAME = "maintenance_history.jsonl"


@dataclass(frozen=True)
class FormalMaintenanceHistoryState:
    intent: Mapping[str, Any] | None = None
    completion: Mapping[str, Any] | None = None


def formal_maintenance_plan_id(
    *,
    cadence_id: str,
    before_revision: str,
    expected_after_revision: str,
    operations: tuple[Mapping[str, Any], ...],
) -> str:
    require_sha256(cadence_id, field_name="cadence_id")
    _require_nonblank(before_revision, "before_revision")
    _require_nonblank(expected_after_revision, "expected_after_revision")
    return canonical_sha256({
        "schema_version": FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION,
        "cadence_id": cadence_id,
        "before_revision": before_revision,
        "expected_after_revision": expected_after_revision,
        "operations": list(operations),
    })


def formal_maintenance_transaction_id(*, cadence_id: str, plan_id: str) -> str:
    require_sha256(cadence_id, field_name="cadence_id")
    require_sha256(plan_id, field_name="plan_id")
    return canonical_sha256({
        "schema_version": FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION,
        "record_type": "transaction_identity",
        "cadence_id": cadence_id,
        "plan_id": plan_id,
    })


def append_formal_maintenance_history(
    path: str | Path,
    record: Mapping[str, Any],
) -> Path:
    payload = _validate_formal_history_record(record)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(history_lock_path(output)), timeout=30.0):
        with output.open("ab") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return output


def load_formal_maintenance_history(
    path: str | Path,
    *,
    cadence_id: str,
) -> FormalMaintenanceHistoryState:
    require_sha256(cadence_id, field_name="cadence_id")
    source = Path(path)
    if not source.exists():
        return FormalMaintenanceHistoryState()
    intent: Mapping[str, Any] | None = None
    completion: Mapping[str, Any] | None = None
    with FileLock(str(history_lock_path(source)), timeout=30.0):
        for line_no, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                candidate = loads_json_strict(raw_line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid maintenance history JSON at line {line_no}") from exc
            if not isinstance(candidate, Mapping):
                raise ValueError(f"invalid maintenance history record at line {line_no}")
            if candidate.get("schema_version") != FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION:
                continue
            parsed = _validate_formal_history_record(candidate)
            if parsed["cadence_id"] != cadence_id:
                continue
            if parsed["record_type"] == "intent":
                if intent is not None:
                    raise ValueError(f"duplicate formal maintenance intent for {cadence_id}")
                intent = parsed
            else:
                if completion is not None:
                    raise ValueError(f"duplicate formal maintenance completion for {cadence_id}")
                completion = parsed
    if intent is not None and completion is not None:
        _require_matching_history_pair(intent, completion)
    return FormalMaintenanceHistoryState(intent, completion)


def formal_intent_record(
    *,
    cadence_id: str,
    plan_id: str,
    transaction_id: str,
    stream_id: str,
    memory_project_key: str,
    before_revision: str,
    expected_after_revision: str,
    operations: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    return _validate_formal_history_record({
        "schema_version": FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION,
        "record_type": "intent",
        "cadence_id": cadence_id,
        "plan_id": plan_id,
        "transaction_id": transaction_id,
        "stream_id": stream_id,
        "memory_project_key": memory_project_key,
        "before_revision": before_revision,
        "expected_after_revision": expected_after_revision,
        "operations": list(operations),
        "updated_at": _now(),
    })


def formal_completion_record(
    *,
    cadence_id: str,
    plan_id: str,
    transaction_id: str,
    stream_id: str,
    memory_project_key: str,
    status: str,
    before_revision: str,
    after_revision: str,
    operations: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    return _validate_formal_history_record({
        "schema_version": FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION,
        "record_type": "completion",
        "cadence_id": cadence_id,
        "plan_id": plan_id,
        "transaction_id": transaction_id,
        "stream_id": stream_id,
        "memory_project_key": memory_project_key,
        "status": status,
        "before_revision": before_revision,
        "after_revision": after_revision,
        "operations": list(operations),
        "updated_at": _now(),
    })


def _validate_formal_history_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    record_type = payload.get("record_type")
    common = {
        "schema_version", "record_type", "cadence_id", "plan_id", "transaction_id",
        "stream_id", "memory_project_key", "before_revision", "operations", "updated_at",
    }
    expected = common | ({"expected_after_revision"} if record_type == "intent" else {
        "status", "after_revision",
    })
    if set(payload) != expected:
        raise ValueError("formal maintenance history fields do not match the schema")
    if payload.get("schema_version") != FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION:
        raise ValueError("unsupported formal maintenance history schema")
    if record_type not in {"intent", "completion"}:
        raise ValueError("unsupported formal maintenance history record type")
    require_sha256(str(payload.get("cadence_id") or ""), field_name="cadence_id")
    require_sha256(str(payload.get("plan_id") or ""), field_name="plan_id")
    require_sha256(str(payload.get("transaction_id") or ""), field_name="transaction_id")
    expected_transaction_id = formal_maintenance_transaction_id(
        cadence_id=str(payload["cadence_id"]),
        plan_id=str(payload["plan_id"]),
    )
    if payload["transaction_id"] != expected_transaction_id:
        raise ValueError("formal maintenance transaction_id does not match cadence and plan")
    for field_name in ("stream_id", "memory_project_key", "before_revision", "updated_at"):
        _require_nonblank(payload.get(field_name), field_name)
    operations = payload.get("operations")
    if not isinstance(operations, list) or any(not isinstance(item, Mapping) for item in operations):
        raise ValueError("formal maintenance operations must be an array of objects")
    if record_type == "intent":
        _require_nonblank(payload.get("expected_after_revision"), "expected_after_revision")
    else:
        if payload.get("status") not in {"committed", "noop"}:
            raise ValueError("formal maintenance completion status must be committed or noop")
        _require_nonblank(payload.get("after_revision"), "after_revision")
    try:
        datetime.fromisoformat(str(payload["updated_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("formal maintenance updated_at must be an ISO timestamp") from exc
    return payload


def _require_matching_history_pair(
    intent: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> None:
    for field_name in (
        "cadence_id", "plan_id", "transaction_id", "stream_id", "memory_project_key",
        "before_revision", "operations",
    ):
        if intent[field_name] != completion[field_name]:
            raise ValueError(f"formal maintenance history {field_name} mismatch")
    if intent["expected_after_revision"] != completion["after_revision"]:
        raise ValueError("formal maintenance history after revision mismatch")


def _require_nonblank(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION",
    "MAINTENANCE_HISTORY_FILENAME",
    "FormalMaintenanceHistoryState",
    "append_formal_maintenance_history",
    "formal_completion_record",
    "formal_intent_record",
    "formal_maintenance_plan_id",
    "formal_maintenance_transaction_id",
    "load_formal_maintenance_history",
]
