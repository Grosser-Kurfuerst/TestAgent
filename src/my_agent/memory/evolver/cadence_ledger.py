"""Persistent per-stream Q-task maintenance cadence and recovery state."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import sqlite3

from filelock import FileLock

from my_agent.json_safety import loads_json_strict
from my_agent.memory.evolver.artifacts import _history_lock_path
from my_agent.memory.evolver.cadence_schema import LEDGER_DDL
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256, require_sha256


CADENCE_SCHEMA_VERSION = "opd-maintenance-cadence-v1"
FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION = "opd-formal-maintenance-history-v1"
MAINTENANCE_HISTORY_FILENAME = "maintenance_history.jsonl"
WRITER_TERMINAL_STATUSES = frozenset({"committed", "no_write", "failed_no_write"})
CADENCE_OPEN_STATUSES = frozenset({"pending", "started"})

ProcessLock = Callable[[], Iterator[None]]


@dataclass(frozen=True)
class CadenceRecord:
    stream_id: str
    memory_project_key: str
    cadence_index: int
    cadence_id: str
    boundary_ordinal: int
    status: str
    maintenance_plan_id: str | None = None
    repository_revision_after: str | None = None


@dataclass(frozen=True)
class CadenceAdvanceResult:
    counted: bool
    task_ordinal: int | None
    cadence: CadenceRecord | None


@dataclass(frozen=True)
class FormalMaintenanceHistoryState:
    intent: Mapping[str, Any] | None = None
    completion: Mapping[str, Any] | None = None


class CadenceLedger:
    """SQLite ledger keyed by ``(stream_id, memory_project_key)``."""

    def __init__(
        self,
        path: str | Path,
        *,
        interval_tasks: int = 30,
        process_lock: ProcessLock | None = None,
    ) -> None:
        if isinstance(interval_tasks, bool) or not isinstance(interval_tasks, int) or interval_tasks < 1:
            raise ValueError("maintenance interval must be a positive integer")
        self.path = Path(path)
        self.interval_tasks = interval_tasks
        self._process_lock = process_lock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            with self._connect() as connection:
                for statement in LEDGER_DDL:
                    connection.execute(statement)
                connection.commit()

    def record_task_completion(
        self,
        *,
        stream_id: str,
        memory_project_key: str,
        task_id: str,
        task_valid: bool,
        outcome_finalized: bool,
        writer_terminal_status: str,
        repository_revision_after_writer: str,
    ) -> CadenceAdvanceResult:
        _require_nonblank(stream_id, "stream_id")
        _require_nonblank(memory_project_key, "memory_project_key")
        _require_nonblank(task_id, "task_id")
        _require_nonblank(repository_revision_after_writer, "repository_revision_after_writer")
        if not isinstance(task_valid, bool) or not isinstance(outcome_finalized, bool):
            raise ValueError("task_valid and outcome_finalized must be booleans")
        if writer_terminal_status not in WRITER_TERMINAL_STATUSES:
            raise ValueError("writer status is not terminal for the cadence ledger")
        if not task_valid or not outcome_finalized:
            return CadenceAdvanceResult(False, None, None)

        with self._locked():
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    """
                    SELECT task_ordinal
                    FROM task_completion
                    WHERE stream_id = ? AND memory_project_key = ? AND task_id = ?
                    """,
                    (stream_id, memory_project_key, task_id),
                ).fetchone()
                if existing is not None:
                    ordinal = int(existing[0])
                    cadence = self._cadence_at_boundary(
                        connection,
                        stream_id=stream_id,
                        memory_project_key=memory_project_key,
                        boundary_ordinal=ordinal,
                    )
                    connection.commit()
                    return CadenceAdvanceResult(False, ordinal, cadence)

                ordinal = int(connection.execute(
                    """
                    SELECT COALESCE(MAX(task_ordinal), 0) + 1
                    FROM task_completion
                    WHERE stream_id = ? AND memory_project_key = ?
                    """,
                    (stream_id, memory_project_key),
                ).fetchone()[0])
                now = _now()
                connection.execute(
                    """
                    INSERT INTO task_completion (
                      stream_id, memory_project_key, task_id, task_ordinal,
                      outcome_finalized, writer_terminal_status,
                      repository_revision_after_writer, updated_at
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        stream_id,
                        memory_project_key,
                        task_id,
                        ordinal,
                        writer_terminal_status,
                        repository_revision_after_writer,
                        now,
                    ),
                )
                cadence: CadenceRecord | None = None
                if ordinal % self.interval_tasks == 0:
                    cadence_index = ordinal // self.interval_tasks
                    cadence_id = stable_cadence_id(
                        stream_id=stream_id,
                        memory_project_key=memory_project_key,
                        interval_tasks=self.interval_tasks,
                        cadence_index=cadence_index,
                    )
                    connection.execute(
                        """
                        INSERT INTO maintenance_cadence (
                          stream_id, memory_project_key, cadence_index, cadence_id,
                          boundary_ordinal, status, maintenance_plan_id,
                          repository_revision_after, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, ?)
                        """,
                        (
                            stream_id,
                            memory_project_key,
                            cadence_index,
                            cadence_id,
                            ordinal,
                            now,
                        ),
                    )
                    cadence = CadenceRecord(
                        stream_id,
                        memory_project_key,
                        cadence_index,
                        cadence_id,
                        ordinal,
                        "pending",
                    )
                connection.commit()
                return CadenceAdvanceResult(True, ordinal, cadence)

    def oldest_open_cadence(
        self,
        *,
        stream_id: str,
        memory_project_key: str,
    ) -> CadenceRecord | None:
        _require_nonblank(stream_id, "stream_id")
        _require_nonblank(memory_project_key, "memory_project_key")
        with self._locked():
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT stream_id, memory_project_key, cadence_index, cadence_id,
                           boundary_ordinal, status, maintenance_plan_id,
                           repository_revision_after
                    FROM maintenance_cadence
                    WHERE stream_id = ? AND memory_project_key = ?
                      AND status IN ('pending', 'started')
                    ORDER BY cadence_index
                    LIMIT 1
                    """,
                    (stream_id, memory_project_key),
                ).fetchone()
        return _cadence_from_row(row) if row is not None else None

    def open_cadences(
        self,
        *,
        memory_project_key: str,
    ) -> tuple[CadenceRecord, ...]:
        _require_nonblank(memory_project_key, "memory_project_key")
        with self._locked():
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT stream_id, memory_project_key, cadence_index, cadence_id,
                           boundary_ordinal, status, maintenance_plan_id,
                           repository_revision_after
                    FROM maintenance_cadence
                    WHERE memory_project_key = ? AND status IN ('pending', 'started')
                    ORDER BY stream_id, cadence_index
                    """,
                    (memory_project_key,),
                ).fetchall()
        return tuple(_cadence_from_row(row) for row in rows)

    def mark_started(self, cadence_id: str) -> CadenceRecord:
        require_sha256(cadence_id, field_name="cadence_id")
        with self._locked():
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE maintenance_cadence
                    SET status = 'started', updated_at = ?
                    WHERE cadence_id = ? AND status IN ('pending', 'started')
                    """,
                    (_now(), cadence_id),
                )
                row = _load_cadence_row(connection, cadence_id)
                connection.commit()
        if row is None:
            raise ValueError(f"unknown cadence_id: {cadence_id}")
        record = _cadence_from_row(row)
        if record.status == "committed":
            return record
        if record.status != "started":
            raise ValueError(f"cadence cannot be started from status {record.status!r}")
        return record

    def mark_committed(
        self,
        cadence_id: str,
        *,
        maintenance_plan_id: str,
        repository_revision_after: str,
    ) -> CadenceRecord:
        require_sha256(cadence_id, field_name="cadence_id")
        require_sha256(maintenance_plan_id, field_name="maintenance_plan_id")
        _require_nonblank(repository_revision_after, "repository_revision_after")
        with self._locked():
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = _load_cadence_row(connection, cadence_id)
                if row is None:
                    raise ValueError(f"unknown cadence_id: {cadence_id}")
                current = _cadence_from_row(row)
                if current.status == "committed":
                    if (
                        current.maintenance_plan_id != maintenance_plan_id
                        or current.repository_revision_after != repository_revision_after
                    ):
                        raise ValueError("committed cadence identity does not match recovery evidence")
                    connection.commit()
                    return current
                connection.execute(
                    """
                    UPDATE maintenance_cadence
                    SET status = 'committed', maintenance_plan_id = ?,
                        repository_revision_after = ?, updated_at = ?
                    WHERE cadence_id = ?
                    """,
                    (maintenance_plan_id, repository_revision_after, _now(), cadence_id),
                )
                committed_row = _load_cadence_row(connection, cadence_id)
                connection.commit()
        assert committed_row is not None
        return _cadence_from_row(committed_row)

    def task_count(self, *, stream_id: str, memory_project_key: str) -> int:
        with self._locked():
            with self._connect() as connection:
                return int(connection.execute(
                    """
                    SELECT COUNT(*) FROM task_completion
                    WHERE stream_id = ? AND memory_project_key = ?
                    """,
                    (stream_id, memory_project_key),
                ).fetchone()[0])

    def cadence_records(
        self,
        *,
        stream_id: str,
        memory_project_key: str,
    ) -> tuple[CadenceRecord, ...]:
        with self._locked():
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT stream_id, memory_project_key, cadence_index, cadence_id,
                           boundary_ordinal, status, maintenance_plan_id,
                           repository_revision_after
                    FROM maintenance_cadence
                    WHERE stream_id = ? AND memory_project_key = ?
                    ORDER BY cadence_index
                    """,
                    (stream_id, memory_project_key),
                ).fetchall()
        return tuple(_cadence_from_row(row) for row in rows)

    def _cadence_at_boundary(
        self,
        connection: sqlite3.Connection,
        *,
        stream_id: str,
        memory_project_key: str,
        boundary_ordinal: int,
    ) -> CadenceRecord | None:
        if boundary_ordinal % self.interval_tasks != 0:
            return None
        row = connection.execute(
            """
            SELECT stream_id, memory_project_key, cadence_index, cadence_id,
                   boundary_ordinal, status, maintenance_plan_id,
                   repository_revision_after
            FROM maintenance_cadence
            WHERE stream_id = ? AND memory_project_key = ? AND boundary_ordinal = ?
            """,
            (stream_id, memory_project_key, boundary_ordinal),
        ).fetchone()
        return _cadence_from_row(row) if row is not None else None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    def _locked(self):
        return self._process_lock() if self._process_lock is not None else nullcontext()


def stable_cadence_id(
    *,
    stream_id: str,
    memory_project_key: str,
    interval_tasks: int,
    cadence_index: int,
) -> str:
    _require_nonblank(stream_id, "stream_id")
    _require_nonblank(memory_project_key, "memory_project_key")
    if interval_tasks < 1 or cadence_index < 1:
        raise ValueError("cadence interval and index must be positive")
    return canonical_sha256({
        "schema_version": CADENCE_SCHEMA_VERSION,
        "stream_id": stream_id,
        "memory_project_key": memory_project_key,
        "interval_tasks": interval_tasks,
        "cadence_index": cadence_index,
    })


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
    with FileLock(str(_history_lock_path(output)), timeout=30.0):
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
    with FileLock(str(_history_lock_path(source)), timeout=30.0):
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


def _load_cadence_row(
    connection: sqlite3.Connection,
    cadence_id: str,
) -> tuple[Any, ...] | None:
    return connection.execute(
        """
        SELECT stream_id, memory_project_key, cadence_index, cadence_id,
               boundary_ordinal, status, maintenance_plan_id,
               repository_revision_after
        FROM maintenance_cadence
        WHERE cadence_id = ?
        """,
        (cadence_id,),
    ).fetchone()


def _cadence_from_row(row: tuple[Any, ...]) -> CadenceRecord:
    return CadenceRecord(
        stream_id=str(row[0]),
        memory_project_key=str(row[1]),
        cadence_index=int(row[2]),
        cadence_id=str(row[3]),
        boundary_ordinal=int(row[4]),
        status=str(row[5]),
        maintenance_plan_id=None if row[6] is None else str(row[6]),
        repository_revision_after=None if row[7] is None else str(row[7]),
    )


def _require_nonblank(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CADENCE_OPEN_STATUSES",
    "CADENCE_SCHEMA_VERSION",
    "FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION",
    "MAINTENANCE_HISTORY_FILENAME",
    "WRITER_TERMINAL_STATUSES",
    "CadenceAdvanceResult",
    "CadenceLedger",
    "CadenceRecord",
    "FormalMaintenanceHistoryState",
    "append_formal_maintenance_history",
    "formal_completion_record",
    "formal_intent_record",
    "formal_maintenance_plan_id",
    "formal_maintenance_transaction_id",
    "load_formal_maintenance_history",
    "stable_cadence_id",
]
