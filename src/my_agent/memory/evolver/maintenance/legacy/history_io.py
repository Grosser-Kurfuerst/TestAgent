"""Legacy maintenance history schema validation and locked JSONL IO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import math
import os

from filelock import FileLock, Timeout as FileLockTimeout

from my_agent.json_safety import loads_json_strict
from my_agent.memory.evolver.maintenance.contracts import (
    MAINTENANCE_POLICY,
    MAINTENANCE_SCHEMA_VERSION,
    MAINTENANCE_SCOPE_MODE,
    MaintenanceAction,
    MaintenanceApplyStatus,
    MaintenancePlan,
    MaintenancePlanError,
    _operation_summary,
)
from my_agent.memory.evolver.maintenance.formal.history import (
    FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION,
)
from my_agent.memory.evolver.maintenance.history_io import history_lock_path
from my_agent.text_safety import sanitize_json_value

_HISTORY_LOCK_TIMEOUT_SECONDS = 30.0


class MaintenanceHistoryLockTimeout(RuntimeError):
    """Raised when maintenance history cannot acquire its process lock."""


@dataclass(frozen=True)
class _MaintenanceHistoryState:
    intent: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None
    audit_error: dict[str, Any] | None = None


def _append_maintenance_history(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    lock_timeout_seconds: float = _HISTORY_LOCK_TIMEOUT_SECONDS,
) -> Path:
    _validate_history_lock_timeout(lock_timeout_seconds)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_json_value(dict(record))
    if not isinstance(payload, dict):
        raise MaintenancePlanError("maintenance history record must be an object")
    _validate_maintenance_history_record(payload)
    lock = FileLock(
        str(history_lock_path(output)),
        timeout=lock_timeout_seconds,
    )
    try:
        with lock:
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    except FileLockTimeout as exc:
        raise MaintenanceHistoryLockTimeout(
            "maintenance history lock acquisition timed out"
        ) from exc
    return output


def _parse_maintenance_history_record(
    payload: Mapping[str, Any],
    *,
    line_no: int,
    plan: MaintenancePlan | None = None,
) -> dict[str, Any] | None:
    record = dict(payload)
    if record.get("schema_version") == FORMAL_MAINTENANCE_HISTORY_SCHEMA_VERSION:
        return None
    try:
        _validate_maintenance_history_record(record)
        if plan is not None:
            _validate_history_record_for_plan(record, plan)
    except MaintenancePlanError as exc:
        raise MaintenancePlanError(
            f"invalid maintenance history at line {line_no}: {exc}"
        ) from exc
    return record


def _validate_maintenance_history_record(record: Mapping[str, Any]) -> None:
    schema_version = _history_nonnegative_int(record, "schema_version")
    if schema_version != MAINTENANCE_SCHEMA_VERSION:
        raise MaintenancePlanError("unsupported schema_version")
    record_type = _history_string(record, "record_type")
    _history_identifier(record, "plan_id", prefix="maint-")

    if record_type == "intent":
        _validate_history_intent(record)
        return
    if record_type == "completion":
        status = _history_status(record, "status")
        if status == MaintenanceApplyStatus.PRE_COMMIT_FAILED:
            _validate_history_pre_commit_completion(record)
        else:
            _validate_history_success_completion(record, status=status)
        return
    if record_type == "audit_error":
        _validate_history_audit_error(record)
        return
    raise MaintenancePlanError("unsupported record_type")


def _validate_history_intent(record: Mapping[str, Any]) -> None:
    _history_exact_fields(record, _HISTORY_INTENT_FIELDS, "intent")
    _validate_history_plan_context(record)
    before_revision = _history_revision(record, "before_revision")
    expected_revision = _history_revision(record, "expected_after_revision")
    if before_revision == expected_revision:
        raise MaintenancePlanError("intent revisions must differ")
    before_count = _history_nonnegative_int(record, "before_count")
    after_count = _history_nonnegative_int(record, "after_count")
    removed_ids = _history_string_list(record, "removed_ids")
    updated_ids = _history_string_list(record, "updated_ids")
    added_ids = _history_string_list(record, "added_ids")
    _history_string(record, "backup_path")
    operation_ids = _history_string_list(record, "operation_ids")
    if not operation_ids:
        raise MaintenancePlanError("intent operation_ids must not be empty")
    for operation_id in operation_ids:
        _validate_history_identifier(operation_id, prefix="op-", name="operation_id")
    _validate_history_mutation_ids(removed_ids, updated_ids, added_ids)
    if after_count != before_count - len(removed_ids) + len(added_ids):
        raise MaintenancePlanError("intent entry counts are inconsistent")


def _validate_history_success_completion(
    record: Mapping[str, Any],
    *,
    status: MaintenanceApplyStatus,
) -> None:
    if status not in {
        MaintenanceApplyStatus.NOOP,
        MaintenanceApplyStatus.COMMITTED,
    }:
        raise MaintenancePlanError("invalid successful completion status")
    _history_exact_fields(record, _HISTORY_SUCCESS_FIELDS, "completion")
    _validate_history_plan_context(record)
    before_revision = _history_revision(record, "before_revision")
    after_revision = _history_revision(record, "after_revision")
    summary = _history_summary(record, "summary")
    operation_actions = _history_operations(record, "operations")
    backup_path = _history_string(record, "backup_path", allow_empty=True)
    mutation_committed = _history_bool(record, "mutation_committed")
    should_retry = _history_bool(record, "should_retry")
    result = _history_mapping(record, "result")
    _history_exact_fields(result, _HISTORY_RESULT_FIELDS, "completion result")

    result_plan_id = _history_identifier(result, "plan_id", prefix="maint-")
    result_status = _history_status(result, "status")
    result_committed = _history_bool(result, "mutation_committed")
    audit_complete = _history_bool(result, "audit_complete")
    result_retry = _history_bool(result, "should_retry")
    result_before_revision = _history_revision(result, "before_revision")
    result_after_revision = _history_revision(result, "after_revision")
    before_count = _history_nonnegative_int(result, "before_count")
    after_count = _history_nonnegative_int(result, "after_count")
    kept = _history_nonnegative_int(result, "kept")
    deleted = _history_nonnegative_int(result, "deleted")
    merged = _history_nonnegative_int(result, "merged")
    promoted = _history_nonnegative_int(result, "promoted")
    removed_ids = _history_string_list(result, "removed_ids")
    updated_ids = _history_string_list(result, "updated_ids")
    added_ids = _history_string_list(result, "added_ids")
    result_backup = _history_string(result, "backup_path", allow_empty=True)
    audit_error_stage = _history_string(
        result,
        "audit_error_stage",
        allow_empty=True,
    )
    audit_error = _history_string(result, "audit_error", allow_empty=True)

    if result_plan_id != record["plan_id"] or result_status != status:
        raise MaintenancePlanError("completion result identity mismatch")
    if (
        result_committed != mutation_committed
        or result_retry != should_retry
        or result_before_revision != before_revision
        or result_after_revision != after_revision
        or result_backup != backup_path
    ):
        raise MaintenancePlanError("completion result fields mismatch")
    if should_retry or not audit_complete or audit_error_stage or audit_error:
        raise MaintenancePlanError("successful completion audit state is inconsistent")
    if [kept, deleted, merged, promoted] != [
        summary[MaintenanceAction.KEEP.value],
        summary[MaintenanceAction.DELETE.value],
        summary[MaintenanceAction.MERGE.value],
        summary[MaintenanceAction.PROMOTE.value],
    ]:
        raise MaintenancePlanError("completion operation counts mismatch")
    action_counts = {action.value: 0 for action in MaintenanceAction}
    for action in operation_actions:
        action_counts[action.value] += 1
    if any(summary[action] != count for action, count in action_counts.items()):
        raise MaintenancePlanError("completion operation summary mismatch")
    if summary["source_entries_removed"] != len(removed_ids):
        raise MaintenancePlanError("completion removed_ids count mismatch")
    if summary["entries_added"] != len(added_ids):
        raise MaintenancePlanError("completion added_ids count mismatch")
    _validate_history_mutation_ids(removed_ids, updated_ids, added_ids)
    if after_count != before_count - len(removed_ids) + len(added_ids):
        raise MaintenancePlanError("completion entry counts are inconsistent")

    if status == MaintenanceApplyStatus.NOOP:
        if mutation_committed or backup_path or removed_ids or updated_ids or added_ids:
            raise MaintenancePlanError("noop completion contains mutation state")
        if before_revision != after_revision or before_count != after_count:
            raise MaintenancePlanError("noop completion changed repository state")
    elif not mutation_committed or not backup_path:
        raise MaintenancePlanError("committed completion lacks mutation state")


def _validate_history_pre_commit_completion(record: Mapping[str, Any]) -> None:
    _history_exact_fields(
        record,
        _HISTORY_PRE_COMMIT_FIELDS,
        "pre-commit completion",
    )
    _validate_history_plan_context(record)
    before_revision = _history_revision(record, "before_revision")
    after_revision = _history_revision(record, "after_revision")
    _history_string(record, "backup_path", allow_empty=True)
    if _history_bool(record, "mutation_committed"):
        raise MaintenancePlanError("pre-commit completion cannot be committed")
    _history_bool(record, "should_retry")
    _history_string(record, "failure_stage")
    _history_string(record, "error")
    if before_revision != after_revision:
        raise MaintenancePlanError("pre-commit completion changed repository revision")


def _validate_history_audit_error(record: Mapping[str, Any]) -> None:
    _history_exact_fields(record, _HISTORY_AUDIT_ERROR_FIELDS, "audit_error")
    if _history_status(record, "status") != MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR:
        raise MaintenancePlanError("audit_error has invalid status")
    if not _history_bool(record, "mutation_committed"):
        raise MaintenancePlanError("audit_error requires a committed mutation")
    if _history_bool(record, "should_retry"):
        raise MaintenancePlanError("audit_error cannot be retryable")
    _history_string(record, "audit_error_stage")
    _history_string(record, "error")


def _validate_history_plan_context(record: Mapping[str, Any]) -> None:
    if _history_string(record, "policy") != MAINTENANCE_POLICY:
        raise MaintenancePlanError("history policy mismatch")
    if _history_string(record, "scope_mode") != MAINTENANCE_SCOPE_MODE:
        raise MaintenancePlanError("history scope_mode mismatch")
    _history_string(record, "memory_project_key")
    timestamp = _history_string(record, "as_of")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MaintenancePlanError("history as_of must be a timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MaintenancePlanError("history as_of must be timezone-aware")


def _validate_history_record_for_plan(
    record: Mapping[str, Any],
    plan: MaintenancePlan,
) -> None:
    if record["plan_id"] != plan.plan_id:
        return
    record_type = record["record_type"]
    if record_type == "audit_error":
        return
    if (
        record["policy"] != plan.policy
        or record["scope_mode"] != plan.scope_mode
        or record["memory_project_key"] != plan.memory_project_key
        or record["as_of"] != plan.as_of
    ):
        raise MaintenancePlanError("history plan context mismatch")
    if record["before_revision"] != plan.repository_revision:
        raise MaintenancePlanError("history before_revision does not match plan")

    removed_ids, updated_ids, added_ids = _plan_mutation_ids(plan)
    if record_type == "intent":
        if tuple(record["operation_ids"]) != tuple(
            operation.operation_id for operation in plan.operations
        ):
            raise MaintenancePlanError("intent operation_ids do not match plan")
        if (
            tuple(record["removed_ids"]) != removed_ids
            or tuple(record["updated_ids"]) != updated_ids
            or tuple(record["added_ids"]) != added_ids
        ):
            raise MaintenancePlanError("intent mutation ids do not match plan")
        return

    status = MaintenanceApplyStatus(record["status"])
    if status == MaintenanceApplyStatus.PRE_COMMIT_FAILED:
        return
    expected_summary = _operation_summary(plan.operations)
    if record["summary"] != expected_summary:
        raise MaintenancePlanError("completion summary does not match plan")
    if record["operations"] != _history_operations_for_plan(plan):
        raise MaintenancePlanError("completion operations do not match plan")
    result = record["result"]
    if (
        tuple(result["removed_ids"]) != removed_ids
        or tuple(result["updated_ids"]) != updated_ids
        or tuple(result["added_ids"]) != added_ids
    ):
        raise MaintenancePlanError("completion mutation ids do not match plan")


def _history_exact_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MaintenancePlanError(
            f"{name} fields mismatch: missing={missing}, extra={extra}"
        )


def _history_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise MaintenancePlanError(f"history {key} must be an object")
    return value


def _history_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise MaintenancePlanError(f"history {key} must be a string")
    return value


def _history_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise MaintenancePlanError(f"history {key} must be boolean")
    return value


def _history_nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MaintenancePlanError(
            f"history {key} must be a non-negative integer"
        )
    return value


def _history_string_list(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise MaintenancePlanError(
            f"history {key} must be a list of non-empty strings"
        )
    result = tuple(value)
    if len(result) != len(set(result)):
        raise MaintenancePlanError(f"history {key} contains duplicates")
    return result


def _history_status(
    payload: Mapping[str, Any],
    key: str,
) -> MaintenanceApplyStatus:
    value = _history_string(payload, key)
    try:
        return MaintenanceApplyStatus(value)
    except ValueError as exc:
        raise MaintenancePlanError(f"history {key} is invalid") from exc


def _history_identifier(
    payload: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
) -> str:
    value = _history_string(payload, key)
    _validate_history_identifier(value, prefix=prefix, name=key)
    return value


def _validate_history_identifier(value: str, *, prefix: str, name: str) -> None:
    digest = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(digest) != 24
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise MaintenancePlanError(f"history {name} is invalid")


def _history_revision(payload: Mapping[str, Any], key: str) -> str:
    value = _history_string(payload, key)
    digest = value.removeprefix("sha256:")
    if (
        not value.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise MaintenancePlanError(f"history {key} must be a sha256 revision")
    return value


def _history_summary(payload: Mapping[str, Any], key: str) -> dict[str, int]:
    summary = _history_mapping(payload, key)
    expected = frozenset({
        *(action.value for action in MaintenanceAction),
        "source_entries_removed",
        "entries_added",
    })
    _history_exact_fields(summary, expected, "completion summary")
    return {
        name: _history_nonnegative_int(summary, name)
        for name in sorted(expected)
    }


def _history_operations(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[MaintenanceAction, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise MaintenancePlanError(f"history {key} must be a list")
    actions: list[MaintenanceAction] = []
    operation_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise MaintenancePlanError("history operations must contain objects")
        _history_exact_fields(item, _HISTORY_OPERATION_FIELDS, "operation")
        operation_id = _history_identifier(item, "operation_id", prefix="op-")
        if operation_id in operation_ids:
            raise MaintenancePlanError("history operations contain duplicate ids")
        operation_ids.add(operation_id)
        action_value = _history_string(item, "action")
        try:
            actions.append(MaintenanceAction(action_value))
        except ValueError as exc:
            raise MaintenancePlanError("history operation action is invalid") from exc
        _history_string_list(item, "source_ids")
        _history_string_list(item, "target_ids")
        _history_string_list(item, "reason_codes")
    return tuple(actions)


def _validate_history_mutation_ids(
    removed_ids: tuple[str, ...],
    updated_ids: tuple[str, ...],
    added_ids: tuple[str, ...],
) -> None:
    if (
        set(removed_ids).intersection(updated_ids)
        or set(removed_ids).intersection(added_ids)
        or set(updated_ids).intersection(added_ids)
    ):
        raise MaintenancePlanError("history mutation id sets overlap")


def _plan_mutation_ids(
    plan: MaintenancePlan,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    removed_ids = tuple(sorted({item for op in plan.operations for item in op.remove_ids}))
    updated_ids = tuple(sorted({
        payload.id
        for op in plan.operations
        for payload in op.replacements
    }))
    added_ids = tuple(sorted({
        payload.id
        for op in plan.operations
        for payload in op.additions
    }))
    return removed_ids, updated_ids, added_ids


def _history_operations_for_plan(plan: MaintenancePlan) -> list[dict[str, Any]]:
    return [
        {
            "operation_id": operation.operation_id,
            "action": operation.action.value,
            "source_ids": list(operation.source_ids),
            "target_ids": list(operation.target_ids),
            "reason_codes": list(operation.reason_codes),
        }
        for operation in plan.operations
    ]


def _load_maintenance_history_state(
    path: str | Path,
    plan: MaintenancePlan,
    *,
    lock_timeout_seconds: float = _HISTORY_LOCK_TIMEOUT_SECONDS,
) -> _MaintenanceHistoryState:
    _validate_history_lock_timeout(lock_timeout_seconds)
    source = Path(path)
    intent: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None
    audit_error: dict[str, Any] | None = None
    lock = FileLock(
        str(history_lock_path(source)),
        timeout=lock_timeout_seconds,
    )
    try:
        with lock:
            if not source.exists():
                return _MaintenanceHistoryState()
            with source.open("r", encoding="utf-8") as handle:
                for line_no, raw in enumerate(handle, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        payload = loads_json_strict(line)
                        if not isinstance(payload, dict):
                            raise TypeError("expected object")
                    except (TypeError, ValueError) as exc:
                        raise MaintenancePlanError(
                            f"invalid maintenance history at line {line_no}: "
                            f"{type(exc).__name__}"
                        ) from exc
                    parsed = _parse_maintenance_history_record(
                        payload,
                        line_no=line_no,
                        plan=plan,
                    )
                    if parsed is None:
                        continue
                    if parsed["plan_id"] != plan.plan_id:
                        continue
                    record_type = parsed["record_type"]
                    if record_type == "intent":
                        if intent is not None:
                            raise MaintenancePlanError("duplicate maintenance intent record")
                        intent = parsed
                    elif record_type == "completion":
                        if completion is not None:
                            raise MaintenancePlanError("duplicate maintenance completion record")
                        completion = parsed
                    elif record_type == "audit_error":
                        audit_error = parsed
    except FileLockTimeout as exc:
        raise MaintenanceHistoryLockTimeout(
            "maintenance history lock acquisition timed out"
        ) from exc
    return _MaintenanceHistoryState(
        intent=intent,
        completion=completion,
        audit_error=audit_error,
    )


__all__ = ["MaintenanceHistoryLockTimeout"]
_HISTORY_INTENT_FIELDS = frozenset({
    "schema_version",
    "record_type",
    "plan_id",
    "policy",
    "scope_mode",
    "memory_project_key",
    "as_of",
    "before_revision",
    "expected_after_revision",
    "before_count",
    "after_count",
    "removed_ids",
    "updated_ids",
    "added_ids",
    "backup_path",
    "operation_ids",
})
_HISTORY_SUCCESS_FIELDS = frozenset({
    "schema_version",
    "record_type",
    "status",
    "plan_id",
    "policy",
    "scope_mode",
    "memory_project_key",
    "as_of",
    "before_revision",
    "after_revision",
    "summary",
    "operations",
    "backup_path",
    "mutation_committed",
    "should_retry",
    "result",
})
_HISTORY_PRE_COMMIT_FIELDS = frozenset({
    "schema_version",
    "record_type",
    "status",
    "plan_id",
    "policy",
    "scope_mode",
    "memory_project_key",
    "as_of",
    "before_revision",
    "after_revision",
    "backup_path",
    "mutation_committed",
    "should_retry",
    "failure_stage",
    "error",
})
_HISTORY_AUDIT_ERROR_FIELDS = frozenset({
    "schema_version",
    "record_type",
    "status",
    "plan_id",
    "mutation_committed",
    "should_retry",
    "audit_error_stage",
    "error",
})
_HISTORY_RESULT_FIELDS = frozenset({
    "plan_id",
    "status",
    "mutation_committed",
    "audit_complete",
    "should_retry",
    "before_revision",
    "after_revision",
    "before_count",
    "after_count",
    "kept",
    "deleted",
    "merged",
    "promoted",
    "removed_ids",
    "updated_ids",
    "added_ids",
    "backup_path",
    "audit_error_stage",
    "audit_error",
})
_HISTORY_OPERATION_FIELDS = frozenset({
    "operation_id",
    "action",
    "source_ids",
    "target_ids",
    "reason_codes",
})


def _validate_history_lock_timeout(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("history lock timeout must be finite and non-negative")
