"""Transactional apply, backup, and audit history for maintenance plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import os

from filelock import FileLock, Timeout as FileLockTimeout

from my_agent.memory.evolver.artifacts import (
    _MaintenanceArtifactGraph,
    _artifact_paths_alias,
    _history_lock_path,
    _maintenance_backup_path,
    _resolve_maintenance_artifact_graph,
    _validate_maintenance_artifact_graph,
)
from my_agent.memory.evolver.contracts import (
    MAINTENANCE_POLICY,
    MAINTENANCE_SCHEMA_VERSION,
    MAINTENANCE_SCOPE_MODE,
    MaintenanceAction,
    MaintenanceApplyResult,
    MaintenanceApplyStatus,
    MaintenancePlan,
    MaintenancePlanError,
    _operation_summary,
    _validated_payload_entry,
    maintenance_plan_json,
)
from my_agent.memory.evolver.planner import (
    _repository_after_operations,
)
from my_agent.memory.evolver.validation import (
    parse_maintenance_plan,
    validate_plan_semantics,
)
from my_agent.memory.long_term import (
    LongTermMemoryStore,
    MemoryStoreLockTimeout,
    MemoryStorePostCommitError,
    MemoryStoreRevisionConflict,
    _atomic_write_tmp_path,
    memory_entries_revision,
)
from my_agent.memory.types import MemoryEntry, MemoryScope
from my_agent.text_safety import sanitize_json_value


_HISTORY_LOCK_TIMEOUT_SECONDS = 30.0


class MaintenanceHistoryLockTimeout(RuntimeError):
    """Raised when maintenance history cannot acquire its process lock."""


@dataclass(frozen=True)
class _MaintenanceHistoryState:
    intent: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None
    audit_error: dict[str, Any] | None = None


def apply_maintenance_plan(
    *,
    store: LongTermMemoryStore,
    plan: MaintenancePlan,
    backup_dir: str | Path,
    history_path: str | Path,
    lock_timeout_seconds: float = 30.0,
) -> MaintenanceApplyResult:
    """Apply a reviewed plan through the stable programmatic API."""
    return _apply_maintenance_plan(
        store=store,
        plan=plan,
        backup_dir=backup_dir,
        history_path=history_path,
        lock_timeout_seconds=lock_timeout_seconds,
    )


def _apply_maintenance_plan(
    *,
    store: LongTermMemoryStore,
    plan: MaintenancePlan,
    backup_dir: str | Path,
    history_path: str | Path,
    lock_timeout_seconds: float = 30.0,
    artifact_graph: _MaintenanceArtifactGraph | None = None,
) -> MaintenanceApplyResult:
    """Apply a reviewed plan under one process lock and one store persist."""
    if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds < 0:
        raise ValueError("lock_timeout_seconds must be finite and non-negative")

    before_revision = plan.repository_revision
    after_revision = plan.repository_revision
    before_count = 0
    after_count = 0
    backup_path = ""
    stage = "plan_validation"
    intent_written = False
    reuse_intent = False
    mutation_committed = False
    backup: Path | None = None

    try:
        plan = _validated_plan_copy(plan)
    except Exception as exc:
        return _pre_commit_failure_result(
            plan=plan,
            stage=stage,
            error=exc,
            before_revision=before_revision,
            before_count=before_count,
            backup_path=backup_path,
            should_retry=False,
        )

    stage = "artifact_validation"
    try:
        core_artifact_graph = _resolve_maintenance_artifact_graph(
            store_path=store.path,
            store_lock_path=store.lock_path,
            history_path=history_path,
            backup_dir=backup_dir,
            plan_id=plan.plan_id,
        )
        if artifact_graph is not None:
            _validate_maintenance_artifact_graph(artifact_graph)
            _validate_supplied_artifact_graph(
                artifact_graph,
                core_graph=core_artifact_graph,
            )
        backup = core_artifact_graph.backup_path
        if backup is None:
            raise MaintenancePlanError("maintenance backup path was not resolved")
    except Exception as exc:
        return _pre_commit_failure_result(
            plan=plan,
            stage=stage,
            error=exc,
            before_revision=before_revision,
            before_count=before_count,
            backup_path=backup_path,
            should_retry=False,
        )

    stage = "lock"
    removed_ids = tuple(sorted({item for op in plan.operations for item in op.remove_ids}))
    updated_ids = tuple(sorted({
        str(payload.get("id") or "")
        for op in plan.operations
        for payload in op.replacements
    }))
    added_ids = tuple(sorted({
        str(payload.get("id") or "")
        for op in plan.operations
        for payload in op.additions
    }))

    try:
        with store.exclusive_process_lock(timeout_seconds=lock_timeout_seconds):
            try:
                stage = "strict_load"
                snapshot = store.load_strict_snapshot()
                before_revision = snapshot.revision
                after_revision = snapshot.revision
                before_count = len(snapshot.entries)
                after_count = before_count

                stage = "history_load"
                try:
                    history_state = _load_maintenance_history_state(
                        history_path,
                        plan,
                        lock_timeout_seconds=lock_timeout_seconds,
                    )
                except MaintenanceHistoryLockTimeout:
                    stage = "history_lock"
                    raise
                terminal = _terminal_history_result(
                    plan,
                    history_state,
                    current_revision=snapshot.revision,
                )
                if terminal is not None:
                    return terminal
                stage = "validation"
                if history_state.intent is not None:
                    intent_before = history_state.intent["before_revision"]
                    if snapshot.revision != intent_before:
                        raise MaintenancePlanError(
                            "incomplete maintenance intent has ambiguous repository revision"
                        )
                    reuse_intent = True
                    intent_written = True
                elif snapshot.revision != plan.repository_revision:
                    raise MemoryStoreRevisionConflict(
                        "reviewed plan repository revision no longer matches"
                    )

                validate_plan_semantics(plan, repository_entries=snapshot.entries)
                _validate_apply_project_boundaries(plan, snapshot.entries)
                next_entries = _repository_after_operations(snapshot.entries, plan.operations)
                after_count = len(next_entries)

                has_mutation = any(
                    operation.remove_ids or operation.replacements or operation.additions
                    for operation in plan.operations
                )
                if not has_mutation:
                    result = _maintenance_apply_result(
                        plan=plan,
                        status=MaintenanceApplyStatus.NOOP,
                        mutation_committed=False,
                        audit_complete=True,
                        should_retry=False,
                        before_revision=before_revision,
                        after_revision=after_revision,
                        before_count=before_count,
                        after_count=after_count,
                    )
                    stage = "history_completion"
                    try:
                        _append_maintenance_history(
                            history_path,
                            _completion_history_record(plan, result),
                            lock_timeout_seconds=lock_timeout_seconds,
                        )
                    except Exception as exc:
                        return replace(
                            result,
                            audit_complete=False,
                            audit_error_stage=stage,
                            audit_error=_safe_error(exc),
                        )
                    return result

                stage = "backup"
                assert backup is not None
                _write_backup_atomic(backup, snapshot.raw_bytes)
                backup_path = str(backup)

                expected_after_revision = memory_entries_revision(next_entries)
                if reuse_intent:
                    assert history_state.intent is not None
                    recorded_after = history_state.intent["expected_after_revision"]
                    if recorded_after and recorded_after != expected_after_revision:
                        raise MaintenancePlanError(
                            "incomplete maintenance intent does not match reviewed plan"
                        )
                else:
                    stage = "audit_intent"
                    try:
                        _append_maintenance_history(
                            history_path,
                            _intent_history_record(
                                plan,
                                before_revision=before_revision,
                                expected_after_revision=expected_after_revision,
                                before_count=before_count,
                                after_count=after_count,
                                removed_ids=removed_ids,
                                updated_ids=updated_ids,
                                added_ids=added_ids,
                                backup_path=backup_path,
                            ),
                            lock_timeout_seconds=lock_timeout_seconds,
                        )
                    except MaintenanceHistoryLockTimeout:
                        stage = "history_lock"
                        raise
                    intent_written = True

                stage = "persist"
                try:
                    after_revision = store.replace_all_atomically(
                        next_entries,
                        expected_revision=before_revision,
                    )
                except MemoryStorePostCommitError as exc:
                    mutation_committed = True
                    after_revision = exc.expected_revision
                    stage = "verify"
                    raise
                mutation_committed = True

                stage = "verify"
                written = store.load_strict_snapshot()
                if written.revision != after_revision or len(written.entries) != after_count:
                    raise MaintenancePlanError("post-commit repository verification mismatch")
                written_ids = {entry.id for entry in written.entries}
                if set(removed_ids).intersection(written_ids):
                    raise MaintenancePlanError("removed maintenance sources remain after commit")
                if not set(updated_ids).union(added_ids).issubset(written_ids):
                    raise MaintenancePlanError("maintenance targets are missing after commit")

                committed = _maintenance_apply_result(
                    plan=plan,
                    status=MaintenanceApplyStatus.COMMITTED,
                    mutation_committed=True,
                    audit_complete=True,
                    should_retry=False,
                    before_revision=before_revision,
                    after_revision=after_revision,
                    before_count=before_count,
                    after_count=after_count,
                    removed_ids=removed_ids,
                    updated_ids=updated_ids,
                    added_ids=added_ids,
                    backup_path=backup_path,
                )
                stage = "history_completion"
                try:
                    _append_maintenance_history(
                        history_path,
                        _completion_history_record(plan, committed),
                        lock_timeout_seconds=lock_timeout_seconds,
                    )
                except Exception as exc:
                    failed = replace(
                        committed,
                        status=MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
                        audit_complete=False,
                        audit_error_stage=stage,
                        audit_error=_safe_error(exc),
                    )
                    _best_effort_history(
                        history_path,
                        _audit_error_history_record(plan, failed),
                        lock_timeout_seconds=lock_timeout_seconds,
                    )
                    return failed
                return committed
            except Exception as exc:
                if mutation_committed:
                    failed = _maintenance_apply_result(
                        plan=plan,
                        status=MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
                        mutation_committed=True,
                        audit_complete=False,
                        should_retry=False,
                        before_revision=before_revision,
                        after_revision=after_revision,
                        before_count=before_count,
                        after_count=after_count,
                        removed_ids=removed_ids,
                        updated_ids=updated_ids,
                        added_ids=added_ids,
                        backup_path=backup_path,
                        audit_error_stage=stage,
                        audit_error=_safe_error(exc),
                    )
                    _best_effort_history(
                        history_path,
                        _audit_error_history_record(plan, failed),
                        lock_timeout_seconds=lock_timeout_seconds,
                    )
                    return failed
                retryable = _is_retryable_pre_commit_failure(stage, exc)
                if intent_written and not retryable:
                    _best_effort_history(
                        history_path,
                        _pre_commit_history_record(
                            plan,
                            before_revision=before_revision,
                            backup_path=backup_path,
                            stage=stage,
                            error=_safe_error(exc),
                            should_retry=False,
                        ),
                        lock_timeout_seconds=lock_timeout_seconds,
                    )
                return _pre_commit_failure_result(
                    plan=plan,
                    stage=stage,
                    error=exc,
                    before_revision=before_revision,
                    before_count=before_count,
                    backup_path=backup_path,
                    should_retry=retryable,
                )
    except MemoryStoreLockTimeout as exc:
        return _pre_commit_failure_result(
            plan=plan,
            stage="lock",
            error=exc,
            before_revision=before_revision,
            before_count=before_count,
            backup_path=backup_path,
            should_retry=True,
        )


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
        str(_history_lock_path(output)),
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


def record_post_commit_audit_error(
    *,
    history_path: str | Path,
    plan: MaintenancePlan,
    result: MaintenanceApplyResult,
    stage: str,
    error: Exception,
    lock_timeout_seconds: float = _HISTORY_LOCK_TIMEOUT_SECONDS,
) -> MaintenanceApplyResult:
    """Record a post-commit sink failure without making mutation retryable."""
    if not result.mutation_committed:
        raise ValueError("post-commit audit errors require a committed mutation")
    if not str(stage or ""):
        raise ValueError("post-commit audit error stage must not be empty")
    updated = replace(
        result,
        status=MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
        audit_complete=False,
        should_retry=False,
        audit_error_stage=str(stage),
        audit_error=_safe_error(error),
    )
    _best_effort_history(
        history_path,
        _audit_error_history_record(plan, updated),
        lock_timeout_seconds=lock_timeout_seconds,
    )
    return updated


def _validated_plan_copy(plan: MaintenancePlan) -> MaintenancePlan:
    """Rebuild a plan so mutable payload aliases cannot bypass identity checks."""
    payload = json.loads(maintenance_plan_json(plan))
    if not isinstance(payload, dict):
        raise MaintenancePlanError("maintenance plan must be a JSON object")
    return parse_maintenance_plan(payload)


def _validate_apply_project_boundaries(
    plan: MaintenancePlan,
    entries: Sequence[MemoryEntry],
) -> None:
    by_id = {entry.id: entry for entry in entries}
    for operation in plan.operations:
        for source_id in operation.source_ids:
            source = by_id[source_id]
            if source.scope == MemoryScope.GLOBAL:
                if operation.action != MaintenanceAction.KEEP:
                    raise MaintenancePlanError("global experience may only be kept")
            elif source.project_key != plan.memory_project_key:
                raise MaintenancePlanError("operation crosses memory project boundary")
        for payload in operation.replacements + operation.additions:
            entry = _validated_payload_entry(payload, "mutation")
            if entry.scope == MemoryScope.GLOBAL:
                raise MaintenancePlanError("maintenance cannot mutate global experience")
            if entry.project_key != plan.memory_project_key:
                raise MaintenancePlanError("mutation payload crosses memory project boundary")


def _write_backup_atomic(path: Path, raw_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = sha256(raw_bytes).hexdigest()
    if path.exists():
        if sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise MaintenancePlanError("existing maintenance backup does not match repository")
        return
    tmp = _atomic_write_tmp_path(path)
    try:
        with tmp.open("wb") as handle:
            handle.write(raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        if sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise MaintenancePlanError("maintenance backup verification failed")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _validate_supplied_artifact_graph(
    graph: _MaintenanceArtifactGraph,
    *,
    core_graph: _MaintenanceArtifactGraph,
) -> None:
    supplied = dict(graph.paths)
    for label, expected_path in core_graph.paths:
        actual_path = supplied.get(label)
        if actual_path is None or not _artifact_paths_alias(actual_path, expected_path):
            raise MaintenancePlanError(
                f"maintenance artifact graph does not match transaction {label}"
            )


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


def _parse_maintenance_history_record(
    payload: Mapping[str, Any],
    *,
    line_no: int,
    plan: MaintenancePlan | None = None,
) -> dict[str, Any]:
    record = dict(payload)
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
        str(payload.get("id") or "")
        for op in plan.operations
        for payload in op.replacements
    }))
    added_ids = tuple(sorted({
        str(payload.get("id") or "")
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
        str(_history_lock_path(source)),
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
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            raise TypeError("expected object")
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise MaintenancePlanError(
                            f"invalid maintenance history at line {line_no}: "
                            f"{type(exc).__name__}"
                        ) from exc
                    parsed = _parse_maintenance_history_record(
                        payload,
                        line_no=line_no,
                        plan=plan,
                    )
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


def _terminal_history_result(
    plan: MaintenancePlan,
    state: _MaintenanceHistoryState,
    *,
    current_revision: str,
) -> MaintenanceApplyResult | None:
    if state.completion is not None:
        result = _apply_result_from_history(plan, state.completion)
        if state.audit_error is not None and result.mutation_committed:
            return replace(
                result,
                status=MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
                audit_complete=False,
                should_retry=False,
                audit_error_stage=state.audit_error["audit_error_stage"],
                audit_error=state.audit_error["error"],
            )
        return result

    if state.intent is None:
        return None
    before_revision = state.intent["before_revision"]
    expected_after_revision = state.intent["expected_after_revision"]
    if current_revision == before_revision:
        return None
    if expected_after_revision and current_revision == expected_after_revision:
        return _apply_result_from_intent(
            plan,
            state.intent,
            status=MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
            audit_error_stage=(
                state.audit_error["audit_error_stage"]
                if state.audit_error is not None
                else "history_recovery"
            ),
            audit_error=(
                state.audit_error["error"]
                if state.audit_error is not None
                else "audit_incomplete"
            ),
        )
    raise MaintenancePlanError(
        "incomplete maintenance intent has ambiguous repository revision"
    )


def _intent_history_record(
    plan: MaintenancePlan,
    *,
    before_revision: str,
    expected_after_revision: str,
    before_count: int,
    after_count: int,
    removed_ids: tuple[str, ...],
    updated_ids: tuple[str, ...],
    added_ids: tuple[str, ...],
    backup_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": MAINTENANCE_SCHEMA_VERSION,
        "record_type": "intent",
        "plan_id": plan.plan_id,
        "policy": plan.policy,
        "scope_mode": plan.scope_mode,
        "memory_project_key": plan.memory_project_key,
        "as_of": plan.as_of,
        "before_revision": before_revision,
        "expected_after_revision": expected_after_revision,
        "before_count": before_count,
        "after_count": after_count,
        "removed_ids": list(removed_ids),
        "updated_ids": list(updated_ids),
        "added_ids": list(added_ids),
        "backup_path": backup_path,
        "operation_ids": [operation.operation_id for operation in plan.operations],
    }


def _completion_history_record(
    plan: MaintenancePlan,
    result: MaintenanceApplyResult,
) -> dict[str, Any]:
    return {
        "schema_version": MAINTENANCE_SCHEMA_VERSION,
        "record_type": "completion",
        "status": result.status.value,
        "plan_id": plan.plan_id,
        "policy": plan.policy,
        "scope_mode": plan.scope_mode,
        "memory_project_key": plan.memory_project_key,
        "as_of": plan.as_of,
        "before_revision": result.before_revision,
        "after_revision": result.after_revision,
        "summary": _operation_summary(plan.operations),
        "operations": _history_operations_for_plan(plan),
        "backup_path": result.backup_path,
        "mutation_committed": result.mutation_committed,
        "should_retry": result.should_retry,
        "result": result.to_dict(),
    }


def _pre_commit_history_record(
    plan: MaintenancePlan,
    *,
    before_revision: str,
    backup_path: str,
    stage: str,
    error: str,
    should_retry: bool,
) -> dict[str, Any]:
    return {
        "schema_version": MAINTENANCE_SCHEMA_VERSION,
        "record_type": "completion",
        "status": MaintenanceApplyStatus.PRE_COMMIT_FAILED.value,
        "plan_id": plan.plan_id,
        "policy": plan.policy,
        "scope_mode": plan.scope_mode,
        "memory_project_key": plan.memory_project_key,
        "as_of": plan.as_of,
        "before_revision": before_revision,
        "after_revision": before_revision,
        "backup_path": backup_path,
        "mutation_committed": False,
        "should_retry": should_retry,
        "failure_stage": stage,
        "error": error,
    }


def _audit_error_history_record(
    plan: MaintenancePlan,
    result: MaintenanceApplyResult,
) -> dict[str, Any]:
    return {
        "schema_version": MAINTENANCE_SCHEMA_VERSION,
        "record_type": "audit_error",
        "status": result.status.value,
        "plan_id": plan.plan_id,
        "mutation_committed": True,
        "should_retry": False,
        "audit_error_stage": result.audit_error_stage,
        "error": result.audit_error,
    }


def _best_effort_history(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    lock_timeout_seconds: float = _HISTORY_LOCK_TIMEOUT_SECONDS,
) -> None:
    try:
        _append_maintenance_history(
            path,
            record,
            lock_timeout_seconds=lock_timeout_seconds,
        )
    except Exception:
        pass


def _apply_result_from_history(
    plan: MaintenancePlan,
    record: Mapping[str, Any],
) -> MaintenanceApplyResult:
    payload = record.get("result")
    if isinstance(payload, Mapping):
        return MaintenanceApplyResult(
            plan_id=payload["plan_id"],
            status=MaintenanceApplyStatus(payload["status"]),
            mutation_committed=payload["mutation_committed"],
            audit_complete=payload["audit_complete"],
            should_retry=payload["should_retry"],
            before_revision=payload["before_revision"],
            after_revision=payload["after_revision"],
            before_count=payload["before_count"],
            after_count=payload["after_count"],
            kept=payload["kept"],
            deleted=payload["deleted"],
            merged=payload["merged"],
            promoted=payload["promoted"],
            removed_ids=tuple(payload["removed_ids"]),
            updated_ids=tuple(payload["updated_ids"]),
            added_ids=tuple(payload["added_ids"]),
            backup_path=payload["backup_path"],
            audit_error_stage=payload["audit_error_stage"],
            audit_error=payload["audit_error"],
        )
    return _maintenance_apply_result(
        plan=plan,
        status=MaintenanceApplyStatus(record["status"]),
        mutation_committed=record["mutation_committed"],
        audit_complete=False,
        should_retry=record["should_retry"],
        before_revision=record["before_revision"],
        after_revision=record["after_revision"],
        before_count=0,
        after_count=0,
        backup_path=record["backup_path"],
    )


def _apply_result_from_intent(
    plan: MaintenancePlan,
    intent: Mapping[str, Any],
    *,
    status: MaintenanceApplyStatus,
    audit_error_stage: str,
    audit_error: str,
) -> MaintenanceApplyResult:
    return _maintenance_apply_result(
        plan=plan,
        status=status,
        mutation_committed=True,
        audit_complete=False,
        should_retry=False,
        before_revision=intent["before_revision"],
        after_revision=intent["expected_after_revision"],
        before_count=intent["before_count"],
        after_count=intent["after_count"],
        removed_ids=tuple(intent["removed_ids"]),
        updated_ids=tuple(intent["updated_ids"]),
        added_ids=tuple(intent["added_ids"]),
        backup_path=intent["backup_path"],
        audit_error_stage=audit_error_stage,
        audit_error=audit_error,
    )


def _maintenance_apply_result(
    *,
    plan: MaintenancePlan,
    status: MaintenanceApplyStatus,
    mutation_committed: bool,
    audit_complete: bool,
    should_retry: bool,
    before_revision: str,
    after_revision: str,
    before_count: int,
    after_count: int,
    removed_ids: tuple[str, ...] = (),
    updated_ids: tuple[str, ...] = (),
    added_ids: tuple[str, ...] = (),
    backup_path: str = "",
    audit_error_stage: str = "",
    audit_error: str = "",
) -> MaintenanceApplyResult:
    counts = _operation_summary(plan.operations)
    return MaintenanceApplyResult(
        plan_id=plan.plan_id,
        status=status,
        mutation_committed=mutation_committed,
        audit_complete=audit_complete,
        should_retry=should_retry,
        before_revision=before_revision,
        after_revision=after_revision,
        before_count=before_count,
        after_count=after_count,
        kept=counts[MaintenanceAction.KEEP.value],
        deleted=counts[MaintenanceAction.DELETE.value],
        merged=counts[MaintenanceAction.MERGE.value],
        promoted=counts[MaintenanceAction.PROMOTE.value],
        removed_ids=removed_ids,
        updated_ids=updated_ids,
        added_ids=added_ids,
        backup_path=backup_path,
        audit_error_stage=audit_error_stage,
        audit_error=audit_error,
    )


def _pre_commit_failure_result(
    *,
    plan: MaintenancePlan,
    stage: str,
    error: Exception,
    before_revision: str,
    before_count: int,
    backup_path: str,
    should_retry: bool,
) -> MaintenanceApplyResult:
    return _maintenance_apply_result(
        plan=plan,
        status=MaintenanceApplyStatus.PRE_COMMIT_FAILED,
        mutation_committed=False,
        audit_complete=False,
        should_retry=should_retry,
        before_revision=before_revision,
        after_revision=before_revision,
        before_count=before_count,
        after_count=before_count,
        backup_path=backup_path,
        audit_error_stage=stage,
        audit_error=_safe_error(error),
    )


def _safe_error(error: Exception) -> str:
    return type(error).__name__[:120]


def _validate_history_lock_timeout(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("history lock timeout must be finite and non-negative")


def _is_retryable_pre_commit_failure(stage: str, error: Exception) -> bool:
    if isinstance(error, (MaintenanceHistoryLockTimeout, MemoryStoreLockTimeout)):
        return True
    return stage in {"backup", "audit_intent", "persist"} and isinstance(
        error,
        OSError,
    )


__all__ = [
    "MaintenanceHistoryLockTimeout",
    "apply_maintenance_plan",
    "record_post_commit_audit_error",
]
