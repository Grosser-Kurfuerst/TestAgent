"""Transactional apply, backup, and audit history for maintenance plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import os

from my_agent.memory.evolver.contracts import (
    MAINTENANCE_SCHEMA_VERSION,
    MaintenanceAction,
    MaintenanceApplyResult,
    MaintenanceApplyStatus,
    MaintenancePlan,
    MaintenancePlanError,
    _as_int,
    _operation_summary,
    _string_tuple,
    _validated_payload_entry,
    maintenance_plan_json,
)
from my_agent.memory.evolver.planner import (
    _repository_after_operations,
    validate_plan_semantics,
)
from my_agent.memory.long_term import (
    LongTermMemoryStore,
    MemoryStoreLockTimeout,
    MemoryStorePostCommitError,
    MemoryStoreRevisionConflict,
    memory_entries_revision,
)
from my_agent.memory.types import MemoryEntry, MemoryScope
from my_agent.text_safety import sanitize_json_value

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

                stage = "validation"
                history_state = _load_maintenance_history_state(history_path, plan.plan_id)
                terminal = _terminal_history_result(
                    plan,
                    history_state,
                    current_revision=snapshot.revision,
                )
                if terminal is not None:
                    return terminal
                if history_state.intent is not None:
                    intent_before = str(history_state.intent.get("before_revision") or "")
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
                        append_maintenance_history(
                            history_path,
                            _completion_history_record(plan, result),
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
                backup = _maintenance_backup_path(backup_dir, plan.plan_id)
                _write_backup_atomic(backup, snapshot.raw_bytes)
                backup_path = str(backup)

                expected_after_revision = memory_entries_revision(next_entries)
                if reuse_intent:
                    recorded_after = str(
                        (history_state.intent or {}).get("expected_after_revision") or ""
                    )
                    if recorded_after and recorded_after != expected_after_revision:
                        raise MaintenancePlanError(
                            "incomplete maintenance intent does not match reviewed plan"
                        )
                else:
                    stage = "audit_intent"
                    append_maintenance_history(
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
                    )
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
                    append_maintenance_history(
                        history_path,
                        _completion_history_record(plan, committed),
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


def append_maintenance_history(
    path: str | Path,
    record: Mapping[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_json_value(dict(record))
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return output


def _validated_plan_copy(plan: MaintenancePlan) -> MaintenancePlan:
    """Rebuild a plan so mutable payload aliases cannot bypass identity checks."""
    payload = json.loads(maintenance_plan_json(plan))
    if not isinstance(payload, dict):
        raise MaintenancePlanError("maintenance plan must be a JSON object")
    return MaintenancePlan.from_dict(payload)


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
    tmp = path.with_suffix(path.suffix + ".tmp")
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


def _maintenance_backup_path(backup_dir: str | Path, plan_id: str) -> Path:
    root = Path(backup_dir).resolve()
    candidate = (root / f"{plan_id}.long_term_memory.jsonl").resolve()
    if candidate.parent != root:
        raise MaintenancePlanError("maintenance backup path escapes backup directory")
    return candidate


def _load_maintenance_history_state(
    path: str | Path,
    plan_id: str,
) -> _MaintenanceHistoryState:
    source = Path(path)
    if not source.exists():
        return _MaintenanceHistoryState()
    intent: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None
    audit_error: dict[str, Any] | None = None
    audit_stages: set[str] = set()
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
                    f"invalid maintenance history at line {line_no}: {type(exc).__name__}"
                ) from exc
            if str(payload.get("plan_id") or "") != plan_id:
                continue
            record_type = str(payload.get("record_type") or "")
            if record_type == "intent":
                if intent is not None:
                    raise MaintenancePlanError("duplicate maintenance intent record")
                intent = payload
            elif record_type == "completion":
                if completion is not None:
                    raise MaintenancePlanError("duplicate maintenance completion record")
                completion = payload
            elif record_type == "audit_error":
                stage = str(payload.get("audit_error_stage") or "")
                if stage in audit_stages:
                    raise MaintenancePlanError("duplicate maintenance audit error record")
                audit_stages.add(stage)
                audit_error = payload
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
                audit_error_stage=str(
                    state.audit_error.get("audit_error_stage") or "history"
                ),
                audit_error=str(state.audit_error.get("error") or "audit_error"),
            )
        return result

    if state.intent is None:
        return None
    before_revision = str(state.intent.get("before_revision") or "")
    expected_after_revision = str(
        state.intent.get("expected_after_revision") or ""
    )
    if current_revision == before_revision:
        return None
    if expected_after_revision and current_revision == expected_after_revision:
        return _apply_result_from_intent(
            plan,
            state.intent,
            status=MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
            audit_error_stage=str(
                (state.audit_error or {}).get("audit_error_stage")
                or "history_recovery"
            ),
            audit_error=str((state.audit_error or {}).get("error") or "audit_incomplete"),
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
        "operations": [
            {
                "operation_id": operation.operation_id,
                "action": operation.action.value,
                "source_ids": list(operation.source_ids),
                "target_ids": list(operation.target_ids),
                "reason_codes": list(operation.reason_codes),
            }
            for operation in plan.operations
        ],
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


def _best_effort_history(path: str | Path, record: Mapping[str, Any]) -> None:
    try:
        append_maintenance_history(path, record)
    except Exception:
        pass


def _apply_result_from_history(
    plan: MaintenancePlan,
    record: Mapping[str, Any],
) -> MaintenanceApplyResult:
    payload = record.get("result")
    if isinstance(payload, Mapping):
        try:
            status = MaintenanceApplyStatus(str(payload.get("status") or ""))
        except ValueError as exc:
            raise MaintenancePlanError("invalid maintenance completion status") from exc
        return MaintenanceApplyResult(
            plan_id=plan.plan_id,
            status=status,
            mutation_committed=bool(payload.get("mutation_committed", False)),
            audit_complete=bool(payload.get("audit_complete", False)),
            should_retry=bool(payload.get("should_retry", False)),
            before_revision=str(payload.get("before_revision") or ""),
            after_revision=str(payload.get("after_revision") or ""),
            before_count=_as_int(payload.get("before_count")),
            after_count=_as_int(payload.get("after_count")),
            kept=_as_int(payload.get("kept")),
            deleted=_as_int(payload.get("deleted")),
            merged=_as_int(payload.get("merged")),
            promoted=_as_int(payload.get("promoted")),
            removed_ids=_string_tuple(payload.get("removed_ids")),
            updated_ids=_string_tuple(payload.get("updated_ids")),
            added_ids=_string_tuple(payload.get("added_ids")),
            backup_path=str(payload.get("backup_path") or ""),
            audit_error_stage=str(payload.get("audit_error_stage") or ""),
            audit_error=str(payload.get("audit_error") or ""),
        )
    try:
        status = MaintenanceApplyStatus(str(record.get("status") or ""))
    except ValueError as exc:
        raise MaintenancePlanError("invalid maintenance completion status") from exc
    return _maintenance_apply_result(
        plan=plan,
        status=status,
        mutation_committed=bool(record.get("mutation_committed", False)),
        audit_complete=status in {MaintenanceApplyStatus.COMMITTED, MaintenanceApplyStatus.NOOP},
        should_retry=bool(record.get("should_retry", False)),
        before_revision=str(record.get("before_revision") or ""),
        after_revision=str(record.get("after_revision") or ""),
        before_count=_as_int(record.get("before_count")),
        after_count=_as_int(record.get("after_count")),
        backup_path=str(record.get("backup_path") or ""),
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
        before_revision=str(intent.get("before_revision") or ""),
        after_revision=str(intent.get("expected_after_revision") or ""),
        before_count=_as_int(intent.get("before_count")),
        after_count=_as_int(intent.get("after_count")),
        removed_ids=_string_tuple(intent.get("removed_ids")),
        updated_ids=_string_tuple(intent.get("updated_ids")),
        added_ids=_string_tuple(intent.get("added_ids")),
        backup_path=str(intent.get("backup_path") or ""),
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


def _is_retryable_pre_commit_failure(stage: str, error: Exception) -> bool:
    if isinstance(error, MemoryStoreLockTimeout):
        return True
    return stage in {"backup", "audit_intent", "persist"} and isinstance(
        error,
        OSError,
    )


__all__ = [
    "append_maintenance_history",
    "apply_maintenance_plan",
]
