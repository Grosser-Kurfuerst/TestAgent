"""Legacy maintenance history recovery and result construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from my_agent.memory.evolver.maintenance.contracts import (
    MAINTENANCE_SCHEMA_VERSION,
    MaintenanceAction,
    MaintenanceApplyResult,
    MaintenanceApplyStatus,
    MaintenancePlan,
    MaintenancePlanError,
    _operation_summary,
)
from my_agent.memory.evolver.maintenance.legacy.history_io import (
    _HISTORY_LOCK_TIMEOUT_SECONDS,
    MaintenanceHistoryLockTimeout,
    _MaintenanceHistoryState,
    _append_maintenance_history,
    _history_operations_for_plan,
)
from my_agent.memory.store_errors import MemoryStoreLockTimeout


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


def _is_retryable_pre_commit_failure(stage: str, error: Exception) -> bool:
    if isinstance(error, (MaintenanceHistoryLockTimeout, MemoryStoreLockTimeout)):
        return True
    return stage in {"backup", "audit_intent", "persist"} and isinstance(
        error,
        OSError,
    )


__all__: list[str] = []
