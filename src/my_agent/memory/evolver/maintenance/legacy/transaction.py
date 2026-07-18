"""Transactional apply, backup, and audit history for reviewed legacy plans."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
import json
import math
import os


from my_agent.memory.evolver.maintenance.legacy.artifacts import (
    _MaintenanceArtifactGraph,
    _artifact_paths_alias,
    _maintenance_backup_path,  # noqa: F401 - compatibility for maintenance fault tests
    _resolve_maintenance_artifact_graph,
    _validate_maintenance_artifact_graph,
)
from my_agent.memory.evolver.maintenance.contracts import (
    MaintenanceAction,
    MaintenanceApplyResult,
    MaintenanceApplyStatus,
    MaintenancePlan,
    MaintenancePlanError,
    _validated_payload_entry,
    maintenance_plan_json,
)
from my_agent.memory.evolver.maintenance.repository_reducer import reduce_repository
from my_agent.memory.experience.repository_rules import experience_memories_revision
from my_agent.memory.evolver.maintenance.legacy.validation import (
    parse_maintenance_plan,
    validate_plan_semantics,
)
from my_agent.memory.evolver.maintenance.legacy.history_io import (
    MaintenanceHistoryLockTimeout,
    _append_maintenance_history,
    _load_maintenance_history_state,
)
from my_agent.memory.evolver.maintenance.legacy.history_state import (
    _audit_error_history_record,
    _best_effort_history,
    _completion_history_record,
    _intent_history_record,
    _is_retryable_pre_commit_failure,
    _maintenance_apply_result,
    _pre_commit_failure_result,
    _pre_commit_history_record,
    _safe_error,
    _terminal_history_result,
)
from my_agent.memory.experience.models import ExperienceMemory
from my_agent.memory.store_errors import (
    MemoryStoreLockTimeout,
    MemoryStorePostCommitError,
    MemoryStoreRevisionConflict,
)
from my_agent.memory.types import MemoryScope

if TYPE_CHECKING:
    from my_agent.memory.experience.repository import ExperienceStore


_HISTORY_LOCK_TIMEOUT_SECONDS = 30.0




def apply_maintenance_plan(
    *,
    store: ExperienceStore,
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
    store: ExperienceStore,
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
        payload.id
        for op in plan.operations
        for payload in op.replacements
    }))
    added_ids = tuple(sorted({
        payload.id
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
                before_count = len(snapshot.memories)
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

                validate_plan_semantics(plan, repository_entries=snapshot.memories)
                _validate_apply_project_boundaries(plan, snapshot.memories)
                next_entries = reduce_repository(snapshot.memories, plan.operations, validate=False)
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

                expected_after_revision = experience_memories_revision(next_entries)
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
                if written.revision != after_revision or len(written.memories) != after_count:
                    raise MaintenancePlanError("post-commit repository verification mismatch")
                written_ids = {entry.id for entry in written.memories}
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
    if result.plan_id != plan.plan_id:
        raise ValueError("post-commit audit result does not match plan")
    if not result.mutation_committed:
        raise ValueError("post-commit audit errors require a committed mutation")
    if result.status not in {
        MaintenanceApplyStatus.COMMITTED,
        MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR,
    }:
        raise ValueError("post-commit audit result must have a committed status")
    if result.should_retry:
        raise ValueError("post-commit audit result cannot be retryable")
    if result.before_revision != plan.repository_revision:
        raise ValueError("post-commit audit result revision does not match plan")
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
    entries: Sequence[ExperienceMemory],
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


def _atomic_write_tmp_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(target.suffix + ".tmp")


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








from my_agent.memory.evolver.maintenance.formal.transaction import (  # noqa: E402
    FormalMaintenanceApplyResult,
    apply_formal_maintenance_operations,
)
from my_agent.memory.evolver.maintenance.cadence.ledger import (  # noqa: E402
    formal_maintenance_transaction_id,
)


__all__ = [
    "FormalMaintenanceApplyResult",
    "MaintenanceHistoryLockTimeout",
    "apply_formal_maintenance_operations",
    "apply_maintenance_plan",
    "formal_maintenance_transaction_id",
    "record_post_commit_audit_error",
]
