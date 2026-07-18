"""Formal staged-operation transaction with intent/completion recovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from my_agent.memory.evolver.maintenance.formal.history import (
    append_formal_maintenance_history,
    formal_completion_record,
    formal_intent_record,
    formal_maintenance_plan_id,
    formal_maintenance_transaction_id,
    load_formal_maintenance_history,
)
from my_agent.memory.evolver.maintenance.contracts import (
    MaintenanceOperation,
    MaintenancePlanError,
)
from my_agent.memory.evolver.maintenance.repository_reducer import (
    reduce_repository,
    validate_formal_operations,
)
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.experience.repository_rules import experience_memories_revision
from my_agent.memory.store_errors import (
    MemoryStorePostCommitError,
    MemoryStoreRevisionConflict,
)


@dataclass(frozen=True)
class FormalMaintenanceApplyResult:
    status: str
    cadence_id: str
    plan_id: str
    transaction_id: str
    before_revision: str
    after_revision: str
    operation_ids: tuple[str, ...]
    error: str = ""


def apply_formal_maintenance_operations(
    *,
    store: ExperienceStore,
    cadence_id: str,
    stream_id: str,
    expected_revision: str,
    project_key: str,
    operations: Sequence[MaintenanceOperation],
    history_path: str | Path,
) -> FormalMaintenanceApplyResult:
    with store.exclusive_process_lock():
        return _apply_formal_maintenance_operations_locked(
            store=store,
            cadence_id=cadence_id,
            stream_id=stream_id,
            expected_revision=expected_revision,
            project_key=project_key,
            operations=operations,
            history_path=history_path,
        )


def _apply_formal_maintenance_operations_locked(
    *,
    store: ExperienceStore,
    cadence_id: str,
    stream_id: str,
    expected_revision: str,
    project_key: str,
    operations: Sequence[MaintenanceOperation],
    history_path: str | Path,
) -> FormalMaintenanceApplyResult:
    snapshot = store.load_strict_snapshot()
    operation_ids = tuple(operation.operation_id for operation in operations)
    operation_payloads = tuple(operation.to_dict() for operation in operations)
    history = load_formal_maintenance_history(history_path, cadence_id=cadence_id)
    history_record = history.completion or history.intent
    if history_record is not None:
        expected_after = str(
            history_record.get("after_revision")
            or history_record.get("expected_after_revision")
            or ""
        )
    elif snapshot.revision != expected_revision:
        plan_id = formal_maintenance_plan_id(
            cadence_id=cadence_id,
            before_revision=expected_revision,
            expected_after_revision=expected_revision,
            operations=operation_payloads,
        )
        return FormalMaintenanceApplyResult(
            "stale",
            cadence_id,
            plan_id,
            formal_maintenance_transaction_id(
                cadence_id=cadence_id,
                plan_id=plan_id,
            ),
            snapshot.revision,
            snapshot.revision,
            operation_ids,
            "repository_revision_changed",
        )
    else:
        validate_formal_operations(
            snapshot.memories,
            operations,
            project_key=project_key,
        )
        next_entries = reduce_repository(
            snapshot.memories,
            operations,
            validate=False,
        )
        expected_after = experience_memories_revision(next_entries)
    plan_id = formal_maintenance_plan_id(
        cadence_id=cadence_id,
        before_revision=expected_revision,
        expected_after_revision=expected_after,
        operations=operation_payloads,
    )
    transaction_id = formal_maintenance_transaction_id(
        cadence_id=cadence_id,
        plan_id=plan_id,
    )
    if history_record is not None:
        _validate_formal_history_identity(
            history_record,
            plan_id=plan_id,
            stream_id=stream_id,
            project_key=project_key,
            transaction_id=transaction_id,
            expected_revision=expected_revision,
            expected_after=expected_after,
            operation_payloads=operation_payloads,
        )
    if history.completion is not None:
        return FormalMaintenanceApplyResult(
            str(history.completion["status"]),
            cadence_id,
            plan_id,
            transaction_id,
            expected_revision,
            str(history.completion["after_revision"]),
            operation_ids,
        )
    if snapshot.revision != expected_revision:
        if history.intent is not None and snapshot.revision == expected_after:
            append_formal_maintenance_history(
                history_path,
                formal_completion_record(
                    cadence_id=cadence_id,
                    plan_id=plan_id,
                    transaction_id=transaction_id,
                    stream_id=stream_id,
                    memory_project_key=project_key,
                    status="committed",
                    before_revision=expected_revision,
                    after_revision=expected_after,
                    operations=operation_payloads,
                ),
            )
            return FormalMaintenanceApplyResult(
                "committed",
                cadence_id,
                plan_id,
                transaction_id,
                expected_revision,
                expected_after,
                operation_ids,
            )
        return FormalMaintenanceApplyResult(
            "stale",
            cadence_id,
            plan_id,
            transaction_id,
            snapshot.revision,
            snapshot.revision,
            operation_ids,
            "repository_revision_changed",
        )
    validate_formal_operations(
        snapshot.memories,
        operations,
        project_key=project_key,
    )
    next_entries = reduce_repository(
        snapshot.memories,
        operations,
        validate=False,
    )
    if experience_memories_revision(next_entries) != expected_after:
        raise MaintenancePlanError(
            "formal maintenance recovery operations do not match the recorded revision"
        )
    if tuple(next_entries) == tuple(
        sorted(snapshot.memories, key=lambda entry: entry.id)
    ):
        if history.intent is not None:
            raise MaintenancePlanError(
                "noop maintenance cannot reuse a mutation intent"
            )
        append_formal_maintenance_history(
            history_path,
            formal_completion_record(
                cadence_id=cadence_id,
                plan_id=plan_id,
                transaction_id=transaction_id,
                stream_id=stream_id,
                memory_project_key=project_key,
                status="noop",
                before_revision=expected_revision,
                after_revision=expected_revision,
                operations=operation_payloads,
            ),
        )
        return FormalMaintenanceApplyResult(
            "noop",
            cadence_id,
            plan_id,
            transaction_id,
            snapshot.revision,
            snapshot.revision,
            operation_ids,
        )
    if history.intent is None:
        append_formal_maintenance_history(
            history_path,
            formal_intent_record(
                cadence_id=cadence_id,
                plan_id=plan_id,
                transaction_id=transaction_id,
                stream_id=stream_id,
                memory_project_key=project_key,
                before_revision=expected_revision,
                expected_after_revision=expected_after,
                operations=operation_payloads,
            ),
        )
    try:
        after_revision = store.replace_all_atomically(
            next_entries,
            expected_revision=expected_revision,
        )
    except MemoryStoreRevisionConflict:
        return FormalMaintenanceApplyResult(
            "stale",
            cadence_id,
            plan_id,
            transaction_id,
            expected_revision,
            store.revision(),
            operation_ids,
            "repository_revision_changed",
        )
    except MemoryStorePostCommitError as exc:
        try:
            recovered = store.load_strict_snapshot()
        except Exception:
            raise exc
        if (
            recovered.revision != exc.expected_revision
            or recovered.revision != expected_after
        ):
            raise exc
        after_revision = recovered.revision
    append_formal_maintenance_history(
        history_path,
        formal_completion_record(
            cadence_id=cadence_id,
            plan_id=plan_id,
            transaction_id=transaction_id,
            stream_id=stream_id,
            memory_project_key=project_key,
            status="committed",
            before_revision=expected_revision,
            after_revision=after_revision,
            operations=operation_payloads,
        ),
    )
    return FormalMaintenanceApplyResult(
        "committed",
        cadence_id,
        plan_id,
        transaction_id,
        expected_revision,
        after_revision,
        operation_ids,
    )


def _validate_formal_history_identity(
    record: Mapping[str, Any],
    *,
    plan_id: str,
    stream_id: str,
    project_key: str,
    transaction_id: str,
    expected_revision: str,
    expected_after: str,
    operation_payloads: tuple[Mapping[str, Any], ...],
) -> None:
    expected = {
        "plan_id": plan_id,
        "transaction_id": transaction_id,
        "stream_id": stream_id,
        "memory_project_key": project_key,
        "before_revision": expected_revision,
        "operations": list(operation_payloads),
    }
    for field_name, value in expected.items():
        if record[field_name] != value:
            raise MaintenancePlanError(
                f"formal maintenance history {field_name} does not match the staged plan"
            )
    recorded_after = (
        record["expected_after_revision"]
        if record["record_type"] == "intent"
        else record["after_revision"]
    )
    if recorded_after != expected_after:
        raise MaintenancePlanError(
            "formal maintenance history after revision does not match the staged plan"
        )


__all__ = [
    "FormalMaintenanceApplyResult",
    "apply_formal_maintenance_operations",
]
