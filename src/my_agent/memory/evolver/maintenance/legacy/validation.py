"""Reviewed-plan parsing and semantic validation for legacy maintenance."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math

from my_agent.json_safety import loads_json_strict
from my_agent.memory.evolver.maintenance.contracts import (
    MaintenanceAction,
    MaintenanceConfig,
    MaintenanceEvidence,
    MaintenanceOperation,
    MaintenancePlan,
    MaintenancePlanError,
    _evidence_float,
    _operation_summary,
    _parse_datetime,
    _validated_payload_entry,
)
from my_agent.memory.evolver.maintenance.legacy.policies import (
    _automatic_maintenance_provenance,
    _merge_replacement_payload,
    _merge_threshold,
    _plan_operations_from_evidence,
    _promoted_target_entry,
    _promotion_eligible,
)
from my_agent.memory.evolver.maintenance.legacy.policy_validation import (
    _validate_operation_conflicts,
)
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceTier,
)
from my_agent.memory.types import MemoryScope


_PLAN_FIELDS = frozenset({
    "schema_version",
    "policy",
    "plan_id",
    "repository_revision",
    "scope_mode",
    "memory_project_key",
    "as_of",
    "config",
    "input_summary",
    "operations",
    "summary",
})


def parse_maintenance_plan(
    data: Mapping[str, Any],
    *,
    repository_entries: Sequence[ExperienceMemory] | None = None,
) -> MaintenancePlan:
    """Parse a reviewed plan and apply the full available semantic contract."""
    actual_fields = frozenset(data)
    if actual_fields != _PLAN_FIELDS:
        missing = sorted(_PLAN_FIELDS - actual_fields)
        extra = sorted(actual_fields - _PLAN_FIELDS)
        raise MaintenancePlanError(
            "maintenance plan fields mismatch: "
            f"missing={missing}, extra={extra}"
        )
    plan = MaintenancePlan.from_dict(data)
    if _canonical_json(data) != _canonical_json(plan.to_dict()):
        raise MaintenancePlanError(
            "maintenance plan must use canonical JSON field types and complete values"
        )
    validate_plan_semantics(plan, repository_entries=repository_entries)
    return plan


def load_maintenance_plan(
    path: str | Path,
    *,
    repository_entries: Sequence[ExperienceMemory] | None = None,
) -> MaintenancePlan:
    source = Path(path)
    try:
        payload = loads_json_strict(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MaintenancePlanError(f"invalid maintenance plan JSON: {exc.msg}") from exc
    except ValueError as exc:
        raise MaintenancePlanError(
            f"invalid maintenance plan JSON: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise MaintenancePlanError("maintenance plan must be a JSON object")
    return parse_maintenance_plan(payload, repository_entries=repository_entries)


def validate_legacy_plan_semantics(
    plan: MaintenancePlan,
    *,
    repository_entries: Sequence[ExperienceMemory] | None = None,
) -> None:
    """Validate action meaning independently from plan and operation digests."""
    as_of = _parse_datetime(plan.as_of)
    if as_of is None or as_of.astimezone(timezone.utc).isoformat() != plan.as_of:
        raise MaintenancePlanError("as_of must be a canonical timezone-aware UTC datetime")
    if plan.summary != _operation_summary(plan.operations):
        raise MaintenancePlanError("plan summary does not match its operations")
    config = MaintenanceConfig.from_dict(plan.config)

    for operation in plan.operations:
        for source_id in operation.source_ids:
            precondition = operation.source_preconditions[source_id]
            scope = MemoryScope(precondition["scope"])
            if scope == MemoryScope.GLOBAL:
                if operation.action != MaintenanceAction.KEEP:
                    raise MaintenancePlanError("global experience may only be kept")
            elif precondition["project_key"] != plan.memory_project_key:
                raise MaintenancePlanError("operation crosses memory project boundary")
        for payload in operation.replacements + operation.additions:
            entry = _validated_payload_entry(payload, "mutation")
            if entry.scope == MemoryScope.GLOBAL:
                raise MaintenancePlanError("maintenance cannot mutate global experience")
            if entry.project_key != plan.memory_project_key:
                raise MaintenancePlanError("mutation payload crosses memory project boundary")

    _validate_operation_conflicts(
        plan.operations,
        repository_entries=repository_entries,
    )
    repository_by_id = (
        {entry.id: entry for entry in repository_entries}
        if repository_entries is not None
        else None
    )
    all_evidence: dict[str, MaintenanceEvidence] = {}
    for operation in plan.operations:
        evidence_by_id = _validated_operation_evidence(operation)
        all_evidence.update(evidence_by_id)
        if operation.action != MaintenanceAction.KEEP:
            for evidence in evidence_by_id.values():
                if not _automatic_maintenance_provenance(evidence.created_by):
                    raise MaintenancePlanError(
                        f"destructive action has protected provenance: {evidence.memory_id}"
                    )
        if operation.action == MaintenanceAction.MERGE:
            _validate_merge_action_semantics(
                operation,
                evidence_by_id=evidence_by_id,
                as_of=as_of,
                repository_by_id=repository_by_id,
                config=config,
            )
        elif operation.action == MaintenanceAction.PROMOTE:
            _validate_promotion_action_semantics(
                operation,
                evidence_by_id=evidence_by_id,
                as_of=as_of,
                repository_by_id=repository_by_id,
                config=config,
            )

    if repository_entries is None:
        return

    assert repository_by_id is not None
    for source_id, evidence in all_evidence.items():
        _validate_snapshot_evidence(repository_by_id[source_id], evidence)
    expected_operations, expected_input_summary = _plan_operations_from_evidence(
        entries=repository_entries,
        evidence_by_id=all_evidence,
        project_key=plan.memory_project_key,
        as_of=as_of,
        config=config,
    )
    if plan.input_summary != expected_input_summary:
        raise MaintenancePlanError("plan input summary does not match repository snapshot")
    if [item.to_dict() for item in plan.operations] != [
        item.to_dict() for item in expected_operations
    ]:
        raise MaintenancePlanError(
            "plan operations do not match deterministic snapshot decisions"
        )


def validate_plan_semantics(
    plan: MaintenancePlan,
    *,
    repository_entries: Sequence[ExperienceMemory] | None = None,
) -> None:
    """Validate only shape, safety, references, preconditions, and conflicts."""

    as_of = _parse_datetime(plan.as_of)
    if as_of is None or as_of.astimezone(timezone.utc).isoformat() != plan.as_of:
        raise MaintenancePlanError("as_of must be a canonical timezone-aware UTC datetime")
    if plan.summary != _operation_summary(plan.operations):
        raise MaintenancePlanError("plan summary does not match its operations")
    MaintenanceConfig.from_dict(plan.config)
    for operation in plan.operations:
        for source_id in operation.source_ids:
            precondition = operation.source_preconditions[source_id]
            scope = MemoryScope(precondition["scope"])
            if scope == MemoryScope.GLOBAL and operation.action != MaintenanceAction.KEEP:
                raise MaintenancePlanError("global experience may only be kept")
            if scope == MemoryScope.PROJECT and precondition["project_key"] != plan.memory_project_key:
                raise MaintenancePlanError("operation crosses memory project boundary")
        for payload in operation.replacements + operation.additions:
            entry = _validated_payload_entry(payload, "mutation")
            if entry.scope == MemoryScope.GLOBAL:
                raise MaintenancePlanError("maintenance cannot mutate global experience")
            if entry.project_key != plan.memory_project_key:
                raise MaintenancePlanError("mutation payload crosses memory project boundary")

    _validate_operation_conflicts(plan.operations, repository_entries=repository_entries)
    repository_by_id = (
        {entry.id: entry for entry in repository_entries}
        if repository_entries is not None
        else None
    )
    for operation in plan.operations:
        evidence_by_id = _validated_operation_evidence(operation)
        if operation.action != MaintenanceAction.KEEP:
            for evidence in evidence_by_id.values():
                if not _automatic_maintenance_provenance(evidence.created_by):
                    raise MaintenancePlanError(
                        f"destructive action has protected provenance: {evidence.memory_id}"
                    )
        if repository_by_id is not None:
            for source_id, evidence in evidence_by_id.items():
                _validate_snapshot_evidence(repository_by_id[source_id], evidence)


def _validate_snapshot_evidence(
    entry: ExperienceMemory,
    evidence: MaintenanceEvidence,
) -> None:
    expected = {
        "memory_id": entry.id,
        "tier": entry.tier.value,
        "scope": entry.scope.value,
        "project_key": entry.project_key,
        "created_by": entry.created_by.value,
        "created_at": entry.created_at.isoformat(),
        "source_task": entry.source_task,
        "writer_confidence": _evidence_float(entry.writer_confidence, "writer_confidence"),
    }
    for name, expected_value in expected.items():
        if getattr(evidence, name) != expected_value:
            raise MaintenancePlanError(
                f"operation evidence does not match repository {name}: {entry.id}"
            )


def _validated_operation_evidence(
    operation: MaintenanceOperation,
) -> dict[str, MaintenanceEvidence]:
    if len(operation.evidence) != len(operation.source_ids):
        raise MaintenancePlanError("operation evidence must cover every source exactly")
    result: dict[str, MaintenanceEvidence] = {}
    for source_id, source_tier, raw in zip(
        operation.source_ids,
        operation.source_tiers,
        operation.evidence,
    ):
        evidence = MaintenanceEvidence.from_dict(raw)
        if raw != evidence.to_dict():
            raise MaintenancePlanError(f"operation evidence is not canonical: {source_id}")
        precondition = operation.source_preconditions[source_id]
        if evidence.memory_id != source_id or evidence.tier != source_tier:
            raise MaintenancePlanError(f"operation evidence identity mismatch: {source_id}")
        if evidence.scope != precondition["scope"]:
            raise MaintenancePlanError(f"operation evidence scope mismatch: {source_id}")
        if evidence.project_key != precondition["project_key"]:
            raise MaintenancePlanError(f"operation evidence project mismatch: {source_id}")
        result[source_id] = evidence
    return result


def _validate_merge_action_semantics(
    operation: MaintenanceOperation,
    *,
    evidence_by_id: Mapping[str, MaintenanceEvidence],
    as_of: datetime,
    repository_by_id: Mapping[str, ExperienceMemory] | None,
    config: MaintenanceConfig,
) -> None:
    if operation.reason_codes != ("near_duplicate_complete_link",):
        raise MaintenancePlanError("merge operation has invalid reason codes")
    if (
        operation.redundancy_score is None
        or not math.isfinite(operation.redundancy_score)
        or not 0.0 <= operation.redundancy_score <= 1.0
    ):
        raise MaintenancePlanError("merge redundancy score must be finite and between 0 and 1")
    tier = ExperienceTier(operation.source_tiers[0])
    if operation.redundancy_score < _merge_threshold(tier, config):
        raise MaintenancePlanError("merge redundancy score is below the configured threshold")

    replacement = _validated_payload_entry(operation.replacements[0], "merge replacement")
    if replacement.created_by != ExperienceCreatedBy.MAINTENANCE:
        raise MaintenancePlanError("merge replacement must be maintenance-created")
    if replacement.maintenance_operation_id != operation.operation_id:
        raise MaintenancePlanError("merge replacement operation id mismatch")

    if repository_by_id is None:
        return
    ordered = [
        (repository_by_id[source_id], evidence_by_id[source_id])
        for source_id in operation.source_ids
    ]
    expected_replacement = _merge_replacement_payload(
        ordered,
        as_of=as_of,
        minimum_score=operation.redundancy_score,
        operation_id=operation.operation_id,
    )
    if operation.replacements[0] != expected_replacement:
        raise MaintenancePlanError("merge replacement does not match deterministic semantics")


def _validate_promotion_action_semantics(
    operation: MaintenanceOperation,
    *,
    evidence_by_id: Mapping[str, MaintenanceEvidence],
    as_of: datetime,
    repository_by_id: Mapping[str, ExperienceMemory] | None,
    config: MaintenanceConfig,
) -> None:
    source_id = operation.source_ids[0]
    evidence = evidence_by_id[source_id]
    replacement = _validated_payload_entry(operation.replacements[0], "promotion source")
    if replacement.promoted_to != operation.target_ids[0]:
        raise MaintenancePlanError(f"promotion source target mismatch: {source_id}")
    if replacement.maintenance_operation_id != operation.operation_id:
        raise MaintenancePlanError(f"promotion source operation mismatch: {source_id}")

    if repository_by_id is not None:
        source = repository_by_id[source_id]
        expected_replacement = replace(
            source,
            promoted_to=operation.target_ids[0],
            maintenance_operation_id=operation.operation_id,
        )
        if operation.replacements[0] != expected_replacement:
            raise MaintenancePlanError("promotion source replacement changes non-lineage fields")
    else:
        source = replace(
            replacement,
            promoted_to="",
            maintenance_operation_id="",
        )

    if source.tier not in {ExperienceTier.TIP, ExperienceTier.TRAJECTORY}:
        raise MaintenancePlanError(f"invalid promotion source tier: {source_id}")
    if not _promotion_eligible(source, evidence, config=config):
        raise MaintenancePlanError(f"promotion source is not eligible: {source_id}")
    if operation.additions:
        if operation.reason_codes != ("promoted_to_skill",):
            raise MaintenancePlanError("new promotion target has invalid reason codes")
        expected_target = _promoted_target_entry(
            source,
            evidence,
            as_of=as_of,
            operation_id=operation.operation_id,
        )
        if operation.target_ids != (expected_target.id,):
            raise MaintenancePlanError("promotion target id is not deterministic")
        if operation.additions[0] != expected_target:
            raise MaintenancePlanError("promotion target does not match deterministic semantics")
    elif operation.reason_codes != ("promotion_linked_existing_skill",):
        raise MaintenancePlanError("existing promotion target has invalid reason codes")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MaintenancePlanError("maintenance plan must contain canonical JSON values") from exc


__all__ = [
    "load_maintenance_plan",
    "parse_maintenance_plan",
    "validate_legacy_plan_semantics",
    "validate_plan_semantics",
]
