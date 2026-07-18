"""Repository semantic validation for deterministic legacy policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from my_agent.memory.evolver.maintenance.contracts import (
    MaintenanceAction,
    MaintenanceOperation,
    MaintenancePlanError,
    _payload_ids,
    _source_precondition,
    _validated_payload_entry,
)
from my_agent.memory.evolver.maintenance.legacy.policies import _promoted_skill_fields
from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.experience.repository_rules import experience_dedup_key
from my_agent.memory.types import MemoryScope, content_fingerprint


def _validate_operation_conflicts(
    operations: Sequence[MaintenanceOperation],
    *,
    repository_entries: Sequence[ExperienceMemory] | None = None,
) -> None:
    source_owners: dict[str, str] = {}
    remove_owners: dict[str, str] = {}
    replacement_owners: dict[str, str] = {}
    addition_owners: dict[str, str] = {}
    for operation in operations:
        for source_id in operation.source_ids:
            _claim_operation_id(source_owners, source_id, operation.operation_id, "source")
        for memory_id in operation.remove_ids:
            _claim_operation_id(remove_owners, memory_id, operation.operation_id, "remove")
        for memory_id in _payload_ids(operation.replacements, "replacement"):
            _claim_operation_id(replacement_owners, memory_id, operation.operation_id, "replacement")
        for memory_id in _payload_ids(operation.additions, "addition"):
            _claim_operation_id(addition_owners, memory_id, operation.operation_id, "addition")

    remove_ids = set(remove_owners)
    replacement_ids = set(replacement_owners)
    addition_ids = set(addition_owners)
    if remove_ids.intersection(replacement_ids | addition_ids):
        raise MaintenancePlanError("remove ids conflict with replacement/addition ids")
    if replacement_ids.intersection(addition_ids):
        raise MaintenancePlanError("replacement ids conflict with addition ids")
    if repository_entries is None:
        return

    repository_by_id: dict[str, ExperienceMemory] = {}
    for entry in repository_entries:
        if entry.id in repository_by_id:
            raise MaintenancePlanError(f"duplicate repository id: {entry.id}")
        expected_fingerprint = content_fingerprint(entry.content)
        if entry.fingerprint and entry.fingerprint != expected_fingerprint:
            raise MaintenancePlanError(f"repository fingerprint mismatch: {entry.id}")
        repository_by_id[entry.id] = entry
    existing_ids = set(repository_by_id)
    if addition_ids.intersection(existing_ids):
        conflict = min(addition_ids.intersection(existing_ids))
        raise MaintenancePlanError(f"addition id already exists in repository: {conflict}")

    for operation in operations:
        for source_id, tier in zip(operation.source_ids, operation.source_tiers):
            entry = repository_by_id.get(source_id)
            if entry is None:
                raise MaintenancePlanError(f"source id is absent from repository: {source_id}")
            actual_tier = entry.tier
            if actual_tier.value != tier:
                raise MaintenancePlanError(f"source tier does not match repository: {source_id}")
            expected = _source_precondition(entry, tier)
            if operation.source_preconditions[source_id] != expected:
                raise MaintenancePlanError(f"source precondition mismatch: {source_id}")

    final_entries = _repository_after_operations(repository_entries, operations, validate=False)
    final_by_id = {entry.id: entry for entry in final_entries}
    for operation in operations:
        if operation.action == MaintenanceAction.MERGE:
            _validate_merge_repository_contract(
                operation,
                repository_by_id=repository_by_id,
                final_by_id=final_by_id,
            )
        if operation.action != MaintenanceAction.PROMOTE:
            continue
        source = repository_by_id[operation.source_ids[0]]
        target = final_by_id.get(operation.target_ids[0])
        if target is None:
            raise MaintenancePlanError(
                f"promotion target is absent after planning: {operation.target_ids[0]}"
            )
        _validate_promotion_target(operation, source=source, target=target, final_by_id=final_by_id)

    seen_dedup_keys: dict[tuple[str, str, str, str], str] = {}
    for entry in final_entries:
        key = experience_dedup_key(entry)
        previous = seen_dedup_keys.get(key)
        if previous is not None:
            raise MaintenancePlanError(
                f"duplicate repository dedup identity after planning: {previous}, {entry.id}"
            )
        seen_dedup_keys[key] = entry.id


def _validate_merge_repository_contract(
    operation: MaintenanceOperation,
    *,
    repository_by_id: Mapping[str, ExperienceMemory],
    final_by_id: Mapping[str, ExperienceMemory],
) -> None:
    sources = [repository_by_id[source_id] for source_id in operation.source_ids]
    tiers = {entry.tier for entry in sources}
    scopes = {entry.scope for entry in sources}
    projects = {
        "" if entry.scope == MemoryScope.GLOBAL else entry.project_key
        for entry in sources
    }
    if len(tiers) != 1 or ExperienceTier.TRAJECTORY in tiers:
        raise MaintenancePlanError("merge repository sources must have one non-trajectory tier")
    if len(scopes) != 1 or len(projects) != 1:
        raise MaintenancePlanError("merge repository sources must share scope and project")

    anchor = repository_by_id[operation.target_ids[0]]
    replacement = final_by_id.get(anchor.id)
    if replacement is None:
        raise MaintenancePlanError(f"merge anchor replacement is absent: {anchor.id}")
    mutable_fields = {"payload", "created_by", "maintenance_operation_id"}
    preserved_fields = (
        field_name
        for field_name in anchor.__dataclass_fields__
        if field_name not in mutable_fields
    )
    if any(
        getattr(replacement, field_name) != getattr(anchor, field_name)
        for field_name in preserved_fields
    ):
        raise MaintenancePlanError(f"merge replacement does not preserve anchor identity: {anchor.id}")


def _repository_after_operations(
    entries: Sequence[ExperienceMemory],
    operations: Sequence[MaintenanceOperation],
    *,
    validate: bool = True,
) -> list[ExperienceMemory]:
    if validate:
        _validate_operation_conflicts(operations, repository_entries=entries)
    by_id = {entry.id: entry for entry in entries}
    for operation in operations:
        for memory_id in operation.remove_ids:
            by_id.pop(memory_id, None)
        for payload in operation.replacements:
            replacement = _validated_payload_entry(payload, "replacement")
            by_id[replacement.id] = replacement
        for payload in operation.additions:
            addition = _validated_payload_entry(payload, "addition")
            by_id[addition.id] = addition
    return sorted(by_id.values(), key=lambda entry: entry.id)


def _validate_promotion_target(
    operation: MaintenanceOperation,
    *,
    source: ExperienceMemory,
    target: ExperienceMemory,
    final_by_id: Mapping[str, ExperienceMemory],
) -> None:
    if source.tier not in {ExperienceTier.TIP, ExperienceTier.TRAJECTORY}:
        raise MaintenancePlanError(f"invalid promotion source tier: {source.id}")
    if target.tier != ExperienceTier.SKILL:
        raise MaintenancePlanError(f"promotion target is not a skill: {target.id}")
    expected_content, _ = _promoted_skill_fields(source)
    expected_fingerprint = content_fingerprint(expected_content)
    if target.fingerprint != expected_fingerprint:
        raise MaintenancePlanError(f"promotion target fingerprint mismatch: {target.id}")
    if target.scope != source.scope:
        raise MaintenancePlanError(f"promotion target scope mismatch: {target.id}")
    if target.scope != MemoryScope.GLOBAL and target.project_key != source.project_key:
        raise MaintenancePlanError(f"promotion target project mismatch: {target.id}")

    updated_source = final_by_id.get(source.id)
    if updated_source is None:
        raise MaintenancePlanError(f"promotion source is absent after planning: {source.id}")
    if updated_source.promoted_to != target.id:
        raise MaintenancePlanError(f"promotion source target metadata mismatch: {source.id}")
    if updated_source.maintenance_operation_id != operation.operation_id:
        raise MaintenancePlanError(f"promotion source operation metadata mismatch: {source.id}")


def _claim_operation_id(
    owners: dict[str, str],
    memory_id: str,
    operation_id: str,
    kind: str,
) -> None:
    previous = owners.get(memory_id)
    if previous is not None:
        raise MaintenancePlanError(
            f"{kind} id {memory_id} is claimed by both {previous} and {operation_id}"
        )
    owners[memory_id] = operation_id


__all__: list[str] = []
