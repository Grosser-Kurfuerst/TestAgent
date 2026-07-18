"""Pure repository reduction and safety-only maintenance validation."""

from __future__ import annotations

from collections.abc import Sequence

from my_agent.memory.evolver.contracts import (
    MaintenanceAction,
    MaintenanceOperation,
    MaintenancePlanError,
    _source_precondition,
)
from my_agent.memory.experience.repository_rules import experience_dedup_key
from my_agent.memory.experience.models import ExperienceCreatedBy, ExperienceMemory, ExperienceTier
from my_agent.memory.types import MemoryScope, content_fingerprint


FORMAL_MUTATION_ACTIONS = frozenset({MaintenanceAction.DELETE, MaintenanceAction.MERGE})


def validate_formal_operations(
    entries: Sequence[ExperienceMemory],
    operations: Sequence[MaintenanceOperation],
    *,
    project_key: str,
) -> None:
    if not project_key:
        raise MaintenancePlanError("formal maintenance requires project_key")
    by_id = _repository_by_id(entries)
    claimed_sources: set[str] = set()
    mutation_ids: set[str] = set()
    for operation in operations:
        if operation.action not in FORMAL_MUTATION_ACTIONS:
            raise MaintenancePlanError(
                "formal maintenance supports only merge and delete staged mutations"
            )
        overlap = claimed_sources.intersection(operation.source_ids)
        if overlap:
            raise MaintenancePlanError(f"maintenance source is used more than once: {min(overlap)}")
        claimed_sources.update(operation.source_ids)
        sources: list[ExperienceMemory] = []
        for source_id, source_tier in zip(operation.source_ids, operation.source_tiers):
            source = by_id.get(source_id)
            if source is None:
                raise MaintenancePlanError(f"source id is absent from repository: {source_id}")
            if source.tier.value != source_tier:
                raise MaintenancePlanError(f"source tier does not match repository: {source_id}")
            if operation.source_preconditions[source_id] != _source_precondition(source, source_tier):
                raise MaintenancePlanError(f"source precondition mismatch: {source_id}")
            _validate_mutable_source(source, project_key=project_key)
            sources.append(source)
        if operation.action == MaintenanceAction.DELETE:
            _validate_delete(operation)
        else:
            _validate_merge(operation, sources=sources, project_key=project_key)
        current_mutation_ids = set(operation.remove_ids)
        current_mutation_ids.update(item.id for item in operation.replacements)
        current_mutation_ids.update(item.id for item in operation.additions)
        conflict = mutation_ids.intersection(current_mutation_ids)
        if conflict:
            raise MaintenancePlanError(f"maintenance mutation id is reused: {min(conflict)}")
        mutation_ids.update(current_mutation_ids)

    final_entries = reduce_repository(entries, operations, validate=False)
    seen_dedup: dict[tuple[str, str, str, str], str] = {}
    for entry in final_entries:
        key = experience_dedup_key(entry)
        previous = seen_dedup.get(key)
        if previous is not None:
            raise MaintenancePlanError(
                f"duplicate repository dedup identity after maintenance: {previous}, {entry.id}"
            )
        seen_dedup[key] = entry.id


def reduce_repository(
    entries: Sequence[ExperienceMemory],
    operations: Sequence[MaintenanceOperation],
    *,
    validate: bool = True,
    project_key: str = "",
) -> list[ExperienceMemory]:
    if validate:
        validate_formal_operations(entries, operations, project_key=project_key)
    by_id = _repository_by_id(entries)
    for operation in operations:
        for memory_id in operation.remove_ids:
            by_id.pop(memory_id, None)
        for replacement in operation.replacements:
            by_id[replacement.id] = replacement
        for addition in operation.additions:
            by_id[addition.id] = addition
    return sorted(by_id.values(), key=lambda entry: entry.id)


def _repository_by_id(entries: Sequence[ExperienceMemory]) -> dict[str, ExperienceMemory]:
    by_id: dict[str, ExperienceMemory] = {}
    for entry in entries:
        if entry.id in by_id:
            raise MaintenancePlanError(f"duplicate repository id: {entry.id}")
        if entry.fingerprint != content_fingerprint(entry.content):
            raise MaintenancePlanError(f"repository fingerprint mismatch: {entry.id}")
        by_id[entry.id] = entry
    return by_id


def _validate_mutable_source(source: ExperienceMemory, *, project_key: str) -> None:
    if source.scope == MemoryScope.GLOBAL:
        raise MaintenancePlanError(f"global experience is protected: {source.id}")
    if source.project_key != project_key:
        raise MaintenancePlanError(f"maintenance source crosses project boundary: {source.id}")
    if source.protected:
        raise MaintenancePlanError(f"protected experience cannot be mutated: {source.id}")
    if source.created_by not in {ExperienceCreatedBy.WRITER, ExperienceCreatedBy.MAINTENANCE}:
        raise MaintenancePlanError(f"manual experience cannot be mutated: {source.id}")


def _validate_delete(operation: MaintenanceOperation) -> None:
    if operation.target_ids or operation.replacements or operation.additions:
        raise MaintenancePlanError("delete cannot contain targets or replacement payloads")
    if set(operation.remove_ids) != set(operation.source_ids):
        raise MaintenancePlanError("delete must remove every source exactly")


def _validate_merge(
    operation: MaintenanceOperation,
    *,
    sources: Sequence[ExperienceMemory],
    project_key: str,
) -> None:
    if len(sources) < 2:
        raise MaintenancePlanError("merge requires at least two sources")
    tiers = {source.tier for source in sources}
    scopes = {source.scope for source in sources}
    projects = {source.project_key for source in sources}
    if len(tiers) != 1 or ExperienceTier.TRAJECTORY in tiers:
        raise MaintenancePlanError("merge sources must share one non-trajectory tier")
    if scopes != {MemoryScope.PROJECT} or projects != {project_key}:
        raise MaintenancePlanError("merge sources must share the formal project scope")
    if len(operation.target_ids) != 1 or operation.target_ids[0] not in operation.source_ids:
        raise MaintenancePlanError("merge target must be one source anchor")
    if len(operation.replacements) != 1 or operation.replacements[0].id != operation.target_ids[0]:
        raise MaintenancePlanError("merge must replace exactly its anchor")
    if set(operation.remove_ids) != set(operation.source_ids) - set(operation.target_ids):
        raise MaintenancePlanError("merge must remove every non-anchor source")
    if operation.additions:
        raise MaintenancePlanError("merge cannot add unrelated entries")
    anchor = next(source for source in sources if source.id == operation.target_ids[0])
    replacement = operation.replacements[0]
    if replacement.tier != anchor.tier:
        raise MaintenancePlanError("merge replacement tier mismatch")
    if replacement.scope != anchor.scope or replacement.project_key != anchor.project_key:
        raise MaintenancePlanError("merge replacement scope/project mismatch")
    if replacement.created_by != ExperienceCreatedBy.MAINTENANCE:
        raise MaintenancePlanError("merge replacement must be maintenance-created")
    if replacement.maintenance_operation_id != operation.operation_id:
        raise MaintenancePlanError("merge replacement operation id mismatch")


__all__ = [
    "FORMAL_MUTATION_ACTIONS",
    "reduce_repository",
    "validate_formal_operations",
]
