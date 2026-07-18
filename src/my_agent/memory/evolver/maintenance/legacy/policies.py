"""Deterministic retention, merge, and promotion policies for legacy maintenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256

from my_agent.memory.evolver.attribution import MemoryAttributionRecord
from my_agent.memory.evolver.maintenance.contracts import (
    AttributionKey,
    MAINTENANCE_POLICY,
    MAINTENANCE_SCHEMA_VERSION,
    MAINTENANCE_SCOPE_MODE,
    MaintenanceAction,
    MaintenanceConfig,
    MaintenanceEvidence,
    MaintenanceOperation,
    MaintenancePlan,
    MaintenancePlanError,
    _operation_id,
    _operation_summary,
    _plan_id,
    _require_aware_datetime,
    _source_precondition,
)
from my_agent.memory.evolver.maintenance.legacy.planner import (
    _maintenance_evidence_for_entry,
    _validate_attribution_mapping,
)
from my_agent.memory.evolver.maintenance.legacy.policy_helpers import (
    _anchor_priority,
    _has_sufficient_retention_evidence,
    _merge_pair_score,
    _merge_threshold,
    _negative_delete_eligible,
    _ordered_payload_union,
    _promotion_priority,
    _stable_title,
    _stale_delete_eligible,
    _successful_step_summaries,
)
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceTier,
    SkillPayload,
    TipPayload,
    TrajectoryPayload,
)
from my_agent.memory.experience.repository_rules import experience_dedup_key
from my_agent.memory.types import MemoryScope, content_fingerprint
from my_agent.memory.token import estimate_tokens


def _build_maintenance_plan(
    *,
    entries: Sequence[ExperienceMemory],
    attribution: Mapping[AttributionKey, MemoryAttributionRecord],
    repository_revision: str,
    project_key: str,
    as_of: datetime,
    config: MaintenanceConfig | None = None,
) -> MaintenancePlan:
    """Build a deterministic single-project maintenance plan."""
    if not project_key:
        raise ValueError("project_key must not be empty")
    if not repository_revision:
        raise ValueError("repository_revision must not be empty")
    as_of_utc = _require_aware_datetime(as_of, "as_of").astimezone(timezone.utc)
    cfg = config or MaintenanceConfig()
    _validate_attribution_mapping(attribution, project_key=project_key)

    evidence_by_id: dict[str, MaintenanceEvidence] = {}
    for entry in entries:
        if not _entry_visible_to_project(entry, project_key):
            continue
        evidence_by_id[entry.id] = _maintenance_evidence_for_entry(
            entry,
            attribution=attribution,
            project_key=project_key,
        )
    operation_tuple, input_summary = _plan_operations_from_evidence(
        entries=entries,
        evidence_by_id=evidence_by_id,
        project_key=project_key,
        as_of=as_of_utc,
        config=cfg,
    )

    summary = _operation_summary(operation_tuple)
    as_of_text = as_of_utc.isoformat()
    config_payload = cfg.to_dict()
    plan_id = _plan_id(
        repository_revision=repository_revision,
        project_key=project_key,
        as_of=as_of_text,
        config=config_payload,
        input_summary=input_summary,
        operations=operation_tuple,
        summary=summary,
    )
    return MaintenancePlan(
        schema_version=MAINTENANCE_SCHEMA_VERSION,
        policy=MAINTENANCE_POLICY,
        plan_id=plan_id,
        repository_revision=repository_revision,
        scope_mode=MAINTENANCE_SCOPE_MODE,
        memory_project_key=project_key,
        as_of=as_of_text,
        config=config_payload,
        input_summary=input_summary,
        operations=operation_tuple,
        summary=summary,
    )


def _plan_operations_from_evidence(
    *,
    entries: Sequence[ExperienceMemory],
    evidence_by_id: Mapping[str, MaintenanceEvidence],
    project_key: str,
    as_of: datetime,
    config: MaintenanceConfig,
) -> tuple[tuple[MaintenanceOperation, ...], dict[str, int]]:
    from my_agent.memory.evolver.maintenance.legacy.policy_validation import (
        _repository_after_operations,
    )

    considered: list[tuple[ExperienceMemory, MaintenanceEvidence]] = []
    for entry in entries:
        if not _entry_visible_to_project(entry, project_key):
            continue
        evidence = evidence_by_id.get(entry.id)
        if evidence is None:
            raise MaintenancePlanError(f"maintenance evidence is missing: {entry.id}")
        considered.append((entry, evidence))
    considered.sort(key=lambda item: (item[1].tier, item[0].id))

    operations: list[MaintenanceOperation] = []
    remaining: list[tuple[ExperienceMemory, MaintenanceEvidence]] = []
    retention_reasons: dict[str, str] = {}
    protected_reasons = {
        "protected_metadata",
        "protected_global",
        "protected_manual",
        "protected_unknown_provenance",
    }
    for entry, evidence in considered:
        action, reason = _retention_decision(entry, evidence, as_of=as_of, config=config)
        retention_reasons[entry.id] = reason
        if action == MaintenanceAction.DELETE or reason in protected_reasons:
            operations.append(_retention_operation(
                entry,
                evidence,
                as_of=as_of,
                config=config,
            ))
        else:
            remaining.append((entry, evidence))

    merge_operations, merged_source_ids = _plan_merge_operations(
        remaining,
        as_of=as_of,
        config=config,
    )
    operations.extend(merge_operations)
    remaining = [item for item in remaining if item[0].id not in merged_source_ids]

    removed_before_promotion = {
        memory_id
        for operation in operations
        for memory_id in operation.remove_ids
    }
    repository_after_merge = _repository_after_operations(entries, operations)
    promotion_operations, promoted_source_ids = _plan_promotion_operations(
        remaining,
        repository_entries=repository_after_merge,
        as_of=as_of,
        config=config,
    )
    operations.extend(promotion_operations)
    remaining = [item for item in remaining if item[0].id not in promoted_source_ids]

    for entry, evidence in remaining:
        reason = retention_reasons[entry.id]
        tier = entry.tier
        if reason == "no_maintenance_rule" and tier in {
            ExperienceTier.TIP,
            ExperienceTier.TRAJECTORY,
        }:
            reason = "promotion_not_ready"
        elif reason == "no_maintenance_rule" and tier in {
            ExperienceTier.SKILL,
            ExperienceTier.TOOL,
        }:
            reason = "merge_not_safe"
        operations.append(_keep_operation(entry, evidence, reason=reason))

    operations.sort(key=_operation_sort_key)
    operation_tuple = tuple(operations)
    input_summary = {
        "entries_total": len(entries),
        "experiences_considered": len(considered),
        "missing_attribution": sum(1 for _, evidence in considered if not evidence.has_attribution),
        "sources_removed_before_promotion": len(removed_before_promotion),
    }
    return operation_tuple, input_summary


def _retention_operation(
    entry: ExperienceMemory,
    evidence: MaintenanceEvidence,
    *,
    as_of: datetime,
    config: MaintenanceConfig,
) -> MaintenanceOperation:
    action, reason = _retention_decision(entry, evidence, as_of=as_of, config=config)
    if action == MaintenanceAction.KEEP:
        return _keep_operation(entry, evidence, reason=reason)

    operation_id = _operation_id(
        action=action,
        source_ids=(entry.id,),
        target_ids=(),
        replacements=(),
        additions=(),
    )
    return MaintenanceOperation(
        operation_id=operation_id,
        action=action,
        source_ids=(entry.id,),
        source_tiers=(evidence.tier,),
        source_preconditions={entry.id: _source_precondition(entry, evidence.tier)},
        reason_codes=(reason,),
        evidence=(evidence.to_dict(),),
        remove_ids=(entry.id,),
    )


def _retention_decision(
    entry: ExperienceMemory,
    evidence: MaintenanceEvidence,
    *,
    as_of: datetime,
    config: MaintenanceConfig,
) -> tuple[MaintenanceAction, str]:
    action = MaintenanceAction.KEEP
    reason = "no_maintenance_rule"

    if entry.protected:
        reason = "protected_metadata"
    elif entry.scope == MemoryScope.GLOBAL:
        reason = "protected_global"
    elif not _automatic_maintenance_provenance(evidence.created_by):
        reason = (
            "protected_manual"
            if evidence.created_by == ExperienceCreatedBy.MANUAL.value
            else "protected_unknown_provenance"
        )
    elif entry.invalidated:
        action = MaintenanceAction.DELETE
        reason = "explicitly_invalidated"
    elif _negative_delete_eligible(evidence, config):
        action = MaintenanceAction.DELETE
        reason = "negative_attribution_with_control"
    elif _stale_delete_eligible(evidence, as_of=as_of, config=config):
        action = MaintenanceAction.DELETE
        reason = "stale_retrieved_never_selected"
    elif evidence.has_attribution and evidence.value > 0:
        reason = "high_value"
    elif not _has_sufficient_retention_evidence(evidence, config):
        reason = "insufficient_attribution_evidence"

    return action, reason


def _keep_operation(
    entry: ExperienceMemory,
    evidence: MaintenanceEvidence,
    *,
    reason: str,
) -> MaintenanceOperation:
    operation_id = _operation_id(
        action=MaintenanceAction.KEEP,
        source_ids=(entry.id,),
        target_ids=(),
        replacements=(),
        additions=(),
    )
    return MaintenanceOperation(
        operation_id=operation_id,
        action=MaintenanceAction.KEEP,
        source_ids=(entry.id,),
        source_tiers=(evidence.tier,),
        source_preconditions={entry.id: _source_precondition(entry, evidence.tier)},
        reason_codes=(reason,),
        evidence=(evidence.to_dict(),),
    )


def _plan_merge_operations(
    candidates: Sequence[tuple[ExperienceMemory, MaintenanceEvidence]],
    *,
    as_of: datetime,
    config: MaintenanceConfig,
) -> tuple[list[MaintenanceOperation], set[str]]:
    groups: dict[tuple[str, str, str], list[tuple[ExperienceMemory, MaintenanceEvidence]]] = {}
    for entry, evidence in candidates:
        tier = entry.tier
        if (
            tier not in {ExperienceTier.TIP, ExperienceTier.SKILL, ExperienceTier.TOOL}
            or evidence.created_by not in {
                ExperienceCreatedBy.WRITER.value,
                ExperienceCreatedBy.MAINTENANCE.value,
            }
        ):
            continue
        group_key = (tier.value, entry.scope.value, entry.project_key)
        groups.setdefault(group_key, []).append((entry, evidence))

    operations: list[MaintenanceOperation] = []
    consumed: set[str] = set()
    for group_key in sorted(groups):
        tier = ExperienceTier(group_key[0])
        threshold = _merge_threshold(tier, config)
        unassigned = sorted(groups[group_key], key=_anchor_priority)
        while unassigned:
            anchor = unassigned.pop(0)
            cluster = [anchor]
            accepted_ids: set[str] = set()
            for candidate in unassigned:
                if len(cluster) >= config.merge_max_cluster_size:
                    break
                if all(
                    _merge_pair_score(candidate[0], member[0]) >= threshold
                    for member in cluster
                ):
                    cluster.append(candidate)
                    accepted_ids.add(candidate[0].id)
            if accepted_ids:
                unassigned = [item for item in unassigned if item[0].id not in accepted_ids]
            if len(cluster) < 2:
                continue
            operation = _merge_operation(cluster, as_of=as_of)
            operations.append(operation)
            consumed.update(operation.source_ids)
    return operations, consumed


def _merge_operation(
    cluster: Sequence[tuple[ExperienceMemory, MaintenanceEvidence]],
    *,
    as_of: datetime,
) -> MaintenanceOperation:
    ordered = sorted(cluster, key=_anchor_priority)
    anchor_entry, _ = ordered[0]
    source_ids = tuple(item[0].id for item in ordered)
    source_tiers = tuple(item[1].tier for item in ordered)
    source_preconditions = {
        entry.id: _source_precondition(entry, evidence.tier)
        for entry, evidence in ordered
    }
    pair_scores = [
        _merge_pair_score(ordered[left][0], ordered[right][0])
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    ]
    minimum_score = round(min(pair_scores), 6)

    provisional_replacement = _merge_replacement_payload(
        ordered,
        as_of=as_of,
        minimum_score=minimum_score,
    )
    operation_id = _operation_id(
        action=MaintenanceAction.MERGE,
        source_ids=source_ids,
        target_ids=(anchor_entry.id,),
        replacements=(provisional_replacement,),
        additions=(),
    )
    replacement = _merge_replacement_payload(
        ordered,
        as_of=as_of,
        minimum_score=minimum_score,
        operation_id=operation_id,
    )
    return MaintenanceOperation(
        operation_id=operation_id,
        action=MaintenanceAction.MERGE,
        source_ids=source_ids,
        source_tiers=source_tiers,
        source_preconditions=source_preconditions,
        target_ids=(anchor_entry.id,),
        reason_codes=("near_duplicate_complete_link",),
        redundancy_score=minimum_score,
        evidence=tuple(evidence.to_dict() for _, evidence in ordered),
        remove_ids=tuple(sorted(source_ids[1:])),
        replacements=(replacement,),
    )


def _merge_replacement_payload(
    ordered: Sequence[tuple[ExperienceMemory, MaintenanceEvidence]],
    *,
    as_of: datetime,
    minimum_score: float,
    operation_id: str = "",
) -> ExperienceMemory:
    anchor_entry = ordered[0][0]
    _ = as_of, minimum_score
    payload = anchor_entry.payload
    if isinstance(payload, SkillPayload):
        payload = replace(
            payload,
            preconditions=_ordered_payload_union(
                ordered,
                field_name="preconditions",
            ),
            steps=_ordered_payload_union(ordered, field_name="steps"),
        )
    return replace(
        anchor_entry,
        payload=payload,
        created_by=ExperienceCreatedBy.MAINTENANCE,
        maintenance_operation_id=operation_id,
    )


def _plan_promotion_operations(
    candidates: Sequence[tuple[ExperienceMemory, MaintenanceEvidence]],
    *,
    repository_entries: Sequence[ExperienceMemory],
    as_of: datetime,
    config: MaintenanceConfig,
) -> tuple[list[MaintenanceOperation], set[str]]:
    eligible = [item for item in candidates if _promotion_eligible(*item, config=config)]
    eligible.sort(key=_promotion_priority)
    eligible = eligible[:config.max_promotions]

    available_repository = list(repository_entries)
    operations: list[MaintenanceOperation] = []
    promoted_sources: set[str] = set()
    for entry, evidence in eligible:
        operation, target = _promotion_operation(
            entry,
            evidence,
            repository_entries=available_repository,
            as_of=as_of,
        )
        operations.append(operation)
        promoted_sources.add(entry.id)
        available_repository = [
            item for item in available_repository if item.id != entry.id
        ]
        available_repository.append(operation.replacements[0])
        if operation.additions:
            available_repository.append(target)
    return operations, promoted_sources


def _promotion_eligible(
    entry: ExperienceMemory,
    evidence: MaintenanceEvidence,
    *,
    config: MaintenanceConfig,
) -> bool:
    tier = entry.tier
    if tier not in {ExperienceTier.TIP, ExperienceTier.TRAJECTORY}:
        return False
    if not _automatic_maintenance_provenance(evidence.created_by):
        return False
    if entry.promoted_to:
        return False
    if not (
        evidence.has_attribution
        and evidence.value >= config.promote_value_threshold
        and evidence.confidence >= config.promote_min_confidence
        and evidence.selected_count >= config.promote_min_selected_count
    ):
        return False
    if tier == ExperienceTier.TRAJECTORY:
        assert isinstance(entry.payload, TrajectoryPayload)
        return entry.payload.outcome.casefold() == "success" and bool(entry.payload.key_learnings)
    return True


def _automatic_maintenance_provenance(created_by: str) -> bool:
    return created_by in {
        ExperienceCreatedBy.WRITER.value,
        ExperienceCreatedBy.MAINTENANCE.value,
    }


def _promotion_operation(
    entry: ExperienceMemory,
    evidence: MaintenanceEvidence,
    *,
    repository_entries: Sequence[ExperienceMemory],
    as_of: datetime,
) -> tuple[MaintenanceOperation, ExperienceMemory]:
    content, _ = _promoted_skill_fields(entry)
    provisional_target = _promoted_target_entry(
        entry,
        evidence,
        as_of=as_of,
    )
    existing_target = _find_existing_skill(repository_entries, provisional_target)
    target = existing_target or provisional_target
    additions = () if existing_target is not None else (provisional_target,)
    reason = (
        "promotion_linked_existing_skill"
        if existing_target is not None
        else "promoted_to_skill"
    )

    provisional_source = replace(entry, promoted_to=target.id)
    operation_id = _operation_id(
        action=MaintenanceAction.PROMOTE,
        source_ids=(entry.id,),
        target_ids=(target.id,),
        replacements=(provisional_source,),
        additions=additions,
    )
    source_replacement = replace(
        entry,
        promoted_to=target.id,
        maintenance_operation_id=operation_id,
    )
    if existing_target is None:
        target = _promoted_target_entry(
            entry,
            evidence,
            as_of=as_of,
            operation_id=operation_id,
        )
        additions = (target,)

    operation = MaintenanceOperation(
        operation_id=operation_id,
        action=MaintenanceAction.PROMOTE,
        source_ids=(entry.id,),
        source_tiers=(evidence.tier,),
        source_preconditions={entry.id: _source_precondition(entry, evidence.tier)},
        target_ids=(target.id,),
        reason_codes=(reason,),
        evidence=(evidence.to_dict(),),
        replacements=(source_replacement,),
        additions=additions,
    )
    return operation, target


def _promoted_target_entry(
    source: ExperienceMemory,
    evidence: MaintenanceEvidence,
    *,
    as_of: datetime,
    operation_id: str = "",
) -> ExperienceMemory:
    content, skill_payload = _promoted_skill_fields(source)
    return ExperienceMemory(
        id=_promoted_skill_id(source.id, content),
        content=content,
        tier=ExperienceTier.SKILL,
        payload=skill_payload,
        scope=source.scope,
        project_key=source.project_key,
        created_at=as_of,
        token_count=estimate_tokens(content),
        fingerprint=content_fingerprint(content),
        source_task=source.source_task,
        run_id=source.run_id,
        stream_id=source.stream_id,
        created_by=ExperienceCreatedBy.MAINTENANCE,
        writer_confidence=round(max(0.0, min(1.0, evidence.writer_confidence)), 6),
        maintenance_operation_id=operation_id,
        parent_id=source.id,
        parent_tier=source.tier,
    )


def _promoted_skill_fields(entry: ExperienceMemory) -> tuple[str, SkillPayload]:
    if entry.tier == ExperienceTier.TIP:
        assert isinstance(entry.payload, TipPayload)
        category = entry.payload.category
        return entry.content, SkillPayload(
            category=category,
            technique=_stable_title(entry.content, fallback=category),
            preconditions=(entry.payload.trigger,),
            steps=(entry.content,),
        )
    if entry.tier == ExperienceTier.TRAJECTORY:
        assert isinstance(entry.payload, TrajectoryPayload)
        learnings = list(entry.payload.key_learnings[:3])
        content = "\n".join(learnings)
        task_description = entry.payload.task_description
        preconditions = (task_description,) if task_description else entry.payload.tags
        steps = tuple(_successful_step_summaries(entry.payload.steps)[:6]) or tuple(learnings)
        return content, SkillPayload(
            category="trajectory_distillation",
            technique=_stable_title(task_description, fallback="Trajectory distillation"),
            preconditions=preconditions,
            steps=steps,
        )
    raise MaintenancePlanError(f"unsupported promotion source tier: {entry.tier!r}")


def _promoted_skill_id(source_id: str, content: str) -> str:
    digest = sha256(f"{source_id}{ExperienceTier.SKILL.value}{content}".encode("utf-8")).hexdigest()
    return f"exp_maint_{digest[:16]}"


def _find_existing_skill(
    repository_entries: Sequence[ExperienceMemory],
    target: ExperienceMemory,
) -> ExperienceMemory | None:
    target_key = experience_dedup_key(target)
    matches = [
        entry
        for entry in repository_entries
        if entry.tier == ExperienceTier.SKILL
        and experience_dedup_key(entry) == target_key
    ]
    return min(matches, key=lambda entry: entry.id) if matches else None






def _operation_sort_key(operation: MaintenanceOperation) -> tuple[int, tuple[str, ...], str]:
    order = {
        MaintenanceAction.DELETE: 0,
        MaintenanceAction.MERGE: 1,
        MaintenanceAction.PROMOTE: 2,
        MaintenanceAction.KEEP: 3,
    }
    return (order[operation.action], operation.source_ids, operation.operation_id)


def _entry_visible_to_project(entry: ExperienceMemory, project_key: str) -> bool:
    return entry.scope == MemoryScope.GLOBAL or (
        entry.scope == MemoryScope.PROJECT and entry.project_key == project_key
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _char_trigrams(value: str) -> set[str]:
    compact = "".join(value.split())
    if not compact:
        return set()
    if len(compact) < 3:
        return {compact}
    return {compact[index:index + 3] for index in range(len(compact) - 2)}

__all__: list[str] = []
