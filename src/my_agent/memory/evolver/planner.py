"""Evidence adaptation and pure deterministic maintenance planning."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence
import json
import math

from my_agent.memory.evolver.attribution import MemoryAttributionRecord
from my_agent.memory.evolver.contracts import (
    AttributionKey,
    MAINTENANCE_POLICY,
    MAINTENANCE_SCHEMA_VERSION,
    MAINTENANCE_SCOPE_MODE,
    MaintenanceAction,
    MaintenanceAttributionError,
    MaintenanceConfig,
    MaintenanceEvidence,
    MaintenanceLookupHit,
    MaintenanceOperation,
    MaintenancePlan,
    MaintenancePlanError,
    _as_float,
    _as_int,
    _operation_id,
    _operation_summary,
    _parse_datetime,
    _payload_ids,
    _plan_id,
    _require_aware_datetime,
    _source_precondition,
    _validated_payload_entry,
    _valid_tier,
)
from my_agent.memory.evolver.types import (
    ExperienceCreatedBy,
    ExperienceTier,
    build_experience_entry,
    experience_tier,
)
from my_agent.memory.long_term import memory_dedup_key
from my_agent.memory.retrieval import tokenize
from my_agent.memory.types import (
    MemoryEntry,
    MemoryScope,
    content_fingerprint,
    normalize_content,
)
from my_agent.text_safety import sanitize_json_value

def load_project_attribution(
    path: str | Path,
    *,
    memory_project_key: str,
) -> dict[AttributionKey, MemoryAttributionRecord]:
    """Strictly load one project's attribution records by composite identity."""
    if not str(memory_project_key or ""):
        raise MaintenanceAttributionError("memory_project_key must not be empty")
    source = Path(path)
    records: dict[AttributionKey, MemoryAttributionRecord] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("expected object")
                record = MemoryAttributionRecord.from_dict(payload)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MaintenanceAttributionError(
                    f"invalid attribution JSONL at line {line_no}: {type(exc).__name__}"
                ) from exc
            tier = _valid_tier(record.tier)
            if not record.memory_id:
                raise MaintenanceAttributionError(f"empty memory_id at line {line_no}")
            if tier is None:
                raise MaintenanceAttributionError(f"invalid tier at line {line_no}")
            if record.memory_project_key != memory_project_key:
                raise MaintenanceAttributionError(f"memory_project_key mismatch at line {line_no}")
            key = (record.memory_id, tier.value, memory_project_key)
            if key in records:
                raise MaintenanceAttributionError(f"duplicate attribution record at line {line_no}")
            records[key] = record
    return records


def maintenance_evidence_for_entry(
    entry: MemoryEntry,
    *,
    attribution: Mapping[AttributionKey, MemoryAttributionRecord],
    project_key: str,
) -> MaintenanceEvidence:
    """Resolve attribution and writer evidence for one experience entry."""
    tier = experience_tier(entry)
    if tier is None:
        raise ValueError("maintenance evidence requires an experience entry")
    if not project_key:
        raise ValueError("project_key must not be empty")

    visible = entry.scope == MemoryScope.GLOBAL or entry.project_key == project_key
    record = attribution.get((entry.id, tier.value, project_key)) if visible else None
    if record is not None and (
        record.memory_id != entry.id
        or record.tier != tier.value
        or record.memory_project_key != project_key
    ):
        record = None
    metadata = entry.metadata
    metadata_project_matches = (
        visible
        and str(metadata.get("evolver_attribution_memory_project_key") or "") == project_key
    )
    metadata_has_attribution = metadata_project_matches and any(
        key in metadata
        for key in (
            "evolver_attribution_version",
            "evolver_value",
            "evolver_confidence",
            "evolver_candidate_count",
            "evolver_selected_count",
            "evolver_not_selected_count",
        )
    )

    value = (
        float(record.value)
        if record is not None
        else _as_float(metadata.get("evolver_value")) if metadata_has_attribution else 0.0
    )
    confidence = (
        float(record.confidence)
        if record is not None
        else _as_float(metadata.get("evolver_confidence")) if metadata_has_attribution else 0.0
    )
    candidate_count = (
        int(record.candidate_count)
        if record is not None
        else _as_int(metadata.get("evolver_candidate_count")) if metadata_has_attribution else 0
    )
    selected_count = (
        int(record.selected_count)
        if record is not None
        else _as_int(metadata.get("evolver_selected_count")) if metadata_has_attribution else 0
    )
    not_selected_count = (
        int(record.not_selected_count)
        if record is not None
        else _as_int(metadata.get("evolver_not_selected_count")) if metadata_has_attribution else 0
    )
    last_used = (
        (record.last_used if record is not None else "")
        or (
            str(metadata.get("evolver_last_used") or "")
            if metadata_has_attribution
            else ""
        )
        or str(metadata.get("last_used") or "")
    )

    return MaintenanceEvidence(
        memory_id=entry.id,
        tier=tier.value,
        scope=entry.scope.value,
        project_key=entry.project_key,
        created_by=str(metadata.get("created_by") or ""),
        created_at=entry.created_at.isoformat(),
        last_used=last_used,
        source_task=str(metadata.get("source_task") or metadata.get("task_id") or ""),
        value=value,
        confidence=confidence,
        candidate_count=candidate_count,
        selected_count=selected_count,
        not_selected_count=not_selected_count,
        writer_confidence=_as_float(metadata.get("confidence")),
        has_attribution=record is not None or metadata_has_attribution,
    )


def lookup_experiences(
    entries: Sequence[MemoryEntry],
    query: str,
    *,
    project_key: str,
    tiers: Collection[str] | None = None,
    limit: int = 20,
) -> list[MaintenanceLookupHit]:
    """Return deterministic lexical lookup hits without mutating usage state."""
    if not project_key:
        raise ValueError("project_key must not be empty")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    requested_tiers: set[str] | None = None
    if tiers is not None:
        requested_tiers = set()
        for value in tiers:
            tier = _valid_tier(str(value))
            if tier is None:
                raise ValueError(f"invalid experience tier: {value!r}")
            requested_tiers.add(tier.value)

    normalized_query = normalize_content(query)
    query_terms = set(tokenize(query))
    if not normalized_query and not query_terms:
        return []

    hits: list[MaintenanceLookupHit] = []
    for entry in entries:
        tier = experience_tier(entry)
        if tier is None or not _entry_visible_to_project(entry, project_key):
            continue
        if requested_tiers is not None and tier.value not in requested_tiers:
            continue
        content_terms = set(tokenize(entry.content))
        matched = tuple(sorted(query_terms.intersection(content_terms)))
        coverage = len(matched) / len(query_terms) if query_terms else 0.0
        substring_bonus = (
            0.25
            if normalized_query and normalized_query in normalize_content(entry.content)
            else 0.0
        )
        score = min(1.0, coverage + substring_bonus)
        if score <= 0:
            continue
        hits.append(
            MaintenanceLookupHit(
                entry=entry,
                tier=tier.value,
                score=round(score, 6),
                matched_terms=matched,
            )
        )
    hits.sort(key=lambda item: (-item.score, item.tier, item.entry.id))
    return hits[:limit] if limit else []


def redundancy_score(left: MemoryEntry, right: MemoryEntry) -> float:
    """Return deterministic lexical redundancy for a merge-safe pair."""
    left_tier = experience_tier(left)
    right_tier = experience_tier(right)
    if left_tier is None or right_tier is None or left_tier != right_tier:
        return 0.0
    if left.scope != right.scope:
        return 0.0
    if left.scope != MemoryScope.GLOBAL and left.project_key != right.project_key:
        return 0.0

    left_normalized = normalize_content(left.content)
    right_normalized = normalize_content(right.content)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0

    token_score = _jaccard(set(tokenize(left.content)), set(tokenize(right.content)))
    trigram_score = _jaccard(_char_trigrams(left_normalized), _char_trigrams(right_normalized))
    return round(min(1.0, max(0.0, token_score, trigram_score)), 6)


def build_maintenance_plan(
    *,
    entries: Sequence[MemoryEntry],
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

    considered: list[tuple[MemoryEntry, MaintenanceEvidence]] = []
    for entry in entries:
        if experience_tier(entry) is None or not _entry_visible_to_project(entry, project_key):
            continue
        considered.append((
            entry,
            maintenance_evidence_for_entry(
                entry,
                attribution=attribution,
                project_key=project_key,
            ),
        ))
    considered.sort(key=lambda item: (item[1].tier, item[0].id))

    operations: list[MaintenanceOperation] = []
    remaining: list[tuple[MemoryEntry, MaintenanceEvidence]] = []
    retention_reasons: dict[str, str] = {}
    protected_reasons = {"protected_metadata", "protected_global", "protected_manual"}
    for entry, evidence in considered:
        action, reason = _retention_decision(entry, evidence, as_of=as_of_utc, config=cfg)
        retention_reasons[entry.id] = reason
        if action == MaintenanceAction.DELETE or reason in protected_reasons:
            operations.append(_retention_operation(
                entry,
                evidence,
                as_of=as_of_utc,
                config=cfg,
            ))
        else:
            remaining.append((entry, evidence))

    merge_operations, merged_source_ids = _plan_merge_operations(
        remaining,
        as_of=as_of_utc,
        config=cfg,
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
        as_of=as_of_utc,
        config=cfg,
    )
    operations.extend(promotion_operations)
    remaining = [item for item in remaining if item[0].id not in promoted_source_ids]

    for entry, evidence in remaining:
        reason = retention_reasons[entry.id]
        tier = experience_tier(entry)
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
    _validate_operation_conflicts(operation_tuple, repository_entries=entries)
    summary = _operation_summary(operation_tuple)
    input_summary = {
        "entries_total": len(entries),
        "experiences_considered": len(considered),
        "missing_attribution": sum(1 for _, evidence in considered if not evidence.has_attribution),
        "sources_removed_before_promotion": len(removed_before_promotion),
    }
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
    plan = MaintenancePlan(
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
    validate_plan_semantics(plan, repository_entries=entries)
    return plan


def _retention_operation(
    entry: MemoryEntry,
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
    entry: MemoryEntry,
    evidence: MaintenanceEvidence,
    *,
    as_of: datetime,
    config: MaintenanceConfig,
) -> tuple[MaintenanceAction, str]:
    metadata = entry.metadata
    action = MaintenanceAction.KEEP
    reason = "no_maintenance_rule"

    if metadata.get("maintenance_protected") is True:
        reason = "protected_metadata"
    elif entry.scope == MemoryScope.GLOBAL:
        reason = "protected_global"
    elif config.protect_manual and evidence.created_by == "manual":
        reason = "protected_manual"
    elif metadata.get("maintenance_invalidated") is True:
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
    entry: MemoryEntry,
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
    candidates: Sequence[tuple[MemoryEntry, MaintenanceEvidence]],
    *,
    as_of: datetime,
    config: MaintenanceConfig,
) -> tuple[list[MaintenanceOperation], set[str]]:
    groups: dict[tuple[str, str, str], list[tuple[MemoryEntry, MaintenanceEvidence]]] = {}
    for entry, evidence in candidates:
        tier = experience_tier(entry)
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
    cluster: Sequence[tuple[MemoryEntry, MaintenanceEvidence]],
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
    ordered: Sequence[tuple[MemoryEntry, MaintenanceEvidence]],
    *,
    as_of: datetime,
    minimum_score: float,
    operation_id: str = "",
) -> dict[str, Any]:
    anchor_entry = ordered[0][0]
    source_ids = tuple(entry.id for entry, _ in ordered)
    metadata = dict(anchor_entry.metadata)
    for key in ("steps", "tags"):
        merged_values = _ordered_metadata_union(ordered, key)
        if merged_values:
            metadata[key] = merged_values
    metadata.update({
        "created_by": ExperienceCreatedBy.MAINTENANCE.value,
        "maintenance_action": MaintenanceAction.MERGE.value,
        "maintenance_policy": MAINTENANCE_POLICY,
        "maintenance_as_of": as_of.isoformat(),
        "maintenance_source_ids": list(source_ids),
        "maintenance_source_fingerprints": {
            entry.id: entry.fingerprint for entry, _ in ordered
        },
        "maintenance_redundancy_min": minimum_score,
        "maintenance_source_evidence": {
            entry.id: evidence.to_dict() for entry, evidence in ordered
        },
    })
    if operation_id:
        metadata["maintenance_operation_id"] = operation_id
    return _entry_payload_with_metadata(anchor_entry, metadata)


def _plan_promotion_operations(
    candidates: Sequence[tuple[MemoryEntry, MaintenanceEvidence]],
    *,
    repository_entries: Sequence[MemoryEntry],
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
        available_repository.append(MemoryEntry.from_dict(operation.replacements[0]))
        if operation.additions:
            available_repository.append(target)
    return operations, promoted_sources


def _promotion_eligible(
    entry: MemoryEntry,
    evidence: MaintenanceEvidence,
    *,
    config: MaintenanceConfig,
) -> bool:
    tier = experience_tier(entry)
    if tier not in {ExperienceTier.TIP, ExperienceTier.TRAJECTORY}:
        return False
    if evidence.created_by == ExperienceCreatedBy.MANUAL.value:
        return False
    if entry.metadata.get("maintenance_promoted_to"):
        return False
    if not (
        evidence.has_attribution
        and evidence.value >= config.promote_value_threshold
        and evidence.confidence >= config.promote_min_confidence
        and evidence.selected_count >= config.promote_min_selected_count
    ):
        return False
    if tier == ExperienceTier.TRAJECTORY:
        learnings = _non_empty_strings(entry.metadata.get("key_learnings"))
        return str(entry.metadata.get("outcome") or "").casefold() == "success" and bool(learnings)
    return True


def _promotion_operation(
    entry: MemoryEntry,
    evidence: MaintenanceEvidence,
    *,
    repository_entries: Sequence[MemoryEntry],
    as_of: datetime,
) -> tuple[MaintenanceOperation, MemoryEntry]:
    content, _ = _promoted_skill_fields(entry)
    target_id = _promoted_skill_id(entry.id, content)
    provisional_target = _promoted_target_entry(
        entry,
        evidence,
        as_of=as_of,
    )
    existing_target = _find_existing_skill(repository_entries, provisional_target)
    target = existing_target or provisional_target
    additions = () if existing_target is not None else (provisional_target.to_dict(),)
    reason = (
        "promotion_linked_existing_skill"
        if existing_target is not None
        else "promoted_to_skill"
    )

    source_metadata = dict(entry.metadata)
    source_metadata.update({
        "maintenance_promoted_to": target.id,
        "maintenance_promoted_at": as_of.isoformat(),
    })
    provisional_source = _entry_payload_with_metadata(entry, source_metadata)
    operation_id = _operation_id(
        action=MaintenanceAction.PROMOTE,
        source_ids=(entry.id,),
        target_ids=(target.id,),
        replacements=(provisional_source,),
        additions=additions,
    )
    source_metadata["maintenance_operation_id"] = operation_id
    source_replacement = _entry_payload_with_metadata(entry, source_metadata)
    if existing_target is None:
        target = _promoted_target_entry(
            entry,
            evidence,
            as_of=as_of,
            operation_id=operation_id,
        )
        additions = (target.to_dict(),)

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
    source: MemoryEntry,
    evidence: MaintenanceEvidence,
    *,
    as_of: datetime,
    operation_id: str = "",
) -> MemoryEntry:
    content, skill_metadata = _promoted_skill_fields(source)
    skill_metadata.update({
        "maintenance_action": MaintenanceAction.PROMOTE.value,
        "maintenance_policy": MAINTENANCE_POLICY,
        "maintenance_as_of": as_of.isoformat(),
        "maintenance_source_ids": [source.id],
        "maintenance_source_fingerprints": {source.id: source.fingerprint},
        "maintenance_source_evidence": {source.id: evidence.to_dict()},
        "maintenance_parent_id": source.id,
        "maintenance_parent_tier": evidence.tier,
        "maintenance_parent_value": evidence.value,
        "maintenance_parent_confidence": evidence.confidence,
        "confidence": round(max(0.0, min(1.0, evidence.writer_confidence)), 6),
    })
    if operation_id:
        skill_metadata["maintenance_operation_id"] = operation_id
    return build_experience_entry(
        id=_promoted_skill_id(source.id, content),
        content=content,
        tier=ExperienceTier.SKILL,
        project_key=source.project_key,
        scope=source.scope,
        source="evolver:maintenance",
        run_id=source.run_id,
        source_task=str(source.metadata.get("source_task") or source.metadata.get("task_id") or ""),
        created_by=ExperienceCreatedBy.MAINTENANCE,
        extra_metadata=skill_metadata,
        created_at=as_of,
    )


def _promoted_skill_fields(entry: MemoryEntry) -> tuple[str, dict[str, Any]]:
    tier = experience_tier(entry)
    metadata = entry.metadata
    if tier == ExperienceTier.TIP:
        category = str(metadata.get("category") or "promoted_tip")
        return entry.content, {
            "category": category,
            "technique": _stable_title(entry.content, fallback=category),
            "preconditions": str(metadata.get("trigger") or ""),
            "steps": [entry.content],
        }
    if tier == ExperienceTier.TRAJECTORY:
        learnings = _non_empty_strings(metadata.get("key_learnings"))[:3]
        content = "\n".join(learnings)
        task_description = str(metadata.get("task_description") or "")
        tags = _non_empty_strings(metadata.get("tags"))
        return content, {
            "category": "trajectory_distillation",
            "technique": _stable_title(task_description, fallback="Trajectory distillation"),
            "preconditions": task_description or ", ".join(tags),
            "steps": _successful_step_summaries(metadata.get("steps"))[:6],
        }
    raise MaintenancePlanError(f"unsupported promotion source tier: {tier!r}")


def _promoted_skill_id(source_id: str, content: str) -> str:
    digest = sha256(f"{source_id}{ExperienceTier.SKILL.value}{content}".encode("utf-8")).hexdigest()
    return f"exp_maint_{digest[:16]}"


def _find_existing_skill(
    repository_entries: Sequence[MemoryEntry],
    target: MemoryEntry,
) -> MemoryEntry | None:
    target_key = memory_dedup_key(target)
    matches = [
        entry
        for entry in repository_entries
        if experience_tier(entry) == ExperienceTier.SKILL
        and memory_dedup_key(entry) == target_key
    ]
    return min(matches, key=lambda entry: entry.id) if matches else None


def _merge_threshold(tier: ExperienceTier, config: MaintenanceConfig) -> float:
    if tier == ExperienceTier.TIP:
        return config.merge_threshold_tip
    if tier == ExperienceTier.SKILL:
        return config.merge_threshold_skill
    if tier == ExperienceTier.TOOL:
        return config.merge_threshold_tool
    raise MaintenancePlanError(f"tier does not support merge: {tier.value}")


def _merge_pair_score(left: MemoryEntry, right: MemoryEntry) -> float:
    if experience_tier(left) == ExperienceTier.TOOL and not _tool_payload_matches(left, right):
        return 0.0
    return redundancy_score(left, right)


def _tool_payload_matches(left: MemoryEntry, right: MemoryEntry) -> bool:
    def payload(entry: MemoryEntry) -> tuple[str, tuple[str, ...]]:
        language = normalize_content(str(entry.metadata.get("language") or ""))
        executable = tuple(
            _normalize_executable_payload(entry.metadata.get(key))
            for key in ("code", "command", "template")
        )
        return language, executable

    left_language, left_executable = payload(left)
    right_language, right_executable = payload(right)
    return bool(
        any(left_executable)
        and left_language == right_language
        and left_executable == right_executable
    )


def _normalize_executable_payload(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _anchor_priority(
    item: tuple[MemoryEntry, MaintenanceEvidence],
) -> tuple[float, float, int, str, str]:
    entry, evidence = item
    created_at = entry.created_at.astimezone(timezone.utc).isoformat()
    return (-evidence.value, -evidence.confidence, -evidence.selected_count, created_at, entry.id)


def _promotion_priority(
    item: tuple[MemoryEntry, MaintenanceEvidence],
) -> tuple[float, float, int, str]:
    entry, evidence = item
    return (-evidence.value, -evidence.confidence, -evidence.selected_count, entry.id)


def _ordered_metadata_union(
    sources: Sequence[tuple[MemoryEntry, MaintenanceEvidence]],
    key: str,
) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for entry, _ in sources:
        value = entry.metadata.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            continue
        for item in value:
            normalized = json.dumps(
                sanitize_json_value(item),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(sanitize_json_value(item))
    return result


def _successful_step_summaries(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    summaries: list[str] = []
    for raw in value:
        if isinstance(raw, str):
            summary = " ".join(raw.split())
        elif isinstance(raw, Mapping):
            status = str(raw.get("status") or raw.get("outcome") or "").casefold()
            if raw.get("success") is False or status in {"failed", "failure", "error"}:
                continue
            action = " ".join(str(raw.get("action") or "").split())
            result = " ".join(str(raw.get("result") or "").split())
            summary = ": ".join(item for item in (action, result) if item)
        else:
            continue
        if summary:
            summaries.append(summary)
    return summaries


def _non_empty_strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [" ".join(str(item).split()) for item in value if str(item).strip()]


def _stable_title(value: str, *, fallback: str, max_chars: int = 120) -> str:
    title = next((" ".join(line.split()) for line in str(value).splitlines() if line.strip()), "")
    return (title or fallback)[:max_chars].rstrip()


def _entry_payload_with_metadata(
    entry: MemoryEntry,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = entry.to_dict()
    payload["metadata"] = sanitize_json_value(dict(metadata))
    return payload


def _negative_delete_eligible(
    evidence: MaintenanceEvidence,
    config: MaintenanceConfig,
) -> bool:
    return bool(
        evidence.has_attribution
        and evidence.value <= config.delete_value_threshold
        and evidence.confidence >= config.delete_min_confidence
        and evidence.candidate_count >= config.delete_min_candidate_count
        and evidence.selected_count >= config.delete_min_selected_count
        and evidence.not_selected_count >= config.delete_min_not_selected_count
    )


def _stale_delete_eligible(
    evidence: MaintenanceEvidence,
    *,
    as_of: datetime,
    config: MaintenanceConfig,
) -> bool:
    if (
        evidence.candidate_count < config.stale_min_candidate_count
        or evidence.selected_count != 0
        or evidence.value > 0
    ):
        return False
    last_seen = _parse_datetime(evidence.last_used) or _parse_datetime(evidence.created_at)
    if last_seen is None:
        return False
    age_days = max(0.0, (as_of - last_seen.astimezone(timezone.utc)).total_seconds() / 86_400)
    return age_days >= config.stale_after_days


def _has_sufficient_retention_evidence(
    evidence: MaintenanceEvidence,
    config: MaintenanceConfig,
) -> bool:
    return bool(
        evidence.has_attribution
        and evidence.confidence >= config.delete_min_confidence
        and evidence.candidate_count >= config.delete_min_candidate_count
    )


def validate_plan_semantics(
    plan: MaintenancePlan,
    *,
    repository_entries: Sequence[MemoryEntry] | None = None,
) -> None:
    """Validate action meaning independently from plan and operation digests."""
    as_of = _parse_datetime(plan.as_of)
    if as_of is None or as_of.astimezone(timezone.utc).isoformat() != plan.as_of:
        raise MaintenancePlanError("as_of must be a canonical timezone-aware UTC datetime")
    expected_summary = _operation_summary(plan.operations)
    if plan.summary != expected_summary:
        raise MaintenancePlanError("plan summary does not match its operations")

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
    for operation in plan.operations:
        evidence_by_id = _validated_operation_evidence(operation)
        if operation.action == MaintenanceAction.MERGE:
            _validate_merge_action_semantics(
                operation,
                evidence_by_id=evidence_by_id,
                as_of=as_of,
                repository_by_id=repository_by_id,
            )
        elif operation.action == MaintenanceAction.PROMOTE:
            _validate_promotion_action_semantics(
                operation,
                evidence_by_id=evidence_by_id,
                as_of=as_of,
                repository_by_id=repository_by_id,
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
    repository_by_id: Mapping[str, MemoryEntry] | None,
) -> None:
    if operation.reason_codes != ("near_duplicate_complete_link",):
        raise MaintenancePlanError("merge operation has invalid reason codes")
    if (
        operation.redundancy_score is None
        or not math.isfinite(operation.redundancy_score)
        or not 0.0 <= operation.redundancy_score <= 1.0
    ):
        raise MaintenancePlanError("merge redundancy score must be finite and between 0 and 1")

    replacement = _validated_payload_entry(operation.replacements[0], "merge replacement")
    metadata = replacement.metadata
    expected_lineage = {
        "created_by": ExperienceCreatedBy.MAINTENANCE.value,
        "maintenance_action": MaintenanceAction.MERGE.value,
        "maintenance_policy": MAINTENANCE_POLICY,
        "maintenance_operation_id": operation.operation_id,
        "maintenance_as_of": as_of.isoformat(),
        "maintenance_source_ids": list(operation.source_ids),
        "maintenance_source_fingerprints": {
            source_id: operation.source_preconditions[source_id]["fingerprint"]
            for source_id in operation.source_ids
        },
        "maintenance_redundancy_min": operation.redundancy_score,
        "maintenance_source_evidence": {
            source_id: evidence_by_id[source_id].to_dict()
            for source_id in operation.source_ids
        },
    }
    for key, expected in expected_lineage.items():
        if metadata.get(key) != expected:
            raise MaintenancePlanError(
                f"merge replacement lineage mismatch for {key}: {replacement.id}"
            )

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
    repository_by_id: Mapping[str, MemoryEntry] | None,
) -> None:
    source_id = operation.source_ids[0]
    evidence = evidence_by_id[source_id]
    replacement = _validated_payload_entry(operation.replacements[0], "promotion source")
    replacement_metadata = replacement.metadata
    expected_source_lineage = {
        "maintenance_promoted_to": operation.target_ids[0],
        "maintenance_promoted_at": as_of.isoformat(),
        "maintenance_operation_id": operation.operation_id,
    }
    for key, expected in expected_source_lineage.items():
        if replacement_metadata.get(key) != expected:
            raise MaintenancePlanError(
                f"promotion source lineage mismatch for {key}: {source_id}"
            )

    if repository_by_id is not None:
        source = repository_by_id[source_id]
        expected_source_metadata = dict(source.metadata)
        expected_source_metadata.update(expected_source_lineage)
        expected_replacement = _entry_payload_with_metadata(source, expected_source_metadata)
        if operation.replacements[0] != expected_replacement:
            raise MaintenancePlanError("promotion source replacement changes non-lineage fields")
    else:
        source_payload = replacement.to_dict()
        source_metadata = dict(replacement_metadata)
        for key in expected_source_lineage:
            source_metadata.pop(key, None)
        source_payload["metadata"] = source_metadata
        source = MemoryEntry.from_dict(source_payload)

    if experience_tier(source) not in {ExperienceTier.TIP, ExperienceTier.TRAJECTORY}:
        raise MaintenancePlanError(f"invalid promotion source tier: {source_id}")
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
        if operation.additions[0] != expected_target.to_dict():
            raise MaintenancePlanError("promotion target does not match deterministic semantics")
    elif operation.reason_codes != ("promotion_linked_existing_skill",):
        raise MaintenancePlanError("existing promotion target has invalid reason codes")


def _validate_operation_conflicts(
    operations: Sequence[MaintenanceOperation],
    *,
    repository_entries: Sequence[MemoryEntry] | None = None,
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

    repository_by_id: dict[str, MemoryEntry] = {}
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
            actual_tier = experience_tier(entry)
            if actual_tier is None or actual_tier.value != tier:
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
        key = memory_dedup_key(entry)
        previous = seen_dedup_keys.get(key)
        if previous is not None:
            raise MaintenancePlanError(
                f"duplicate repository dedup identity after planning: {previous}, {entry.id}"
            )
        seen_dedup_keys[key] = entry.id


def _validate_merge_repository_contract(
    operation: MaintenanceOperation,
    *,
    repository_by_id: Mapping[str, MemoryEntry],
    final_by_id: Mapping[str, MemoryEntry],
) -> None:
    sources = [repository_by_id[source_id] for source_id in operation.source_ids]
    tiers = {experience_tier(entry) for entry in sources}
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
    preserved_fields = (
        replacement.id == anchor.id,
        replacement.content == anchor.content,
        replacement.fingerprint == anchor.fingerprint,
        replacement.created_at == anchor.created_at,
        replacement.scope == anchor.scope,
        replacement.project_key == anchor.project_key,
        replacement.run_id == anchor.run_id,
        experience_tier(replacement) == experience_tier(anchor),
    )
    if not all(preserved_fields):
        raise MaintenancePlanError(f"merge replacement does not preserve anchor identity: {anchor.id}")


def _repository_after_operations(
    entries: Sequence[MemoryEntry],
    operations: Sequence[MaintenanceOperation],
    *,
    validate: bool = True,
) -> list[MemoryEntry]:
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
    source: MemoryEntry,
    target: MemoryEntry,
    final_by_id: Mapping[str, MemoryEntry],
) -> None:
    if experience_tier(source) not in {ExperienceTier.TIP, ExperienceTier.TRAJECTORY}:
        raise MaintenancePlanError(f"invalid promotion source tier: {source.id}")
    if experience_tier(target) != ExperienceTier.SKILL:
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
    if str(updated_source.metadata.get("maintenance_promoted_to") or "") != target.id:
        raise MaintenancePlanError(f"promotion source target metadata mismatch: {source.id}")
    if str(updated_source.metadata.get("maintenance_operation_id") or "") != operation.operation_id:
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


def _operation_sort_key(operation: MaintenanceOperation) -> tuple[int, tuple[str, ...], str]:
    order = {
        MaintenanceAction.DELETE: 0,
        MaintenanceAction.MERGE: 1,
        MaintenanceAction.PROMOTE: 2,
        MaintenanceAction.KEEP: 3,
    }
    return (order[operation.action], operation.source_ids, operation.operation_id)


def _entry_visible_to_project(entry: MemoryEntry, project_key: str) -> bool:
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


__all__ = [
    "build_maintenance_plan",
    "load_project_attribution",
    "lookup_experiences",
    "maintenance_evidence_for_entry",
    "redundancy_score",
    "validate_plan_semantics",
]
