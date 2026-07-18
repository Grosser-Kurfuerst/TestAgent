"""Evidence adaptation and pure deterministic maintenance planning."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence
import json
import math

from my_agent.json_safety import loads_json_strict
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
    _evidence_float,
    _evidence_int,
    _operation_id,
    _operation_summary,
    _parse_datetime,
    _payload_ids,
    _plan_id,
    _require_aware_datetime,
    _source_precondition,
    _validated_payload_entry,
    _valid_tier,
    _validate_evidence_values,
)
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceTrajectoryStep,
    ExperienceTier,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.experience.repository_rules import experience_dedup_key
from my_agent.memory.experience_retrieval import tokenize_experience_text
from my_agent.memory.types import (
    MemoryScope,
    content_fingerprint,
    normalize_content,
)
from my_agent.memory.token import estimate_tokens


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
                payload = loads_json_strict(line)
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
            _validate_attribution_record(payload, record, line_no=line_no)
            key = (record.memory_id, tier.value, memory_project_key)
            if key in records:
                raise MaintenanceAttributionError(f"duplicate attribution record at line {line_no}")
            records[key] = record
    return records


def _validate_attribution_record(
    payload: Mapping[str, Any],
    record: MemoryAttributionRecord,
    *,
    line_no: int,
) -> None:
    for name in ("memory_id", "tier", "memory_project_key"):
        raw = payload.get(name)
        if not isinstance(raw, str) or not raw:
            raise MaintenanceAttributionError(
                f"{name} must be a non-empty string at line {line_no}"
            )
    for name in ("candidate_count", "selected_count", "not_selected_count"):
        raw = payload.get(name, 0)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise MaintenanceAttributionError(
                f"{name} must be a non-negative integer at line {line_no}"
            )
    for name in ("value", "confidence"):
        _validate_attribution_number(
            payload.get(name),
            name=name,
            line_no=line_no,
        )
    for name in (
        "success_when_selected",
        "success_when_candidate_not_selected",
        "reward_when_selected",
        "reward_when_candidate_not_selected",
    ):
        raw = payload.get(name)
        if raw is not None:
            _validate_attribution_number(raw, name=name, line_no=line_no)
    raw_last_used = payload.get("last_used", "")
    if not isinstance(raw_last_used, str):
        raise MaintenanceAttributionError(
            f"last_used must be a string at line {line_no}"
        )
    _validate_attribution_record_domain(record, context=f"at line {line_no}")


def _validate_attribution_record_domain(
    record: MemoryAttributionRecord,
    *,
    context: str,
) -> None:
    if not isinstance(record.memory_id, str) or not record.memory_id:
        raise MaintenanceAttributionError(f"memory_id must be a non-empty string {context}")
    if _valid_tier(record.tier) is None:
        raise MaintenanceAttributionError(f"tier is invalid {context}")
    if not isinstance(record.memory_project_key, str) or not record.memory_project_key:
        raise MaintenanceAttributionError(
            f"memory_project_key must be a non-empty string {context}"
        )
    try:
        _validate_evidence_values(
            value=record.value,
            confidence=record.confidence,
            candidate_count=record.candidate_count,
            selected_count=record.selected_count,
            not_selected_count=record.not_selected_count,
            writer_confidence=0.0,
            has_attribution=True,
            last_used=record.last_used,
        )
    except MaintenancePlanError as exc:
        raise MaintenanceAttributionError(
            f"invalid attribution evidence {context}: {exc}"
        ) from exc

    optional_values = {
        "success_when_selected": record.success_when_selected,
        "success_when_candidate_not_selected": (
            record.success_when_candidate_not_selected
        ),
        "reward_when_selected": record.reward_when_selected,
        "reward_when_candidate_not_selected": (
            record.reward_when_candidate_not_selected
        ),
    }
    for name, value in optional_values.items():
        if value is not None:
            _validate_attribution_number(value, name=name, context=context)
    for name, value in (
        ("success_when_selected", record.success_when_selected),
        (
            "success_when_candidate_not_selected",
            record.success_when_candidate_not_selected,
        ),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise MaintenanceAttributionError(
                f"{name} is out of range {context}"
            )


def _validate_attribution_number(
    value: Any,
    *,
    name: str,
    line_no: int | None = None,
    context: str = "",
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        location = context or f"at line {line_no}"
        raise MaintenanceAttributionError(
            f"{name} must be a finite JSON number {location}"
        )


def _validate_attribution_mapping(
    attribution: Mapping[AttributionKey, MemoryAttributionRecord],
    *,
    project_key: str,
) -> None:
    for key, record in attribution.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or any(not isinstance(item, str) or not item for item in key)
        ):
            raise MaintenanceAttributionError(
                "attribution keys must be non-empty (memory_id, tier, project_key) strings"
            )
        if not isinstance(record, MemoryAttributionRecord):
            raise MaintenanceAttributionError(
                f"attribution value must be MemoryAttributionRecord for key {key!r}"
            )
        _validate_attribution_record_domain(record, context=f"for key {key!r}")
        expected_key = (
            record.memory_id,
            record.tier,
            record.memory_project_key,
        )
        if key != expected_key:
            raise MaintenanceAttributionError(
                f"attribution key does not match record identity: {key!r}"
            )
        if record.memory_project_key != project_key:
            raise MaintenanceAttributionError(
                f"attribution record crosses project boundary: {key!r}"
            )


def maintenance_evidence_for_entry(
    entry: ExperienceMemory,
    *,
    attribution: Mapping[AttributionKey, MemoryAttributionRecord],
    project_key: str,
) -> MaintenanceEvidence:
    """Resolve attribution and writer evidence for one experience entry."""
    _validate_attribution_mapping(attribution, project_key=project_key)
    return _maintenance_evidence_for_entry(
        entry,
        attribution=attribution,
        project_key=project_key,
    )


def _maintenance_evidence_for_entry(
    entry: ExperienceMemory,
    *,
    attribution: Mapping[AttributionKey, MemoryAttributionRecord],
    project_key: str,
) -> MaintenanceEvidence:
    tier = entry.tier
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
    if record is not None:
        value = _evidence_float(record.value, "value")
        confidence = _evidence_float(record.confidence, "confidence")
        candidate_count = _evidence_int(record.candidate_count, "candidate_count")
        selected_count = _evidence_int(record.selected_count, "selected_count")
        not_selected_count = _evidence_int(
            record.not_selected_count,
            "not_selected_count",
        )
    else:
        value = entry.attribution_value
        confidence = entry.attribution_confidence
        candidate_count = entry.candidate_count
        selected_count = entry.selected_count
        not_selected_count = entry.not_selected_count
    has_attribution = bool(
        record is not None
        or entry.attribution_updated_at is not None
        or candidate_count > 0
        or confidence != 0.0
        or value != 0.0
    )
    last_used = (record.last_used if record is not None else "") or (
        entry.last_used.isoformat() if entry.last_used is not None else ""
    )

    return MaintenanceEvidence(
        memory_id=entry.id,
        tier=tier.value,
        scope=entry.scope.value,
        project_key=entry.project_key,
        created_by=entry.created_by.value,
        created_at=entry.created_at.isoformat(),
        last_used=last_used,
        source_task=entry.source_task,
        value=value,
        confidence=confidence,
        candidate_count=candidate_count,
        selected_count=selected_count,
        not_selected_count=not_selected_count,
        writer_confidence=entry.writer_confidence,
        has_attribution=has_attribution,
    )


def lookup_experiences(
    entries: Sequence[ExperienceMemory],
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
    query_terms = set(tokenize_experience_text(query))
    if not normalized_query and not query_terms:
        return []

    hits: list[MaintenanceLookupHit] = []
    for entry in entries:
        tier = entry.tier
        if not _entry_visible_to_project(entry, project_key):
            continue
        if requested_tiers is not None and tier.value not in requested_tiers:
            continue
        content_terms = set(tokenize_experience_text(entry.content))
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
                memory=entry,
                tier=tier.value,
                score=round(score, 6),
                matched_terms=matched,
            )
        )
    hits.sort(key=lambda item: (-item.score, item.tier, item.memory.id))
    return hits[:limit] if limit else []


def redundancy_score(left: ExperienceMemory, right: ExperienceMemory) -> float:
    """Return deterministic lexical redundancy for a merge-safe pair."""
    left_tier = left.tier
    right_tier = right.tier
    if left_tier != right_tier:
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

    token_score = _jaccard(
        set(tokenize_experience_text(left.content)),
        set(tokenize_experience_text(right.content)),
    )
    trigram_score = _jaccard(_char_trigrams(left_normalized), _char_trigrams(right_normalized))
    return round(min(1.0, max(0.0, token_score, trigram_score)), 6)


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


def _merge_threshold(tier: ExperienceTier, config: MaintenanceConfig) -> float:
    if tier == ExperienceTier.TIP:
        return config.merge_threshold_tip
    if tier == ExperienceTier.SKILL:
        return config.merge_threshold_skill
    if tier == ExperienceTier.TOOL:
        return config.merge_threshold_tool
    raise MaintenancePlanError(f"tier does not support merge: {tier.value}")


def _merge_pair_score(left: ExperienceMemory, right: ExperienceMemory) -> float:
    if left.tier == ExperienceTier.TOOL and not _tool_payload_matches(left, right):
        return 0.0
    return redundancy_score(left, right)


def _tool_payload_matches(left: ExperienceMemory, right: ExperienceMemory) -> bool:
    def payload(entry: ExperienceMemory) -> tuple[str, tuple[str, ...]]:
        assert isinstance(entry.payload, ToolPayload)
        language = normalize_content(entry.payload.language)
        executable = (
            _normalize_executable_payload(entry.payload.code),
            _normalize_executable_payload(entry.payload.command),
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
    item: tuple[ExperienceMemory, MaintenanceEvidence],
) -> tuple[float, float, int, str, str]:
    entry, evidence = item
    created_at = entry.created_at.astimezone(timezone.utc).isoformat()
    return (-evidence.value, -evidence.confidence, -evidence.selected_count, created_at, entry.id)


def _promotion_priority(
    item: tuple[ExperienceMemory, MaintenanceEvidence],
) -> tuple[float, float, int, str]:
    entry, evidence = item
    return (-evidence.value, -evidence.confidence, -evidence.selected_count, entry.id)


def _ordered_payload_union(
    sources: Sequence[tuple[ExperienceMemory, MaintenanceEvidence]],
    *,
    field_name: str,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for entry, _ in sources:
        if not isinstance(entry.payload, SkillPayload):
            raise MaintenancePlanError("skill merge payload type mismatch")
        value = getattr(entry.payload, field_name)
        for item in value:
            normalized = " ".join(item.split())
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
            if len(result) >= 64:
                return tuple(result)
    return tuple(result)


def _successful_step_summaries(value: Sequence[ExperienceTrajectoryStep]) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    summaries: list[str] = []
    for raw in value:
        if raw.reward is not None and raw.reward < 0:
            continue
        action = " ".join(raw.action.split())
        result = " ".join(raw.result.split())
        summary = ": ".join(item for item in (action, result) if item)
        if summary:
            summaries.append(summary)
    return summaries


def _stable_title(value: str, *, fallback: str, max_chars: int = 120) -> str:
    title = next((" ".join(line.split()) for line in str(value).splitlines() if line.strip()), "")
    return (title or fallback)[:max_chars].rstrip()


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
        not evidence.has_attribution
        or evidence.candidate_count <= 0
        or evidence.candidate_count < config.stale_min_candidate_count
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


__all__ = [
    "load_project_attribution",
    "lookup_experiences",
    "maintenance_evidence_for_entry",
    "redundancy_score",
]
