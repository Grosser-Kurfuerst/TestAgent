"""Deterministic contracts and evidence adapters for memory maintenance.

The planner is intentionally pure: repository I/O and transactional apply are
separate concerns added by later Phase 6 iterations.  This module owns the
stable schema shared by planning, review, audit, and future dataset export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence
import json

from my_agent.memory.evolver.attribution import MemoryAttributionRecord
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


MAINTENANCE_SCHEMA_VERSION = 1
MAINTENANCE_POLICY = "rule_attribution_redundancy_v1"
MAINTENANCE_SCOPE_MODE = "single_project"

AttributionKey = tuple[str, str, str]


class MaintenanceError(ValueError):
    """Base class for maintenance contract and input errors."""


class MaintenanceAttributionError(MaintenanceError):
    """Raised when project attribution JSONL violates the strict schema."""


class MaintenancePlanError(MaintenanceError):
    """Raised when a serialized or generated plan violates its invariants."""


class MaintenanceAction(str, Enum):
    KEEP = "keep"
    DELETE = "delete"
    MERGE = "merge"
    PROMOTE = "promote"


class MaintenanceApplyStatus(str, Enum):
    NOOP = "noop"
    PRE_COMMIT_FAILED = "pre_commit_failed"
    COMMITTED = "committed"
    COMMITTED_WITH_AUDIT_ERROR = "committed_with_audit_error"


@dataclass(frozen=True)
class MaintenanceConfig:
    delete_value_threshold: float = -0.05
    delete_min_confidence: float = 0.50
    delete_min_candidate_count: int = 4
    delete_min_selected_count: int = 1
    delete_min_not_selected_count: int = 1

    stale_after_days: int = 90
    stale_min_candidate_count: int = 6

    merge_threshold_tip: float = 0.86
    merge_threshold_skill: float = 0.86
    merge_threshold_tool: float = 0.94
    merge_max_cluster_size: int = 5
    max_merged_content_chars: int = 1_600

    promote_value_threshold: float = 0.10
    promote_min_confidence: float = 0.70
    promote_min_selected_count: int = 3
    max_promotions: int = 10

    protect_manual: bool = True

    def __post_init__(self) -> None:
        _validate_range("delete_value_threshold", self.delete_value_threshold, -1.0, 1.0)
        _validate_range("delete_min_confidence", self.delete_min_confidence, 0.0, 1.0)
        _validate_range("merge_threshold_tip", self.merge_threshold_tip, 0.0, 1.0)
        _validate_range("merge_threshold_skill", self.merge_threshold_skill, 0.0, 1.0)
        _validate_range("merge_threshold_tool", self.merge_threshold_tool, 0.0, 1.0)
        _validate_range("promote_value_threshold", self.promote_value_threshold, -1.0, 1.0)
        _validate_range("promote_min_confidence", self.promote_min_confidence, 0.0, 1.0)
        for name in (
            "delete_min_candidate_count",
            "delete_min_selected_count",
            "delete_min_not_selected_count",
            "stale_after_days",
            "stale_min_candidate_count",
            "promote_min_selected_count",
            "max_promotions",
        ):
            _validate_non_negative_int(name, getattr(self, name))
        _validate_positive_int("max_merged_content_chars", self.max_merged_content_chars)
        _validate_positive_int("merge_max_cluster_size", self.merge_max_cluster_size)
        if self.merge_max_cluster_size < 2:
            raise ValueError("merge_max_cluster_size must be at least 2")
        if not isinstance(self.protect_manual, bool):
            raise ValueError("protect_manual must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return sanitize_json_value(asdict(self))  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaintenanceConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown maintenance config fields: {', '.join(unknown)}")
        return cls(**dict(data))


@dataclass(frozen=True)
class MaintenanceEvidence:
    memory_id: str
    tier: str
    scope: str
    project_key: str
    created_by: str
    created_at: str
    last_used: str = ""
    source_task: str = ""
    value: float = 0.0
    confidence: float = 0.0
    candidate_count: int = 0
    selected_count: int = 0
    not_selected_count: int = 0
    writer_confidence: float = 0.0
    has_attribution: bool = False

    def to_dict(self) -> dict[str, Any]:
        return sanitize_json_value(asdict(self))  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaintenanceEvidence":
        return cls(
            memory_id=str(data.get("memory_id") or ""),
            tier=str(data.get("tier") or ""),
            scope=str(data.get("scope") or ""),
            project_key=str(data.get("project_key") or ""),
            created_by=str(data.get("created_by") or ""),
            created_at=str(data.get("created_at") or ""),
            last_used=str(data.get("last_used") or ""),
            source_task=str(data.get("source_task") or ""),
            value=_as_float(data.get("value")),
            confidence=_as_float(data.get("confidence")),
            candidate_count=_as_int(data.get("candidate_count")),
            selected_count=_as_int(data.get("selected_count")),
            not_selected_count=_as_int(data.get("not_selected_count")),
            writer_confidence=_as_float(data.get("writer_confidence")),
            has_attribution=bool(data.get("has_attribution", False)),
        )


@dataclass(frozen=True)
class MaintenanceLookupHit:
    entry: MemoryEntry
    tier: str
    score: float
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaintenanceOperation:
    operation_id: str
    action: MaintenanceAction
    source_ids: tuple[str, ...]
    source_tiers: tuple[str, ...]
    source_preconditions: dict[str, dict[str, str]]
    target_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    redundancy_score: float | None = None
    evidence: tuple[dict[str, Any], ...] = ()
    remove_ids: tuple[str, ...] = ()
    replacements: tuple[dict[str, Any], ...] = ()
    additions: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise MaintenancePlanError("operation_id must not be empty")
        if not self.source_ids:
            raise MaintenancePlanError("maintenance operation must have at least one source")
        if len(self.source_ids) != len(self.source_tiers):
            raise MaintenancePlanError("source_ids and source_tiers must have the same length")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise MaintenancePlanError("source_ids must be unique")
        if set(self.source_preconditions) != set(self.source_ids):
            raise MaintenancePlanError("source_preconditions must cover every source id exactly")
        required = {"fingerprint", "tier", "scope", "project_key"}
        for source_id, source_tier in zip(self.source_ids, self.source_tiers):
            if _valid_tier(source_tier) is None:
                raise MaintenancePlanError(f"invalid source tier for {source_id}")
            precondition = self.source_preconditions[source_id]
            if not isinstance(precondition, Mapping):
                raise MaintenancePlanError(f"source precondition for {source_id} must be an object")
            missing = sorted(required - set(precondition))
            if missing:
                raise MaintenancePlanError(
                    f"source precondition for {source_id} is missing: {', '.join(missing)}"
                )
            if not str(precondition.get("fingerprint") or ""):
                raise MaintenancePlanError(f"source precondition for {source_id} has empty fingerprint")
            if str(precondition.get("tier") or "") != source_tier:
                raise MaintenancePlanError(f"source tier mismatch for {source_id}")
            try:
                scope = MemoryScope(str(precondition.get("scope") or ""))
            except ValueError as exc:
                raise MaintenancePlanError(f"invalid source scope for {source_id}") from exc
            if scope == MemoryScope.SESSION:
                raise MaintenancePlanError(f"session memory cannot be maintained: {source_id}")
            if scope == MemoryScope.PROJECT and not str(precondition.get("project_key") or ""):
                raise MaintenancePlanError(
                    f"project source precondition for {source_id} requires project_key"
                )
        _validate_operation_shape(self)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation_id": self.operation_id,
            "action": self.action.value,
            "source_ids": list(self.source_ids),
            "source_tiers": list(self.source_tiers),
            "source_preconditions": {
                key: dict(self.source_preconditions[key])
                for key in sorted(self.source_preconditions)
            },
            "target_ids": list(self.target_ids),
            "reason_codes": list(self.reason_codes),
            "redundancy_score": self.redundancy_score,
            "evidence": [dict(item) for item in self.evidence],
            "remove_ids": list(self.remove_ids),
            "replacements": [dict(item) for item in self.replacements],
            "additions": [dict(item) for item in self.additions],
        }
        return sanitize_json_value(payload)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaintenanceOperation":
        try:
            action = MaintenanceAction(str(data.get("action") or ""))
        except ValueError as exc:
            raise MaintenancePlanError(f"invalid maintenance action: {data.get('action')!r}") from exc
        raw_preconditions = data.get("source_preconditions") or {}
        if not isinstance(raw_preconditions, Mapping):
            raise MaintenancePlanError("source_preconditions must be an object")
        return cls(
            operation_id=str(data.get("operation_id") or ""),
            action=action,
            source_ids=_string_tuple(data.get("source_ids")),
            source_tiers=_string_tuple(data.get("source_tiers")),
            source_preconditions={
                str(key): {str(k): str(v) for k, v in dict(value).items()}
                for key, value in raw_preconditions.items()
                if isinstance(value, Mapping)
            },
            target_ids=_string_tuple(data.get("target_ids")),
            reason_codes=_string_tuple(data.get("reason_codes")),
            redundancy_score=(
                None if data.get("redundancy_score") is None else _as_float(data.get("redundancy_score"))
            ),
            evidence=_mapping_tuple(data.get("evidence")),
            remove_ids=_string_tuple(data.get("remove_ids")),
            replacements=_mapping_tuple(data.get("replacements")),
            additions=_mapping_tuple(data.get("additions")),
        )


@dataclass(frozen=True)
class MaintenancePlan:
    schema_version: int
    policy: str
    plan_id: str
    repository_revision: str
    scope_mode: str
    memory_project_key: str
    as_of: str
    config: dict[str, Any]
    input_summary: dict[str, Any]
    operations: tuple[MaintenanceOperation, ...]
    summary: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != MAINTENANCE_SCHEMA_VERSION:
            raise MaintenancePlanError(f"unsupported maintenance schema version: {self.schema_version}")
        if self.policy != MAINTENANCE_POLICY:
            raise MaintenancePlanError(f"unsupported maintenance policy: {self.policy!r}")
        if self.scope_mode != MAINTENANCE_SCOPE_MODE:
            raise MaintenancePlanError(f"unsupported maintenance scope mode: {self.scope_mode!r}")
        if not self.memory_project_key:
            raise MaintenancePlanError("memory_project_key must not be empty")
        if not self.plan_id:
            raise MaintenancePlanError("plan_id must not be empty")
        if not self.repository_revision:
            raise MaintenancePlanError("repository_revision must not be empty")
        if not self.as_of:
            raise MaintenancePlanError("as_of must not be empty")
        normalized_config = MaintenanceConfig.from_dict(self.config).to_dict()
        object.__setattr__(self, "config", normalized_config)
        operation_ids = [item.operation_id for item in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise MaintenancePlanError("operation_id values must be unique within a plan")
        _validate_operation_conflicts(self.operations)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "plan_id": self.plan_id,
            "repository_revision": self.repository_revision,
            "scope_mode": self.scope_mode,
            "memory_project_key": self.memory_project_key,
            "as_of": self.as_of,
            "config": dict(self.config),
            "input_summary": dict(self.input_summary),
            "operations": [item.to_dict() for item in self.operations],
            "summary": dict(self.summary),
        }
        return sanitize_json_value(payload)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaintenancePlan":
        raw_operations = data.get("operations") or []
        if not isinstance(raw_operations, Sequence) or isinstance(raw_operations, (str, bytes)):
            raise MaintenancePlanError("operations must be an array")
        config = MaintenanceConfig.from_dict(_require_mapping(data.get("config"), "config"))
        return cls(
            schema_version=_as_int(data.get("schema_version")),
            policy=str(data.get("policy") or ""),
            plan_id=str(data.get("plan_id") or ""),
            repository_revision=str(data.get("repository_revision") or ""),
            scope_mode=str(data.get("scope_mode") or ""),
            memory_project_key=str(data.get("memory_project_key") or ""),
            as_of=str(data.get("as_of") or ""),
            config=config.to_dict(),
            input_summary=dict(_require_mapping(data.get("input_summary"), "input_summary")),
            operations=tuple(
                MaintenanceOperation.from_dict(_require_mapping(item, "operation"))
                for item in raw_operations
            ),
            summary=dict(_require_mapping(data.get("summary"), "summary")),
        )


@dataclass(frozen=True)
class MaintenanceApplyResult:
    plan_id: str
    status: MaintenanceApplyStatus
    mutation_committed: bool
    audit_complete: bool
    should_retry: bool
    before_revision: str
    after_revision: str
    before_count: int
    after_count: int
    kept: int
    deleted: int
    merged: int
    promoted: int
    removed_ids: tuple[str, ...] = ()
    updated_ids: tuple[str, ...] = ()
    added_ids: tuple[str, ...] = ()
    backup_path: str = ""
    audit_error_stage: str = ""
    audit_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return sanitize_json_value(payload)  # type: ignore[return-value]


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


def maintenance_plan_json(plan: MaintenancePlan) -> str:
    """Return canonical, byte-stable pretty JSON for a maintenance plan."""
    return json.dumps(
        plan.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_maintenance_plan(plan: MaintenancePlan, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(maintenance_plan_json(plan), encoding="utf-8")
    return output


def load_maintenance_plan(path: str | Path) -> MaintenancePlan:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MaintenancePlanError(f"invalid maintenance plan JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise MaintenancePlanError("maintenance plan must be a JSON object")
    return MaintenancePlan.from_dict(payload)


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
    source_fingerprints = {entry.id: entry.fingerprint for entry, _ in ordered}
    source_evidence = {entry.id: evidence.to_dict() for entry, evidence in ordered}
    pair_scores = [
        _merge_pair_score(ordered[left][0], ordered[right][0])
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    ]
    minimum_score = round(min(pair_scores), 6)

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
        "maintenance_source_fingerprints": source_fingerprints,
        "maintenance_redundancy_min": minimum_score,
        "maintenance_source_evidence": source_evidence,
    })
    provisional_replacement = _entry_payload_with_metadata(anchor_entry, metadata)
    operation_id = _operation_id(
        action=MaintenanceAction.MERGE,
        source_ids=source_ids,
        target_ids=(anchor_entry.id,),
        replacements=(provisional_replacement,),
        additions=(),
    )
    metadata["maintenance_operation_id"] = operation_id
    replacement = _entry_payload_with_metadata(anchor_entry, metadata)
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
    content, skill_metadata = _promoted_skill_fields(entry)
    target_id = _promoted_skill_id(entry.id, content)
    lineage = {
        "maintenance_action": MaintenanceAction.PROMOTE.value,
        "maintenance_policy": MAINTENANCE_POLICY,
        "maintenance_as_of": as_of.isoformat(),
        "maintenance_source_ids": [entry.id],
        "maintenance_source_fingerprints": {entry.id: entry.fingerprint},
        "maintenance_source_evidence": {entry.id: evidence.to_dict()},
        "maintenance_parent_id": entry.id,
        "maintenance_parent_tier": evidence.tier,
        "maintenance_parent_value": evidence.value,
        "maintenance_parent_confidence": evidence.confidence,
    }
    skill_metadata.update(lineage)
    skill_metadata["confidence"] = round(max(0.0, min(1.0, evidence.writer_confidence)), 6)
    provisional_target = build_experience_entry(
        id=target_id,
        content=content,
        tier=ExperienceTier.SKILL,
        project_key=entry.project_key,
        scope=entry.scope,
        source="evolver:maintenance",
        run_id=entry.run_id,
        source_task=str(entry.metadata.get("source_task") or entry.metadata.get("task_id") or ""),
        created_by=ExperienceCreatedBy.MAINTENANCE,
        extra_metadata=skill_metadata,
        created_at=as_of,
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
        target_metadata = dict(provisional_target.metadata)
        target_metadata["maintenance_operation_id"] = operation_id
        target = MemoryEntry.from_dict(_entry_payload_with_metadata(provisional_target, target_metadata))
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


def _validate_operation_shape(operation: MaintenanceOperation) -> None:
    target_ids = operation.target_ids
    remove_ids = operation.remove_ids
    replacement_ids = _payload_ids(operation.replacements, "replacement")
    addition_ids = _payload_ids(operation.additions, "addition")
    replacement_entries = tuple(
        _validated_payload_entry(payload, "replacement")
        for payload in operation.replacements
    )
    addition_entries = tuple(
        _validated_payload_entry(payload, "addition")
        for payload in operation.additions
    )
    if len(set(target_ids)) != len(target_ids):
        raise MaintenancePlanError("target_ids must be unique within an operation")
    if len(set(remove_ids)) != len(remove_ids):
        raise MaintenancePlanError("remove_ids must be unique within an operation")
    if len(set(replacement_ids)) != len(replacement_ids):
        raise MaintenancePlanError("replacement ids must be unique within an operation")
    if len(set(addition_ids)) != len(addition_ids):
        raise MaintenancePlanError("addition ids must be unique within an operation")
    expected_operation_id = _operation_id(
        action=operation.action,
        source_ids=operation.source_ids,
        target_ids=target_ids,
        replacements=operation.replacements,
        additions=operation.additions,
    )
    if operation.operation_id != expected_operation_id:
        raise MaintenancePlanError("operation_id does not match its deterministic payload")

    mutation_entries: tuple[MemoryEntry, ...] = ()
    if operation.action == MaintenanceAction.MERGE:
        mutation_entries = replacement_entries
    elif operation.action == MaintenanceAction.PROMOTE:
        mutation_entries = replacement_entries + addition_entries
    for entry in mutation_entries:
        if str(entry.metadata.get("maintenance_operation_id") or "") != operation.operation_id:
            raise MaintenancePlanError(
                f"mutation payload operation id mismatch: {entry.id}"
            )

    if operation.action == MaintenanceAction.KEEP:
        if target_ids or remove_ids or replacement_ids or addition_ids:
            raise MaintenancePlanError("keep operation cannot contain mutation payloads")
        return
    if operation.action == MaintenanceAction.DELETE:
        if target_ids or replacement_ids or addition_ids or set(remove_ids) != set(operation.source_ids):
            raise MaintenancePlanError("delete operation must remove every source and nothing else")
        return
    if operation.action == MaintenanceAction.MERGE:
        if len(operation.source_ids) < 2:
            raise MaintenancePlanError("merge operation requires at least two sources")
        if len(set(operation.source_tiers)) != 1:
            raise MaintenancePlanError("merge operation sources must have one tier")
        if operation.source_tiers[0] == ExperienceTier.TRAJECTORY.value:
            raise MaintenancePlanError("trajectory entries cannot be merged")
        if len(target_ids) != 1 or target_ids[0] not in operation.source_ids:
            raise MaintenancePlanError("merge target must be exactly one source anchor")
        if replacement_ids != target_ids:
            raise MaintenancePlanError("merge must replace exactly its anchor target")
        if set(remove_ids) != set(operation.source_ids) - set(target_ids) or addition_ids:
            raise MaintenancePlanError("merge must remove every non-anchor source")
        return
    if operation.action == MaintenanceAction.PROMOTE:
        if len(operation.source_ids) != 1 or len(target_ids) != 1:
            raise MaintenancePlanError("promote operation requires one source and one target")
        if remove_ids or replacement_ids != operation.source_ids:
            raise MaintenancePlanError("promote must replace its source without removing it")
        if len(addition_ids) > 1 or (addition_ids and addition_ids != target_ids):
            raise MaintenancePlanError("promote addition must be its target")


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


def _validated_payload_entry(payload: Mapping[str, Any], kind: str) -> MemoryEntry:
    content = str(payload.get("content") or "")
    declared_fingerprint = str(payload.get("fingerprint") or "")
    expected_fingerprint = content_fingerprint(content)
    if declared_fingerprint and declared_fingerprint != expected_fingerprint:
        memory_id = str(payload.get("id") or "")
        raise MaintenancePlanError(f"{kind} fingerprint mismatch: {memory_id}")
    return MemoryEntry.from_dict(dict(payload))


def _payload_ids(payloads: Sequence[Mapping[str, Any]], kind: str) -> tuple[str, ...]:
    result: list[str] = []
    for payload in payloads:
        memory_id = str(payload.get("id") or "")
        if not memory_id:
            raise MaintenancePlanError(f"{kind} payload requires a non-empty id")
        result.append(memory_id)
    return tuple(result)


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


def _source_precondition(entry: MemoryEntry, tier: str) -> dict[str, str]:
    return {
        "fingerprint": entry.fingerprint,
        "tier": tier,
        "scope": entry.scope.value,
        "project_key": entry.project_key,
    }


def _operation_id(
    *,
    action: MaintenanceAction,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    replacements: Sequence[Mapping[str, Any]],
    additions: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "policy": MAINTENANCE_POLICY,
        "action": action.value,
        "source_ids": sorted(str(item) for item in source_ids),
        "target_ids": sorted(str(item) for item in target_ids),
        "replacement_fingerprints": sorted(
            str(item.get("fingerprint") or "") for item in replacements
        ),
        "addition_fingerprints": sorted(
            str(item.get("fingerprint") or "") for item in additions
        ),
    }
    return f"op-{_stable_digest(payload)[:24]}"


def _plan_id(
    *,
    repository_revision: str,
    project_key: str,
    as_of: str,
    config: Mapping[str, Any],
    input_summary: Mapping[str, Any],
    operations: Sequence[MaintenanceOperation],
    summary: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": MAINTENANCE_SCHEMA_VERSION,
        "policy": MAINTENANCE_POLICY,
        "repository_revision": repository_revision,
        "scope_mode": MAINTENANCE_SCOPE_MODE,
        "memory_project_key": project_key,
        "as_of": as_of,
        "config": dict(config),
        "input_summary": dict(input_summary),
        "operations": [item.to_dict() for item in operations],
        "summary": dict(summary),
    }
    return f"maint-{_stable_digest(payload)[:24]}"


def _operation_summary(operations: Sequence[MaintenanceOperation]) -> dict[str, int]:
    counts = {action.value: 0 for action in MaintenanceAction}
    for operation in operations:
        counts[operation.action.value] += 1
    return {
        **counts,
        "source_entries_removed": sum(len(item.remove_ids) for item in operations),
        "entries_added": sum(len(item.additions) for item in operations),
    }


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


def _stable_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        sanitize_json_value(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _require_aware_datetime(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value


def _parse_datetime(value: str) -> datetime | None:
    if not str(value or ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _validate_range(name: str, value: Any, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not minimum <= float(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _validate_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _mapping_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise MaintenancePlanError("operation mapping arrays must contain objects")
        result.append(dict(item))
    return tuple(result)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaintenancePlanError(f"{name} must be an object")
    return value


def _valid_tier(value: str) -> ExperienceTier | None:
    try:
        return ExperienceTier(str(value))
    except ValueError:
        return None


__all__ = [
    "AttributionKey",
    "MAINTENANCE_POLICY",
    "MAINTENANCE_SCHEMA_VERSION",
    "MAINTENANCE_SCOPE_MODE",
    "MaintenanceAction",
    "MaintenanceApplyResult",
    "MaintenanceApplyStatus",
    "MaintenanceAttributionError",
    "MaintenanceConfig",
    "MaintenanceError",
    "MaintenanceEvidence",
    "MaintenanceLookupHit",
    "MaintenanceOperation",
    "MaintenancePlan",
    "MaintenancePlanError",
    "load_maintenance_plan",
    "load_project_attribution",
    "lookup_experiences",
    "build_maintenance_plan",
    "maintenance_evidence_for_entry",
    "maintenance_plan_json",
    "redundancy_score",
    "write_maintenance_plan",
]
