"""Deterministic contracts and evidence adapters for memory maintenance.

The planner is intentionally pure: repository I/O and transactional apply are
separate concerns added by later Phase 6 iterations.  This module owns the
stable schema shared by planning, review, audit, and future dataset export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from my_agent.memory.evolver.attribution import MemoryAttributionRecord
from my_agent.memory.evolver.types import ExperienceTier, experience_tier
from my_agent.memory.types import MemoryEntry, MemoryScope
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
    "MaintenanceOperation",
    "MaintenancePlan",
    "MaintenancePlanError",
    "load_maintenance_plan",
    "load_project_attribution",
    "maintenance_evidence_for_entry",
    "maintenance_plan_json",
    "write_maintenance_plan",
]
