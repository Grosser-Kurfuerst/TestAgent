"""Evidence adaptation and pure deterministic legacy maintenance planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import math

from my_agent.json_safety import loads_json_strict
from my_agent.opd_data.legacy.attribution import MemoryAttributionRecord
from my_agent.memory.evolver.maintenance.contracts import (
    AttributionKey,
    MaintenanceAttributionError,
    MaintenanceEvidence,
    MaintenancePlanError,
    _evidence_float,
    _evidence_int,
    _valid_tier,
    _validate_evidence_values,
)
from my_agent.memory.experience.models import (
    ExperienceMemory,
)
from my_agent.memory.types import (
    MemoryScope,
)
from my_agent.memory.evolver.maintenance.lookup import (
    lookup_experiences as _shared_lookup_experiences,
    redundancy_score as _shared_redundancy_score,
)


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






lookup_experiences = _shared_lookup_experiences
redundancy_score = _shared_redundancy_score


__all__ = [
    "load_project_attribution",
    "lookup_experiences",
    "maintenance_evidence_for_entry",
    "redundancy_score",
]
