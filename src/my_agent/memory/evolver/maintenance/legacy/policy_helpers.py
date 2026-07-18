"""Deterministic helper rules for legacy maintenance policies."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from my_agent.memory.evolver.maintenance.contracts import (
    MaintenanceConfig,
    MaintenanceEvidence,
    MaintenancePlanError,
    _parse_datetime,
)
from my_agent.memory.evolver.maintenance.legacy.planner import redundancy_score
from my_agent.memory.experience.models import (
    ExperienceMemory,
    ExperienceTrajectoryStep,
    ExperienceTier,
    SkillPayload,
    ToolPayload,
)
from my_agent.memory.types import normalize_content


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


__all__: list[str] = []
