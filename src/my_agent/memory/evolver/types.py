from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType
from my_agent.text_safety import sanitize_json_value


EVOLVER_SCHEMA_VERSION = 1


class ExperienceTier(str, Enum):
    TRAJECTORY = "trajectory"
    TIP = "tip"
    SKILL = "skill"
    TOOL = "tool"


class ExperienceCreatedBy(str, Enum):
    MANUAL = "manual"
    WRITER = "writer"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True)
class ExperienceTrajectoryStep:
    step_num: int
    observation: str = ""
    action: str = ""
    action_params: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    reward: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step_num": int(self.step_num),
            "observation": str(self.observation),
            "action": str(self.action),
            "action_params": dict(self.action_params),
            "result": str(self.result),
        }
        if self.reward is not None:
            payload["reward"] = float(self.reward)
        return sanitize_json_value(payload)


@dataclass(frozen=True)
class ExperienceRecord:
    id: str
    content: str
    tier: ExperienceTier
    source_task: str = ""
    created_by: ExperienceCreatedBy = ExperienceCreatedBy.MANUAL
    run_id: str = ""
    project_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_experience_tier(value: str | ExperienceTier | None) -> ExperienceTier | None:
    if isinstance(value, ExperienceTier):
        return value
    if value is None:
        return None
    try:
        return ExperienceTier(str(value))
    except ValueError:
        return None


def experience_tier(entry: MemoryEntry) -> ExperienceTier | None:
    return normalize_experience_tier(entry.metadata.get("evolver_tier"))


def is_experience_entry(entry: MemoryEntry) -> bool:
    return experience_tier(entry) is not None


def experience_metadata(
    *,
    tier: ExperienceTier | str,
    source_task: str = "",
    created_by: ExperienceCreatedBy | str = ExperienceCreatedBy.MANUAL,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_tier = _require_tier(tier)
    resolved_created_by = _require_created_by(created_by)
    payload = dict(extra or {})
    payload.update(
        {
            "evolver_schema_version": EVOLVER_SCHEMA_VERSION,
            "evolver_tier": resolved_tier.value,
            "source_task": str(source_task or ""),
            "created_by": resolved_created_by.value,
        }
    )
    return payload


def build_experience_entry(
    *,
    id: str,
    content: str,
    tier: ExperienceTier | str,
    project_key: str,
    scope: MemoryScope = MemoryScope.PROJECT,
    source: str = "",
    run_id: str = "",
    source_task: str = "",
    created_by: ExperienceCreatedBy | str = ExperienceCreatedBy.MANUAL,
    extra_metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> MemoryEntry:
    if not str(content).strip():
        raise ValueError("experience content must not be empty")
    resolved_tier = _require_tier(tier)
    metadata = experience_metadata(
        tier=resolved_tier,
        source_task=source_task,
        created_by=created_by,
        extra=extra_metadata,
    )
    return MemoryEntry.build(
        id=id,
        content=content,
        type=MemoryType.FACT,
        scope=scope,
        source=source or f"evolver:{resolved_tier.value}",
        token_count=estimate_tokens(content),
        created_at=created_at,
        project_key=project_key,
        run_id=run_id,
        metadata=metadata,
    )


def experience_record_from_entry(entry: MemoryEntry) -> ExperienceRecord | None:
    tier = experience_tier(entry)
    if tier is None:
        return None
    metadata = dict(entry.metadata)
    return ExperienceRecord(
        id=entry.id,
        content=entry.content,
        tier=tier,
        source_task=str(metadata.get("source_task") or metadata.get("task_id") or ""),
        created_by=_normalize_created_by(metadata.get("created_by")) or ExperienceCreatedBy.MANUAL,
        run_id=entry.run_id,
        project_key=entry.project_key,
        metadata=metadata,
    )


def _require_tier(value: str | ExperienceTier) -> ExperienceTier:
    tier = normalize_experience_tier(value)
    if tier is None:
        raise ValueError(f"invalid experience tier: {value!r}")
    return tier


def _normalize_created_by(value: str | ExperienceCreatedBy | None) -> ExperienceCreatedBy | None:
    if isinstance(value, ExperienceCreatedBy):
        return value
    if value is None:
        return None
    try:
        return ExperienceCreatedBy(str(value))
    except ValueError:
        return None


def _require_created_by(value: str | ExperienceCreatedBy) -> ExperienceCreatedBy:
    created_by = _normalize_created_by(value)
    if created_by is None:
        raise ValueError(f"invalid experience created_by: {value!r}")
    return created_by


__all__ = [
    "EVOLVER_SCHEMA_VERSION",
    "ExperienceCreatedBy",
    "ExperienceRecord",
    "ExperienceTier",
    "ExperienceTrajectoryStep",
    "build_experience_entry",
    "experience_metadata",
    "experience_record_from_entry",
    "experience_tier",
    "is_experience_entry",
    "normalize_experience_tier",
]
