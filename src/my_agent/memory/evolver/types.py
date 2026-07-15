from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType, content_fingerprint
from my_agent.text_safety import sanitize_json_value


EVOLVER_SCHEMA_VERSION = 1

_MAX_COLLECTION_ITEMS = 64
_MAX_TRAJECTORY_STEPS = 256
_MAX_SHORT_TEXT_CHARS = 1_000
_MAX_LONG_TEXT_CHARS = 20_000


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

    def __post_init__(self) -> None:
        if isinstance(self.step_num, bool) or not isinstance(self.step_num, int):
            raise ValueError("trajectory step_num must be an integer")
        object.__setattr__(
            self,
            "observation",
            _normalize_text(self.observation, "trajectory observation", max_chars=_MAX_LONG_TEXT_CHARS),
        )
        object.__setattr__(
            self,
            "action",
            _normalize_text(self.action, "trajectory action", max_chars=_MAX_SHORT_TEXT_CHARS),
        )
        object.__setattr__(
            self,
            "action_params",
            _json_object_copy(self.action_params, "trajectory action_params"),
        )
        object.__setattr__(
            self,
            "result",
            _normalize_text(self.result, "trajectory result", max_chars=_MAX_LONG_TEXT_CHARS),
        )
        object.__setattr__(self, "reward", _optional_finite_float(self.reward, "trajectory reward"))

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
class TrajectoryPayload:
    task_description: str
    steps: tuple[ExperienceTrajectoryStep, ...]
    outcome: str
    total_reward: float | None = None
    key_learnings: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_description",
            _normalize_text(
                self.task_description,
                "trajectory task_description",
                required=True,
                max_chars=_MAX_LONG_TEXT_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "outcome",
            _normalize_text(
                self.outcome,
                "trajectory outcome",
                required=True,
                max_chars=_MAX_SHORT_TEXT_CHARS,
            ),
        )
        object.__setattr__(self, "steps", _normalize_trajectory_steps(self.steps))
        object.__setattr__(
            self,
            "total_reward",
            _optional_finite_float(self.total_reward, "trajectory total_reward"),
        )
        object.__setattr__(
            self,
            "key_learnings",
            _normalize_text_tuple(self.key_learnings, "trajectory key_learnings"),
        )
        object.__setattr__(self, "tags", _normalize_text_tuple(self.tags, "trajectory tags"))


@dataclass(frozen=True)
class TipPayload:
    category: str
    severity: str
    trigger: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category",
            _normalize_text(self.category, "tip category", required=True, max_chars=_MAX_SHORT_TEXT_CHARS),
        )
        severity = _normalize_text(
            self.severity,
            "tip severity",
            required=True,
            max_chars=_MAX_SHORT_TEXT_CHARS,
        ).casefold()
        if severity not in {"info", "warning", "critical"}:
            raise ValueError("tip severity must be one of: info, warning, critical")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(
            self,
            "trigger",
            _normalize_text(self.trigger, "tip trigger", required=True, max_chars=_MAX_LONG_TEXT_CHARS),
        )


@dataclass(frozen=True)
class SkillPayload:
    category: str
    technique: str
    preconditions: tuple[str, ...]
    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category",
            _normalize_text(self.category, "skill category", required=True, max_chars=_MAX_SHORT_TEXT_CHARS),
        )
        object.__setattr__(
            self,
            "technique",
            _normalize_text(
                self.technique,
                "skill technique",
                required=True,
                max_chars=_MAX_LONG_TEXT_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "preconditions",
            _normalize_text_tuple(self.preconditions, "skill preconditions"),
        )
        steps = _normalize_text_tuple(self.steps, "skill steps")
        if not steps:
            raise ValueError("skill steps must contain at least one non-empty item")
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True)
class ToolPayload:
    name: str
    language: str
    code: str
    input_description: str = ""
    output_description: str = ""
    command: str = ""
    args_schema: dict[str, Any] = field(default_factory=dict)
    repo_context: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_text(self.name, "tool name", required=True, max_chars=_MAX_SHORT_TEXT_CHARS),
        )
        object.__setattr__(
            self,
            "language",
            _normalize_text(self.language, "tool language", max_chars=_MAX_SHORT_TEXT_CHARS),
        )
        code = _normalize_text(self.code, "tool code", max_chars=_MAX_LONG_TEXT_CHARS)
        command = _normalize_text(self.command, "tool command", max_chars=_MAX_LONG_TEXT_CHARS)
        if not code and not command:
            raise ValueError("tool code or command must be non-empty")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "command", command)
        object.__setattr__(
            self,
            "input_description",
            _normalize_text(
                self.input_description,
                "tool input_description",
                max_chars=_MAX_LONG_TEXT_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "output_description",
            _normalize_text(
                self.output_description,
                "tool output_description",
                max_chars=_MAX_LONG_TEXT_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "args_schema",
            _json_object_copy(self.args_schema, "tool args_schema"),
        )
        object.__setattr__(
            self,
            "repo_context",
            _normalize_text(self.repo_context, "tool repo_context", max_chars=_MAX_LONG_TEXT_CHARS),
        )


ExperiencePayload = TrajectoryPayload | TipPayload | SkillPayload | ToolPayload


@dataclass(frozen=True)
class ExperienceMemory:
    id: str
    content: str
    tier: ExperienceTier
    payload: ExperiencePayload
    scope: MemoryScope
    project_key: str
    created_at: datetime
    token_count: int
    fingerprint: str

    source_task: str = ""
    run_id: str = ""
    stream_id: str = ""
    created_by: ExperienceCreatedBy = ExperienceCreatedBy.MANUAL

    writer_confidence: float = 1.0
    attribution_value: float = 0.0
    attribution_confidence: float = 0.0
    candidate_count: int = 0
    selected_count: int = 0
    not_selected_count: int = 0
    success_when_selected: float | None = None
    success_when_candidate_not_selected: float | None = None
    reward_when_selected: float | None = None
    reward_when_candidate_not_selected: float | None = None
    last_used: datetime | None = None
    attribution_updated_at: datetime | None = None

    protected: bool = False
    invalidated: bool = False
    promoted_to: str = ""
    maintenance_operation_id: str = ""
    parent_id: str = ""
    parent_tier: ExperienceTier | None = None

    def __post_init__(self) -> None:
        memory_id = _normalize_text(self.id, "experience id", required=True, max_chars=_MAX_SHORT_TEXT_CHARS)
        content = _normalize_text(self.content, "experience content", required=True, max_chars=_MAX_LONG_TEXT_CHARS)
        object.__setattr__(self, "id", memory_id)
        object.__setattr__(self, "content", content)

        if not isinstance(self.tier, ExperienceTier):
            raise ValueError("experience tier must be an ExperienceTier")
        expected_payload_type = _PAYLOAD_TYPE_BY_TIER[self.tier]
        if type(self.payload) is not expected_payload_type:
            raise ValueError(
                f"experience tier {self.tier.value!r} requires payload "
                f"{expected_payload_type.__name__}"
            )

        if not isinstance(self.scope, MemoryScope) or self.scope == MemoryScope.SESSION:
            raise ValueError("experience scope must be global or project")
        project_key = _normalize_text(
            self.project_key,
            "experience project_key",
            max_chars=_MAX_LONG_TEXT_CHARS,
        )
        if self.scope == MemoryScope.GLOBAL and project_key:
            raise ValueError("global experience project_key must be empty")
        if self.scope == MemoryScope.PROJECT and not project_key:
            raise ValueError("project experience project_key must be non-empty")
        object.__setattr__(self, "project_key", project_key)

        _require_aware_datetime(self.created_at, "experience created_at")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count < 0:
            raise ValueError("experience token_count must be a non-negative integer")
        fingerprint = _normalize_text(
            self.fingerprint,
            "experience fingerprint",
            required=True,
            max_chars=_MAX_SHORT_TEXT_CHARS,
        )
        if fingerprint != content_fingerprint(content):
            raise ValueError("experience fingerprint does not match content")
        object.__setattr__(self, "fingerprint", fingerprint)

        for field_name in ("source_task", "run_id", "stream_id"):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(
                    getattr(self, field_name),
                    f"experience {field_name}",
                    max_chars=_MAX_SHORT_TEXT_CHARS,
                ),
            )
        if not isinstance(self.created_by, ExperienceCreatedBy):
            raise ValueError("experience created_by must be an ExperienceCreatedBy")

        object.__setattr__(
            self,
            "writer_confidence",
            _bounded_float(self.writer_confidence, "experience writer_confidence", 0.0, 1.0),
        )
        object.__setattr__(
            self,
            "attribution_value",
            _bounded_float(self.attribution_value, "experience attribution_value", -1.0, 1.0),
        )
        object.__setattr__(
            self,
            "attribution_confidence",
            _bounded_float(
                self.attribution_confidence,
                "experience attribution_confidence",
                0.0,
                1.0,
            ),
        )
        counts = {
            name: _nonnegative_int(getattr(self, name), f"experience {name}")
            for name in ("candidate_count", "selected_count", "not_selected_count")
        }
        if counts["candidate_count"] != counts["selected_count"] + counts["not_selected_count"]:
            raise ValueError("experience attribution counts are inconsistent")
        for name, value in counts.items():
            object.__setattr__(self, name, value)

        for field_name in ("success_when_selected", "success_when_candidate_not_selected"):
            object.__setattr__(
                self,
                field_name,
                _optional_bounded_float(getattr(self, field_name), f"experience {field_name}", 0.0, 1.0),
            )
        for field_name in ("reward_when_selected", "reward_when_candidate_not_selected"):
            object.__setattr__(
                self,
                field_name,
                _optional_finite_float(getattr(self, field_name), f"experience {field_name}"),
            )
        for field_name in ("last_used", "attribution_updated_at"):
            value = getattr(self, field_name)
            if value is not None:
                _require_aware_datetime(value, f"experience {field_name}")

        for field_name in ("protected", "invalidated"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"experience {field_name} must be boolean")
        for field_name in ("promoted_to", "maintenance_operation_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(
                    getattr(self, field_name),
                    f"experience {field_name}",
                    max_chars=_MAX_SHORT_TEXT_CHARS,
                ),
            )
        if self.parent_tier is not None and not isinstance(self.parent_tier, ExperienceTier):
            raise ValueError("experience parent_tier must be an ExperienceTier or None")


_PAYLOAD_TYPE_BY_TIER: dict[ExperienceTier, type[ExperiencePayload]] = {
    ExperienceTier.TRAJECTORY: TrajectoryPayload,
    ExperienceTier.TIP: TipPayload,
    ExperienceTier.SKILL: SkillPayload,
    ExperienceTier.TOOL: ToolPayload,
}


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


def _normalize_text(
    value: Any,
    name: str,
    *,
    required: bool = False,
    max_chars: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = sanitize_json_value(value).strip()
    if required and not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > max_chars:
        raise ValueError(f"{name} exceeds {max_chars} characters")
    return normalized


def _normalize_text_tuple(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if len(values) > _MAX_COLLECTION_ITEMS:
        raise ValueError(f"{name} exceeds {_MAX_COLLECTION_ITEMS} items")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        item = _normalize_text(
            value,
            f"{name}[{index}]",
            max_chars=_MAX_SHORT_TEXT_CHARS,
        )
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _normalize_trajectory_steps(values: Any) -> tuple[ExperienceTrajectoryStep, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("trajectory steps must be a list or tuple")
    if not values:
        raise ValueError("trajectory steps must contain at least one step")
    if len(values) > _MAX_TRAJECTORY_STEPS:
        raise ValueError(f"trajectory steps exceeds {_MAX_TRAJECTORY_STEPS} items")
    steps: list[ExperienceTrajectoryStep] = []
    for step in values:
        if not isinstance(step, ExperienceTrajectoryStep):
            raise ValueError("trajectory steps must contain ExperienceTrajectoryStep values")
        steps.append(step)
    steps.sort(key=lambda step: step.step_num)
    if len({step.step_num for step in steps}) != len(steps):
        raise ValueError("trajectory step_num values must be unique")
    return tuple(steps)


def _json_object_copy(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    try:
        encoded = json.dumps(
            sanitize_json_value(value),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by the input check
        raise ValueError(f"{name} must be a JSON object")
    return decoded


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    normalized = _finite_float(value, name)
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _optional_finite_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


def _optional_bounded_float(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    return _bounded_float(value, name, minimum, maximum)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_aware_datetime(value: Any, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


__all__ = [
    "EVOLVER_SCHEMA_VERSION",
    "ExperienceCreatedBy",
    "ExperienceMemory",
    "ExperiencePayload",
    "ExperienceRecord",
    "ExperienceTier",
    "ExperienceTrajectoryStep",
    "SkillPayload",
    "TipPayload",
    "ToolPayload",
    "TrajectoryPayload",
    "build_experience_entry",
    "experience_metadata",
    "experience_record_from_entry",
    "experience_tier",
    "is_experience_entry",
    "normalize_experience_tier",
]
