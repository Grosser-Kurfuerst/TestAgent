from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from my_agent.memory.evolver.types import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperiencePayload,
    ExperienceTier,
    ExperienceTrajectoryStep,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.types import MemoryScope
from my_agent.text_safety import sanitize_json_value


EXPERIENCE_SCHEMA_VERSION = 2

_REQUIRED_TOP_LEVEL_FIELDS = frozenset({
    "schema_version",
    "id",
    "content",
    "tier",
    "payload",
    "scope",
    "project_key",
    "created_at",
    "token_count",
    "fingerprint",
})
_OPTIONAL_TOP_LEVEL_FIELDS = frozenset({
    "source_task",
    "run_id",
    "stream_id",
    "created_by",
    "writer_confidence",
    "attribution_value",
    "attribution_confidence",
    "candidate_count",
    "selected_count",
    "not_selected_count",
    "success_when_selected",
    "success_when_candidate_not_selected",
    "reward_when_selected",
    "reward_when_candidate_not_selected",
    "last_used",
    "attribution_updated_at",
    "protected",
    "invalidated",
    "promoted_to",
    "maintenance_operation_id",
    "parent_id",
    "parent_tier",
})
_TOP_LEVEL_FIELDS = _REQUIRED_TOP_LEVEL_FIELDS | _OPTIONAL_TOP_LEVEL_FIELDS


def experience_to_dict(memory: ExperienceMemory) -> dict[str, Any]:
    if not isinstance(memory, ExperienceMemory):
        raise TypeError("memory must be an ExperienceMemory")
    payload: dict[str, Any] = {
        "schema_version": EXPERIENCE_SCHEMA_VERSION,
        "id": memory.id,
        "content": memory.content,
        "tier": memory.tier.value,
        "payload": experience_payload_to_dict(memory.payload),
        "scope": memory.scope.value,
        "project_key": memory.project_key,
        "created_at": memory.created_at.isoformat(),
        "token_count": memory.token_count,
        "fingerprint": memory.fingerprint,
        "source_task": memory.source_task,
        "run_id": memory.run_id,
        "stream_id": memory.stream_id,
        "created_by": memory.created_by.value,
        "writer_confidence": memory.writer_confidence,
        "attribution_value": memory.attribution_value,
        "attribution_confidence": memory.attribution_confidence,
        "candidate_count": memory.candidate_count,
        "selected_count": memory.selected_count,
        "not_selected_count": memory.not_selected_count,
        "success_when_selected": memory.success_when_selected,
        "success_when_candidate_not_selected": memory.success_when_candidate_not_selected,
        "reward_when_selected": memory.reward_when_selected,
        "reward_when_candidate_not_selected": memory.reward_when_candidate_not_selected,
        "last_used": _datetime_to_string(memory.last_used),
        "attribution_updated_at": _datetime_to_string(memory.attribution_updated_at),
        "protected": memory.protected,
        "invalidated": memory.invalidated,
        "promoted_to": memory.promoted_to,
        "maintenance_operation_id": memory.maintenance_operation_id,
        "parent_id": memory.parent_id,
        "parent_tier": memory.parent_tier.value if memory.parent_tier is not None else None,
    }
    sanitized = sanitize_json_value(payload)
    json.dumps(sanitized, ensure_ascii=False, allow_nan=False)
    return sanitized


def experience_from_dict(payload: Mapping[str, Any]) -> ExperienceMemory:
    data = _mapping_copy(payload, "experience")
    _validate_fields(
        data,
        required=_REQUIRED_TOP_LEVEL_FIELDS,
        allowed=_TOP_LEVEL_FIELDS,
        name="experience",
    )
    schema_version = _required_int(data["schema_version"], "experience schema_version")
    if schema_version != EXPERIENCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported experience schema version: {schema_version}; "
            "clear legacy long-term memory before using the typed experience store"
        )

    tier = _enum_value(ExperienceTier, data["tier"], "experience tier")
    concrete_payload = _payload_from_dict(tier, data["payload"])
    return ExperienceMemory(
        id=_required_string(data["id"], "experience id"),
        content=_required_string(data["content"], "experience content"),
        tier=tier,
        payload=concrete_payload,
        scope=_enum_value(MemoryScope, data["scope"], "experience scope"),
        project_key=_required_string(data["project_key"], "experience project_key", allow_empty=True),
        created_at=_datetime_from_value(data["created_at"], "experience created_at"),
        token_count=_required_int(data["token_count"], "experience token_count"),
        fingerprint=_required_string(data["fingerprint"], "experience fingerprint"),
        source_task=_optional_string(data, "source_task", ""),
        run_id=_optional_string(data, "run_id", ""),
        stream_id=_optional_string(data, "stream_id", ""),
        created_by=_optional_enum(data, "created_by", ExperienceCreatedBy, ExperienceCreatedBy.MANUAL),
        writer_confidence=_optional_number(data, "writer_confidence", 1.0),
        attribution_value=_optional_number(data, "attribution_value", 0.0),
        attribution_confidence=_optional_number(data, "attribution_confidence", 0.0),
        candidate_count=_optional_int(data, "candidate_count", 0),
        selected_count=_optional_int(data, "selected_count", 0),
        not_selected_count=_optional_int(data, "not_selected_count", 0),
        success_when_selected=_optional_nullable_number(data, "success_when_selected"),
        success_when_candidate_not_selected=_optional_nullable_number(
            data,
            "success_when_candidate_not_selected",
        ),
        reward_when_selected=_optional_nullable_number(data, "reward_when_selected"),
        reward_when_candidate_not_selected=_optional_nullable_number(
            data,
            "reward_when_candidate_not_selected",
        ),
        last_used=_optional_datetime(data, "last_used"),
        attribution_updated_at=_optional_datetime(data, "attribution_updated_at"),
        protected=_optional_bool(data, "protected", False),
        invalidated=_optional_bool(data, "invalidated", False),
        promoted_to=_optional_string(data, "promoted_to", ""),
        maintenance_operation_id=_optional_string(data, "maintenance_operation_id", ""),
        parent_id=_optional_string(data, "parent_id", ""),
        parent_tier=_optional_nullable_enum(data, "parent_tier", ExperienceTier),
    )


def experience_canonical_json(memory: ExperienceMemory) -> str:
    return json.dumps(
        experience_to_dict(memory),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def experience_payload_to_dict(payload: ExperiencePayload) -> dict[str, Any]:
    """Serialize one concrete tier payload using the canonical v2 schema."""
    if isinstance(payload, TrajectoryPayload):
        return {
            "task_description": payload.task_description,
            "steps": [_trajectory_step_to_dict(step) for step in payload.steps],
            "outcome": payload.outcome,
            "total_reward": payload.total_reward,
            "key_learnings": list(payload.key_learnings),
            "tags": list(payload.tags),
        }
    if isinstance(payload, TipPayload):
        return {
            "category": payload.category,
            "severity": payload.severity,
            "trigger": payload.trigger,
        }
    if isinstance(payload, SkillPayload):
        return {
            "category": payload.category,
            "technique": payload.technique,
            "preconditions": list(payload.preconditions),
            "steps": list(payload.steps),
        }
    if isinstance(payload, ToolPayload):
        return {
            "name": payload.name,
            "language": payload.language,
            "code": payload.code,
            "input_description": payload.input_description,
            "output_description": payload.output_description,
            "command": payload.command,
            "args_schema": dict(payload.args_schema),
            "repo_context": payload.repo_context,
        }
    raise TypeError(f"unsupported experience payload: {type(payload).__name__}")


def experience_payload_from_dict(
    tier: ExperienceTier,
    payload: Mapping[str, Any],
) -> ExperiencePayload:
    """Parse a writer/manual payload through the same strict store boundary."""
    if not isinstance(tier, ExperienceTier):
        raise ValueError("experience payload tier must be an ExperienceTier")
    return _payload_from_dict(tier, payload)


def _trajectory_step_to_dict(step: ExperienceTrajectoryStep) -> dict[str, Any]:
    return {
        "step_num": step.step_num,
        "observation": step.observation,
        "action": step.action,
        "action_params": dict(step.action_params),
        "result": step.result,
        "reward": step.reward,
    }


def _payload_from_dict(tier: ExperienceTier, value: Any):
    data = _mapping_copy(value, f"{tier.value} payload")
    if tier == ExperienceTier.TRAJECTORY:
        _validate_fields(
            data,
            required=frozenset({"task_description", "steps", "outcome"}),
            allowed=frozenset({
                "task_description",
                "steps",
                "outcome",
                "total_reward",
                "key_learnings",
                "tags",
            }),
            name="trajectory payload",
        )
        raw_steps = _required_list(data["steps"], "trajectory payload steps")
        return TrajectoryPayload(
            task_description=_required_string(data["task_description"], "trajectory task_description"),
            steps=tuple(_trajectory_step_from_dict(item) for item in raw_steps),
            outcome=_required_string(data["outcome"], "trajectory outcome"),
            total_reward=_optional_nullable_number(data, "total_reward"),
            key_learnings=_optional_string_tuple(data, "key_learnings"),
            tags=_optional_string_tuple(data, "tags"),
        )
    if tier == ExperienceTier.TIP:
        fields = frozenset({"category", "severity", "trigger"})
        _validate_fields(data, required=fields, allowed=fields, name="tip payload")
        return TipPayload(
            category=_required_string(data["category"], "tip category"),
            severity=_required_string(data["severity"], "tip severity"),
            trigger=_required_string(data["trigger"], "tip trigger"),
        )
    if tier == ExperienceTier.SKILL:
        fields = frozenset({"category", "technique", "preconditions", "steps"})
        _validate_fields(data, required=fields, allowed=fields, name="skill payload")
        return SkillPayload(
            category=_required_string(data["category"], "skill category"),
            technique=_required_string(data["technique"], "skill technique"),
            preconditions=_string_tuple(data["preconditions"], "skill preconditions"),
            steps=_string_tuple(data["steps"], "skill steps"),
        )
    if tier == ExperienceTier.TOOL:
        _validate_fields(
            data,
            required=frozenset({"name", "language", "code"}),
            allowed=frozenset({
                "name",
                "language",
                "code",
                "input_description",
                "output_description",
                "command",
                "args_schema",
                "repo_context",
            }),
            name="tool payload",
        )
        return ToolPayload(
            name=_required_string(data["name"], "tool name"),
            language=_required_string(data["language"], "tool language", allow_empty=True),
            code=_required_string(data["code"], "tool code", allow_empty=True),
            input_description=_optional_string(data, "input_description", ""),
            output_description=_optional_string(data, "output_description", ""),
            command=_optional_string(data, "command", ""),
            args_schema=_optional_mapping(data, "args_schema"),
            repo_context=_optional_string(data, "repo_context", ""),
        )
    raise ValueError(f"unsupported experience tier: {tier.value}")  # pragma: no cover


def _trajectory_step_from_dict(value: Any) -> ExperienceTrajectoryStep:
    data = _mapping_copy(value, "trajectory step")
    _validate_fields(
        data,
        required=frozenset({"step_num"}),
        allowed=frozenset({"step_num", "observation", "action", "action_params", "result", "reward"}),
        name="trajectory step",
    )
    return ExperienceTrajectoryStep(
        step_num=_required_int(data["step_num"], "trajectory step_num"),
        observation=_optional_string(data, "observation", ""),
        action=_optional_string(data, "action", ""),
        action_params=_optional_mapping(data, "action_params"),
        result=_optional_string(data, "result", ""),
        reward=_optional_nullable_number(data, "reward"),
    )


def _mapping_copy(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return dict(value)


def _validate_fields(
    data: Mapping[str, Any],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    name: str,
) -> None:
    actual = frozenset(data)
    missing = sorted(required - actual)
    extra = sorted(actual - allowed)
    if missing or extra:
        raise ValueError(f"{name} fields mismatch: missing={missing}, extra={extra}")


def _required_string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _optional_string(data: Mapping[str, Any], key: str, default: str) -> str:
    if key not in data:
        return default
    return _required_string(data[key], f"experience {key}", allow_empty=True)


def _required_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_int(data: Mapping[str, Any], key: str, default: int) -> int:
    if key not in data:
        return default
    return _required_int(data[key], f"experience {key}")


def _required_number(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return value


def _optional_number(data: Mapping[str, Any], key: str, default: float) -> int | float:
    if key not in data:
        return default
    return _required_number(data[key], f"experience {key}")


def _optional_nullable_number(data: Mapping[str, Any], key: str) -> int | float | None:
    value = data.get(key)
    if value is None:
        return None
    return _required_number(value, f"experience {key}")


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _optional_bool(data: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in data:
        return default
    return _required_bool(data[key], f"experience {key}")


def _required_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    items = _required_list(value, name)
    return tuple(_required_string(item, f"{name} item", allow_empty=True) for item in items)


def _optional_string_tuple(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    if key not in data:
        return ()
    return _string_tuple(data[key], f"experience {key}")


def _optional_mapping(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key not in data:
        return {}
    return _mapping_copy(data[key], f"experience {key}")


def _enum_value(enum_type, value: Any, name: str):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc


def _optional_enum(data: Mapping[str, Any], key: str, enum_type, default):
    if key not in data:
        return default
    return _enum_value(enum_type, data[key], f"experience {key}")


def _optional_nullable_enum(data: Mapping[str, Any], key: str, enum_type):
    value = data.get(key)
    if value is None:
        return None
    return _enum_value(enum_type, value, f"experience {key}")


def _datetime_from_value(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an ISO datetime string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO datetime string") from exc


def _optional_datetime(data: Mapping[str, Any], key: str) -> datetime | None:
    value = data.get(key)
    if value is None:
        return None
    return _datetime_from_value(value, f"experience {key}")


def _datetime_to_string(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = [
    "EXPERIENCE_SCHEMA_VERSION",
    "experience_canonical_json",
    "experience_from_dict",
    "experience_payload_from_dict",
    "experience_payload_to_dict",
    "experience_to_dict",
]
