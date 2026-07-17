"""Formal evaluator and decision-event contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from my_agent.policy.identity import PolicyIdentity, require_sha256
from my_agent.training.role_views import CanonicalMessage, CanonicalTool, TaskOutcomeRef


DECISION_EVENT_SCHEMA_VERSION = "opd-decision-v2"
_ROLES = frozenset({"selection", "action", "writing", "maintenance"})
_PURPOSES = frozenset({"fast_loop_evidence", "opd_learner"})


@dataclass(frozen=True)
class EvaluatorIdentity:
    name: str
    version: str
    evaluator_hash: str

    def __post_init__(self) -> None:
        _require_nonblank_string(self.name, "evaluator name")
        _require_nonblank_string(self.version, "evaluator version")
        require_sha256(self.evaluator_hash, field_name="evaluator_hash")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "evaluator_hash": self.evaluator_hash,
        }


@dataclass(frozen=True)
class AuthoritativeTaskOutcome:
    task_id: str
    task_group: str
    task_valid: bool
    resolved: bool
    reward: float
    evaluator: EvaluatorIdentity
    outcome_finalized: bool = True

    def __post_init__(self) -> None:
        _require_nonblank_string(self.task_id, "task_id")
        _require_nonblank_string(self.task_group, "task_group")
        if not isinstance(self.evaluator, EvaluatorIdentity):
            raise ValueError("authoritative outcome requires EvaluatorIdentity")
        if not isfinite(self.reward):
            raise ValueError("authoritative outcome reward must be finite")

    def require_formal(self) -> None:
        if not self.outcome_finalized:
            raise ValueError("formal OPD requires outcome_finalized=true")

    def to_ref(self) -> TaskOutcomeRef:
        self.require_formal()
        return TaskOutcomeRef(
            task_id=self.task_id,
            task_group=self.task_group,
            reward=self.reward,
            resolved=self.resolved,
            evaluator_name=self.evaluator.name,
            evaluator_version=self.evaluator.version,
            evaluator_hash=self.evaluator.evaluator_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_group": self.task_group,
            "task_valid": self.task_valid,
            "resolved": self.resolved,
            "reward": self.reward,
            "evaluator_name": self.evaluator.name,
            "evaluator_version": self.evaluator.version,
            "evaluator_hash": self.evaluator.evaluator_hash,
            "outcome_finalized": self.outcome_finalized,
        }


@dataclass(frozen=True)
class DecisionEvent:
    role: str
    purpose: str
    decision_id: str
    trajectory_id: str
    turn_index: int
    step_index: int
    task_id: str
    task_group: str
    stream_id: str
    memory_project_key: str
    run_id: str
    policy_identity: PolicyIdentity
    repository_revision: str
    candidate_snapshot_hash: str
    canonical_messages: tuple[CanonicalMessage, ...]
    canonical_tools: tuple[CanonicalTool, ...]
    rendered_prompt_hash: str
    prompt_token_ids: tuple[int, ...]
    raw_completion: str
    completion_token_ids: tuple[int, ...]
    assistant_loss_mask: tuple[int, ...]
    parsed_output: Mapping[str, Any]
    retry_of: str | None
    status: str
    schema_version: str = DECISION_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported decision event schema: {self.schema_version!r}")
        if self.role not in _ROLES:
            raise ValueError(f"unsupported decision role: {self.role!r}")
        if self.purpose not in _PURPOSES:
            raise ValueError(f"unsupported decision purpose: {self.purpose!r}")
        if not isinstance(self.policy_identity, PolicyIdentity):
            raise ValueError("decision event requires PolicyIdentity")
        for field_name in (
            "decision_id", "trajectory_id", "task_id", "task_group", "stream_id",
            "memory_project_key", "run_id", "repository_revision", "status",
        ):
            _require_nonblank_string(getattr(self, field_name), f"decision event {field_name}")
        for field_name in ("turn_index", "step_index"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"decision event {field_name} must be a non-negative integer")
        for field_name in ("candidate_snapshot_hash", "rendered_prompt_hash"):
            require_sha256(getattr(self, field_name), field_name=field_name)
        _validate_token_ids(self.prompt_token_ids, "prompt_token_ids")
        _validate_token_ids(self.completion_token_ids, "completion_token_ids")
        if len(self.assistant_loss_mask) != len(self.completion_token_ids):
            raise ValueError("assistant_loss_mask must align with completion_token_ids")
        if any(value not in (0, 1) for value in self.assistant_loss_mask):
            raise ValueError("assistant_loss_mask must contain only 0 or 1")

    @property
    def policy_identity_hash(self) -> str:
        return self.policy_identity.identity_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "purpose": self.purpose,
            "decision_id": self.decision_id,
            "trajectory_id": self.trajectory_id,
            "turn_index": self.turn_index,
            "step_index": self.step_index,
            "task_id": self.task_id,
            "task_group": self.task_group,
            "stream_id": self.stream_id,
            "memory_project_key": self.memory_project_key,
            "run_id": self.run_id,
            "policy_identity": self.policy_identity.to_dict(),
            "policy_identity_hash": self.policy_identity_hash,
            "repository_revision": self.repository_revision,
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "canonical_messages": [item.to_dict() for item in self.canonical_messages],
            "canonical_tools": [item.to_dict() for item in self.canonical_tools],
            "rendered_prompt_hash": self.rendered_prompt_hash,
            "prompt_token_ids": list(self.prompt_token_ids),
            "raw_completion": self.raw_completion,
            "completion_token_ids": list(self.completion_token_ids),
            "assistant_loss_mask": list(self.assistant_loss_mask),
            "parsed_output": dict(self.parsed_output),
            "retry_of": self.retry_of,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionEvent":
        fields = (
            "schema_version", "role", "purpose", "decision_id", "trajectory_id",
            "turn_index", "step_index", "task_id", "task_group", "stream_id",
            "memory_project_key", "run_id", "policy_identity", "policy_identity_hash",
            "repository_revision", "candidate_snapshot_hash", "canonical_messages",
            "canonical_tools", "rendered_prompt_hash", "prompt_token_ids", "raw_completion",
            "completion_token_ids", "assistant_loss_mask", "parsed_output", "retry_of", "status",
        )
        _require_exact_fields(data, fields, "decision event")
        identity_data = _mapping(data["policy_identity"], "policy_identity")
        identity = PolicyIdentity.from_dict(identity_data)
        if data["policy_identity_hash"] != identity.identity_hash:
            raise ValueError("policy_identity_hash does not match policy_identity")
        parsed_output = _mapping(data["parsed_output"], "parsed_output")
        retry_of = data["retry_of"]
        if retry_of is not None and not isinstance(retry_of, str):
            raise ValueError("retry_of must be a string or null")
        return cls(
            schema_version=_required_string(data["schema_version"], "schema_version"),
            role=_required_string(data["role"], "role"),
            purpose=_required_string(data["purpose"], "purpose"),
            decision_id=_required_string(data["decision_id"], "decision_id"),
            trajectory_id=_required_string(data["trajectory_id"], "trajectory_id"),
            turn_index=_strict_int(data["turn_index"], "turn_index"),
            step_index=_strict_int(data["step_index"], "step_index"),
            task_id=_required_string(data["task_id"], "task_id"),
            task_group=_required_string(data["task_group"], "task_group"),
            stream_id=_required_string(data["stream_id"], "stream_id"),
            memory_project_key=_required_string(data["memory_project_key"], "memory_project_key"),
            run_id=_required_string(data["run_id"], "run_id"),
            policy_identity=identity,
            repository_revision=_required_string(data["repository_revision"], "repository_revision"),
            candidate_snapshot_hash=_required_string(data["candidate_snapshot_hash"], "candidate_snapshot_hash"),
            canonical_messages=_object_tuple(data["canonical_messages"], CanonicalMessage.from_dict, "canonical_messages"),
            canonical_tools=_object_tuple(data["canonical_tools"], CanonicalTool.from_dict, "canonical_tools"),
            rendered_prompt_hash=_required_string(data["rendered_prompt_hash"], "rendered_prompt_hash"),
            prompt_token_ids=_int_tuple(data["prompt_token_ids"], "prompt_token_ids"),
            raw_completion=_string(data["raw_completion"], "raw_completion"),
            completion_token_ids=_int_tuple(data["completion_token_ids"], "completion_token_ids"),
            assistant_loss_mask=_int_tuple(data["assistant_loss_mask"], "assistant_loss_mask"),
            parsed_output=parsed_output,
            retry_of=retry_of,
            status=_required_string(data["status"], "status"),
        )


def _validate_token_ids(values: tuple[int, ...], field_name: str) -> None:
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError(f"{field_name} must contain non-negative integer token IDs")


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _required_string(value: Any, field_name: str) -> str:
    _require_nonblank_string(value, field_name)
    return value


def _require_nonblank_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _strict_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return tuple(_strict_int(item, field_name) for item in value)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _object_tuple(value: Any, loader: Any, field_name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return tuple(loader(_mapping(item, field_name)) for item in value)


def _require_exact_fields(data: Mapping[str, Any], fields: tuple[str, ...], name: str) -> None:
    missing = [field for field in fields if field not in data]
    unknown = sorted(set(data) - set(fields))
    if missing or unknown:
        raise ValueError(f"invalid {name} fields: missing={missing}, unknown={unknown}")


__all__ = [
    "DECISION_EVENT_SCHEMA_VERSION",
    "AuthoritativeTaskOutcome",
    "DecisionEvent",
    "EvaluatorIdentity",
]
