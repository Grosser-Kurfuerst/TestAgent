"""Typed public and privileged views used by OPD learner generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Mapping, TypeVar
import json

from my_agent.policy.identity import canonical_sha256, require_sha256


ROLE_VIEW_SCHEMA_VERSION = "opd-role-view-v1"
SELECTED_MEMORY_CONTEXT_HEADER = "[Selected evolver memory - frozen for this task]"
_T = TypeVar("_T")


@dataclass(frozen=True)
class CanonicalToolCall:
    call_id: str
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        _require_nonblank(self.call_id, "call_id")
        _require_nonblank(self.name, "name")
        _require_canonical_json(self.arguments_json, field_name="arguments_json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments_json": self.arguments_json,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalToolCall":
        _require_fields(data, ("call_id", "name", "arguments_json"), "canonical tool call")
        return cls(
            _required_string(data["call_id"], "call_id"),
            _required_string(data["name"], "name"),
            _required_string(data["arguments_json"], "arguments_json"),
        )


@dataclass(frozen=True)
class CanonicalMessage:
    role: str
    content: str
    name: str = ""
    tool_call_id: str = ""
    tool_calls: tuple[CanonicalToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"invalid canonical message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError("canonical message content must be a string")
        if not isinstance(self.name, str) or not isinstance(self.tool_call_id, str):
            raise ValueError("canonical message name and tool_call_id must be strings")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": [item.to_dict() for item in self.tool_calls],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalMessage":
        _require_fields(
            data,
            ("role", "content", "name", "tool_call_id", "tool_calls"),
            "canonical message",
        )
        return cls(
            role=_required_string(data["role"], "role"),
            content=_string(data["content"], "content"),
            name=_string(data["name"], "name"),
            tool_call_id=_string(data["tool_call_id"], "tool_call_id"),
            tool_calls=_tuple_from(data["tool_calls"], CanonicalToolCall.from_dict, "tool_calls"),
        )


def without_selected_memory_context(
    messages: tuple[CanonicalMessage, ...],
) -> tuple[CanonicalMessage, ...]:
    return tuple(
        message
        for message in messages
        if not (
            message.role == "system"
            and message.content.startswith(SELECTED_MEMORY_CONTEXT_HEADER)
        )
    )


@dataclass(frozen=True)
class CanonicalTool:
    name: str
    description: str
    parameters_json: str
    schema_hash: str

    def __post_init__(self) -> None:
        _require_nonblank(self.name, "name")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        _require_canonical_json(self.parameters_json, field_name="parameters_json")
        require_sha256(self.schema_hash, field_name="schema_hash")
        if self.schema_hash != canonical_sha256(json.loads(self.parameters_json)):
            raise ValueError("schema_hash does not match parameters_json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters_json": self.parameters_json,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalTool":
        _require_fields(
            data,
            ("name", "description", "parameters_json", "schema_hash"),
            "canonical tool",
        )
        return cls(
            _required_string(data["name"], "name"),
            _string(data["description"], "description"),
            _required_string(data["parameters_json"], "parameters_json"),
            _required_string(data["schema_hash"], "schema_hash"),
        )


@dataclass(frozen=True)
class CandidateSnapshotEntry:
    label: str
    memory_id: str
    tier: str
    content: str
    retrieval_score: float
    rank: int
    token_count: int

    def __post_init__(self) -> None:
        for field_name in ("label", "memory_id", "tier"):
            _require_nonblank(getattr(self, field_name), field_name)
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        _require_finite(self.retrieval_score, "retrieval_score")
        _require_positive_int(self.rank, "rank")
        _require_nonnegative_int(self.token_count, "token_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "memory_id": self.memory_id,
            "tier": self.tier,
            "content": self.content,
            "retrieval_score": self.retrieval_score,
            "rank": self.rank,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateSnapshotEntry":
        fields = ("label", "memory_id", "tier", "content", "retrieval_score", "rank", "token_count")
        _require_fields(data, fields, "candidate snapshot entry")
        return cls(
            label=_required_string(data["label"], "label"),
            memory_id=_required_string(data["memory_id"], "memory_id"),
            tier=_required_string(data["tier"], "tier"),
            content=_string(data["content"], "content"),
            retrieval_score=float(data["retrieval_score"]),
            rank=_strict_int(data["rank"], "rank"),
            token_count=_strict_int(data["token_count"], "token_count"),
        )


@dataclass(frozen=True)
class MemoryValueEvidence:
    memory_id: str
    tier: str
    attribution: float
    gamma: float
    memory_score: float
    status: str

    def __post_init__(self) -> None:
        for field_name in ("memory_id", "tier", "status"):
            _require_nonblank(getattr(self, field_name), field_name)
        for field_name in ("attribution", "gamma", "memory_score"):
            _require_finite(float(getattr(self, field_name)), field_name)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "tier": self.tier,
            "attribution": self.attribution,
            "gamma": self.gamma,
            "memory_score": self.memory_score,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryValueEvidence":
        fields = ("memory_id", "tier", "attribution", "gamma", "memory_score", "status")
        _require_fields(data, fields, "memory value evidence")
        return cls(
            memory_id=_required_string(data["memory_id"], "memory_id"),
            tier=_required_string(data["tier"], "tier"),
            attribution=float(data["attribution"]),
            gamma=float(data["gamma"]),
            memory_score=float(data["memory_score"]),
            status=_required_string(data["status"], "status"),
        )


@dataclass(frozen=True)
class CanonicalTrajectoryStep:
    step_index: int
    observation: str
    action: str
    arguments_json: str
    result: str
    reward: float | None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.step_index, "step_index")
        for field_name in ("observation", "action", "result"):
            if not isinstance(getattr(self, field_name), str):
                raise ValueError(f"{field_name} must be a string")
        _require_canonical_json(self.arguments_json, field_name="arguments_json")
        if self.reward is not None:
            _require_finite(self.reward, "reward")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "observation": self.observation,
            "action": self.action,
            "arguments_json": self.arguments_json,
            "result": self.result,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalTrajectoryStep":
        fields = ("step_index", "observation", "action", "arguments_json", "result", "reward")
        _require_fields(data, fields, "canonical trajectory step")
        reward = data["reward"]
        return cls(
            step_index=_strict_int(data["step_index"], "step_index"),
            observation=_string(data["observation"], "observation"),
            action=_string(data["action"], "action"),
            arguments_json=_required_string(data["arguments_json"], "arguments_json"),
            result=_string(data["result"], "result"),
            reward=None if reward is None else float(reward),
        )


@dataclass(frozen=True)
class TrajectoryEvidence:
    trajectory_id: str
    task_group: str
    outcome: str
    reward: float
    steps: tuple[CanonicalTrajectoryStep, ...]

    def __post_init__(self) -> None:
        for field_name in ("trajectory_id", "task_group", "outcome"):
            _require_nonblank(getattr(self, field_name), field_name)
        _require_finite(self.reward, "reward")
        if tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("trajectory step_index values must be contiguous from zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_group": self.task_group,
            "outcome": self.outcome,
            "reward": self.reward,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectoryEvidence":
        fields = ("trajectory_id", "task_group", "outcome", "reward", "steps")
        _require_fields(data, fields, "trajectory evidence")
        return cls(
            trajectory_id=_required_string(data["trajectory_id"], "trajectory_id"),
            task_group=_required_string(data["task_group"], "task_group"),
            outcome=_required_string(data["outcome"], "outcome"),
            reward=float(data["reward"]),
            steps=_tuple_from(data["steps"], CanonicalTrajectoryStep.from_dict, "steps"),
        )


@dataclass(frozen=True)
class RepositorySnapshotRef:
    repository_revision: str
    memory_project_key: str
    memory_ids: tuple[str, ...]
    snapshot_hash: str

    def __post_init__(self) -> None:
        _require_nonblank(self.repository_revision, "repository_revision")
        _require_nonblank(self.memory_project_key, "memory_project_key")
        if any(not item for item in self.memory_ids) or len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("memory_ids must contain unique non-empty IDs")
        require_sha256(self.snapshot_hash, field_name="snapshot_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_revision": self.repository_revision,
            "memory_project_key": self.memory_project_key,
            "memory_ids": list(self.memory_ids),
            "snapshot_hash": self.snapshot_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepositorySnapshotRef":
        fields = ("repository_revision", "memory_project_key", "memory_ids", "snapshot_hash")
        _require_fields(data, fields, "repository snapshot ref")
        return cls(
            repository_revision=_required_string(data["repository_revision"], "repository_revision"),
            memory_project_key=_required_string(data["memory_project_key"], "memory_project_key"),
            memory_ids=_string_tuple(data["memory_ids"], "memory_ids"),
            snapshot_hash=_required_string(data["snapshot_hash"], "snapshot_hash"),
        )


@dataclass(frozen=True)
class TaskOutcomeRef:
    task_id: str
    task_group: str
    reward: float
    resolved: bool
    evaluator_name: str
    evaluator_version: str
    evaluator_hash: str

    def __post_init__(self) -> None:
        for field_name in ("task_id", "task_group", "evaluator_name", "evaluator_version"):
            _require_nonblank(getattr(self, field_name), field_name)
        _require_finite(self.reward, "reward")
        require_sha256(self.evaluator_hash, field_name="evaluator_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_group": self.task_group,
            "reward": self.reward,
            "resolved": self.resolved,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "evaluator_hash": self.evaluator_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskOutcomeRef":
        fields = (
            "task_id", "task_group", "reward", "resolved",
            "evaluator_name", "evaluator_version", "evaluator_hash",
        )
        _require_fields(data, fields, "task outcome ref")
        if not isinstance(data["resolved"], bool):
            raise ValueError("resolved must be a boolean")
        for field_name in (
            "task_id",
            "task_group",
            "evaluator_name",
            "evaluator_version",
            "evaluator_hash",
        ):
            if not isinstance(data[field_name], str) or not data[field_name].strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        return cls(
            task_id=data["task_id"],
            task_group=data["task_group"],
            reward=float(data["reward"]),
            resolved=data["resolved"],
            evaluator_name=data["evaluator_name"],
            evaluator_version=data["evaluator_version"],
            evaluator_hash=data["evaluator_hash"],
        )


@dataclass(frozen=True)
class MemoryDiagnostic:
    memory_id: str
    tier: str
    memory_score: float
    gamma: float
    candidate_count: int
    selected_count: int
    last_used: str

    def __post_init__(self) -> None:
        _require_nonblank(self.memory_id, "memory_id")
        _require_nonblank(self.tier, "tier")
        _require_finite(self.memory_score, "memory_score")
        _require_finite(self.gamma, "gamma")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        _require_nonnegative_int(self.candidate_count, "candidate_count")
        _require_nonnegative_int(self.selected_count, "selected_count")
        if not isinstance(self.last_used, str):
            raise ValueError("last_used must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "tier": self.tier,
            "memory_score": self.memory_score,
            "gamma": self.gamma,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryDiagnostic":
        fields = ("memory_id", "tier", "memory_score", "gamma", "candidate_count", "selected_count", "last_used")
        _require_fields(data, fields, "memory diagnostic")
        return cls(
            memory_id=_required_string(data["memory_id"], "memory_id"),
            tier=_required_string(data["tier"], "tier"),
            memory_score=float(data["memory_score"]),
            gamma=float(data["gamma"]),
            candidate_count=_strict_int(data["candidate_count"], "candidate_count"),
            selected_count=_strict_int(data["selected_count"], "selected_count"),
            last_used=_string(data["last_used"], "last_used"),
        )


@dataclass(frozen=True)
class RedundancyDiagnostic:
    left_memory_id: str
    right_memory_id: str
    score: float

    def __post_init__(self) -> None:
        _require_nonblank(self.left_memory_id, "left_memory_id")
        _require_nonblank(self.right_memory_id, "right_memory_id")
        if self.left_memory_id == self.right_memory_id:
            raise ValueError("redundancy pair must contain two different memory IDs")
        _require_finite(self.score, "score")

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_memory_id": self.left_memory_id,
            "right_memory_id": self.right_memory_id,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RedundancyDiagnostic":
        fields = ("left_memory_id", "right_memory_id", "score")
        _require_fields(data, fields, "redundancy diagnostic")
        return cls(
            _required_string(data["left_memory_id"], "left_memory_id"),
            _required_string(data["right_memory_id"], "right_memory_id"),
            float(data["score"]),
        )


@dataclass(frozen=True)
class SelectionPublic:
    task: str
    candidates: tuple[CandidateSnapshotEntry, ...]
    token_budget: int

    def __post_init__(self) -> None:
        _require_nonblank(self.task, "task")
        _require_positive_int(self.token_budget, "token_budget")

    def to_dict(self) -> dict[str, Any]:
        return _view_dict("selection_public", {
            "task": self.task,
            "candidates": [item.to_dict() for item in self.candidates],
            "token_budget": self.token_budget,
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectionPublic":
        payload = _view_payload(data, "selection_public", ("task", "candidates", "token_budget"))
        return cls(
            _required_string(payload["task"], "task"),
            _tuple_from(payload["candidates"], CandidateSnapshotEntry.from_dict, "candidates"),
            _strict_int(payload["token_budget"], "token_budget"),
        )


@dataclass(frozen=True)
class SelectionHindsight:
    candidate_values: tuple[MemoryValueEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return _view_dict("selection_hindsight", {
            "candidate_values": [item.to_dict() for item in self.candidate_values],
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectionHindsight":
        payload = _view_payload(data, "selection_hindsight", ("candidate_values",))
        return cls(_tuple_from(payload["candidate_values"], MemoryValueEvidence.from_dict, "candidate_values"))


@dataclass(frozen=True)
class ActionPublic:
    task: str
    tools: tuple[CanonicalTool, ...]
    prefix_messages: tuple[CanonicalMessage, ...]

    def __post_init__(self) -> None:
        _require_nonblank(self.task, "task")

    def to_dict(self) -> dict[str, Any]:
        return _view_dict("action_public", {
            "task": self.task,
            "tools": [item.to_dict() for item in self.tools],
            "prefix_messages": [item.to_dict() for item in self.prefix_messages],
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionPublic":
        payload = _view_payload(data, "action_public", ("task", "tools", "prefix_messages"))
        return cls(
            _required_string(payload["task"], "task"),
            _tuple_from(payload["tools"], CanonicalTool.from_dict, "tools"),
            _tuple_from(payload["prefix_messages"], CanonicalMessage.from_dict, "prefix_messages"),
        )


@dataclass(frozen=True)
class ActionHindsight:
    positive_memories: tuple[MemoryValueEvidence, ...]
    successful_trajectory: TrajectoryEvidence

    def to_dict(self) -> dict[str, Any]:
        return _view_dict("action_hindsight", {
            "positive_memories": [item.to_dict() for item in self.positive_memories],
            "successful_trajectory": self.successful_trajectory.to_dict(),
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionHindsight":
        payload = _view_payload(data, "action_hindsight", ("positive_memories", "successful_trajectory"))
        return cls(
            _tuple_from(payload["positive_memories"], MemoryValueEvidence.from_dict, "positive_memories"),
            TrajectoryEvidence.from_dict(_mapping(payload["successful_trajectory"], "successful_trajectory")),
        )


@dataclass(frozen=True)
class WritingPublic:
    task: str
    trajectory: TrajectoryEvidence
    reward: float
    selected_memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonblank(self.task, "task")
        _require_finite(self.reward, "reward")
        if (
            any(not isinstance(item, str) or not item for item in self.selected_memory_ids)
            or len(set(self.selected_memory_ids)) != len(self.selected_memory_ids)
        ):
            raise ValueError("selected_memory_ids must contain unique non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return _view_dict("writing_public", {
            "task": self.task,
            "trajectory": self.trajectory.to_dict(),
            "reward": self.reward,
            "selected_memory_ids": list(self.selected_memory_ids),
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WritingPublic":
        payload = _view_payload(data, "writing_public", ("task", "trajectory", "reward", "selected_memory_ids"))
        return cls(
            _required_string(payload["task"], "task"),
            TrajectoryEvidence.from_dict(_mapping(payload["trajectory"], "trajectory")),
            float(payload["reward"]),
            _string_tuple(payload["selected_memory_ids"], "selected_memory_ids"),
        )


@dataclass(frozen=True)
class WritingHindsight:
    written_memory_values: tuple[MemoryValueEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return _view_dict("writing_hindsight", {
            "written_memory_values": [item.to_dict() for item in self.written_memory_values],
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WritingHindsight":
        payload = _view_payload(data, "writing_hindsight", ("written_memory_values",))
        return cls(_tuple_from(payload["written_memory_values"], MemoryValueEvidence.from_dict, "written_memory_values"))


@dataclass(frozen=True)
class MaintenancePublic:
    repository_snapshot: RepositorySnapshotRef
    history_window: tuple[TaskOutcomeRef, ...]
    tools: tuple[CanonicalTool, ...]

    def to_dict(self) -> dict[str, Any]:
        return _view_dict("maintenance_public", {
            "repository_snapshot": self.repository_snapshot.to_dict(),
            "history_window": [item.to_dict() for item in self.history_window],
            "tools": [item.to_dict() for item in self.tools],
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaintenancePublic":
        payload = _view_payload(data, "maintenance_public", ("repository_snapshot", "history_window", "tools"))
        return cls(
            RepositorySnapshotRef.from_dict(_mapping(payload["repository_snapshot"], "repository_snapshot")),
            _tuple_from(payload["history_window"], TaskOutcomeRef.from_dict, "history_window"),
            _tuple_from(payload["tools"], CanonicalTool.from_dict, "tools"),
        )


@dataclass(frozen=True)
class MaintenanceHindsight:
    memory_diagnostics: tuple[MemoryDiagnostic, ...]
    redundancy_diagnostics: tuple[RedundancyDiagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return _view_dict("maintenance_hindsight", {
            "memory_diagnostics": [item.to_dict() for item in self.memory_diagnostics],
            "redundancy_diagnostics": [item.to_dict() for item in self.redundancy_diagnostics],
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaintenanceHindsight":
        payload = _view_payload(data, "maintenance_hindsight", ("memory_diagnostics", "redundancy_diagnostics"))
        return cls(
            _tuple_from(payload["memory_diagnostics"], MemoryDiagnostic.from_dict, "memory_diagnostics"),
            _tuple_from(payload["redundancy_diagnostics"], RedundancyDiagnostic.from_dict, "redundancy_diagnostics"),
        )


RoleView = (
    SelectionPublic | SelectionHindsight | ActionPublic | ActionHindsight |
    WritingPublic | WritingHindsight | MaintenancePublic | MaintenanceHindsight
)


def role_view_hash(view: RoleView) -> str:
    return canonical_sha256(view.to_dict())


def _view_dict(view_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ROLE_VIEW_SCHEMA_VERSION,
        "view_type": view_type,
        **payload,
    }


def _view_payload(
    data: Mapping[str, Any],
    view_type: str,
    fields: tuple[str, ...],
) -> Mapping[str, Any]:
    _require_fields(data, ("schema_version", "view_type", *fields), view_type)
    if data["schema_version"] != ROLE_VIEW_SCHEMA_VERSION:
        raise ValueError(f"unsupported role view schema: {data['schema_version']!r}")
    if data["view_type"] != view_type:
        raise ValueError(f"expected view_type={view_type!r}")
    return data


def _require_fields(data: Mapping[str, Any], expected: tuple[str, ...], schema_name: str) -> None:
    missing = [name for name in expected if name not in data]
    unknown = sorted(set(data) - set(expected))
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(f"invalid {schema_name} fields ({'; '.join(parts)})")


def _tuple_from(value: Any, loader: Callable[[Mapping[str, Any]], _T], field_name: str) -> tuple[_T, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return tuple(loader(_mapping(item, field_name)) for item in value)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must contain JSON objects")
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be an array of strings")
    return tuple(value)


def _strict_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _required_string(value: Any, field_name: str) -> str:
    _require_nonblank(value, field_name)
    return value


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_nonnegative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_canonical_json(value: str, *, field_name: str) -> None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    canonical = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical != value:
        raise ValueError(f"{field_name} must already use canonical JSON serialization")


__all__ = [
    "ROLE_VIEW_SCHEMA_VERSION",
    "SELECTED_MEMORY_CONTEXT_HEADER",
    "ActionHindsight",
    "ActionPublic",
    "CandidateSnapshotEntry",
    "CanonicalMessage",
    "CanonicalTool",
    "CanonicalToolCall",
    "CanonicalTrajectoryStep",
    "MaintenanceHindsight",
    "MaintenancePublic",
    "MemoryDiagnostic",
    "MemoryValueEvidence",
    "RedundancyDiagnostic",
    "RepositorySnapshotRef",
    "SelectionHindsight",
    "SelectionPublic",
    "TaskOutcomeRef",
    "TrajectoryEvidence",
    "WritingHindsight",
    "WritingPublic",
    "role_view_hash",
    "without_selected_memory_context",
]
