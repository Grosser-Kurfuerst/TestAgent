"""Versioned evidence and learner dataset schemas for OPD collection rounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from my_agent.opd_ablation import (
    MAIN_ABLATION_RECIPE_HASH,
    ablation_excluded_roles,
    ablation_recipe_hash,
    ablation_uses_replay,
)
from my_agent.policy.identity import PolicyIdentity, canonical_sha256, require_sha256
from my_agent.training.role_views import (
    CandidateSnapshotEntry,
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCall,
    RedundancyDiagnostic,
    RepositorySnapshotRef,
    TaskOutcomeRef,
    TrajectoryEvidence,
)


OPD_EVIDENCE_SCHEMA_VERSION = "opd-round-evidence-v1"
OPD_MAINTENANCE_ATTEMPT_SCHEMA_VERSION = "opd-maintenance-attempt-v1"
OPD_LEARNER_SCHEMA_VERSION = "opd-learner-sample-v1"
OPD_EXPORT_MANIFEST_SCHEMA_VERSION = "opd-export-manifest-v2"
DATASET_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class ActionDecisionEvidence:
    decision_id: str
    turn_index: int
    step_index: int
    prefix_messages: tuple[CanonicalMessage, ...]
    tools: tuple[CanonicalTool, ...]
    expected_tool_calls: tuple[CanonicalToolCall, ...]
    observation_messages: tuple[CanonicalMessage, ...]

    def __post_init__(self) -> None:
        _require_nonblank(self.decision_id, "action decision_id")
        if self.turn_index < 0 or self.step_index < 0:
            raise ValueError("action decision indexes must be non-negative")
        if not self.prefix_messages:
            raise ValueError("action decision prefix must not be empty")
        if any(message.role != "tool" for message in self.observation_messages):
            raise ValueError("action observation_messages must contain only tool messages")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "turn_index": self.turn_index,
            "step_index": self.step_index,
            "prefix_messages": [item.to_dict() for item in self.prefix_messages],
            "tools": [item.to_dict() for item in self.tools],
            "expected_tool_calls": [item.to_dict() for item in self.expected_tool_calls],
            "observation_messages": [item.to_dict() for item in self.observation_messages],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionDecisionEvidence":
        _require_exact(
            data,
            {
                "decision_id", "turn_index", "step_index", "prefix_messages", "tools",
                "expected_tool_calls", "observation_messages",
            },
            "action decision evidence",
        )
        return cls(
            decision_id=_string(data["decision_id"], "decision_id"),
            turn_index=_int(data["turn_index"], "turn_index"),
            step_index=_int(data["step_index"], "step_index"),
            prefix_messages=_objects(
                data["prefix_messages"], CanonicalMessage.from_dict, "prefix_messages"
            ),
            tools=_objects(data["tools"], CanonicalTool.from_dict, "tools"),
            expected_tool_calls=_objects(
                data["expected_tool_calls"], CanonicalToolCall.from_dict, "expected_tool_calls"
            ),
            observation_messages=_objects(
                data["observation_messages"], CanonicalMessage.from_dict, "observation_messages"
            ),
        )


@dataclass(frozen=True)
class TaskEvidence:
    collection_round: int
    task_ordinal: int
    split: str
    task: str
    task_id: str
    task_group: str
    trajectory_id: str
    stream_id: str
    memory_project_key: str
    policy_identity: PolicyIdentity
    repository_snapshot_hash: str
    candidate_snapshot_hash: str
    candidates: tuple[CandidateSnapshotEntry, ...]
    selected_memory_ids: tuple[str, ...]
    trajectory: TrajectoryEvidence
    written_memory_ids: tuple[str, ...]
    selection_decision_id: str
    action_decisions: tuple[ActionDecisionEvidence, ...]
    writing_decision_id: str | None
    selection_token_budget: int
    schema_version: str = OPD_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_round(self.collection_round, self.task_ordinal)
        _require_split(self.split)
        for field_name in (
            "task", "task_id", "task_group", "trajectory_id", "stream_id",
            "memory_project_key", "selection_decision_id",
        ):
            _require_nonblank(getattr(self, field_name), field_name)
        if not isinstance(self.policy_identity, PolicyIdentity):
            raise ValueError("task evidence requires PolicyIdentity")
        require_sha256(self.repository_snapshot_hash, field_name="repository_snapshot_hash")
        require_sha256(self.candidate_snapshot_hash, field_name="candidate_snapshot_hash")
        candidate_ids = tuple(item.memory_id for item in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("task evidence candidates must have unique memory IDs")
        if self.candidate_snapshot_hash != canonical_sha256(
            [item.to_dict() for item in self.candidates]
        ):
            raise ValueError("task evidence candidate_snapshot_hash mismatch")
        _require_unique_strings(self.selected_memory_ids, "selected_memory_ids")
        if any(memory_id not in set(candidate_ids) for memory_id in self.selected_memory_ids):
            raise ValueError("selected memory must reference the candidate snapshot")
        _require_unique_strings(self.written_memory_ids, "written_memory_ids")
        action_ids = tuple(item.decision_id for item in self.action_decisions)
        _require_unique_strings(action_ids, "action_decision_ids", allow_empty=False)
        action_indexes = tuple(
            (item.turn_index, item.step_index) for item in self.action_decisions
        )
        if (
            action_indexes != tuple(sorted(action_indexes))
            or len(set(action_indexes)) != len(action_indexes)
        ):
            raise ValueError("action decisions must have unique ordered turn/step indexes")
        if self.writing_decision_id is not None:
            _require_nonblank(self.writing_decision_id, "writing_decision_id")
        if self.trajectory.trajectory_id != self.trajectory_id:
            raise ValueError("task evidence trajectory identity mismatch")
        if self.trajectory.task_group != self.task_group:
            raise ValueError("task evidence trajectory group mismatch")
        if self.selection_token_budget < 1:
            raise ValueError("selection_token_budget must be positive")

    @property
    def evidence_id(self) -> str:
        return canonical_sha256(self._payload())

    @property
    def source_decision_ids(self) -> tuple[str, ...]:
        values = (
            self.selection_decision_id,
            *(item.decision_id for item in self.action_decisions),
        )
        if self.writing_decision_id is not None:
            values = (*values, self.writing_decision_id)
        return values

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_round": self.collection_round,
            "task_ordinal": self.task_ordinal,
            "split": self.split,
            "task": self.task,
            "task_id": self.task_id,
            "task_group": self.task_group,
            "trajectory_id": self.trajectory_id,
            "stream_id": self.stream_id,
            "memory_project_key": self.memory_project_key,
            "policy_identity": self.policy_identity.to_dict(),
            "policy_identity_hash": self.policy_identity.identity_hash,
            "repository_snapshot_hash": self.repository_snapshot_hash,
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_memory_ids": list(self.selected_memory_ids),
            "trajectory": self.trajectory.to_dict(),
            "written_memory_ids": list(self.written_memory_ids),
            "selection_decision_id": self.selection_decision_id,
            "action_decisions": [item.to_dict() for item in self.action_decisions],
            "writing_decision_id": self.writing_decision_id,
            "selection_token_budget": self.selection_token_budget,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, **self._payload()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskEvidence":
        expected = {
            "evidence_id", "schema_version", "collection_round", "task_ordinal", "split",
            "task", "task_id", "task_group", "trajectory_id", "stream_id",
            "memory_project_key", "policy_identity", "policy_identity_hash",
            "repository_snapshot_hash", "candidate_snapshot_hash", "candidates",
            "selected_memory_ids", "trajectory", "written_memory_ids",
            "selection_decision_id", "action_decisions", "writing_decision_id",
            "selection_token_budget",
        }
        _require_exact(data, expected, "task evidence")
        identity = _identity(data)
        writing_id = data["writing_decision_id"]
        if writing_id is not None and not isinstance(writing_id, str):
            raise ValueError("writing_decision_id must be a string or null")
        record = cls(
            collection_round=_int(data["collection_round"], "collection_round"),
            task_ordinal=_int(data["task_ordinal"], "task_ordinal"),
            split=_string(data["split"], "split"),
            task=_string(data["task"], "task"),
            task_id=_string(data["task_id"], "task_id"),
            task_group=_string(data["task_group"], "task_group"),
            trajectory_id=_string(data["trajectory_id"], "trajectory_id"),
            stream_id=_string(data["stream_id"], "stream_id"),
            memory_project_key=_string(data["memory_project_key"], "memory_project_key"),
            policy_identity=identity,
            repository_snapshot_hash=_string(
                data["repository_snapshot_hash"], "repository_snapshot_hash"
            ),
            candidate_snapshot_hash=_string(
                data["candidate_snapshot_hash"], "candidate_snapshot_hash"
            ),
            candidates=_objects(data["candidates"], CandidateSnapshotEntry.from_dict, "candidates"),
            selected_memory_ids=_strings(data["selected_memory_ids"], "selected_memory_ids"),
            trajectory=TrajectoryEvidence.from_dict(_mapping(data["trajectory"], "trajectory")),
            written_memory_ids=_strings(data["written_memory_ids"], "written_memory_ids"),
            selection_decision_id=_string(
                data["selection_decision_id"], "selection_decision_id"
            ),
            action_decisions=_objects(
                data["action_decisions"], ActionDecisionEvidence.from_dict, "action_decisions"
            ),
            writing_decision_id=writing_id,
            selection_token_budget=_int(
                data["selection_token_budget"], "selection_token_budget"
            ),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
        if data["evidence_id"] != record.evidence_id:
            raise ValueError("task evidence_id mismatch")
        return record


@dataclass(frozen=True)
class TaskOutcomeEvidence:
    collection_round: int
    task_ordinal: int
    trajectory_id: str
    stream_id: str
    memory_project_key: str
    outcome: TaskOutcomeRef
    task_valid: bool
    outcome_finalized: bool
    schema_version: str = OPD_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_round(self.collection_round, self.task_ordinal)
        for field_name in ("trajectory_id", "stream_id", "memory_project_key"):
            _require_nonblank(getattr(self, field_name), field_name)
        if not isinstance(self.task_valid, bool) or not isinstance(self.outcome_finalized, bool):
            raise ValueError("outcome evidence booleans are invalid")

    @property
    def outcome_id(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_round": self.collection_round,
            "task_ordinal": self.task_ordinal,
            "trajectory_id": self.trajectory_id,
            "stream_id": self.stream_id,
            "memory_project_key": self.memory_project_key,
            "outcome": self.outcome.to_dict(),
            "task_valid": self.task_valid,
            "outcome_finalized": self.outcome_finalized,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"outcome_id": self.outcome_id, **self._payload()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskOutcomeEvidence":
        expected = {
            "outcome_id", "schema_version", "collection_round", "task_ordinal",
            "trajectory_id", "stream_id", "memory_project_key", "outcome",
            "task_valid", "outcome_finalized",
        }
        _require_exact(data, expected, "task outcome evidence")
        record = cls(
            collection_round=_int(data["collection_round"], "collection_round"),
            task_ordinal=_int(data["task_ordinal"], "task_ordinal"),
            trajectory_id=_string(data["trajectory_id"], "trajectory_id"),
            stream_id=_string(data["stream_id"], "stream_id"),
            memory_project_key=_string(data["memory_project_key"], "memory_project_key"),
            outcome=TaskOutcomeRef.from_dict(_mapping(data["outcome"], "outcome")),
            task_valid=_bool(data["task_valid"], "task_valid"),
            outcome_finalized=_bool(data["outcome_finalized"], "outcome_finalized"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
        if data["outcome_id"] != record.outcome_id:
            raise ValueError("outcome_id mismatch")
        return record


@dataclass(frozen=True)
class RepositoryMemoryEvidence:
    memory_id: str
    tier: str
    content: str
    candidate_count: int
    selected_count: int
    last_used: str

    def __post_init__(self) -> None:
        _require_nonblank(self.memory_id, "memory_id")
        _require_nonblank(self.tier, "tier")
        if not isinstance(self.content, str) or not isinstance(self.last_used, str):
            raise ValueError("repository memory content/last_used must be strings")
        if self.candidate_count < 0 or self.selected_count < 0:
            raise ValueError("repository memory usage counts must be non-negative")
        if self.selected_count > self.candidate_count:
            raise ValueError("selected_count cannot exceed candidate_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "tier": self.tier,
            "content": self.content,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepositoryMemoryEvidence":
        expected = {
            "memory_id", "tier", "content", "candidate_count", "selected_count", "last_used",
        }
        _require_exact(data, expected, "repository memory evidence")
        return cls(
            memory_id=_string(data["memory_id"], "memory_id"),
            tier=_string(data["tier"], "tier"),
            content=_string(data["content"], "content"),
            candidate_count=_int(data["candidate_count"], "candidate_count"),
            selected_count=_int(data["selected_count"], "selected_count"),
            last_used=_string(data["last_used"], "last_used"),
        )


@dataclass(frozen=True)
class RepositoryEvidence:
    collection_round: int
    event_ordinal: int
    stream_id: str
    previous_revision: str | None
    snapshot: RepositorySnapshotRef
    memories: tuple[RepositoryMemoryEvidence, ...]
    schema_version: str = OPD_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        if self.collection_round < 0:
            raise ValueError("collection_round must be non-negative")
        if self.event_ordinal < 0:
            raise ValueError("repository event_ordinal must be non-negative")
        _require_nonblank(self.stream_id, "stream_id")
        if self.previous_revision is not None:
            _require_nonblank(self.previous_revision, "previous_revision")
        memory_ids = tuple(item.memory_id for item in self.memories)
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("repository evidence memory IDs must be unique")
        if set(memory_ids) != set(self.snapshot.memory_ids):
            raise ValueError("repository evidence memories do not match snapshot IDs")

    @property
    def repository_event_id(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_round": self.collection_round,
            "event_ordinal": self.event_ordinal,
            "stream_id": self.stream_id,
            "previous_revision": self.previous_revision,
            "snapshot": self.snapshot.to_dict(),
            "memories": [item.to_dict() for item in self.memories],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"repository_event_id": self.repository_event_id, **self._payload()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepositoryEvidence":
        expected = {
            "repository_event_id", "schema_version", "collection_round", "event_ordinal",
            "stream_id", "previous_revision", "snapshot", "memories",
        }
        _require_exact(data, expected, "repository evidence")
        record = cls(
            collection_round=_int(data["collection_round"], "collection_round"),
            event_ordinal=_int(data["event_ordinal"], "event_ordinal"),
            stream_id=_string(data["stream_id"], "stream_id"),
            previous_revision=(
                None
                if data["previous_revision"] is None
                else _string(data["previous_revision"], "previous_revision")
            ),
            snapshot=RepositorySnapshotRef.from_dict(_mapping(data["snapshot"], "snapshot")),
            memories=_objects(
                data["memories"], RepositoryMemoryEvidence.from_dict, "memories"
            ),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
        if data["repository_event_id"] != record.repository_event_id:
            raise ValueError("repository_event_id mismatch")
        return record


@dataclass(frozen=True)
class MaintenanceAttemptEvidence:
    collection_round: int
    split: str
    cadence_id: str
    attempt_index: int
    status: str
    task_group: str
    stream_id: str
    memory_project_key: str
    repository_snapshot_hash: str
    as_of_task_ordinal: int
    outcome_ids: tuple[str, ...]
    redundancy_diagnostics: tuple[RedundancyDiagnostic, ...]
    decision_ids: tuple[str, ...] = ()
    reason: str = ""
    schema_version: str = OPD_MAINTENANCE_ATTEMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPD_MAINTENANCE_ATTEMPT_SCHEMA_VERSION:
            raise ValueError("unsupported maintenance attempt schema")
        if self.collection_round < 0 or self.attempt_index < 1 or self.as_of_task_ordinal < 1:
            raise ValueError("maintenance attempt indexes must be positive")
        _require_split(self.split)
        for field_name in ("cadence_id", "task_group", "stream_id", "memory_project_key"):
            _require_nonblank(getattr(self, field_name), field_name)
        require_sha256(self.cadence_id, field_name="cadence_id")
        require_sha256(self.repository_snapshot_hash, field_name="repository_snapshot_hash")
        if self.status not in {
            "started", "committed", "noop", "aborted", "stale", "abandoned",
        }:
            raise ValueError("unsupported maintenance attempt status")
        _require_unique_strings(self.outcome_ids, "outcome_ids")
        _require_unique_strings(self.decision_ids, "decision_ids")
        if self.status == "started" and (self.decision_ids or self.reason):
            raise ValueError("started maintenance attempt cannot be terminal")

    @property
    def attempt_id(self) -> str:
        return canonical_sha256({
            "schema_version": self.schema_version,
            "cadence_id": self.cadence_id,
            "attempt_index": self.attempt_index,
        })

    @property
    def attempt_event_id(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_round": self.collection_round,
            "split": self.split,
            "cadence_id": self.cadence_id,
            "attempt_id": self.attempt_id,
            "attempt_index": self.attempt_index,
            "status": self.status,
            "task_group": self.task_group,
            "stream_id": self.stream_id,
            "memory_project_key": self.memory_project_key,
            "repository_snapshot_hash": self.repository_snapshot_hash,
            "as_of_task_ordinal": self.as_of_task_ordinal,
            "outcome_ids": list(self.outcome_ids),
            "redundancy_diagnostics": [item.to_dict() for item in self.redundancy_diagnostics],
            "decision_ids": list(self.decision_ids),
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"attempt_event_id": self.attempt_event_id, **self._payload()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaintenanceAttemptEvidence":
        expected = {
            "attempt_event_id", "schema_version", "collection_round", "split",
            "cadence_id", "attempt_id", "attempt_index", "status", "task_group",
            "stream_id", "memory_project_key", "repository_snapshot_hash",
            "as_of_task_ordinal", "outcome_ids", "redundancy_diagnostics",
            "decision_ids", "reason",
        }
        _require_exact(data, expected, "maintenance attempt evidence")
        record = cls(
            collection_round=_int(data["collection_round"], "collection_round"),
            split=_string(data["split"], "split"),
            cadence_id=_string(data["cadence_id"], "cadence_id"),
            attempt_index=_int(data["attempt_index"], "attempt_index"),
            status=_string(data["status"], "status"),
            task_group=_string(data["task_group"], "task_group"),
            stream_id=_string(data["stream_id"], "stream_id"),
            memory_project_key=_string(data["memory_project_key"], "memory_project_key"),
            repository_snapshot_hash=_string(
                data["repository_snapshot_hash"], "repository_snapshot_hash"
            ),
            as_of_task_ordinal=_int(data["as_of_task_ordinal"], "as_of_task_ordinal"),
            outcome_ids=_strings(data["outcome_ids"], "outcome_ids"),
            redundancy_diagnostics=_objects(
                data["redundancy_diagnostics"],
                RedundancyDiagnostic.from_dict,
                "redundancy_diagnostics",
            ),
            decision_ids=_strings(data["decision_ids"], "decision_ids"),
            reason=_string(data["reason"], "reason"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
        if data["attempt_id"] != record.attempt_id:
            raise ValueError("maintenance attempt_id mismatch")
        if data["attempt_event_id"] != record.attempt_event_id:
            raise ValueError("maintenance attempt_event_id mismatch")
        return record


@dataclass(frozen=True)
class MaintenanceEvidence:
    collection_round: int
    as_of_task_ordinal: int
    split: str
    cadence_id: str
    attempt_id: str
    task_group: str
    stream_id: str
    memory_project_key: str
    policy_identity: PolicyIdentity
    repository_snapshot_hash: str
    outcome_ids: tuple[str, ...]
    tools: tuple[CanonicalTool, ...]
    redundancy_diagnostics: tuple[RedundancyDiagnostic, ...]
    decision_ids: tuple[str, ...]
    schema_version: str = OPD_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        if self.collection_round < 0:
            raise ValueError("collection_round must be non-negative")
        if self.as_of_task_ordinal < 1:
            raise ValueError("maintenance as_of_task_ordinal must be positive")
        _require_split(self.split)
        for field_name in (
            "cadence_id", "attempt_id", "task_group", "stream_id", "memory_project_key",
        ):
            _require_nonblank(getattr(self, field_name), field_name)
        require_sha256(self.cadence_id, field_name="cadence_id")
        require_sha256(self.attempt_id, field_name="attempt_id")
        require_sha256(self.repository_snapshot_hash, field_name="repository_snapshot_hash")
        _require_unique_strings(self.outcome_ids, "outcome_ids")
        _require_unique_strings(self.decision_ids, "decision_ids", allow_empty=False)

    @property
    def evidence_id(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_round": self.collection_round,
            "as_of_task_ordinal": self.as_of_task_ordinal,
            "split": self.split,
            "cadence_id": self.cadence_id,
            "attempt_id": self.attempt_id,
            "task_group": self.task_group,
            "stream_id": self.stream_id,
            "memory_project_key": self.memory_project_key,
            "policy_identity": self.policy_identity.to_dict(),
            "policy_identity_hash": self.policy_identity.identity_hash,
            "repository_snapshot_hash": self.repository_snapshot_hash,
            "outcome_ids": list(self.outcome_ids),
            "tools": [item.to_dict() for item in self.tools],
            "redundancy_diagnostics": [item.to_dict() for item in self.redundancy_diagnostics],
            "decision_ids": list(self.decision_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, **self._payload()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaintenanceEvidence":
        expected = {
            "evidence_id", "schema_version", "collection_round", "as_of_task_ordinal",
            "split", "cadence_id", "attempt_id", "task_group", "stream_id",
            "memory_project_key", "policy_identity",
            "policy_identity_hash", "repository_snapshot_hash", "outcome_ids", "tools",
            "redundancy_diagnostics", "decision_ids",
        }
        _require_exact(data, expected, "maintenance evidence")
        record = cls(
            collection_round=_int(data["collection_round"], "collection_round"),
            as_of_task_ordinal=_int(data["as_of_task_ordinal"], "as_of_task_ordinal"),
            split=_string(data["split"], "split"),
            cadence_id=_string(data["cadence_id"], "cadence_id"),
            attempt_id=_string(data["attempt_id"], "attempt_id"),
            task_group=_string(data["task_group"], "task_group"),
            stream_id=_string(data["stream_id"], "stream_id"),
            memory_project_key=_string(data["memory_project_key"], "memory_project_key"),
            policy_identity=_identity(data),
            repository_snapshot_hash=_string(
                data["repository_snapshot_hash"], "repository_snapshot_hash"
            ),
            outcome_ids=_strings(data["outcome_ids"], "outcome_ids"),
            tools=_objects(data["tools"], CanonicalTool.from_dict, "tools"),
            redundancy_diagnostics=_objects(
                data["redundancy_diagnostics"],
                RedundancyDiagnostic.from_dict,
                "redundancy_diagnostics",
            ),
            decision_ids=_strings(data["decision_ids"], "decision_ids"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
        if data["evidence_id"] != record.evidence_id:
            raise ValueError("maintenance evidence_id mismatch")
        return record


@dataclass(frozen=True)
class RuntimeExclusionEvidence:
    collection_round: int
    task_ordinal: int
    split: str
    role: str
    reason: str
    task_id: str
    trajectory_id: str
    stream_id: str
    memory_project_key: str
    decision_ids: tuple[str, ...]
    schema_version: str = OPD_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_round(self.collection_round, self.task_ordinal)
        _require_split(self.split)
        if self.role not in {"selection", "action", "writing", "maintenance"}:
            raise ValueError("unsupported runtime exclusion role")
        for field_name in (
            "reason", "task_id", "trajectory_id", "stream_id", "memory_project_key"
        ):
            _require_nonblank(getattr(self, field_name), field_name)
        _require_unique_strings(self.decision_ids, "decision_ids")

    @property
    def exclusion_id(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_round": self.collection_round,
            "task_ordinal": self.task_ordinal,
            "split": self.split,
            "role": self.role,
            "reason": self.reason,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "stream_id": self.stream_id,
            "memory_project_key": self.memory_project_key,
            "decision_ids": list(self.decision_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"exclusion_id": self.exclusion_id, **self._payload()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeExclusionEvidence":
        expected = {
            "exclusion_id", "schema_version", "collection_round", "task_ordinal",
            "split", "role", "reason", "task_id", "trajectory_id", "stream_id",
            "memory_project_key", "decision_ids",
        }
        _require_exact(data, expected, "runtime exclusion evidence")
        record = cls(
            collection_round=_int(data["collection_round"], "collection_round"),
            task_ordinal=_int(data["task_ordinal"], "task_ordinal"),
            split=_string(data["split"], "split"),
            role=_string(data["role"], "role"),
            reason=_string(data["reason"], "reason"),
            task_id=_string(data["task_id"], "task_id"),
            trajectory_id=_string(data["trajectory_id"], "trajectory_id"),
            stream_id=_string(data["stream_id"], "stream_id"),
            memory_project_key=_string(data["memory_project_key"], "memory_project_key"),
            decision_ids=_strings(data["decision_ids"], "decision_ids"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
        if data["exclusion_id"] != record.exclusion_id:
            raise ValueError("runtime exclusion_id mismatch")
        return record


@dataclass(frozen=True)
class LearnerSample:
    role: str
    collection_round: int
    split: str
    task_group: str
    stream_id: str
    memory_project_key: str
    source_evidence_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    policy_identity: PolicyIdentity
    student_public_view: Mapping[str, Any]
    teacher_hindsight_view: Mapping[str, Any]
    canonical_student_messages: tuple[CanonicalMessage, ...]
    canonical_teacher_messages: tuple[CanonicalMessage, ...]
    canonical_tools: tuple[CanonicalTool, ...]
    student_raw_completion: str
    student_prompt_token_ids: tuple[int, ...]
    student_completion_token_ids: tuple[int, ...]
    assistant_loss_mask: tuple[int, ...]
    public_prefix_hash: str
    student_prompt_hash: str
    teacher_prompt_hash: str
    schema_version: str = OPD_LEARNER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPD_LEARNER_SCHEMA_VERSION:
            raise ValueError("unsupported learner sample schema")
        if self.role not in {"selection", "action", "writing", "maintenance"}:
            raise ValueError("unsupported learner role")
        if self.collection_round < 0:
            raise ValueError("collection_round must be non-negative")
        _require_split(self.split)
        for field_name in ("task_group", "stream_id", "memory_project_key"):
            _require_nonblank(getattr(self, field_name), field_name)
        _require_unique_strings(self.source_evidence_ids, "source_evidence_ids", allow_empty=False)
        _require_unique_strings(self.evidence_refs, "evidence_refs", allow_empty=False)
        if not isinstance(self.student_public_view, Mapping):
            raise ValueError("student_public_view must be an object")
        if not isinstance(self.teacher_hindsight_view, Mapping):
            raise ValueError("teacher_hindsight_view must be an object")
        if tuple(self.canonical_teacher_messages[: len(self.canonical_student_messages)]) != (
            self.canonical_student_messages
        ):
            raise ValueError("teacher messages must preserve the exact student public prefix")
        if len(self.canonical_teacher_messages) != len(self.canonical_student_messages) + 1:
            raise ValueError("teacher messages must add exactly one hindsight message")
        if len(self.assistant_loss_mask) != len(self.student_completion_token_ids):
            raise ValueError("assistant_loss_mask must align with completion token IDs")
        if any(value not in (0, 1) for value in self.assistant_loss_mask):
            raise ValueError("assistant_loss_mask must be binary")
        for field_name in ("public_prefix_hash", "student_prompt_hash", "teacher_prompt_hash"):
            require_sha256(getattr(self, field_name), field_name=field_name)
        expected_prefix_hash = canonical_sha256({
            "messages": [item.to_dict() for item in self.canonical_student_messages],
            "tools": [item.to_dict() for item in self.canonical_tools],
        })
        if self.public_prefix_hash != expected_prefix_hash:
            raise ValueError("public_prefix_hash mismatch")

    @property
    def sample_id(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "collection_round": self.collection_round,
            "split": self.split,
            "task_group": self.task_group,
            "stream_id": self.stream_id,
            "memory_project_key": self.memory_project_key,
            "source_evidence_ids": list(self.source_evidence_ids),
            "evidence_refs": list(self.evidence_refs),
            "policy_identity": self.policy_identity.to_dict(),
            "policy_identity_hash": self.policy_identity.identity_hash,
            "student_public_view": dict(self.student_public_view),
            "teacher_hindsight_view": dict(self.teacher_hindsight_view),
            "canonical_student_messages": [item.to_dict() for item in self.canonical_student_messages],
            "canonical_teacher_messages": [item.to_dict() for item in self.canonical_teacher_messages],
            "canonical_tools": [item.to_dict() for item in self.canonical_tools],
            "student_raw_completion": self.student_raw_completion,
            "student_prompt_token_ids": list(self.student_prompt_token_ids),
            "student_completion_token_ids": list(self.student_completion_token_ids),
            "assistant_loss_mask": list(self.assistant_loss_mask),
            "public_prefix_hash": self.public_prefix_hash,
            "student_prompt_hash": self.student_prompt_hash,
            "teacher_prompt_hash": self.teacher_prompt_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, **self._payload()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearnerSample":
        expected = {
            "sample_id", "schema_version", "role", "collection_round", "split",
            "task_group", "stream_id", "memory_project_key", "source_evidence_ids",
            "evidence_refs", "policy_identity", "policy_identity_hash", "student_public_view",
            "teacher_hindsight_view", "canonical_student_messages",
            "canonical_teacher_messages", "canonical_tools", "student_raw_completion",
            "student_prompt_token_ids", "student_completion_token_ids", "assistant_loss_mask",
            "public_prefix_hash", "student_prompt_hash", "teacher_prompt_hash",
        }
        _require_exact(data, expected, "learner sample")
        record = cls(
            role=_string(data["role"], "role"),
            collection_round=_int(data["collection_round"], "collection_round"),
            split=_string(data["split"], "split"),
            task_group=_string(data["task_group"], "task_group"),
            stream_id=_string(data["stream_id"], "stream_id"),
            memory_project_key=_string(data["memory_project_key"], "memory_project_key"),
            source_evidence_ids=_strings(data["source_evidence_ids"], "source_evidence_ids"),
            evidence_refs=_strings(data["evidence_refs"], "evidence_refs"),
            policy_identity=_identity(data),
            student_public_view=dict(_mapping(data["student_public_view"], "student_public_view")),
            teacher_hindsight_view=dict(
                _mapping(data["teacher_hindsight_view"], "teacher_hindsight_view")
            ),
            canonical_student_messages=_objects(
                data["canonical_student_messages"],
                CanonicalMessage.from_dict,
                "canonical_student_messages",
            ),
            canonical_teacher_messages=_objects(
                data["canonical_teacher_messages"],
                CanonicalMessage.from_dict,
                "canonical_teacher_messages",
            ),
            canonical_tools=_objects(data["canonical_tools"], CanonicalTool.from_dict, "canonical_tools"),
            student_raw_completion=_string(
                data["student_raw_completion"], "student_raw_completion"
            ),
            student_prompt_token_ids=_ints(
                data["student_prompt_token_ids"], "student_prompt_token_ids"
            ),
            student_completion_token_ids=_ints(
                data["student_completion_token_ids"], "student_completion_token_ids"
            ),
            assistant_loss_mask=_ints(data["assistant_loss_mask"], "assistant_loss_mask"),
            public_prefix_hash=_string(data["public_prefix_hash"], "public_prefix_hash"),
            student_prompt_hash=_string(data["student_prompt_hash"], "student_prompt_hash"),
            teacher_prompt_hash=_string(data["teacher_prompt_hash"], "teacher_prompt_hash"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
        if data["sample_id"] != record.sample_id:
            raise ValueError("learner sample_id mismatch")
        return record


@dataclass(frozen=True)
class ExportManifest:
    collection_round: int
    trainer_initialization_identity: PolicyIdentity
    learner_dataset_hash: str
    sample_count: int
    role_counts: Mapping[str, int]
    split_counts: Mapping[str, int]
    task_group_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    source_hashes: Mapping[str, str]
    writing_score_decisions: tuple[Mapping[str, Any], ...]
    exclusions: tuple[Mapping[str, Any], ...]
    ablation: str = ""
    ablation_recipe_hash: str = MAIN_ABLATION_RECIPE_HASH
    sample_policy_identity_hashes: tuple[str, ...] = ()
    sample_collection_rounds: tuple[int, ...] = ()
    current_checkpoint_only: bool = True
    replay_enabled: bool = False
    schema_version: str = OPD_EXPORT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPD_EXPORT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported export manifest schema")
        if self.collection_round < 0 or self.sample_count < 0:
            raise ValueError("export manifest counts are invalid")
        normalized_ablation = self.ablation.strip().lower()
        if normalized_ablation != self.ablation:
            raise ValueError("export manifest ablation must be normalized")
        if self.ablation_recipe_hash != ablation_recipe_hash(normalized_ablation):
            raise ValueError("export manifest ablation recipe hash mismatch")
        if not self.sample_policy_identity_hashes:
            object.__setattr__(
                self,
                "sample_policy_identity_hashes",
                (self.trainer_initialization_identity.identity_hash,),
            )
        if not self.sample_collection_rounds:
            object.__setattr__(self, "sample_collection_rounds", (self.collection_round,))
        if len(set(self.sample_policy_identity_hashes)) != len(
            self.sample_policy_identity_hashes
        ):
            raise ValueError("sample policy identity hashes must be unique")
        for identity_hash in self.sample_policy_identity_hashes:
            require_sha256(identity_hash, field_name="sample policy identity hash")
        if len(set(self.sample_collection_rounds)) != len(self.sample_collection_rounds):
            raise ValueError("sample collection rounds must be unique")
        if any(round_index < 0 for round_index in self.sample_collection_rounds):
            raise ValueError("sample collection rounds must be non-negative")
        if ablation_uses_replay(normalized_ablation):
            if self.current_checkpoint_only or not self.replay_enabled:
                raise ValueError("replay ablation manifest must declare off-policy replay")
            if set(self.sample_collection_rounds) != {0, 1}:
                raise ValueError("replay D0+D1 manifest must bind collection rounds 0 and 1")
            if (
                len(self.sample_policy_identity_hashes) < 2
                or self.trainer_initialization_identity.identity_hash
                not in self.sample_policy_identity_hashes
            ):
                raise ValueError("replay D0+D1 manifest must bind old and current policies")
            required_sources = {
                "d0_learner_dataset", "d0_export_manifest",
                "d1_learner_dataset", "d1_export_manifest",
            }
            if not required_sources.issubset(self.source_hashes):
                raise ValueError("replay D0+D1 manifest lacks source dataset hashes")
        elif not self.current_checkpoint_only or self.replay_enabled:
            raise ValueError("formal export manifest must be strict current-checkpoint only")
        elif self.sample_policy_identity_hashes != (
            self.trainer_initialization_identity.identity_hash,
        ) or self.sample_collection_rounds != (self.collection_round,):
            raise ValueError("current-checkpoint manifest contains mixed sample provenance")
        require_sha256(self.learner_dataset_hash, field_name="learner_dataset_hash")
        for source_hash in self.source_hashes.values():
            require_sha256(source_hash, field_name="source_hash")
        if sum(self.role_counts.values()) != self.sample_count:
            raise ValueError("role_counts do not match sample_count")
        if ablation_excluded_roles(normalized_ablation).intersection(self.role_counts):
            raise ValueError("ablation dataset contains an explicitly disabled role")
        if normalized_ablation in {"no_attribution", "similarity_only"}:
            required_sources = {"attribution_input", "attribution_effective"}
            if not required_sources.issubset(self.source_hashes):
                raise ValueError("attribution ablation lacks input/effective evidence hashes")
            if self.source_hashes["attribution_input"] == self.source_hashes[
                "attribution_effective"
            ]:
                raise ValueError("attribution ablation did not change effective evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_round": self.collection_round,
            "trainer_initialization_identity": self.trainer_initialization_identity.to_dict(),
            "trainer_initialization_identity_hash": (
                self.trainer_initialization_identity.identity_hash
            ),
            "learner_dataset_hash": self.learner_dataset_hash,
            "sample_count": self.sample_count,
            "role_counts": dict(sorted(self.role_counts.items())),
            "split_counts": dict(sorted(self.split_counts.items())),
            "task_group_counts": dict(sorted(self.task_group_counts.items())),
            "outcome_counts": dict(sorted(self.outcome_counts.items())),
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "writing_score_decisions": [dict(item) for item in self.writing_score_decisions],
            "exclusions": [dict(item) for item in self.exclusions],
            "ablation": self.ablation,
            "ablation_recipe_hash": self.ablation_recipe_hash,
            "sample_policy_identity_hashes": list(self.sample_policy_identity_hashes),
            "sample_collection_rounds": list(self.sample_collection_rounds),
            "current_checkpoint_only": self.current_checkpoint_only,
            "replay_enabled": self.replay_enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExportManifest":
        expected = {
            "schema_version", "collection_round", "trainer_initialization_identity",
            "trainer_initialization_identity_hash", "learner_dataset_hash", "sample_count",
            "role_counts", "split_counts", "task_group_counts", "outcome_counts",
            "source_hashes", "writing_score_decisions", "exclusions",
            "ablation", "ablation_recipe_hash", "sample_policy_identity_hashes",
            "sample_collection_rounds",
            "current_checkpoint_only", "replay_enabled",
        }
        _require_exact(data, expected, "export manifest")
        identity = PolicyIdentity.from_dict(_mapping(
            data["trainer_initialization_identity"],
            "trainer_initialization_identity",
        ))
        if data["trainer_initialization_identity_hash"] != identity.identity_hash:
            raise ValueError("trainer initialization identity hash mismatch")
        writing = _mapping_tuple(data["writing_score_decisions"], "writing_score_decisions")
        exclusions = _mapping_tuple(data["exclusions"], "exclusions")
        return cls(
            collection_round=_int(data["collection_round"], "collection_round"),
            trainer_initialization_identity=identity,
            learner_dataset_hash=_string(data["learner_dataset_hash"], "learner_dataset_hash"),
            sample_count=_int(data["sample_count"], "sample_count"),
            role_counts=_count_mapping(data["role_counts"], "role_counts"),
            split_counts=_count_mapping(data["split_counts"], "split_counts"),
            task_group_counts=_count_mapping(data["task_group_counts"], "task_group_counts"),
            outcome_counts=_count_mapping(data["outcome_counts"], "outcome_counts"),
            source_hashes=_string_mapping(data["source_hashes"], "source_hashes"),
            writing_score_decisions=writing,
            exclusions=exclusions,
            ablation=_string(data["ablation"], "ablation"),
            ablation_recipe_hash=_string(
                data["ablation_recipe_hash"], "ablation_recipe_hash"
            ),
            sample_policy_identity_hashes=_strings(
                data["sample_policy_identity_hashes"],
                "sample_policy_identity_hashes",
            ),
            sample_collection_rounds=_ints(
                data["sample_collection_rounds"], "sample_collection_rounds"
            ),
            current_checkpoint_only=_bool(
                data["current_checkpoint_only"], "current_checkpoint_only"
            ),
            replay_enabled=_bool(data["replay_enabled"], "replay_enabled"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )


def _identity(data: Mapping[str, Any]) -> PolicyIdentity:
    identity = PolicyIdentity.from_dict(_mapping(data["policy_identity"], "policy_identity"))
    if data["policy_identity_hash"] != identity.identity_hash:
        raise ValueError("policy identity hash mismatch")
    return identity


def _require_schema(value: str) -> None:
    if value != OPD_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported round evidence schema")


def _require_round(collection_round: int, task_ordinal: int) -> None:
    if collection_round < 0 or task_ordinal < 1:
        raise ValueError("round evidence ordinal is invalid")


def _require_split(value: str) -> None:
    if value not in DATASET_SPLITS:
        raise ValueError(f"unsupported dataset split: {value!r}")


def _require_nonblank(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_unique_strings(
    values: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool = True,
) -> None:
    if (not allow_empty and not values) or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")


def _require_exact(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise ValueError(f"{name} fields do not match the versioned schema")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _objects(value: Any, loader, field_name: str):
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return tuple(loader(_mapping(item, field_name)) for item in value)


def _mapping_tuple(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return tuple(dict(_mapping(item, field_name)) for item in value)


def _count_mapping(value: Any, field_name: str) -> Mapping[str, int]:
    payload = _mapping(value, field_name)
    result: dict[str, int] = {}
    for key, count in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        parsed = _int(count, f"{field_name}.{key}")
        if parsed < 0:
            raise ValueError(f"{field_name} counts must be non-negative")
        result[key] = parsed
    return result


def _string_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    payload = _mapping(value, field_name)
    result: dict[str, str] = {}
    for key, item in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        result[key] = _string(item, f"{field_name}.{key}")
    return result


def _strings(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be an array of strings")
    return tuple(value)


def _ints(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value
    ):
        raise ValueError(f"{field_name} must be an array of non-negative integers")
    return tuple(value)


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


__all__ = [
    "DATASET_SPLITS",
    "OPD_EVIDENCE_SCHEMA_VERSION",
    "OPD_EXPORT_MANIFEST_SCHEMA_VERSION",
    "OPD_LEARNER_SCHEMA_VERSION",
    "ExportManifest",
    "LearnerSample",
    "MaintenanceEvidence",
    "RepositoryEvidence",
    "RepositoryMemoryEvidence",
    "TaskEvidence",
    "TaskOutcomeEvidence",
]
