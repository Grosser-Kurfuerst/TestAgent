"""Immutable task-scoped state for one formal evolver episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from my_agent.memory.evolver.writer import ExperienceWriteStep
from my_agent.policy.identity import PolicyIdentity, require_sha256
from my_agent.training.role_views import CandidateSnapshotEntry


@dataclass(frozen=True)
class TaskEvolverSession:
    task_id: str
    task_group: str
    trajectory_id: str
    stream_id: str
    memory_project_key: str
    policy_identity: PolicyIdentity
    repository_revision: str
    candidate_snapshot_hash: str
    selected_memory_ids: tuple[str, ...]
    rendered_memory_context: str
    candidate_snapshot: tuple[CandidateSnapshotEntry, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "task_id",
            "task_group",
            "trajectory_id",
            "stream_id",
            "memory_project_key",
            "repository_revision",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task evolver session {field_name} must not be empty")
        if not isinstance(self.policy_identity, PolicyIdentity):
            raise ValueError("task evolver session requires PolicyIdentity")
        require_sha256(self.candidate_snapshot_hash, field_name="candidate_snapshot_hash")
        candidate_ids = tuple(item.memory_id for item in self.candidate_snapshot)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate snapshot memory IDs must be unique")
        if any(memory_id not in set(candidate_ids) for memory_id in self.selected_memory_ids):
            raise ValueError("selected memory IDs must reference the candidate snapshot")
        if len(set(self.selected_memory_ids)) != len(self.selected_memory_ids):
            raise ValueError("selected memory IDs must be unique")


@dataclass(frozen=True)
class AgentEpisodeArtifact:
    session: TaskEvolverSession
    trace_path: Path
    stop_reason: str
    final_answer: str
    tool_history: tuple[ExperienceWriteStep, ...]
    task: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.session, TaskEvolverSession):
            raise ValueError("agent episode requires TaskEvolverSession")
        if not isinstance(self.trace_path, Path):
            raise ValueError("agent episode trace_path must be a Path")


@dataclass(frozen=True)
class EvolverFinalizeResult:
    writer_status: str
    written_memory_ids: tuple[str, ...]
    repository_revision_after: str
    cadence_id: str | None = None
    maintenance_status: str | None = None


__all__ = ["AgentEpisodeArtifact", "EvolverFinalizeResult", "TaskEvolverSession"]
