"""Typed public prompt construction for the formal maintenance agent."""

from __future__ import annotations

from collections.abc import Sequence

from my_agent.memory.experience.serialization import experience_to_dict
from my_agent.memory.experience.models import ExperienceMemory
from my_agent.policy.contracts import DecisionRequest
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.training.role_views import (
    CanonicalMessage,
    CanonicalTool,
    MaintenancePublic,
    RepositorySnapshotRef,
    TaskOutcomeRef,
)


def repository_snapshot_ref(
    entries: Sequence[ExperienceMemory],
    *,
    repository_revision: str,
    project_key: str,
    stream_id: str = "",
) -> RepositorySnapshotRef:
    visible = tuple(sorted(entries, key=lambda entry: entry.id))
    serialized = [experience_to_dict(entry) for entry in visible]
    return RepositorySnapshotRef(
        repository_revision=repository_revision,
        memory_project_key=project_key,
        memory_ids=tuple(entry.id for entry in visible),
        snapshot_hash=canonical_sha256({
            "repository_revision": repository_revision,
            "memory_project_key": project_key,
            "stream_id": stream_id,
            "memories": serialized,
        }),
    )


def maintenance_initial_messages(public: MaintenancePublic) -> tuple[CanonicalMessage, ...]:
    return (
        CanonicalMessage(
            "system",
            "Maintain the repository only through lookup, merge, delete, and finish tools. "
            "Issue exactly one tool call per turn and no prose. All mutations are staged until "
            "finish. Use source_ids only from repository_snapshot.memory_ids or earlier lookup "
            "hits; never invent memory IDs. Use lookup before changing memories whose content "
            "has not been inspected. Merge only same-tier memories and preserve that tier's "
            "payload schema. Delete only when the supplied repository/history evidence supports "
            "removal. If the repository is empty or no safe change is needed, call finish with "
            "a short summary.",
        ),
        CanonicalMessage(
            "user",
            canonical_json_bytes({"public_view": public.to_dict()}).decode("utf-8"),
        ),
    )


def build_maintenance_request(
    *,
    messages: tuple[CanonicalMessage, ...],
    tools: tuple[CanonicalTool, ...],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> DecisionRequest:
    return DecisionRequest(
        role="maintenance",
        purpose="fast_loop_evidence",
        messages=messages,
        tools=tools,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def maintenance_public_view(
    entries: Sequence[ExperienceMemory],
    *,
    repository_revision: str,
    project_key: str,
    stream_id: str = "",
    history_window: tuple[TaskOutcomeRef, ...],
    tools: tuple[CanonicalTool, ...],
) -> MaintenancePublic:
    return MaintenancePublic(
        repository_snapshot=repository_snapshot_ref(
            entries,
            repository_revision=repository_revision,
            project_key=project_key,
            stream_id=stream_id,
        ),
        history_window=history_window,
        tools=tools,
    )


__all__ = [
    "build_maintenance_request",
    "maintenance_initial_messages",
    "maintenance_public_view",
    "repository_snapshot_ref",
]
