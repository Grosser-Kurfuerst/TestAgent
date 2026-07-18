"""Immutable neutral indexes for an Experience repository snapshot."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier
from my_agent.memory.experience.repository_rules import (
    ExperienceDedupKey,
    experience_dedup_key,
)
from my_agent.memory.types import MemoryScope


@dataclass(frozen=True)
class ExperienceRepositoryIndexSnapshot:
    """Repository identity, deduplication, and visibility state for one revision."""

    revision: str
    by_id: Mapping[str, ExperienceMemory]
    dedup_ids: Mapping[ExperienceDedupKey, str]
    global_ids_by_tier: Mapping[ExperienceTier, tuple[str, ...]]
    project_ids_by_tier: Mapping[tuple[str, ExperienceTier], tuple[str, ...]]


ExperienceStoreIndexSnapshot = ExperienceRepositoryIndexSnapshot


def build_repository_index_snapshot(
    memories: Sequence[ExperienceMemory],
    *,
    revision: str,
) -> ExperienceRepositoryIndexSnapshot:
    by_id: dict[str, ExperienceMemory] = {}
    dedup_ids: dict[ExperienceDedupKey, str] = {}
    global_ids: dict[ExperienceTier, list[str]] = {tier: [] for tier in ExperienceTier}
    project_ids: dict[tuple[str, ExperienceTier], list[str]] = defaultdict(list)

    for memory in sorted(memories, key=lambda item: item.id):
        by_id[memory.id] = memory
        dedup_ids[experience_dedup_key(memory)] = memory.id
        if memory.scope == MemoryScope.GLOBAL:
            global_ids[memory.tier].append(memory.id)
        else:
            project_ids[(memory.project_key, memory.tier)].append(memory.id)

    return ExperienceRepositoryIndexSnapshot(
        revision=revision,
        by_id=MappingProxyType(dict(by_id)),
        dedup_ids=MappingProxyType(dict(dedup_ids)),
        global_ids_by_tier=MappingProxyType({
            tier: tuple(sorted(global_ids[tier])) for tier in ExperienceTier
        }),
        project_ids_by_tier=MappingProxyType({
            key: tuple(sorted(ids))
            for key, ids in sorted(
                project_ids.items(),
                key=lambda item: (item[0][0], item[0][1].value),
            )
        }),
    )


def visible_ids_for_tier(
    index: ExperienceRepositoryIndexSnapshot,
    *,
    project_key: str,
    tier: ExperienceTier,
) -> tuple[str, ...]:
    visible = set(index.global_ids_by_tier.get(tier, ()))
    if project_key:
        visible.update(index.project_ids_by_tier.get((project_key, tier), ()))
    return tuple(sorted(visible))


def visible_memories_for_tier(
    index: ExperienceRepositoryIndexSnapshot,
    *,
    project_key: str,
    tier: ExperienceTier,
    include_invalidated: bool = False,
) -> tuple[ExperienceMemory, ...]:
    return tuple(
        index.by_id[memory_id]
        for memory_id in visible_ids_for_tier(
            index,
            project_key=project_key,
            tier=tier,
        )
        if include_invalidated or not index.by_id[memory_id].invalidated
    )


__all__ = [
    "ExperienceRepositoryIndexSnapshot",
    "ExperienceStoreIndexSnapshot",
    "build_repository_index_snapshot",
    "visible_ids_for_tier",
    "visible_memories_for_tier",
]
