"""Canonical repository identity and revision rules for typed experiences."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Sequence

from my_agent.memory.evolver.serialization import experience_to_dict
from my_agent.memory.evolver.types import ExperienceMemory
from my_agent.memory.types import MemoryScope


ExperienceDedupKey = tuple[str, str, str, str]


def experience_dedup_key(memory: ExperienceMemory) -> ExperienceDedupKey:
    """Return the single tier-aware repository dedup identity."""
    if not isinstance(memory, ExperienceMemory):
        raise TypeError("memory must be an ExperienceMemory")
    project_key = "" if memory.scope == MemoryScope.GLOBAL else memory.project_key
    return (memory.scope.value, project_key, memory.tier.value, memory.fingerprint)


def experience_memories_revision(memories: Sequence[ExperienceMemory]) -> str:
    """Hash the canonical, id-ordered typed repository representation."""
    payload = [
        experience_to_dict(memory)
        for memory in sorted(memories, key=lambda item: item.id)
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = [
    "ExperienceDedupKey",
    "experience_dedup_key",
    "experience_memories_revision",
]
