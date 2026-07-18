"""Compatibility facade for Experience repository identity rules."""

from my_agent.memory.experience.repository_rules import (
    ExperienceDedupKey,
    experience_dedup_key,
    experience_memories_revision,
)

__all__ = [
    "ExperienceDedupKey",
    "experience_dedup_key",
    "experience_memories_revision",
]
