"""Unified memory storage and retrieval for the agent.

The runtime uses :class:`~my_agent.memory.manager.MemoryManager` for memory
storage, retrieval, and compression primitives. Prompt assembly and
context-window budgeting live in :mod:`my_agent.context`.

Phase 3 ships the data model, short/long-term storage, config wiring,
retrieval, map-reduce compression, fact extraction, and the
:class:`MemoryManager` entry point.
"""

from __future__ import annotations

from my_agent.memory.compression import MemoryCompressor
from my_agent.memory.experience_store import (
    EXPERIENCE_LOCK_FILE,
    EXPERIENCE_STORAGE_FILE,
    ExperienceStore,
    ExperienceStoreIndexSnapshot,
    ExperienceStoreSnapshot,
)
from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperiencePayload,
    ExperienceRecord,
    ExperienceTier,
    ExperienceTrajectoryStep,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
    build_experience_entry,
    experience_record_from_entry,
    experience_tier,
    is_experience_entry,
)
from my_agent.memory.manager import MemoryManager
from my_agent.memory.noop import NoopMemoryManager
from my_agent.memory.types import (
    CompressionResult,
    MemoryContext,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RetrievalHit,
    content_fingerprint,
    normalize_content,
)

__all__ = [
    "CompressionResult",
    "ExperienceCreatedBy",
    "ExperienceStore",
    "ExperienceStoreIndexSnapshot",
    "ExperienceStoreSnapshot",
    "ExperienceMemory",
    "ExperiencePayload",
    "ExperienceRecord",
    "ExperienceTier",
    "ExperienceTrajectoryStep",
    "EXPERIENCE_LOCK_FILE",
    "EXPERIENCE_STORAGE_FILE",
    "SkillPayload",
    "TipPayload",
    "ToolPayload",
    "TrajectoryPayload",
    "MemoryCompressor",
    "MemoryContext",
    "MemoryEntry",
    "MemoryManager",
    "NoopMemoryManager",
    "MemoryScope",
    "MemoryStatus",
    "MemoryType",
    "RetrievalHit",
    "build_experience_entry",
    "content_fingerprint",
    "experience_record_from_entry",
    "experience_tier",
    "is_experience_entry",
    "normalize_content",
]
