"""Unified memory storage and retrieval for the agent.

The runtime uses :class:`~my_agent.memory.manager.MemoryManager` for memory
storage, retrieval, and compression primitives. Prompt assembly and
context-window budgeting live in :mod:`my_agent.context`.

Phase 3 ships the data model, short/long-term storage, config wiring,
retrieval, map-reduce compression, and the
:class:`MemoryManager` entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from my_agent.memory.api import MemoryService
from my_agent.memory.experience.models import (
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
from my_agent.memory.types import (
    CompressionResult,
    MemoryContext,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RetrievalHit,
)

if TYPE_CHECKING:
    from my_agent.memory.disabled import DisabledMemoryManager
    from my_agent.memory.manager import MemoryManager
    from my_agent.memory.noop import NoopMemoryManager


def __getattr__(name: str) -> Any:
    if name == "MemoryManager":
        from my_agent.memory.manager import MemoryManager

        globals()[name] = MemoryManager
        return MemoryManager
    if name == "NoopMemoryManager":
        from my_agent.memory.disabled import NoopMemoryManager

        globals()[name] = NoopMemoryManager
        return NoopMemoryManager
    if name == "DisabledMemoryManager":
        from my_agent.memory.disabled import DisabledMemoryManager

        globals()[name] = DisabledMemoryManager
        return DisabledMemoryManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CompressionResult",
    "ExperienceCreatedBy",
    "ExperienceMemory",
    "ExperiencePayload",
    "ExperienceTier",
    "ExperienceTrajectoryStep",
    "SkillPayload",
    "TipPayload",
    "ToolPayload",
    "TrajectoryPayload",
    "MemoryContext",
    "MemoryEntry",
    "MemoryManager",
    "MemoryService",
    "DisabledMemoryManager",
    "NoopMemoryManager",
    "MemoryScope",
    "MemoryStatus",
    "MemoryType",
    "RetrievalHit",
]
