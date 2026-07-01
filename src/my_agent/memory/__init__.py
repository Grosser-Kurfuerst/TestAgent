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
    "MemoryCompressor",
    "MemoryContext",
    "MemoryEntry",
    "MemoryManager",
    "NoopMemoryManager",
    "MemoryScope",
    "MemoryStatus",
    "MemoryType",
    "RetrievalHit",
    "content_fingerprint",
    "normalize_content",
]
