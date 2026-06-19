"""Unified memory and context-budget management for the agent.

The Agent only talks to :class:`~my_agent.memory.manager.MemoryManager`; the
remaining modules (types, token, short/long-term storage, retrieval and
compression) are implementation details that can be swapped without changing
the call sites.

Phase 3.1-3.2 ships the data model, short/long-term storage, config wiring,
retrieval/context injection and the :class:`MemoryManager` facade subset that
does not require compression (``build_context_for_query``, ``save_fact``,
``append_*``, ``status``). Map-reduce compression, fact extraction, runtime
wiring and the remaining facade methods land in phases 3.3-3.6.
"""

from __future__ import annotations

from my_agent.memory.manager import MemoryManager
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
    "MemoryContext",
    "MemoryEntry",
    "MemoryManager",
    "MemoryScope",
    "MemoryStatus",
    "MemoryType",
    "RetrievalHit",
    "content_fingerprint",
    "normalize_content",
]
