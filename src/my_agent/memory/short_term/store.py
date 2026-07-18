"""Ordered in-session memory storage."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from my_agent.memory.types import MemoryEntry, MemoryType


class ShortTermMemory:
    """Ordered short-term conversation memory with FIFO eviction."""

    def __init__(self, *, max_tokens: int, max_entries: int = 500) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1.")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1.")
        self._max_tokens = max_tokens
        self._max_entries = max_entries
        self._entries: deque[MemoryEntry] = deque()
        self._tokens = 0

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def append(self, entry: MemoryEntry) -> list[MemoryEntry]:
        """Append an entry and return FIFO evictions, oldest first."""
        self._entries.append(entry)
        self._tokens += max(0, entry.token_count)
        return self._evict()

    def extend(self, entries: Iterable[MemoryEntry]) -> list[MemoryEntry]:
        evicted: list[MemoryEntry] = []
        for entry in entries:
            evicted.extend(self.append(entry))
        return evicted

    def all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def recent_turns(self, retain_user_turns: int) -> list[MemoryEntry]:
        """Return entries belonging to the most recent user turns."""
        if retain_user_turns < 1:
            return []
        entries = list(self._entries)
        user_indices = [idx for idx, entry in enumerate(entries) if _is_user_boundary(entry)]
        if not user_indices:
            return entries
        if len(user_indices) <= retain_user_turns:
            start = user_indices[0]
        else:
            start = user_indices[-retain_user_turns]
        return entries[start:]

    def old_entries_for_compression(self, retain_user_turns: int) -> list[MemoryEntry]:
        """Return the prefix that may be compressed without splitting a turn."""
        if retain_user_turns < 1:
            return list(self._entries)
        entries = list(self._entries)
        user_indices = [idx for idx, entry in enumerate(entries) if _is_user_boundary(entry)]
        if not user_indices:
            return []
        if len(user_indices) <= retain_user_turns:
            if len(user_indices) != 1:
                return []
            prefix_end = user_indices[0] + 1
            tail_entries = max(2, retain_user_turns * 2)
            if len(entries) - prefix_end <= tail_entries:
                return []
            split_idx = max(prefix_end, len(entries) - tail_entries)
            while split_idx > prefix_end and entries[split_idx].source.startswith("tool:"):
                split_idx -= 1
            return entries[prefix_end:split_idx]
        start = user_indices[-retain_user_turns]
        return entries[:start]

    def replace_old_entries_with_summary(
        self,
        old_ids: set[str],
        summary: MemoryEntry,
    ) -> None:
        """Replace compressed entries with a summary at their first position."""
        if summary.type != MemoryType.SUMMARY:
            raise ValueError("summary entry must have MemoryType.SUMMARY.")
        old_id_set = set(old_ids)
        kept: list[MemoryEntry] = []
        removed_tokens = 0
        inserted_summary = False
        for entry in self._entries:
            if entry.id in old_id_set:
                removed_tokens += max(0, entry.token_count)
                if not inserted_summary:
                    kept.append(summary)
                    inserted_summary = True
            else:
                kept.append(entry)
        if not inserted_summary:
            kept.insert(0, summary)
        self._entries = deque(kept)
        self._tokens = max(0, self._tokens - removed_tokens)
        self._tokens += max(0, summary.token_count)
        self._evict()

    def clear(self) -> list[MemoryEntry]:
        """Remove and return all entries."""
        removed = list(self._entries)
        self._entries.clear()
        self._tokens = 0
        return removed

    def token_count(self) -> int:
        return self._tokens

    def fork(self) -> "ShortTermMemory":
        """Create an empty task-local store with the same hard limits."""
        return ShortTermMemory(
            max_tokens=self._max_tokens,
            max_entries=self._max_entries,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def _evict(self) -> list[MemoryEntry]:
        evicted: list[MemoryEntry] = []
        while self._entries and (
            len(self._entries) > self._max_entries or self._tokens > self._max_tokens
        ):
            if len(self._entries) == 1:
                break
            oldest = self._entries.popleft()
            self._tokens = max(0, self._tokens - max(0, oldest.token_count))
            evicted.append(oldest)
        return evicted


def _is_user_boundary(entry: MemoryEntry) -> bool:
    return entry.source in {"user", "task_goal"}


__all__ = ["ShortTermMemory"]
