from __future__ import annotations

import unittest
from datetime import datetime, timezone

from my_agent.memory.short_term import ShortTermMemory
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType


NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


def _entry(memory_id: str, content: str, *, source: str = "user") -> MemoryEntry:
    return MemoryEntry.build(
        id=memory_id,
        content=content,
        type=MemoryType.CONVERSATION,
        scope=MemoryScope.SESSION,
        source=source,
        token_count=estimate_tokens(content),
        created_at=NOW,
    )


class ShortTermMemoryTests(unittest.TestCase):
    def test_fifo_limits_and_clear(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=2)
        memory.append(_entry("a", "alpha"))
        memory.append(_entry("b", "beta"))
        evicted = memory.append(_entry("c", "gamma"))
        self.assertEqual([item.id for item in evicted], ["a"])
        self.assertEqual([item.id for item in memory.all()], ["b", "c"])
        self.assertEqual([item.id for item in memory.clear()], ["b", "c"])
        self.assertEqual(memory.token_count(), 0)

    def test_single_oversized_entry_is_retained(self) -> None:
        memory = ShortTermMemory(max_tokens=1, max_entries=10)
        memory.append(_entry("only", "x" * 100))
        self.assertEqual([item.id for item in memory.all()], ["only"])

    def test_recent_turns_and_compression_split_preserve_turn_boundaries(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=100)
        for index in range(4):
            memory.append(_entry(f"u{index}", f"user {index}", source="user"))
            memory.append(_entry(f"a{index}", f"assistant {index}", source="assistant"))
            memory.append(_entry(f"t{index}", f"tool {index}", source="tool:read"))
        self.assertEqual(
            [item.id for item in memory.recent_turns(2)],
            ["u2", "a2", "t2", "u3", "a3", "t3"],
        )
        self.assertEqual(
            [item.id for item in memory.old_entries_for_compression(2)],
            ["u0", "a0", "t0", "u1", "a1", "t1"],
        )

    def test_summary_replacement_keeps_recent_entries(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=100)
        for index in range(3):
            memory.append(_entry(f"u{index}", f"user {index}", source="user"))
            memory.append(_entry(f"a{index}", f"assistant {index}", source="assistant"))
        old = memory.old_entries_for_compression(1)
        summary = MemoryEntry.build(
            id="summary",
            content="compressed",
            type=MemoryType.SUMMARY,
            scope=MemoryScope.SESSION,
            source="compressor",
            token_count=estimate_tokens("compressed"),
            created_at=NOW,
        )
        memory.replace_old_entries_with_summary({item.id for item in old}, summary)
        self.assertEqual(
            [item.id for item in memory.all()],
            ["summary", "u2", "a2"],
        )


if __name__ == "__main__":
    unittest.main()
