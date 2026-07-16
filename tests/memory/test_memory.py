from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory import (
    MemoryEntry,
    MemoryScope,
    MemoryType,
    content_fingerprint,
    normalize_content,
)
from my_agent.memory.short_term import ShortTermMemory
from my_agent.memory.token import estimate_tokens


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


class TokenAndFingerprintTests(unittest.TestCase):
    def test_token_estimation_and_fingerprints_remain_stable(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("中文中文中文中文"), estimate_tokens("aaaaaaaa"))
        self.assertEqual(normalize_content("Hello   World"), normalize_content("hello world"))
        self.assertEqual(
            content_fingerprint("Hello World"),
            content_fingerprint("  hello   world "),
        )
        self.assertNotEqual(content_fingerprint("alpha"), content_fingerprint("beta"))


class MemoryEntryTests(unittest.TestCase):
    def test_short_term_entry_round_trip_preserves_fields(self) -> None:
        entry = _entry("mem-1", "用户消息")
        self.assertEqual(MemoryEntry.from_dict(entry.to_dict()), entry)
        self.assertEqual(entry.scope, MemoryScope.SESSION)
        self.assertEqual(entry.type, MemoryType.CONVERSATION)

    def test_naive_legacy_short_term_datetime_is_restored_as_utc(self) -> None:
        payload = {
            "id": "mem-2",
            "content": "summary",
            "type": "summary",
            "scope": "session",
            "source": "compressor",
            "created_at": "2026-06-18T12:00:00",
            "token_count": 1,
        }
        entry = MemoryEntry.from_dict(payload)
        self.assertEqual(entry.created_at, NOW)


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
