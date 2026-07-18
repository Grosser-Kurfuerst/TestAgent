from __future__ import annotations

import unittest
from datetime import datetime, timezone

from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import (
    MemoryEntry,
    MemoryScope,
    MemoryType,
    content_fingerprint,
    normalize_content,
)


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


if __name__ == "__main__":
    unittest.main()
