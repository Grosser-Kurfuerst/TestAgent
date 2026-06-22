from __future__ import annotations

import json
import threading
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.memory import (
    MemoryEntry,
    MemoryScope,
    MemoryType,
    content_fingerprint,
    normalize_content,
)
from my_agent.memory.long_term import LongTermMemoryStore
from my_agent.memory.retrieval import MemoryRetriever, tokenize
from my_agent.memory.short_term import ShortTermMemory
from my_agent.memory.token import estimate_tokens


NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


def _fact(
    content: str,
    *,
    id: str = "mem_x",
    scope: MemoryScope = MemoryScope.PROJECT,
    source: str = "manual",
    project_key: str = "/repo",
    created_at: datetime | None = None,
) -> MemoryEntry:
    return MemoryEntry.build(
        id=id,
        content=content,
        type=MemoryType.FACT,
        scope=scope,
        source=source,
        token_count=estimate_tokens(content),
        project_key=project_key,
        created_at=created_at or NOW,
    )


class TokenEstimationTests(unittest.TestCase):
    def test_chinese_uses_tighter_ratio_than_english(self) -> None:
        # 8 Chinese characters should cost more tokens than 8 ASCII characters.
        self.assertGreater(estimate_tokens("中文中文中文中文"), estimate_tokens("aaaaaaaa"))

    def test_empty_string_is_zero(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_objects_serialize_then_estimate(self) -> None:
        tokens = estimate_tokens({"a": "bcde"})
        self.assertGreater(tokens, 0)


class FingerprintTests(unittest.TestCase):
    def test_normalize_collapses_case_and_whitespace(self) -> None:
        self.assertEqual(normalize_content("Hello   World"), normalize_content("hello world"))

    def test_fingerprint_is_stable_across_case_and_whitespace(self) -> None:
        self.assertEqual(content_fingerprint("Hello World"), content_fingerprint("  hello   world "))

    def test_different_content_has_different_fingerprint(self) -> None:
        self.assertNotEqual(content_fingerprint("alpha"), content_fingerprint("beta"))


class MemoryEntryTests(unittest.TestCase):
    def test_round_trip_preserves_created_at_and_fields(self) -> None:
        entry = _fact("用户偏好：回答中文", id="mem_1", created_at=NOW)
        payload = entry.to_dict()

        restored = MemoryEntry.from_dict(payload)
        self.assertEqual(restored, entry)
        self.assertEqual(restored.created_at, NOW)
        self.assertEqual(restored.type, MemoryType.FACT)
        self.assertEqual(restored.scope, MemoryScope.PROJECT)
        self.assertEqual(restored.fingerprint, entry.fingerprint)

    def test_from_dict_restores_naive_datetime_as_utc(self) -> None:
        payload = {
            "id": "mem_2",
            "content": "fact",
            "type": "fact",
            "scope": "global",
            "source": "manual",
            "created_at": "2026-06-18T12:00:00",
            "token_count": 1,
        }
        entry = MemoryEntry.from_dict(payload)
        self.assertEqual(entry.created_at, datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc))


class ShortTermMemoryTests(unittest.TestCase):
    def _entry(self, id: str, content: str, *, source: str = "user") -> MemoryEntry:
        return MemoryEntry.build(
            id=id,
            content=content,
            type=MemoryType.CONVERSATION,
            scope=MemoryScope.SESSION,
            source=source,
            token_count=estimate_tokens(content),
        )

    def test_fifo_evicts_oldest_when_max_entries_exceeded(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=2)
        memory.append(self._entry("e0", "alpha"))
        memory.append(self._entry("e1", "beta"))
        evicted = memory.append(self._entry("e2", "gamma"))

        self.assertEqual([entry.id for entry in memory.all()], ["e1", "e2"])
        self.assertEqual([entry.id for entry in evicted], ["e0"])

    def test_fifo_evicts_oldest_when_max_tokens_exceeded(self) -> None:
        memory = ShortTermMemory(max_tokens=4, max_entries=100)
        big = "x" * 40
        evicted = memory.append(self._entry("a", big))
        self.assertEqual(evicted, [])
        # A second large entry pushes us over the budget; the first is dropped.
        evicted = memory.append(self._entry("b", big))
        self.assertEqual([entry.id for entry in memory.all()], ["b"])
        self.assertEqual([entry.id for entry in evicted], ["a"])

    def test_single_oversized_entry_is_not_evicted(self) -> None:
        memory = ShortTermMemory(max_tokens=1, max_entries=10)
        memory.append(self._entry("only", "x" * 100))
        self.assertEqual(len(memory), 1)

    def test_recent_turns_keeps_last_n_user_turns(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=100)

        def turn(i: int) -> None:
            memory.append(self._entry(f"u{i}", f"user {i}", source="user"))
            memory.append(self._entry(f"a{i}", f"asst {i}", source="assistant"))
            memory.append(self._entry(f"t{i}", f"tool {i}", source="tool:read"))

        for i in range(4):
            turn(i)

        recent = memory.recent_turns(2)
        self.assertEqual([entry.id for entry in recent], ["u2", "a2", "t2", "u3", "a3", "t3"])

    def test_old_entries_for_compression_splits_on_user_boundary(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=100)

        def turn(i: int) -> None:
            memory.append(self._entry(f"u{i}", f"user {i}", source="user"))
            memory.append(self._entry(f"a{i}", f"asst {i}", source="assistant"))
            memory.append(self._entry(f"t{i}", f"tool {i}", source="tool:read"))

        for i in range(4):
            turn(i)

        old = memory.old_entries_for_compression(2)
        # Split must fall before u2 so the tool_call/tool_result pair of turn 2
        # is never severed from its assistant message.
        self.assertEqual([entry.id for entry in old], ["u0", "a0", "t0", "u1", "a1", "t1"])
        self.assertNotIn("u2", [entry.id for entry in old])

    def test_replace_old_entries_with_summary(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=100)
        for i in range(3):
            memory.append(self._entry(f"u{i}", f"user {i}", source="user"))
            memory.append(self._entry(f"a{i}", f"asst {i}", source="assistant"))
        old = memory.old_entries_for_compression(1)
        summary = MemoryEntry.build(
            id="sum1",
            content="compressed summary",
            type=MemoryType.SUMMARY,
            scope=MemoryScope.SESSION,
            source="compressor",
            token_count=estimate_tokens("compressed summary"),
        )
        memory.replace_old_entries_with_summary({entry.id for entry in old}, summary)

        ids = [entry.id for entry in memory.all()]
        self.assertEqual(ids[0], "sum1")
        self.assertNotIn("u0", ids)
        self.assertEqual(ids[1:], ["u2", "a2"])

    def test_replace_rejects_non_summary_entry(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=100)
        memory.append(self._entry("u0", "user 0", source="user"))
        not_summary = self._entry("bad", "bad")
        with self.assertRaises(ValueError):
            memory.replace_old_entries_with_summary(set(), not_summary)

    def test_clear_returns_entries_and_resets_tokens(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=100)
        memory.append(self._entry("u0", "user 0", source="user"))
        removed = memory.clear()
        self.assertEqual([entry.id for entry in removed], ["u0"])
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.token_count(), 0)

    def test_extend_appends_multiple_and_reports_evicted(self) -> None:
        memory = ShortTermMemory(max_tokens=1_000_000, max_entries=2)
        evicted = memory.extend([self._entry("a", "a"), self._entry("b", "b"), self._entry("c", "c")])
        self.assertEqual([entry.id for entry in memory.all()], ["b", "c"])
        self.assertEqual([entry.id for entry in evicted], ["a"])


class LongTermMemoryStoreTests(unittest.TestCase):
    def _store(self, dir_path: Path, *, traces: list | None = None) -> LongTermMemoryStore:
        store = LongTermMemoryStore(dir_path / "long_term_memory.jsonl")
        if traces is not None:
            store._trace_sink = lambda event, payload: traces.append((event, payload))  # type: ignore[method-assign]
        return store

    def test_add_deduplicates_same_content_same_scope_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            first, created1 = store.add(_fact("用户偏好：回答中文", id="f1"))
            same, created2 = store.add(_fact("用户偏好：回答中文", id="f2"))

            self.assertTrue(created1)
            self.assertFalse(created2)
            self.assertEqual(same.id, "f1")
            self.assertEqual(len(store), 1)

    def test_duplicate_does_not_change_original_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            original = _fact("fact text", id="f1", created_at=NOW - timedelta(days=10))
            store.add(original)
            later = _fact("fact text", id="f2", created_at=NOW)
            stored, created = store.add(later)

            self.assertFalse(created)
            self.assertEqual(stored.created_at, NOW - timedelta(days=10))

    def test_global_scope_ignores_project_key_for_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            store.add(_fact("global fact", id="g1", scope=MemoryScope.GLOBAL, project_key="/a"))
            stored, created = store.add(
                _fact("global fact", id="g2", scope=MemoryScope.GLOBAL, project_key="/b")
            )
            self.assertFalse(created)
            self.assertEqual(stored.id, "g1")

    def test_different_projects_keep_separate_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            store.add(_fact("shared fact", id="p1", project_key="/a"))
            _, created = store.add(_fact("shared fact", id="p2", project_key="/b"))
            self.assertTrue(created)

    def test_concurrent_adds_persist_without_tmp_file_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = self._store(base)
            store.load()
            errors: list[BaseException] = []

            def add_fact(index: int) -> None:
                try:
                    store.add(_fact(f"parallel fact {index}", id=f"f{index}"))
                except BaseException as exc:  # noqa: BLE001 - test records thread failures.
                    errors.append(exc)

            threads = [threading.Thread(target=add_fact, args=(index,)) for index in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(len(store), 12)
            reloaded = self._store(base)
            reloaded.load()
            self.assertEqual(len(reloaded), 12)

    def test_add_computes_fingerprint_for_blank_entry(self) -> None:
        # add() is a public store boundary (plan §6: "add() 先计算
        # fingerprint"). A caller that constructs a MemoryEntry directly —
        # bypassing MemoryEntry.build(), which fills a blank fingerprint —
        # must not collide with another blank-fingerprint entry of different
        # content.
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            first = MemoryEntry(
                id="raw1", content="first raw fact about cats",
                type=MemoryType.FACT, scope=MemoryScope.PROJECT, source="manual",
                created_at=NOW, token_count=4, project_key="/repo",
            )
            second = MemoryEntry(
                id="raw2", content="second raw fact about dogs",
                type=MemoryType.FACT, scope=MemoryScope.PROJECT, source="manual",
                created_at=NOW, token_count=4, project_key="/repo",
            )
            stored_first, created_first = store.add(first)
            stored_second, created_second = store.add(second)

            self.assertTrue(created_first)
            self.assertTrue(created_second)
            self.assertEqual(len(store), 2)
            self.assertEqual(stored_first.fingerprint, content_fingerprint("first raw fact about cats"))
            self.assertEqual(stored_second.fingerprint, content_fingerprint("second raw fact about dogs"))

    def test_add_deduplicates_blank_entries_with_same_content(self) -> None:
        # The flip side: two blank-fingerprint entries with the SAME content
        # are a true duplicate and must still collapse to the original.
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            first = MemoryEntry(
                id="raw1", content="identical raw fact",
                type=MemoryType.FACT, scope=MemoryScope.PROJECT, source="manual",
                created_at=NOW - timedelta(days=2), token_count=4, project_key="/repo",
            )
            second = MemoryEntry(
                id="raw2", content="identical raw fact",
                type=MemoryType.FACT, scope=MemoryScope.PROJECT, source="manual",
                created_at=NOW, token_count=4, project_key="/repo",
            )
            store.add(first)
            stored, created = store.add(second)

            self.assertFalse(created)
            self.assertEqual(len(store), 1)
            self.assertEqual(stored.id, "raw1")
            # Original created_at preserved on a duplicate save.
            self.assertEqual(stored.created_at, NOW - timedelta(days=2))
            self.assertEqual(stored.fingerprint, content_fingerprint("identical raw fact"))

    def test_add_passes_through_well_formed_entry_unchanged(self) -> None:
        # An entry built via MemoryEntry.build() already carries a matching
        # fingerprint; add() must return it as-is rather than rebuild a copy.
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            built = _fact("well-formed fact", id="wf1")
            stored, created = store.add(built)

            self.assertTrue(created)
            self.assertIs(stored, built)
            self.assertEqual(stored.fingerprint, built.fingerprint)

    def test_manual_upgrade_rolls_back_when_persist_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            auto = _fact(
                "用户偏好：回答中文",
                id="auto",
                source="fact_extractor",
                created_at=NOW - timedelta(days=1),
            )
            store.add(auto)
            original_persist = store._persist

            def fail_persist() -> None:
                raise OSError("disk full")

            store._persist = fail_persist  # type: ignore[method-assign]
            with self.assertRaises(OSError):
                store.add(_fact("用户偏好：回答中文", id="manual", source="manual", created_at=NOW))

            restored = store.all()[0]
            self.assertEqual(restored.id, "auto")
            self.assertEqual(restored.source, "fact_extractor")
            self.assertEqual(restored.created_at, NOW - timedelta(days=1))

            store._persist = original_persist  # type: ignore[method-assign]
            upgraded, created = store.add(_fact("用户偏好：回答中文", id="manual", source="manual", created_at=NOW))
            self.assertFalse(created)
            self.assertEqual(upgraded.id, "auto")
            self.assertEqual(upgraded.source, "manual")
            self.assertEqual(upgraded.created_at, NOW - timedelta(days=1))

    def test_reload_preserves_count_and_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            created_at = NOW - timedelta(days=5)
            entry = store.add(_fact("persist me", id="f1", created_at=created_at))[0]

            reloaded = self._store(Path(tmp))
            reloaded.load()
            self.assertEqual(len(reloaded), 1)
            restored = reloaded.all()[0]
            self.assertEqual(restored.id, entry.id)
            self.assertEqual(restored.created_at, created_at)
            self.assertEqual(restored.fingerprint, entry.fingerprint)

    def test_load_dedupe_keeps_earliest_created_at_when_newer_appears_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            newer = _fact("shared fact", id="n", created_at=NOW)
            older = _fact("shared fact", id="o", created_at=NOW - timedelta(days=17))
            # File lists the newer entry first, the older entry second.
            with path.open("w", encoding="utf-8") as file:
                file.write(json.dumps(newer.to_dict(), ensure_ascii=False) + "\n")
                file.write(json.dumps(older.to_dict(), ensure_ascii=False) + "\n")

            store = LongTermMemoryStore(path)
            store.load()

            self.assertEqual(len(store), 1)
            kept = store.all()[0]
            # The earliest created_at must win (plan §6: duplicate saves do not
            # change the original created_at), even though it appeared later.
            self.assertEqual(kept.id, "o")
            self.assertEqual(kept.created_at, NOW - timedelta(days=17))

    def test_load_backfills_fingerprint_for_legacy_lines(self) -> None:
        # Older JSONL lines (or a hand-edited file) may omit the fingerprint
        # field. from_dict() must recompute it from content so load-time
        # dedup still collapses true duplicates and the stored entry carries a
        # stable fingerprint.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            dup_a = {
                "id": "a", "content": "legacy fact", "type": "fact",
                "scope": "project", "source": "manual",
                "created_at": NOW.isoformat(), "token_count": 4,
                "project_key": "/repo", "metadata": {},
            }
            dup_b = {
                "id": "b", "content": "legacy fact", "type": "fact",
                "scope": "project", "source": "manual",
                "created_at": (NOW - timedelta(days=17)).isoformat(),
                "token_count": 4, "project_key": "/repo", "metadata": {},
            }
            distinct = {
                "id": "c", "content": "other fact", "type": "fact",
                "scope": "project", "source": "manual",
                "created_at": NOW.isoformat(), "token_count": 4,
                "project_key": "/repo", "metadata": {},
            }
            with path.open("w", encoding="utf-8") as file:
                for entry in (dup_a, dup_b, distinct):
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")

            store = LongTermMemoryStore(path)
            store.load()

            entries = store.all()
            self.assertEqual(len(entries), 2)
            ids = {entry.id for entry in entries}
            # Earliest created_at wins for the duplicate pair.
            self.assertEqual(ids, {"b", "c"})
            for entry in entries:
                self.assertEqual(entry.fingerprint, content_fingerprint(entry.content))

    def test_bad_jsonl_line_is_skipped_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            good = _fact("good fact", id="g1").to_dict()
            second = _fact("second good", id="g2").to_dict()
            path.write_text(
                json.dumps(good, ensure_ascii=False) + "\n"
                + "this is not json\n"
                + json.dumps(second, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            traces: list = []
            store = LongTermMemoryStore(path, trace_sink=lambda event, payload: traces.append((event, payload)))
            store.load()

            self.assertEqual(len(store), 2)
            events = {event for event, _ in traces}
            self.assertIn("memory.load_skipped", events)

    def test_search_candidates_filters_by_project_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            store.add(_fact("repo a fact", id="a1", project_key="/repo_a"))
            store.add(_fact("repo b fact", id="b1", project_key="/repo_b"))
            store.add(_fact("global fact", id="g1", scope=MemoryScope.GLOBAL, project_key=""))

            visible_a = store.search_candidates(project_key="/repo_a")
            ids = {entry.id for entry in visible_a}
            self.assertEqual(ids, {"a1", "g1"})
            self.assertNotIn("b1", ids)

    def test_clear_with_scope_only_removes_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            store.add(_fact("project fact", id="p1", project_key="/repo"))
            store.add(_fact("global fact", id="g1", scope=MemoryScope.GLOBAL))

            removed = store.clear(scope=MemoryScope.GLOBAL)
            self.assertEqual(removed, 1)
            ids = {entry.id for entry in store.all()}
            self.assertEqual(ids, {"p1"})

    def test_atomic_save_writes_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            store.load()
            store.add(_fact("line one", id="l1"))
            store.add(_fact("line two", id="l2"))

            lines = store.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                payload = json.loads(line)
                self.assertIn("id", payload)
                self.assertIn("fingerprint", payload)
            self.assertFalse(store.path.with_suffix(".jsonl.tmp").exists())

    def test_add_without_load_does_not_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            store = self._store(Path(tmp))
            store.load()
            store.add(_fact("OLD FACT", id="old"))

            # A brand-new store with no load() must not clobber the existing
            # file when it adds a new entry.
            unready = LongTermMemoryStore(path)
            unready.add(_fact("NEW FACT", id="new"))

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            ids = {json.loads(line)["id"] for line in lines}
            self.assertEqual(ids, {"old", "new"})

    def test_all_without_load_reads_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            store = self._store(Path(tmp))
            store.load()
            store.add(_fact("persisted fact", id="p1", project_key="/repo"))

            unready = LongTermMemoryStore(path)
            self.assertEqual(len(unready), 1)
            self.assertEqual(unready.all(project_key="/repo")[0].id, "p1")


class RetrievalTests(unittest.TestCase):
    def _store_with(self, entries: list[MemoryEntry]) -> tuple[LongTermMemoryStore, Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = LongTermMemoryStore(Path(tmp.name) / "long_term_memory.jsonl")
        store.load()
        for entry in entries:
            store.add(entry)
        return store, Path(tmp.name)

    def test_tokenize_matches_english_words_and_chinese_spans(self) -> None:
        tokens = tokenize("Fix the subtract bug in calculator.py, 计算器")
        self.assertIn("subtract", tokens)
        self.assertIn("calculator.py", tokens)
        self.assertIn("计算器", tokens)

    def test_retrieve_returns_relevant_long_term_hits(self) -> None:
        store, _ = self._store_with([
            _fact("用户偏好：回答中文，先给结论", id="r1"),
            _fact("项目使用 FastAPI 构建 REST 接口", id="r2"),
        ])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "用户偏好 回答中文",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].entry.id, "r1")
        self.assertGreater(hits[0].score, 0)

    def test_recent_entry_outscores_30_day_old_same_content(self) -> None:
        store, _ = self._store_with([
            _fact("用户偏好：回答中文", id="recent", created_at=NOW),
            _fact("用户偏好：回答中文", id="old", created_at=NOW - timedelta(days=30)),
        ])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "用户偏好 回答中文",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        # Dedup keeps only the earliest (oldest) entry, so the recent one is
        # the one that survives a second add of identical content. Verify the
        # surviving entry is scored with full time-decay = 1.0.
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].time_decay, 1.0, places=6)

    def test_newer_outranks_older_when_both_present(self) -> None:
        store, _ = self._store_with([
            _fact("项目使用 FastAPI 构建接口", id="newer", created_at=NOW),
            _fact("项目使用 Django 构建接口", id="older", created_at=NOW - timedelta(days=29)),
        ])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "项目 接口",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].entry.id, "newer")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_long_term_weight_beats_short_term_same_score(self) -> None:
        store, _ = self._store_with([_fact("用户偏好：回答中文", id="lt", created_at=NOW)])
        short_term = ShortTermMemory(max_tokens=1_000_000, max_entries=100)
        short_term.append(
            MemoryEntry.build(
                id="st",
                content="用户偏好：回答中文",
                type=MemoryType.CONVERSATION,
                scope=MemoryScope.SESSION,
                source="user",
                token_count=estimate_tokens("用户偏好：回答中文"),
                created_at=NOW,
            )
        )
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "用户偏好 回答中文",
            short_term=short_term,
            long_term=store,
            project_key="/repo",
            limit=8,
            include_short_term=True,
        )
        self.assertEqual(hits[0].entry.id, "lt")
        self.assertEqual(hits[0].source_weight, 1.2)
        # Short-term candidate exists with same base/time-decay but weight 1.0.
        short_hit = next(hit for hit in hits if hit.entry.id == "st")
        self.assertAlmostEqual(short_hit.source_weight, 1.0)
        self.assertGreater(hits[0].score, short_hit.score)

    def test_project_scope_only_injects_same_repo(self) -> None:
        store, _ = self._store_with([
            _fact("repo a fact about config", id="a", project_key="/repo_a"),
            _fact("repo b fact about config", id="b", project_key="/repo_b"),
        ])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "config",
            short_term=None,
            long_term=store,
            project_key="/repo_a",
            limit=8,
        )
        ids = {hit.entry.id for hit in hits}
        self.assertEqual(ids, {"a"})

    def test_global_scope_visible_to_all_repos(self) -> None:
        store, _ = self._store_with([
            _fact("global rule about config", id="g", scope=MemoryScope.GLOBAL, project_key=""),
        ])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "config",
            short_term=None,
            long_term=store,
            project_key="/any_repo",
            limit=8,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].entry.id, "g")

    def test_build_context_respects_token_budget(self) -> None:
        store, _ = self._store_with([
            _fact("用户偏好：回答中文，先给结论" * 3, id="big"),
        ])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "用户偏好",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        ctx = retriever.build_context(hits, max_tokens=60)
        self.assertLessEqual(ctx.estimated_tokens, 60)
        self.assertTrue(ctx.injected_text.startswith("Relevant long-term memory:"))

    def test_build_context_never_exceeds_budget_even_when_header_alone_overflows(self) -> None:
        store, _ = self._store_with([_fact("用户偏好：回答中文", id="r")])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "用户偏好",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        self.assertEqual(len(hits), 1)
        # Budget too small for even the header: must inject nothing, not a
        # truncated header that exceeds the budget.
        ctx = retriever.build_context(hits, max_tokens=1)
        self.assertEqual(ctx.injected_text, "")
        self.assertEqual(ctx.estimated_tokens, 0)
        self.assertEqual(ctx.hits, [])

    def test_build_context_returns_empty_when_no_hit_fits_budget(self) -> None:
        store, _ = self._store_with([_fact("用户偏好：回答中文" * 4, id="big")])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "用户偏好",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        # Header fits but the single large hit does not: no empty memory
        # message is injected.
        ctx = retriever.build_context(hits, max_tokens=10)
        self.assertEqual(ctx.injected_text, "")
        self.assertEqual(ctx.estimated_tokens, 0)
        self.assertEqual(ctx.hits, [])

    def test_build_context_skips_oversized_hit_and_keeps_fittable_later_hit(self) -> None:
        store, _ = self._store_with([
            _fact("big hit " * 20, id="big"),
            _fact("small hit", id="small"),
        ])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "hit",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        # Both match with equal score; the big hit sorts first.
        self.assertEqual([hit.entry.id for hit in hits], ["big", "small"])
        # The big hit does not fit, but the small one does: the oversized hit
        # is skipped (not used to terminate) so the relevant, fittable memory
        # is still injected.
        ctx = retriever.build_context(hits, max_tokens=25)
        self.assertIn("small", ctx.injected_text)
        self.assertNotIn("big hit", ctx.injected_text)
        self.assertEqual([hit.entry.id for hit in ctx.hits], ["small"])
        self.assertLessEqual(ctx.estimated_tokens, 25)

    def test_build_context_never_exceeds_budget_with_many_small_hits(self) -> None:
        # Many small fittable hits joined with "\n" separators must still keep
        # the final estimated_tokens within budget (the separator tokens are
        # counted by estimate_tokens, so the admission check must count them
        # too).
        store, _ = self._store_with([_fact(f"hit {i}", id=f"h{i}") for i in range(20)])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "hit",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=20,
        )
        ctx = retriever.build_context(hits, max_tokens=57)
        self.assertGreater(len(ctx.hits), 1)
        self.assertLessEqual(ctx.estimated_tokens, 57)

    def test_build_context_empty_when_no_hits(self) -> None:
        retriever = MemoryRetriever(now=NOW)
        ctx = retriever.build_context([], max_tokens=100)
        self.assertEqual(ctx.injected_text, "")
        self.assertEqual(ctx.estimated_tokens, 0)
        self.assertEqual(ctx.hits, [])

    def test_build_context_header_generalizes_with_short_term_hits(self) -> None:
        store, _ = self._store_with([_fact("用户偏好：回答中文", id="lt", created_at=NOW)])
        short_term = ShortTermMemory(max_tokens=1_000_000, max_entries=100)
        short_term.append(
            MemoryEntry.build(
                id="st",
                content="用户偏好：回答中文",
                type=MemoryType.CONVERSATION,
                scope=MemoryScope.SESSION,
                source="user",
                token_count=estimate_tokens("用户偏好：回答中文"),
                created_at=NOW,
            )
        )
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "用户偏好 回答中文",
            short_term=short_term,
            long_term=store,
            project_key="/repo",
            limit=8,
            include_short_term=True,
        )
        ctx = retriever.build_context(hits, max_tokens=500)
        # Mixed sources use the general header so it is not factually wrong.
        self.assertTrue(ctx.injected_text.startswith("Relevant memory:"))

    def test_retrieve_default_excludes_short_term(self) -> None:
        store, _ = self._store_with([])
        short_term = ShortTermMemory(max_tokens=1_000_000, max_entries=100)
        short_term.append(
            MemoryEntry.build(
                id="st",
                content="用户偏好：回答中文",
                type=MemoryType.CONVERSATION,
                scope=MemoryScope.SESSION,
                source="user",
                token_count=estimate_tokens("用户偏好：回答中文"),
                created_at=NOW,
            )
        )
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "用户偏好 回答中文",
            short_term=short_term,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        # Default injection uses long-term only; short-term is not injected.
        self.assertEqual(hits, [])

    def test_query_with_no_match_returns_empty(self) -> None:
        store, _ = self._store_with([_fact("项目使用 FastAPI", id="r1")])
        retriever = MemoryRetriever(now=NOW)
        hits = retriever.retrieve(
            "completely unrelated query xyz",
            short_term=None,
            long_term=store,
            project_key="/repo",
            limit=8,
        )
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
