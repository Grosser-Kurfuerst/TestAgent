from __future__ import annotations

# ruff: noqa: E402 - tests add the src layout before importing project modules

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver.attribution import MemoryAttributionRecord
from my_agent.memory.evolver.serialization import experience_canonical_json, experience_to_dict
from my_agent.memory.evolver.types import (
    ExperienceMemory,
    ExperienceTier,
    SkillPayload,
    TipPayload,
)
from my_agent.memory.experience_retrieval import ExperienceRetriever, experience_index_terms
from my_agent.memory.experience.retrieval.lexical import build_lexical_index
from my_agent.memory.experience_store import (
    EXPERIENCE_LOCK_FILE,
    EXPERIENCE_STORAGE_FILE,
    ExperienceStore,
    MemoryStoreLoadError,
    MemoryStorePostCommitError,
    MemoryStoreRevisionConflict,
    experience_memories_revision,
)
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType, content_fingerprint


NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


def _payload(tier: ExperienceTier):
    if tier == ExperienceTier.TIP:
        return TipPayload("testing", "warning", "focused pytest failure")
    if tier == ExperienceTier.SKILL:
        return SkillPayload(
            "testing",
            "focused-to-full validation",
            ("a focused target exists",),
            ("run pytest focused", "run pytest full"),
        )
    raise AssertionError(f"unsupported fixture tier: {tier}")


def _memory(
    memory_id: str,
    content: str,
    *,
    tier: ExperienceTier = ExperienceTier.TIP,
    project_key: str = "/repo",
    scope: MemoryScope = MemoryScope.PROJECT,
    created_at: datetime = NOW,
    invalidated: bool = False,
) -> ExperienceMemory:
    return ExperienceMemory(
        id=memory_id,
        content=content,
        tier=tier,
        payload=_payload(tier),
        scope=scope,
        project_key="" if scope == MemoryScope.GLOBAL else project_key,
        created_at=created_at,
        token_count=8,
        fingerprint=content_fingerprint(content),
        source_task="task-1",
        run_id="run-1",
        stream_id="stream-1",
        writer_confidence=0.9,
        invalidated=invalidated,
    )


class ExperienceStoreTests(unittest.TestCase):
    def test_from_dir_uses_typed_storage_and_lock_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            self.assertEqual(store.path.name, EXPERIENCE_STORAGE_FILE)
            self.assertEqual(store.lock_path.name, EXPERIENCE_LOCK_FILE)
            self.assertFalse(store.path.exists())

    def test_add_round_trips_all_fields_and_rejects_plain_memory_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            memory = _memory("tip", "Run focused pytest before the suite.")

            stored, created = store.add(memory)
            restored = ExperienceStore.from_dir(tmp).load_strict_snapshot().memories[0]

            self.assertTrue(created)
            self.assertEqual(stored, memory)
            self.assertEqual(restored, memory)
            self.assertEqual(store.path.read_text(encoding="utf-8"), experience_canonical_json(memory) + "\n")

            for memory_type in MemoryType:
                legacy_entry = MemoryEntry.build(
                    id=f"legacy-{memory_type.value}",
                    content=f"legacy {memory_type.value}",
                    type=memory_type,
                    scope=MemoryScope.PROJECT,
                    source="legacy",
                    token_count=3,
                    project_key="/repo",
                    created_at=NOW,
                )
                with self.subTest(memory_type=memory_type), self.assertRaises(TypeError):
                    store.add(legacy_entry)  # type: ignore[arg-type]

    def test_same_tier_dedup_preserves_first_record_and_cross_tier_coexists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            first = _memory("first", "same content", created_at=NOW)
            duplicate = _memory("duplicate", "same content", created_at=NOW - timedelta(days=1))
            skill = _memory("skill", "same content", tier=ExperienceTier.SKILL)

            stored_first, first_created = store.add(first)
            stored_duplicate, duplicate_created = store.add(duplicate)
            _, skill_created = store.add(skill)

            self.assertTrue(first_created)
            self.assertFalse(duplicate_created)
            self.assertEqual(stored_duplicate.id, stored_first.id)
            self.assertEqual(stored_duplicate.created_at, NOW)
            self.assertTrue(skill_created)
            self.assertEqual({memory.id for memory in store.all()}, {"first", "skill"})

    def test_visibility_tier_buckets_and_filters_match_repository_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            memories = (
                _memory("repo-tip", "repo tip", project_key="/repo"),
                _memory("other-tip", "other tip", project_key="/other"),
                _memory("repo-skill", "repo skill", tier=ExperienceTier.SKILL, project_key="/repo"),
                _memory("global-tip", "global tip", scope=MemoryScope.GLOBAL),
            )
            for memory in memories:
                store.add(memory)

            expected_tip_ids = tuple(sorted(
                memory.id
                for memory in memories
                if memory.tier == ExperienceTier.TIP
                and (memory.scope == MemoryScope.GLOBAL or memory.project_key == "/repo")
            ))
            self.assertEqual(
                store.visible_ids_for_tier(project_key="/repo", tier=ExperienceTier.TIP),
                expected_tip_ids,
            )
            self.assertEqual(
                {memory.id for memory in store.all(project_key="/repo")},
                {"repo-tip", "repo-skill", "global-tip"},
            )
            self.assertEqual(
                {memory.id for memory in store.all(tiers=frozenset({ExperienceTier.SKILL}))},
                {"repo-skill"},
            )

    def test_index_snapshot_is_revision_coupled_and_deeply_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            memory = _memory("tip", "pytest focused failure")
            store.add(memory)
            index = store.index_snapshot()
            snapshot = store.load_strict_snapshot()

            self.assertEqual(index.revision, snapshot.revision)
            self.assertEqual(index.by_id["tip"], memory)
            self.assertEqual(index.dedup_ids[next(iter(index.dedup_ids))], "tip")
            with self.assertRaises(TypeError):
                index.by_id["new"] = memory  # type: ignore[index]
            with self.assertRaises(TypeError):
                index.global_ids_by_tier[ExperienceTier.TIP] = ("new",)  # type: ignore[index]

    def test_invalidated_memory_remains_by_id_but_leaves_lexical_postings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            active = _memory("active", "pytest focused active")
            invalidated = _memory("invalid", "pytest focused invalid", invalidated=True)
            store.add(active)
            store.add(invalidated)
            index = store.index_snapshot()
            lexical = build_lexical_index(index)

            self.assertIn("invalid", index.by_id)
            self.assertIn("invalid", lexical.searchable_text_by_id)
            for term in experience_index_terms(invalidated):
                self.assertNotIn("invalid", lexical.postings_by_tier[ExperienceTier.TIP].get(term, ()))
            self.assertTrue(any(
                "active" in ids
                for ids in lexical.postings_by_tier[ExperienceTier.TIP].values()
            ))

    def test_add_replace_and_cross_process_style_refresh_publish_new_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ExperienceStore.from_dir(tmp)
            reader = ExperienceStore.from_dir(tmp)
            writer.add(_memory("first", "first pytest memory"))
            self.assertEqual(set(reader.index_snapshot().by_id), {"first"})

            writer.add(_memory("second", "second pytest memory"))
            self.assertEqual(set(reader.index_snapshot().by_id), {"first", "second"})

            snapshot = writer.load_strict_snapshot()
            replacement = replace(snapshot.memories[1], invalidated=True)
            writer.replace_all_atomically(
                (snapshot.memories[0], replacement),
                expected_revision=snapshot.revision,
            )
            refreshed = reader.index_snapshot()
            lexical = build_lexical_index(refreshed)
            self.assertEqual(refreshed.revision, writer.revision())
            self.assertFalse(any(
                "second" in ids
                for ids in lexical.postings_by_tier[ExperienceTier.TIP].values()
            ))

    def test_bulk_append_is_atomic_and_reports_existing_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            first = _memory("first", "first pytest memory")
            duplicate = _memory("duplicate", first.content)
            second = _memory("second", "second pytest memory")
            store.add(first)
            revision = store.revision()

            result = store.append_all_atomically(
                (duplicate, second),
                expected_revision=revision,
            )

            self.assertEqual(result.appended, (second,))
            self.assertEqual(result.duplicate_ids, (first.id,))
            self.assertEqual(result.revision, store.revision())
            self.assertEqual(
                [memory.id for memory in store.all(project_key="/repo")],
                ["first", "second"],
            )

    def test_bulk_append_rejects_stale_revision_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(_memory("first", "first pytest memory"))
            before = store.path.read_bytes()

            with self.assertRaises(MemoryStoreRevisionConflict):
                store.append_all_atomically(
                    (_memory("second", "second pytest memory"),),
                    expected_revision="sha256:stale",
                )

            self.assertEqual(store.path.read_bytes(), before)

    def test_attribution_update_preserves_non_attribution_fields_and_guards_scope_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            original = _memory("tip", "attributed pytest memory")
            store.add(original)
            record = MemoryAttributionRecord(
                memory_id="tip",
                tier="tip",
                memory_project_key="/repo",
                candidate_count=3,
                selected_count=2,
                not_selected_count=1,
                success_when_selected=1.0,
                success_when_candidate_not_selected=0.0,
                reward_when_selected=0.888888888,
                reward_when_candidate_not_selected=0.222222222,
                value=0.123456789,
                confidence=0.876543219,
                last_used=NOW.isoformat(),
            )

            self.assertFalse(store.update_attribution(
                record,
                project_key="/other",
                expected_tier=ExperienceTier.TIP,
            ))
            self.assertFalse(store.update_attribution(
                record,
                project_key="/repo",
                expected_tier=ExperienceTier.SKILL,
            ))
            before_update = datetime.now(timezone.utc)
            self.assertTrue(store.update_attribution(
                record,
                project_key="/repo",
                expected_tier=ExperienceTier.TIP,
            ))
            after_update = datetime.now(timezone.utc)
            updated = store.get("tip")
            self.assertIsNotNone(updated)
            assert updated is not None
            self.assertEqual(updated.attribution_value, 0.123457)
            self.assertEqual(updated.attribution_confidence, 0.876543)
            self.assertEqual(updated.reward_when_selected, 0.888889)
            self.assertEqual(updated.reward_when_candidate_not_selected, 0.222222)
            self.assertEqual(updated.last_used, NOW)
            self.assertIsNotNone(updated.attribution_updated_at)
            assert updated.attribution_updated_at is not None
            self.assertIsNotNone(updated.attribution_updated_at.utcoffset())
            self.assertLessEqual(before_update, updated.attribution_updated_at)
            self.assertLessEqual(updated.attribution_updated_at, after_update)
            for field_name in (
                "id",
                "content",
                "tier",
                "payload",
                "scope",
                "project_key",
                "created_at",
                "writer_confidence",
                "source_task",
                "run_id",
                "stream_id",
                "invalidated",
            ):
                self.assertEqual(getattr(updated, field_name), getattr(original, field_name))

    def test_permissive_load_skips_bad_lines_while_strict_load_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / EXPERIENCE_STORAGE_FILE
            good = experience_canonical_json(_memory("good", "good pytest memory"))
            path.write_text(good + "\n{bad}\n[]\n", encoding="utf-8")
            events: list[tuple[str, dict]] = []
            runtime = ExperienceStore(path, trace_sink=lambda name, payload: events.append((name, payload)))

            runtime.load()

            self.assertEqual([memory.id for memory in runtime.all()], ["good"])
            self.assertEqual(sum(name == "memory.load_skipped" for name, _ in events), 2)
            with self.assertRaises(MemoryStoreLoadError):
                ExperienceStore(path).load_strict_snapshot()

    def test_strict_load_rejects_unknown_schema_duplicate_id_and_dedup(self) -> None:
        first = experience_to_dict(_memory("first", "shared pytest memory"))
        second = experience_to_dict(_memory("second", "other pytest memory"))
        cases = {
            "schema": [dict(first, schema_version=1)],
            "duplicate-id": [first, dict(second, id="first")],
            "duplicate-dedup": [first, dict(first, id="second")],
            "session": [dict(first, scope="session")],
            "unknown-tier": [dict(first, tier="future")],
        }
        for name, payloads in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / EXPERIENCE_STORAGE_FILE
                path.write_text(
                    "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads),
                    encoding="utf-8",
                )
                with self.assertRaises(MemoryStoreLoadError):
                    ExperienceStore(path).load_strict_snapshot()

    def test_atomic_replace_rejects_stale_revision_and_duplicate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            first = _memory("first", "first memory")
            store.add(first)
            snapshot = store.load_strict_snapshot()
            before = store.path.read_bytes()

            with self.assertRaises(MemoryStoreRevisionConflict):
                store.replace_all_atomically(
                    (_memory("replacement", "replacement memory"),),
                    expected_revision="sha256:stale",
                )
            duplicate = _memory("duplicate", first.content)
            with self.assertRaises(MemoryStoreLoadError):
                store.replace_all_atomically(
                    (first, duplicate),
                    expected_revision=snapshot.revision,
                )
            self.assertEqual(store.path.read_bytes(), before)

    def test_persistent_post_commit_index_failure_keeps_old_snapshot_readable_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events: list[tuple[str, dict]] = []
            store = ExperienceStore.from_dir(
                tmp,
                trace_sink=lambda name, payload: events.append((name, payload)),
            )
            store.add(_memory("first", "first memory"))
            old_index = store.index_snapshot()
            import my_agent.memory.experience_store as store_module

            original_build = store_module._build_index_snapshot
            build_calls = 0

            def fail_after_precommit_build(*args, **kwargs):
                nonlocal build_calls
                build_calls += 1
                if build_calls >= 2:
                    raise RuntimeError("index failed")
                return original_build(*args, **kwargs)

            second = _memory("second", "second memory")
            expected_revision = experience_memories_revision((store.get("first"), second))  # type: ignore[arg-type]
            fresh = ExperienceStore(store.path)
            with patch.object(store_module, "_build_index_snapshot", side_effect=fail_after_precommit_build):
                with self.assertRaises(MemoryStorePostCommitError) as raised:
                    store.add(second)

                served_index = store.index_snapshot()
                self.assertEqual(served_index.revision, old_index.revision)
                self.assertEqual(set(served_index.by_id), {"first"})
                self.assertIs(store.index_snapshot(), served_index)
                self.assertEqual([memory.id for memory in store.all(project_key="/repo")], ["first"])
                retriever = ExperienceRetriever(now=NOW)
                hits = retriever.retrieve_candidates(
                    "first",
                    store=store,
                    project_key="/repo",
                    top_k_per_tier=5,
                )
                self.assertEqual([hit.entry.id for hit in hits], ["first"])
                self.assertEqual(retriever.last_metrics.repository_revision, old_index.revision)
                self.assertEqual(retriever.last_metrics.retrieval_fallback, "")
                self.assertIsNone(store.get("second"))

                failures = [
                    payload
                    for event, payload in events
                    if event == "memory.experience_index_rebuild_failed"
                ]
                self.assertTrue(failures)
                self.assertTrue(all(payload["using_previous_snapshot"] for payload in failures))
                self.assertTrue(all(payload["fallback_revision"] == old_index.revision for payload in failures))

                with self.assertRaises(MemoryStoreLoadError):
                    fresh.load()

            self.assertEqual(set(old_index.by_id), {"first"})
            self.assertEqual(raised.exception.expected_revision, expected_revision)
            recovered = store.index_snapshot()
            self.assertEqual(set(recovered.by_id), {"first", "second"})
            self.assertEqual(recovered.revision, experience_memories_revision(tuple(recovered.by_id.values())))


if __name__ == "__main__":
    unittest.main()
