from __future__ import annotations

import json
import multiprocessing
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.long_term import (
    LongTermMemoryStore,
    MemoryStoreLoadError,
    MemoryStoreLockTimeout,
    MemoryStoreRevisionConflict,
    memory_entries_revision,
)
from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    MaintenanceApplyStatus,
    apply_maintenance_plan,
    build_experience_entry,
    build_maintenance_plan,
)
from my_agent.memory.types import (
    MemoryEntry,
    MemoryScope,
    MemoryType,
    content_fingerprint,
)


NOW = datetime(2026, 7, 11, tzinfo=timezone.utc)


def _fact(memory_id: str, content: str, *, project_key: str = "/repo") -> MemoryEntry:
    return MemoryEntry.build(
        id=memory_id,
        content=content,
        type=MemoryType.FACT,
        scope=MemoryScope.PROJECT,
        source="manual",
        token_count=4,
        project_key=project_key,
        created_at=NOW,
    )


def _process_add(path: str, memory_id: str, start, results) -> None:
    try:
        start.wait(10)
        store = LongTermMemoryStore(Path(path), lock_timeout_seconds=10)
        store.add(_fact(memory_id, f"process fact {memory_id}"))
        results.put((memory_id, ""))
    except BaseException as exc:
        results.put((memory_id, f"{type(exc).__name__}: {exc}"))


def _process_add_runtime_experience(
    path: str,
    start,
    attempting,
    acquired,
    results,
) -> None:
    try:
        start.wait(10)
        store = LongTermMemoryStore(Path(path), lock_timeout_seconds=10)
        attempting.set()
        with store.exclusive_process_lock():
            acquired.set()
            _, created = store.add(build_experience_entry(
                id="runtime-writer-tip",
                content="Runtime writer preserves this concurrent experience.",
                tier="tip",
                project_key="/repo",
                created_at=NOW,
                created_by=ExperienceCreatedBy.WRITER,
            ))
        results.put((created, ""))
    except BaseException as exc:
        results.put((False, f"{type(exc).__name__}: {exc}"))


class StrictSnapshotTests(unittest.TestCase):
    def test_resident_reader_refreshes_after_another_store_replaces_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = LongTermMemoryStore.from_dir(tmp)
            writer.add(_fact("stale", "stale resident fact"))
            reader = LongTermMemoryStore.from_dir(tmp)
            self.assertEqual([entry.id for entry in reader.all()], ["stale"])

            writer.clear()

            self.assertEqual(writer.all(), [])
            self.assertEqual(reader.all(), [])
            self.assertEqual(len(reader), 0)

    def test_resident_reader_fails_closed_after_corrupt_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = LongTermMemoryStore.from_dir(tmp)
            writer.add(_fact("resident", "resident fact"))
            reader = LongTermMemoryStore.from_dir(tmp)
            self.assertEqual([entry.id for entry in reader.all()], ["resident"])
            corrupt = writer.path.with_suffix(".replacement")
            corrupt.write_bytes(b"{bad}\n")
            corrupt.replace(writer.path)

            with self.assertRaises(MemoryStoreLoadError):
                reader.all()
            with self.assertRaises(MemoryStoreLoadError):
                reader.all()

            recovered = writer.path.with_suffix(".recovered")
            recovered.write_text(
                json.dumps(_fact("recovered", "recovered fact").to_dict()) + "\n",
                encoding="utf-8",
            )
            recovered.replace(writer.path)

            self.assertEqual([entry.id for entry in reader.all()], ["recovered"])

    def test_add_rejects_duplicate_id_before_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore.from_dir(tmp)
            store.add(_fact("same-id", "first fact"))
            before = store.path.read_bytes()

            with self.assertRaises(MemoryStoreLoadError):
                store.add(_fact("same-id", "different fact"))

            self.assertEqual(store.path.read_bytes(), before)
            self.assertEqual([entry.content for entry in store.all()], ["first fact"])

    def test_metadata_update_rejects_tier_dedup_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore.from_dir(tmp)
            tip = build_experience_entry(
                id="tip",
                content="same experience",
                tier="tip",
                project_key="/repo",
                created_at=NOW,
                created_by=ExperienceCreatedBy.WRITER,
            )
            skill = build_experience_entry(
                id="skill",
                content="same experience",
                tier="skill",
                project_key="/repo",
                created_at=NOW,
                created_by=ExperienceCreatedBy.WRITER,
            )
            store.add(tip)
            store.add(skill)
            before = store.path.read_bytes()

            with self.assertRaises(MemoryStoreLoadError):
                store.update_metadata_by_id(
                    "tip",
                    {"evolver_tier": "skill"},
                    project_key="/repo",
                    expected_tier="tip",
                )

            self.assertEqual(store.path.read_bytes(), before)
            self.assertEqual(
                {entry.id: entry.metadata["evolver_tier"] for entry in store.all()},
                {"tip": "tip", "skill": "skill"},
            )

    def test_strict_load_rejects_corruption_duplicates_and_forged_fingerprint(self) -> None:
        first = _fact("first", "shared fact").to_dict()
        second = _fact("second", "other fact").to_dict()
        cases = {
            "bad-json": b"{bad}\n",
            "non-object": b"[]\n",
            "bad-entry": json.dumps({**first, "scope": "future"}).encode() + b"\n",
            "duplicate-id": (
                json.dumps(first) + "\n" + json.dumps({**second, "id": "first"}) + "\n"
            ).encode(),
            "duplicate-dedup": (
                json.dumps(first) + "\n" + json.dumps({**first, "id": "second"}) + "\n"
            ).encode(),
            "forged-fingerprint": (
                json.dumps({**first, "fingerprint": "f" * 64}) + "\n"
            ).encode(),
            "duplicate-json-key": (
                json.dumps(first)[:-1] + ', "id": "hidden-id"}\n'
            ).encode(),
        }
        for name, raw in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "long_term_memory.jsonl"
                path.write_bytes(raw)
                store = LongTermMemoryStore(path)

                with self.assertRaises(MemoryStoreLoadError):
                    store.load_strict_snapshot()

                self.assertEqual(path.read_bytes(), raw)

    def test_blank_legacy_fingerprint_is_stably_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            payload = _fact("legacy", "legacy fact").to_dict()
            payload["fingerprint"] = ""
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            snapshot = LongTermMemoryStore(path).load_strict_snapshot()

            self.assertEqual(
                snapshot.entries[0].fingerprint,
                content_fingerprint("legacy fact"),
            )
            self.assertEqual(snapshot.raw_bytes, path.read_bytes())

    def test_non_finite_metadata_is_rejected_before_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore.from_dir(tmp)
            baseline = _fact("baseline", "baseline fact")
            store.add(baseline)
            before = store.path.read_bytes()
            invalid = replace(
                _fact("invalid", "invalid numeric metadata"),
                metadata={"score": float("nan")},
            )

            with self.assertRaises(ValueError):
                store.add(invalid)
            with self.assertRaises(ValueError):
                memory_entries_revision([invalid])

            self.assertEqual(store.path.read_bytes(), before)
            self.assertEqual([entry.id for entry in store.all()], ["baseline"])

    def test_temp_cleanup_failure_does_not_change_committed_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore.from_dir(tmp)
            original_unlink = Path.unlink

            def fail_tmp_cleanup(path: Path, *args, **kwargs):
                if path.name.endswith(".tmp"):
                    raise OSError("cleanup unavailable")
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", new=fail_tmp_cleanup):
                stored, created = store.add(_fact("committed", "committed fact"))

            self.assertTrue(created)
            self.assertEqual(stored.id, "committed")
            self.assertEqual([entry.id for entry in store.all()], ["committed"])
            self.assertEqual(
                [entry.id for entry in LongTermMemoryStore(store.path).all()],
                ["committed"],
            )

    def test_revision_covers_every_persisted_field(self) -> None:
        base = _fact("base", "base content")
        base_revision = memory_entries_revision([base])
        variants = [
            replace(base, id="changed-id"),
            replace(base, content="changed", fingerprint=content_fingerprint("changed")),
            replace(base, type=MemoryType.SUMMARY),
            replace(base, scope=MemoryScope.GLOBAL),
            replace(base, source="writer"),
            replace(base, created_at=NOW + timedelta(seconds=1)),
            replace(base, token_count=99),
            replace(base, project_key="other-project"),
            replace(base, run_id="run-2"),
            replace(base, metadata={"key": "value"}),
            replace(base, fingerprint="f" * 64),
        ]

        for variant in variants:
            self.assertNotEqual(memory_entries_revision([variant]), base_revision)

    def test_mutation_strict_reload_refuses_to_rewrite_bad_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            good = json.dumps(_fact("good", "good fact").to_dict())
            raw = (good + "\n{bad}\n").encode()
            path.write_bytes(raw)
            store = LongTermMemoryStore(path)
            store.load()

            with self.assertRaises(MemoryStoreLoadError):
                store.add(_fact("new", "new fact"))

            self.assertEqual(path.read_bytes(), raw)

    def test_atomic_replace_rejects_stale_revision_and_duplicate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore.from_dir(tmp)
            store.add(_fact("first", "first fact"))
            snapshot = store.load_strict_snapshot()
            before = store.path.read_bytes()

            with self.assertRaises(MemoryStoreRevisionConflict):
                store.replace_all_atomically(
                    [_fact("replacement", "replacement fact")],
                    expected_revision="sha256:stale",
                )
            self.assertEqual(store.path.read_bytes(), before)

            duplicate = _fact("duplicate", "first fact")
            with self.assertRaises(MemoryStoreLoadError):
                store.replace_all_atomically(
                    [snapshot.entries[0], duplicate],
                    expected_revision=snapshot.revision,
                )
            self.assertEqual(store.path.read_bytes(), before)


class ProcessLockTests(unittest.TestCase):
    def test_process_lock_timeout_must_be_finite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(constructor_timeout=value), self.assertRaises(ValueError):
                    LongTermMemoryStore(path, lock_timeout_seconds=value)

            store = LongTermMemoryStore(path)
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(lock_timeout=value), self.assertRaises(ValueError):
                    with store.exclusive_process_lock(timeout_seconds=value):
                        self.fail("non-finite timeout acquired the process lock")

    def test_two_processes_add_without_lost_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "long_term_memory.jsonl")
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_process_add,
                    args=(path, f"process-{index}", start, results),
                )
                for index in range(2)
            ]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(15)

            outcomes = [results.get(timeout=2) for _ in processes]
            self.assertEqual([process.exitcode for process in processes], [0, 0])
            self.assertEqual([error for _, error in outcomes], ["", ""])
            snapshot = LongTermMemoryStore(Path(path)).load_strict_snapshot()
            self.assertEqual(
                {entry.id for entry in snapshot.entries},
                {"process-0", "process-1"},
            )

    def test_writer_waits_for_shared_process_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            holder = LongTermMemoryStore(path)
            writer = LongTermMemoryStore(path, lock_timeout_seconds=2)
            done = threading.Event()

            def add_entry() -> None:
                writer.add(_fact("writer", "writer fact"))
                done.set()

            with holder.exclusive_process_lock():
                thread = threading.Thread(target=add_entry)
                thread.start()
                time.sleep(0.1)
                self.assertFalse(done.is_set())
            thread.join(3)

            self.assertTrue(done.is_set())
            self.assertEqual({entry.id for entry in writer.all()}, {"writer"})

    def test_runtime_writer_and_maintenance_do_not_lose_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LongTermMemoryStore.from_dir(root)
            store.add(build_experience_entry(
                id="invalidated-tip",
                content="This invalidated experience should be removed.",
                tier="tip",
                project_key="/repo",
                created_at=NOW,
                created_by=ExperienceCreatedBy.WRITER,
                extra_metadata={"maintenance_invalidated": True},
            ))
            snapshot = store.load_strict_snapshot()
            plan = build_maintenance_plan(
                entries=snapshot.entries,
                attribution={},
                repository_revision=snapshot.revision,
                project_key="/repo",
                as_of=NOW,
            )
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            attempting = context.Event()
            acquired = context.Event()
            results = context.Queue()
            writer = context.Process(
                target=_process_add_runtime_experience,
                args=(str(store.path), start, attempting, acquired, results),
            )
            original_load = store.load_strict_snapshot
            writer_started = False

            def load_while_writer_waits():
                nonlocal writer_started
                current = original_load()
                if not writer_started:
                    writer_started = True
                    writer.start()
                    start.set()
                    self.assertTrue(attempting.wait(10))
                    self.assertFalse(acquired.wait(0.1))
                return current

            with patch.object(
                store,
                "load_strict_snapshot",
                side_effect=load_while_writer_waits,
            ):
                apply_result = apply_maintenance_plan(
                    store=store,
                    plan=plan,
                    backup_dir=root / "maintenance_backups",
                    history_path=root / "maintenance_history.jsonl",
                )
                self.assertEqual(
                    apply_result.status,
                    MaintenanceApplyStatus.COMMITTED,
                )

            writer.join(15)
            self.assertEqual(writer.exitcode, 0)
            self.assertTrue(acquired.is_set())
            created, error = results.get(timeout=2)
            self.assertTrue(created)
            self.assertEqual(error, "")
            final_ids = {
                entry.id for entry in store.load_strict_snapshot().entries
            }
            self.assertNotIn("invalidated-tip", final_ids)
            self.assertIn("runtime-writer-tip", final_ids)

    def test_writer_lock_timeout_fails_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_term_memory.jsonl"
            holder = LongTermMemoryStore(path)
            writer = LongTermMemoryStore(path, lock_timeout_seconds=0.05)
            errors: list[BaseException] = []

            def add_entry() -> None:
                try:
                    writer.add(_fact("writer", "writer fact"))
                except BaseException as exc:
                    errors.append(exc)

            with holder.exclusive_process_lock():
                thread = threading.Thread(target=add_entry)
                thread.start()
                thread.join(2)

            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], MemoryStoreLockTimeout)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
