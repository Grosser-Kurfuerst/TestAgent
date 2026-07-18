from __future__ import annotations

import multiprocessing
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.experience.models import ExperienceMemory, ExperienceTier, TipPayload
from my_agent.memory.experience_store import (
    EXPERIENCE_STORAGE_FILE,
    ExperienceStore,
    MemoryStoreLockTimeout,
)
from my_agent.memory.types import MemoryScope, content_fingerprint


NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


def _memory(memory_id: str) -> ExperienceMemory:
    content = f"Concurrent typed memory {memory_id}"
    return ExperienceMemory(
        id=memory_id,
        content=content,
        tier=ExperienceTier.TIP,
        payload=TipPayload("concurrency", "warning", "concurrent writer"),
        scope=MemoryScope.PROJECT,
        project_key="/repo",
        created_at=NOW,
        token_count=6,
        fingerprint=content_fingerprint(content),
    )


def _process_add(path: str, memory_id: str, start, results) -> None:
    try:
        start.wait(10)
        store = ExperienceStore(path, lock_timeout_seconds=10)
        _, created = store.add(_memory(memory_id))
        results.put((memory_id, created, ""))
    except BaseException as exc:
        results.put((memory_id, False, f"{type(exc).__name__}: {exc}"))


class ExperienceStoreConcurrencyTests(unittest.TestCase):
    def test_two_processes_add_without_lost_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / EXPERIENCE_STORAGE_FILE)
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(target=_process_add, args=(path, f"process-{index}", start, results))
                for index in range(2)
            ]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(15)

            outcomes = [results.get(timeout=2) for _ in processes]
            self.assertEqual([process.exitcode for process in processes], [0, 0])
            self.assertEqual([error for _, _, error in outcomes], ["", ""])
            self.assertTrue(all(created for _, created, _ in outcomes))
            snapshot = ExperienceStore(path).load_strict_snapshot()
            self.assertEqual(
                {memory.id for memory in snapshot.memories},
                {"process-0", "process-1"},
            )
            self.assertEqual(
                ExperienceStore(path).index_snapshot().revision,
                snapshot.revision,
            )

    def test_writer_waits_for_shared_process_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / EXPERIENCE_STORAGE_FILE
            holder = ExperienceStore(path)
            writer = ExperienceStore(path, lock_timeout_seconds=2)
            done = threading.Event()

            def add_memory() -> None:
                writer.add(_memory("writer"))
                done.set()

            with holder.exclusive_process_lock():
                thread = threading.Thread(target=add_memory)
                thread.start()
                time.sleep(0.1)
                self.assertFalse(done.is_set())
            thread.join(3)

            self.assertTrue(done.is_set())
            self.assertEqual(set(writer.index_snapshot().by_id), {"writer"})

    def test_lock_timeout_is_finite_and_does_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / EXPERIENCE_STORAGE_FILE
            for timeout in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    ExperienceStore(path, lock_timeout_seconds=timeout)

            holder = ExperienceStore(path)
            writer = ExperienceStore(path, lock_timeout_seconds=0.05)
            errors: list[BaseException] = []

            def add_memory() -> None:
                try:
                    writer.add(_memory("writer"))
                except BaseException as exc:
                    errors.append(exc)

            with holder.exclusive_process_lock():
                thread = threading.Thread(target=add_memory)
                thread.start()
                thread.join(2)

            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], MemoryStoreLockTimeout)
            self.assertFalse(path.exists())

    def test_atomic_replace_failure_leaves_complete_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(_memory("first"))
            snapshot = store.load_strict_snapshot()
            before = store.path.read_bytes()

            with patch.object(Path, "replace", side_effect=OSError("replace unavailable")):
                with self.assertRaises(OSError):
                    store.replace_all_atomically(
                        (_memory("replacement"),),
                        expected_revision=snapshot.revision,
                    )

            self.assertEqual(store.path.read_bytes(), before)
            self.assertFalse(store.path.with_suffix(store.path.suffix + ".tmp").exists())
            self.assertEqual(
                [memory.id for memory in ExperienceStore(store.path).load_strict_snapshot().memories],
                ["first"],
            )


if __name__ == "__main__":
    unittest.main()
