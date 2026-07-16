from __future__ import annotations

import json
import math
import os
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

from filelock import FileLock, Timeout as FileLockTimeout

from my_agent.json_safety import loads_json_strict
from my_agent.memory.experience_attribution import replace_experience_attribution
from my_agent.memory.evolver.serialization import (
    EXPERIENCE_SCHEMA_VERSION,
    experience_canonical_json,
    experience_from_dict,
    experience_to_dict,
)
from my_agent.memory.evolver.repository_rules import (
    ExperienceDedupKey,
    experience_dedup_key,
    experience_memories_revision,
)
from my_agent.memory.evolver.types import ExperienceMemory, ExperienceTier
from my_agent.memory.experience_retrieval import experience_index_terms, experience_searchable_text
from my_agent.memory.store_errors import (
    MemoryStoreLoadError,
    MemoryStoreLockTimeout,
    MemoryStorePostCommitError,
    MemoryStoreRevisionConflict,
)
from my_agent.memory.types import MemoryScope

if TYPE_CHECKING:
    from my_agent.memory.evolver.attribution import MemoryAttributionRecord


TraceSink = Callable[[str, dict[str, Any]], None]
FileGeneration = tuple[int, int, int]
DedupKey = ExperienceDedupKey

EXPERIENCE_STORAGE_FILE = "experience_memory.jsonl"
EXPERIENCE_LOCK_FILE = ".experience_memory.lock"
LEGACY_LONG_TERM_STORAGE_FILE = "long_term_memory.jsonl"


class _ExperienceIndexBuildError(MemoryStoreLoadError):
    """Internal marker for a complete snapshot whose index could not be built."""


@dataclass(frozen=True)
class ExperienceStoreSnapshot:
    memories: tuple[ExperienceMemory, ...]
    raw_bytes: bytes
    revision: str


@dataclass(frozen=True)
class ExperienceStoreIndexSnapshot:
    revision: str
    by_id: Mapping[str, ExperienceMemory]
    dedup_ids: Mapping[DedupKey, str]
    global_ids_by_tier: Mapping[ExperienceTier, tuple[str, ...]]
    project_ids_by_tier: Mapping[tuple[str, ExperienceTier], tuple[str, ...]]
    lexical_postings_by_tier: Mapping[
        ExperienceTier,
        Mapping[str, frozenset[str]],
    ]
    searchable_text_by_id: Mapping[str, str]


class ExperienceStore:
    """Typed four-tier JSONL repository with revision-coupled indexes."""

    def __init__(
        self,
        path: str | Path,
        *,
        trace_sink: TraceSink | None = None,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self.path = Path(path)
        legacy_path = self.path.parent / LEGACY_LONG_TERM_STORAGE_FILE
        if self.path.name == EXPERIENCE_STORAGE_FILE and legacy_path.exists():
            raise MemoryStoreLoadError(
                "legacy long_term_memory.jsonl exists; stop memory processes and run "
                "the explicit four-tier memory reset before starting this runtime"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds < 0:
            raise ValueError("lock_timeout_seconds must be finite and non-negative")
        self._trace_sink = trace_sink
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._lock = threading.RLock()
        self.lock_path = self.path.parent / EXPERIENCE_LOCK_FILE
        self._process_lock = FileLock(self.lock_path)
        self._memories: list[ExperienceMemory] = []
        self._loaded = False
        self._loaded_generation: FileGeneration | None = None
        empty_revision = experience_memories_revision(())
        self._index = _build_index_snapshot((), revision=empty_revision)

    @classmethod
    def from_dir(
        cls,
        directory: str | Path,
        *,
        trace_sink: TraceSink | None = None,
        lock_timeout_seconds: float = 30.0,
    ) -> "ExperienceStore":
        return cls(
            Path(directory) / EXPERIENCE_STORAGE_FILE,
            trace_sink=trace_sink,
            lock_timeout_seconds=lock_timeout_seconds,
        )

    def load(self) -> None:
        """Load valid lines permissively for runtime startup."""
        with self.exclusive_process_lock():
            with self._lock:
                try:
                    self._load_permissive_locked()
                except _ExperienceIndexBuildError:
                    if not self._loaded:
                        raise
                    self._loaded_generation = None

    @contextmanager
    def exclusive_process_lock(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[None]:
        timeout = self._lock_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        try:
            with self._process_lock.acquire(timeout=timeout):
                yield
        except FileLockTimeout as exc:
            raise MemoryStoreLockTimeout(
                f"timed out acquiring experience store lock after {timeout:g}s"
            ) from exc

    def load_strict_snapshot(self) -> ExperienceStoreSnapshot:
        with self.exclusive_process_lock():
            with self._lock:
                return self._load_strict_snapshot_locked()

    def revision(self) -> str:
        return self.load_strict_snapshot().revision

    def add(self, memory: ExperienceMemory) -> tuple[ExperienceMemory, bool]:
        if not isinstance(memory, ExperienceMemory):
            raise TypeError("experience store only accepts ExperienceMemory")
        _validate_strict_memories((memory,))
        with self.exclusive_process_lock():
            with self._lock:
                self._load_strict_snapshot_locked()
                duplicate_id = self._index.dedup_ids.get(experience_dedup_key(memory))
                if duplicate_id is not None:
                    return self._index.by_id[duplicate_id], False
                self._commit_memories_locked((*self._memories, memory))
                return self._index.by_id[memory.id], True

    def get(self, memory_id: str) -> ExperienceMemory | None:
        self._refresh_for_read()
        with self._lock:
            return self._index.by_id.get(str(memory_id or ""))

    def all(
        self,
        *,
        project_key: str | None = None,
        tiers: frozenset[ExperienceTier] | None = None,
    ) -> list[ExperienceMemory]:
        if tiers is not None and any(not isinstance(tier, ExperienceTier) for tier in tiers):
            raise ValueError("tiers must contain only ExperienceTier values")
        self._refresh_for_read()
        with self._lock:
            return [
                memory
                for memory in self._memories
                if (project_key is None or _is_visible(memory, project_key))
                and (tiers is None or memory.tier in tiers)
            ]

    def visible_ids_for_tier(
        self,
        *,
        project_key: str,
        tier: ExperienceTier,
    ) -> tuple[str, ...]:
        if not isinstance(project_key, str):
            raise ValueError("project_key must be a string")
        if not isinstance(tier, ExperienceTier):
            raise ValueError("tier must be an ExperienceTier")
        self._refresh_for_read()
        with self._lock:
            visible = set(self._index.global_ids_by_tier.get(tier, ()))
            if project_key:
                visible.update(self._index.project_ids_by_tier.get((project_key, tier), ()))
            return tuple(sorted(visible))

    def index_snapshot(self) -> ExperienceStoreIndexSnapshot:
        self._refresh_for_read()
        with self._lock:
            return self._index

    def update_attribution(
        self,
        record: "MemoryAttributionRecord",
        *,
        project_key: str | None,
        expected_tier: ExperienceTier,
        all_projects: bool = False,
        updated_at: datetime | str | None = None,
    ) -> bool:
        if not isinstance(expected_tier, ExperienceTier):
            raise ValueError("expected_tier must be an ExperienceTier")
        memory_id = getattr(record, "memory_id", None)
        if not isinstance(memory_id, str) or not memory_id:
            return False
        record_tier = getattr(record, "tier", expected_tier.value)
        if record_tier != expected_tier.value:
            return False

        with self.exclusive_process_lock():
            with self._lock:
                self._load_strict_snapshot_locked()
                target = self._index.by_id.get(memory_id)
                if target is None or target.tier != expected_tier:
                    return False
                if not all_projects and project_key is not None and not _is_visible(target, project_key):
                    return False
                replacement = replace_experience_attribution(
                    target,
                    record,
                    updated_at=updated_at,
                )
                next_memories = [
                    replacement if memory.id == memory_id else memory
                    for memory in self._memories
                ]
                self._commit_memories_locked(next_memories)
                return True

    def replace_all_atomically(
        self,
        memories: Sequence[ExperienceMemory],
        *,
        expected_revision: str,
    ) -> str:
        replacements = tuple(memories)
        _validate_strict_memories(replacements)
        with self.exclusive_process_lock():
            with self._lock:
                current = self._load_strict_snapshot_locked()
                if current.revision != expected_revision:
                    raise MemoryStoreRevisionConflict(
                        f"experience revision changed: expected {expected_revision}, got {current.revision}"
                    )
                written = self._commit_memories_locked(replacements)
                return written.revision

    def __len__(self) -> int:
        self._refresh_for_read()
        with self._lock:
            return len(self._memories)

    def _commit_memories_locked(
        self,
        memories: Sequence[ExperienceMemory],
    ) -> ExperienceStoreSnapshot:
        replacements = tuple(memories)
        _validate_strict_memories(replacements)
        expected_revision = experience_memories_revision(replacements)
        try:
            self._persist_memories(replacements)
        except MemoryStorePostCommitError:
            self._loaded_generation = None
            raise
        try:
            written = self._load_strict_snapshot_locked()
            if written.revision != expected_revision:
                raise MemoryStoreLoadError("written experience revision mismatch")
            return written
        except Exception as exc:
            self._loaded_generation = None
            if not isinstance(exc, _ExperienceIndexBuildError):
                self._trace_index_rebuild_failed(
                    expected_revision=expected_revision,
                    error=exc,
                )
            raise MemoryStorePostCommitError(
                "experience repository committed but post-commit verification failed",
                expected_revision=expected_revision,
            ) from exc

    def _load_permissive_locked(self) -> None:
        memories: list[ExperienceMemory] = []
        skipped = 0
        raw_bytes = self.path.read_bytes() if self.path.exists() else b""
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryStoreLoadError("invalid experience JSONL: UnicodeDecodeError") from exc
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("expected object")
                memory = experience_from_dict(payload)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                skipped += 1
                self._trace_load_skip(line_no, exc)
                continue
            memories.append(memory)
        memories, duplicate_skips = _dedupe_permissive(memories)
        skipped += duplicate_skips
        revision = experience_memories_revision(memories)
        snapshot = ExperienceStoreSnapshot(tuple(memories), raw_bytes, revision)
        self._publish_snapshot_locked(snapshot)
        if skipped:
            self._trace(
                "memory.load_skipped_summary",
                {
                    "file": str(self.path),
                    "storage_file": self.path.name,
                    "experience_schema_version": EXPERIENCE_SCHEMA_VERSION,
                    "skipped": skipped,
                    "loaded": len(memories),
                },
            )

    def _load_strict_snapshot_locked(self) -> ExperienceStoreSnapshot:
        raw_bytes = self.path.read_bytes() if self.path.exists() else b""
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryStoreLoadError("invalid experience JSONL: UnicodeDecodeError") from exc

        memories: list[ExperienceMemory] = []
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = loads_json_strict(line)
                if not isinstance(payload, Mapping):
                    raise TypeError("expected object")
                memories.append(experience_from_dict(payload))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise MemoryStoreLoadError(
                    f"invalid experience JSONL at line {line_no}: {type(exc).__name__}"
                ) from exc

        _validate_strict_memories(memories)
        revision = experience_memories_revision(memories)
        snapshot = ExperienceStoreSnapshot(tuple(memories), raw_bytes, revision)
        self._publish_snapshot_locked(snapshot)
        return snapshot

    def _publish_snapshot_locked(self, snapshot: ExperienceStoreSnapshot) -> None:
        # Build every view first; only publish after the whole immutable index succeeds.
        try:
            next_index = _build_index_snapshot(snapshot.memories, revision=snapshot.revision)
            if next_index.revision != snapshot.revision:  # pragma: no cover - construction invariant
                raise MemoryStoreLoadError("experience index revision mismatch")
        except Exception as exc:
            self._trace_index_rebuild_failed(
                expected_revision=snapshot.revision,
                error=exc,
            )
            raise _ExperienceIndexBuildError(
                f"experience index build failed for revision {snapshot.revision}"
            ) from exc
        self._memories = list(snapshot.memories)
        self._index = next_index
        self._loaded = True
        self._loaded_generation = self._file_generation()

    def _refresh_for_read(self) -> None:
        generation = self._file_generation()
        with self._lock:
            if self._loaded and self._loaded_generation == generation:
                return
            if self._loaded and generation is None and not self.path.parent.exists():
                return
        with self.exclusive_process_lock():
            with self._lock:
                generation = self._file_generation()
                if self._loaded and self._loaded_generation == generation:
                    return
                try:
                    if self._loaded:
                        self._load_strict_snapshot_locked()
                    else:
                        self._load_permissive_locked()
                except _ExperienceIndexBuildError:
                    if not self._loaded:
                        raise
                    # Keep serving the previous complete snapshot. Leaving the
                    # generation unresolved makes the next read retry the rebuild.
                    self._loaded_generation = None

    def _persist_memories(self, memories: Sequence[ExperienceMemory]) -> None:
        tmp = _atomic_write_tmp_path(self.path)
        expected_revision = experience_memories_revision(memories)
        replaced = False
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                for memory in memories:
                    handle.write(experience_canonical_json(memory) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(self.path)
            replaced = True
        except Exception as exc:
            if replaced:
                raise MemoryStorePostCommitError(
                    "experience repository replaced but post-commit finalization failed",
                    expected_revision=expected_revision,
                ) from exc
            raise
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _file_generation(self) -> FileGeneration | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))

    def _trace_load_skip(self, line_no: int, error: BaseException) -> None:
        self._trace(
            "memory.load_skipped",
            {
                "file": str(self.path),
                "storage_file": self.path.name,
                "experience_schema_version": EXPERIENCE_SCHEMA_VERSION,
                "line": line_no,
                "error": f"{type(error).__name__}: {error}",
            },
        )

    def _trace_index_rebuild_failed(
        self,
        *,
        expected_revision: str,
        error: BaseException,
    ) -> None:
        self._trace(
            "memory.experience_index_rebuild_failed",
            {
                "experience_schema_version": EXPERIENCE_SCHEMA_VERSION,
                "storage_file": self.path.name,
                "expected_revision": expected_revision,
                "fallback_revision": self._index.revision if self._loaded else "",
                "using_previous_snapshot": self._loaded,
                "error": f"{type(error).__name__}: {error}",
            },
        )

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_sink is not None:
            try:
                self._trace_sink(event, payload)
            except Exception:
                pass


def _validate_strict_memories(memories: Sequence[ExperienceMemory]) -> None:
    ids: set[str] = set()
    dedup_ids: dict[DedupKey, str] = {}
    for memory in memories:
        if not isinstance(memory, ExperienceMemory):
            raise MemoryStoreLoadError("experience repository only accepts ExperienceMemory")
        # Re-serialize at the trust boundary to catch nested post-construction mutation.
        experience_to_dict(memory)
        if memory.id in ids:
            raise MemoryStoreLoadError(f"duplicate experience id: {memory.id}")
        ids.add(memory.id)
        key = experience_dedup_key(memory)
        previous = dedup_ids.get(key)
        if previous is not None:
            raise MemoryStoreLoadError(
                f"duplicate experience dedup identity: {previous}, {memory.id}"
            )
        dedup_ids[key] = memory.id


def _build_index_snapshot(
    memories: Sequence[ExperienceMemory],
    *,
    revision: str,
) -> ExperienceStoreIndexSnapshot:
    by_id: dict[str, ExperienceMemory] = {}
    dedup_ids: dict[DedupKey, str] = {}
    global_ids: dict[ExperienceTier, list[str]] = {tier: [] for tier in ExperienceTier}
    project_ids: dict[tuple[str, ExperienceTier], list[str]] = defaultdict(list)
    postings: dict[ExperienceTier, dict[str, set[str]]] = {
        tier: defaultdict(set) for tier in ExperienceTier
    }
    searchable_text: dict[str, str] = {}

    for memory in sorted(memories, key=lambda item: item.id):
        by_id[memory.id] = memory
        dedup_ids[experience_dedup_key(memory)] = memory.id
        if memory.scope == MemoryScope.GLOBAL:
            global_ids[memory.tier].append(memory.id)
        else:
            project_ids[(memory.project_key, memory.tier)].append(memory.id)
        text = experience_searchable_text(memory)
        searchable_text[memory.id] = text
        if memory.invalidated:
            continue
        for term in experience_index_terms(memory):
            postings[memory.tier][term].add(memory.id)

    frozen_postings: dict[ExperienceTier, Mapping[str, frozenset[str]]] = {}
    for tier in ExperienceTier:
        frozen_postings[tier] = MappingProxyType({
            term: frozenset(ids)
            for term, ids in sorted(postings[tier].items())
        })

    return ExperienceStoreIndexSnapshot(
        revision=revision,
        by_id=MappingProxyType(dict(by_id)),
        dedup_ids=MappingProxyType(dict(dedup_ids)),
        global_ids_by_tier=MappingProxyType({
            tier: tuple(sorted(global_ids[tier])) for tier in ExperienceTier
        }),
        project_ids_by_tier=MappingProxyType({
            key: tuple(sorted(ids)) for key, ids in sorted(
                project_ids.items(),
                key=lambda item: (item[0][0], item[0][1].value),
            )
        }),
        lexical_postings_by_tier=MappingProxyType(frozen_postings),
        searchable_text_by_id=MappingProxyType(dict(searchable_text)),
    )


def _dedupe_permissive(
    memories: Sequence[ExperienceMemory],
) -> tuple[list[ExperienceMemory], int]:
    unique: list[ExperienceMemory] = []
    id_positions: dict[str, int] = {}
    dedup_positions: dict[DedupKey, int] = {}
    skipped = 0
    for memory in memories:
        if memory.id in id_positions:
            skipped += 1
            continue
        key = experience_dedup_key(memory)
        position = dedup_positions.get(key)
        if position is None:
            id_positions[memory.id] = len(unique)
            dedup_positions[key] = len(unique)
            unique.append(memory)
            continue
        skipped += 1
        existing = unique[position]
        if memory.created_at < existing.created_at:
            id_positions.pop(existing.id, None)
            id_positions[memory.id] = position
            unique[position] = memory
    return unique, skipped


def _is_visible(memory: ExperienceMemory, project_key: str) -> bool:
    if memory.scope == MemoryScope.GLOBAL:
        return True
    return bool(project_key) and memory.project_key == project_key


def _atomic_write_tmp_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(target.suffix + ".tmp")


__all__ = [
    "EXPERIENCE_LOCK_FILE",
    "EXPERIENCE_STORAGE_FILE",
    "LEGACY_LONG_TERM_STORAGE_FILE",
    "ExperienceStore",
    "ExperienceStoreIndexSnapshot",
    "ExperienceStoreSnapshot",
    "MemoryStoreLoadError",
    "MemoryStoreLockTimeout",
    "MemoryStorePostCommitError",
    "MemoryStoreRevisionConflict",
    "experience_dedup_key",
    "experience_memories_revision",
]
