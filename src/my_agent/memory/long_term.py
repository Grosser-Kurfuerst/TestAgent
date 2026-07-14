from __future__ import annotations

import json
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from filelock import FileLock, Timeout as FileLockTimeout

from my_agent.json_safety import loads_json_strict
from my_agent.memory.types import MemoryEntry, MemoryScope, content_fingerprint
from my_agent.text_safety import sanitize_json_value


TraceSink = Callable[[str, dict[str, Any]], None]
FileGeneration = tuple[int, int, int]

STORAGE_FILE = "long_term_memory.jsonl"
LOCK_FILE = ".long_term_memory.lock"


class MemoryStoreLockTimeout(RuntimeError):
    """Raised when the directory-scoped process lock cannot be acquired."""


class MemoryStoreLoadError(ValueError):
    """Raised when a strict repository snapshot is malformed or ambiguous."""


class MemoryStoreRevisionConflict(RuntimeError):
    """Raised when an atomic replace was planned against a stale revision."""


class MemoryStorePostCommitError(RuntimeError):
    """Raised when replacement occurred but write verification did not finish."""

    def __init__(self, message: str, *, expected_revision: str) -> None:
        super().__init__(message)
        self.expected_revision = expected_revision


@dataclass(frozen=True)
class MemoryStoreSnapshot:
    entries: tuple[MemoryEntry, ...]
    raw_bytes: bytes
    revision: str


class LongTermMemoryStore:
    """Persistent cross-session fact store backed by a JSONL file.

    Semantics (see plan §6):

    * Loaded fully at startup; malformed lines are skipped and traced.
    * ``add()`` deduplicates ordinary memories by ``scope`` + ``project_key`` +
      ``fingerprint`` and experience memories by the same key plus
      ``evolver_tier`` (global scope ignores ``project_key``). Duplicate writes
      return the original entry and ``False`` so the caller can report
      "already exists".
    * The original ``created_at`` is always preserved on a hit, so time-decay
      retrieval is not polluted by repeated saves.
    * Persistence uses an atomic temp-file + ``Path.replace``.
    """

    def __init__(
        self,
        path: Path,
        *,
        trace_sink: TraceSink | None = None,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds < 0:
            raise ValueError("lock_timeout_seconds must be finite and non-negative")
        self._trace_sink = trace_sink
        self._entries: list[MemoryEntry] = []
        self._loaded = False
        self._loaded_generation: FileGeneration | None = None
        self._lock = threading.RLock()
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self.lock_path = self.path.parent / LOCK_FILE
        self._process_lock = FileLock(self.lock_path)

    @classmethod
    def from_dir(
        cls,
        directory: str | Path,
        *,
        trace_sink: TraceSink | None = None,
        lock_timeout_seconds: float = 30.0,
    ) -> "LongTermMemoryStore":
        return cls(
            Path(directory) / STORAGE_FILE,
            trace_sink=trace_sink,
            lock_timeout_seconds=lock_timeout_seconds,
        )

    def load(self) -> None:
        """Load entries from disk with the startup-compatible permissive parser."""
        with self.exclusive_process_lock():
            with self._lock:
                self._load_permissive_locked()

    @contextmanager
    def exclusive_process_lock(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[None]:
        """Hold the directory-scoped lock shared by every store mutation."""
        timeout = self._lock_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        try:
            with self._process_lock.acquire(timeout=timeout):
                yield
        except FileLockTimeout as exc:
            raise MemoryStoreLockTimeout(
                f"timed out acquiring memory store lock after {timeout:g}s"
            ) from exc

    def load_strict_snapshot(self) -> MemoryStoreSnapshot:
        """Load a fail-closed snapshot suitable for planning or mutation."""
        with self.exclusive_process_lock():
            with self._lock:
                return self._load_strict_snapshot_locked()

    def revision(self) -> str:
        return self.load_strict_snapshot().revision

    def replace_all_atomically(
        self,
        entries: Sequence[MemoryEntry],
        *,
        expected_revision: str,
    ) -> str:
        """Replace the full repository once after a strict revision check."""
        replacement_entries = tuple(entries)
        _validate_strict_entries(replacement_entries)
        expected_next_revision = memory_entries_revision(replacement_entries)
        with self.exclusive_process_lock():
            with self._lock:
                current = self._load_strict_snapshot_locked()
                if current.revision != expected_revision:
                    raise MemoryStoreRevisionConflict(
                        f"memory revision changed: expected {expected_revision}, got {current.revision}"
                    )
                previous_entries = list(self._entries)
                self._entries = list(replacement_entries)
                self._loaded = True
                try:
                    self._persist()
                except MemoryStorePostCommitError:
                    self._loaded_generation = None
                    raise
                except Exception:
                    self._entries = previous_entries
                    self._loaded = True
                    raise
                try:
                    written = self._load_strict_snapshot_locked()
                    if written.revision != expected_next_revision:
                        raise MemoryStoreLoadError("written memory revision mismatch")
                except Exception as exc:
                    self._entries = list(replacement_entries)
                    self._loaded = True
                    raise MemoryStorePostCommitError(
                        "memory replaced but post-commit verification failed",
                        expected_revision=expected_next_revision,
                    ) from exc
                return written.revision

    def add(self, entry: MemoryEntry) -> tuple[MemoryEntry, bool]:
        """Add an entry, deduplicating by scope/project/tier/fingerprint.

        Returns ``(stored_entry, created)``. On a duplicate the original entry
        (with its original ``created_at``) is returned and ``created`` is
        ``False``; nothing is persisted.

        The fingerprint is recomputed from ``entry.content`` at this boundary
        (plan §6: "add() 先计算 fingerprint") so callers that construct a
        ``MemoryEntry`` directly — with a blank or untrusted fingerprint —
        cannot collide. ``MemoryEntry.build()`` already fills a matching
        fingerprint, so well-formed entries pass through unchanged.
        """
        with self.exclusive_process_lock():
            with self._lock:
                self._load_strict_snapshot_locked()
                entry = _normalize_fingerprint(entry)
                existing = self._find_duplicate(entry)
                if existing is not None:
                    upgraded = _manual_upgrade(existing, entry)
                    if upgraded is not None:
                        snapshot = list(self._entries)
                        self._replace_entry(existing, upgraded)
                        try:
                            _validate_strict_entries(self._entries)
                            self._persist()
                        except MemoryStorePostCommitError:
                            self._loaded_generation = None
                            raise
                        except Exception:
                            self._entries = snapshot
                            raise
                        return upgraded, False
                    return existing, False
                snapshot = list(self._entries)
                self._entries.append(entry)
                try:
                    _validate_strict_entries(self._entries)
                    self._persist()
                except MemoryStorePostCommitError:
                    self._loaded_generation = None
                    raise
                except Exception:
                    self._entries = snapshot
                    raise
                return entry, True

    def all(self, *, project_key: str | None = None) -> list[MemoryEntry]:
        self._refresh_for_read()
        with self._lock:
            if project_key is None:
                return list(self._entries)
            return [entry for entry in self._entries if _is_visible(entry, project_key)]

    def search_candidates(self, *, project_key: str | None = None) -> list[MemoryEntry]:
        """Entries visible to ``project_key`` for retrieval scoring."""
        return self.all(project_key=project_key)

    def update_metadata_by_id(
        self,
        memory_id: str,
        metadata: dict[str, Any],
        *,
        project_key: str | None = None,
        expected_tier: str | None = None,
        all_projects: bool = False,
    ) -> bool:
        """Update only ``metadata`` for one entry, preserving identity fields.

        ``project_key`` applies the same visibility boundary as retrieval unless
        ``all_projects`` is explicit. This is used by offline attribution
        write-back so a global usage log cannot accidentally mutate another
        stream's project memories.
        """
        if not str(memory_id or ""):
            return False
        with self.exclusive_process_lock():
            with self._lock:
                self._load_strict_snapshot_locked()
                target_index: int | None = None
                for idx, entry in enumerate(self._entries):
                    if entry.id != memory_id:
                        continue
                    if not all_projects and project_key is not None and not _is_visible(entry, project_key):
                        continue
                    if expected_tier is not None and str(entry.metadata.get("evolver_tier") or "") != str(expected_tier):
                        continue
                    target_index = idx
                    break
                if target_index is None:
                    return False

                entry = self._entries[target_index]
                next_metadata = dict(entry.metadata)
                next_metadata.update(metadata)
                replacement = replace(entry, metadata=sanitize_json_value(next_metadata))
                snapshot = list(self._entries)
                self._entries[target_index] = replacement
                try:
                    _validate_strict_entries(self._entries)
                    self._persist()
                except MemoryStorePostCommitError:
                    self._loaded_generation = None
                    raise
                except Exception:
                    self._entries = snapshot
                    raise
                return True

    def clear(self, *, scope: MemoryScope | None = None, project_key: str | None = None) -> int:
        with self.exclusive_process_lock():
            with self._lock:
                self._load_strict_snapshot_locked()
                snapshot = list(self._entries)
                before = len(self._entries)
                if scope is None and project_key is None:
                    self._entries = []
                else:
                    self._entries = [
                        entry for entry in self._entries
                        if not _matches_filter(entry, scope=scope, project_key=project_key)
                    ]
                removed = before - len(self._entries)
                if removed:
                    try:
                        _validate_strict_entries(self._entries)
                        self._persist()
                    except MemoryStorePostCommitError:
                        self._loaded_generation = None
                        raise
                    except Exception:
                        self._entries = snapshot
                        raise
                return removed

    def __len__(self) -> int:
        self._refresh_for_read()
        with self._lock:
            return len(self._entries)

    def _load_permissive_locked(self) -> None:
        self._entries = []
        skipped = 0
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as file:
                for line_no, raw in enumerate(file, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            raise ValueError("not an object")
                        entry = _entry_from_payload(payload)
                    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
                        skipped += 1
                        self._trace("memory.load_skipped", {
                            "file": str(self.path),
                            "line": line_no,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        continue
                    self._entries.append(entry)
        self._dedupe_in_place()
        self._loaded = True
        self._loaded_generation = self._file_generation()
        if skipped:
            self._trace("memory.load_skipped_summary", {
                "file": str(self.path),
                "skipped": skipped,
                "loaded": len(self._entries),
            })

    def _load_strict_snapshot_locked(self) -> MemoryStoreSnapshot:
        raw_bytes = self.path.read_bytes() if self.path.exists() else b""
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryStoreLoadError("invalid memory JSONL: UnicodeDecodeError") from exc

        entries: list[MemoryEntry] = []
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = loads_json_strict(line)
                if not isinstance(payload, dict):
                    raise TypeError("expected object")
                content = str(payload.get("content") or "")
                declared_fingerprint = str(payload.get("fingerprint") or "")
                expected_fingerprint = content_fingerprint(content)
                if declared_fingerprint and declared_fingerprint != expected_fingerprint:
                    raise ValueError("fingerprint mismatch")
                entry = _entry_from_payload(payload)
            except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise MemoryStoreLoadError(
                    f"invalid memory JSONL at line {line_no}: {type(exc).__name__}"
                ) from exc
            entries.append(entry)

        try:
            _validate_strict_entries(entries)
        except MemoryStoreLoadError:
            raise
        except ValueError as exc:
            raise MemoryStoreLoadError(f"invalid memory repository: {type(exc).__name__}") from exc
        snapshot = MemoryStoreSnapshot(
            entries=tuple(entries),
            raw_bytes=raw_bytes,
            revision=memory_entries_revision(entries),
        )
        self._entries = list(entries)
        self._loaded = True
        self._loaded_generation = self._file_generation()
        return snapshot

    def _refresh_for_read(self) -> None:
        """Reload a resident cache after another process replaces the store."""
        generation = self._file_generation()
        with self._lock:
            if self._loaded and self._loaded_generation == generation:
                return
            if self._loaded and generation is None and not self.path.parent.exists():
                # A loaded store remains usable after an owning temporary
                # workspace is torn down. Deleting only the storage file while
                # its directory remains is still treated as a real generation
                # change and reloads to an empty repository.
                return

        # Mutations use the same process -> thread lock order. Recheck after
        # acquiring both locks so a writer cannot replace the file between the
        # generation comparison and the reload.
        with self.exclusive_process_lock():
            with self._lock:
                generation = self._file_generation()
                if self._loaded and self._loaded_generation == generation:
                    return
                if self._loaded:
                    self._load_strict_snapshot_locked()
                else:
                    self._load_permissive_locked()

    def _file_generation(self) -> FileGeneration | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))

    def _find_duplicate(self, entry: MemoryEntry) -> MemoryEntry | None:
        for existing in self._entries:
            if _is_duplicate(existing, entry):
                return existing
        return None

    def _replace_entry(self, existing: MemoryEntry, replacement: MemoryEntry) -> None:
        for idx, entry in enumerate(self._entries):
            if entry is existing or entry.id == existing.id:
                self._entries[idx] = replacement
                return

    def _dedupe_in_place(self) -> None:
        """Collapse accidental duplicates loaded from a hand-edited file.

        When two loaded entries share a dedup key, the one with the earliest
        ``created_at`` wins (plan §6: "重复保存不改变原始 created_at"). We
        track the winning entry's position in ``unique`` so a later, older
        duplicate actually replaces the earlier, newer one in the output list
        — not just in the bookkeeping dict.
        """
        seen: dict[tuple[str, str, str, str], int] = {}
        unique: list[MemoryEntry] = []
        for entry in self._entries:
            key = memory_dedup_key(entry)
            if key not in seen:
                seen[key] = len(unique)
                unique.append(entry)
                continue
            first = unique[seen[key]]
            if entry.created_at < first.created_at:
                unique[seen[key]] = entry
        self._entries = unique

    def _persist(self) -> None:
        tmp = _atomic_write_tmp_path(self.path)
        expected_revision = ""
        replaced = False
        try:
            expected_revision = memory_entries_revision(self._entries)
            with tmp.open("w", encoding="utf-8") as file:
                for entry in self._entries:
                    payload = sanitize_json_value(entry.to_dict())
                    file.write(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                file.flush()
                os.fsync(file.fileno())
            tmp.replace(self.path)
            replaced = True
            self._loaded_generation = self._file_generation()
        except Exception as exc:
            if replaced:
                self._loaded_generation = None
                raise MemoryStorePostCommitError(
                    "memory replaced but post-commit finalization failed",
                    expected_revision=expected_revision,
                ) from exc
            raise
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                # Temp cleanup cannot change the commit status of the store.
                pass

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_sink is not None:
            try:
                self._trace_sink(event, payload)
            except Exception:
                # Tracing must never break memory operations.
                pass


def _atomic_write_tmp_path(path: str | Path) -> Path:
    """Return the fixed sidecar used for atomic replacement of ``path``."""
    target = Path(path)
    return target.with_suffix(target.suffix + ".tmp")


def _entry_from_payload(payload: dict[str, Any]) -> MemoryEntry:
    return MemoryEntry.from_dict(payload)


def _normalize_fingerprint(entry: MemoryEntry) -> MemoryEntry:
    """Return ``entry`` with a fingerprint derived from its content.

    ``MemoryEntry`` is frozen and its ``fingerprint`` defaults to ``""``, so a
    caller that builds one directly (bypassing ``MemoryEntry.build()``) would
    otherwise dedup against every other blank-fingerprint entry. Recomputing
    here keeps the dedup contract honest at the store boundary. Entries built
    via ``MemoryEntry.build()`` already carry a matching fingerprint and are
    returned untouched to avoid a needless copy.
    """
    expected = content_fingerprint(entry.content)
    if entry.fingerprint == expected:
        return entry
    return replace(entry, fingerprint=expected)


def _manual_upgrade(existing: MemoryEntry, candidate: MemoryEntry) -> MemoryEntry | None:
    """Upgrade an auto-extracted duplicate when the user manually saves it.

    Manual saves are treated as explicit user confirmation. The stable identity
    and original timestamp are preserved so dedupe/time-decay semantics remain
    intact, but the content/source/metadata are updated to the manual entry.
    """
    if not _is_manual_entry(candidate) or not _is_auto_extracted_entry(existing):
        return None
    return replace(
        candidate,
        id=existing.id,
        created_at=existing.created_at,
        fingerprint=existing.fingerprint,
    )


def _is_manual_entry(entry: MemoryEntry) -> bool:
    return entry.source == "manual" or entry.metadata.get("source") == "manual"


def _is_auto_extracted_entry(entry: MemoryEntry) -> bool:
    return entry.source == "fact_extractor" or entry.metadata.get("source") == "fact_extractor"


def _is_visible(entry: MemoryEntry, project_key: str) -> bool:
    if entry.scope == MemoryScope.GLOBAL:
        return True
    return bool(project_key) and entry.project_key == project_key


def _is_duplicate(existing: MemoryEntry, candidate: MemoryEntry) -> bool:
    return memory_dedup_key(existing) == memory_dedup_key(candidate)


def memory_dedup_key(entry: MemoryEntry) -> tuple[str, str, str, str]:
    """Return the repository-wide dedup identity for ``entry``.

    Experience tiers are separate repositories in OPD-Evolver semantics, so a
    promoted skill may coexist with its source tip/trajectory even when their
    normalized contents (and fingerprints) are identical. Ordinary memories
    retain the original scope/project/fingerprint behavior via an empty tier
    component.
    """
    project_key = "" if entry.scope == MemoryScope.GLOBAL else entry.project_key
    raw_tier = str(entry.metadata.get("evolver_tier") or "")
    tier = raw_tier if raw_tier in {"trajectory", "tip", "skill", "tool"} else ""
    return (entry.scope.value, project_key, tier, entry.fingerprint)


def memory_entries_revision(entries: Sequence[MemoryEntry]) -> str:
    payload = [entry.to_dict() for entry in sorted(entries, key=lambda item: item.id)]
    canonical = json.dumps(
        sanitize_json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"


def _validate_strict_entries(entries: Sequence[MemoryEntry]) -> None:
    ids: set[str] = set()
    dedup_keys: dict[tuple[str, str, str, str], str] = {}
    for entry in entries:
        if not entry.id:
            raise MemoryStoreLoadError("memory entry id must not be empty")
        if entry.id in ids:
            raise MemoryStoreLoadError(f"duplicate memory id: {entry.id}")
        ids.add(entry.id)
        expected_fingerprint = content_fingerprint(entry.content)
        if entry.fingerprint != expected_fingerprint:
            raise MemoryStoreLoadError(f"fingerprint mismatch for memory id: {entry.id}")
        key = memory_dedup_key(entry)
        previous = dedup_keys.get(key)
        if previous is not None:
            raise MemoryStoreLoadError(
                f"duplicate memory dedup identity: {previous}, {entry.id}"
            )
        dedup_keys[key] = entry.id


def _matches_filter(entry: MemoryEntry, *, scope: MemoryScope | None, project_key: str | None) -> bool:
    if scope is not None and entry.scope != scope:
        return False
    if project_key is not None and entry.scope != MemoryScope.GLOBAL and entry.project_key != project_key:
        return False
    return True


__all__ = [
    "LOCK_FILE",
    "STORAGE_FILE",
    "LongTermMemoryStore",
    "MemoryStoreLoadError",
    "MemoryStoreLockTimeout",
    "MemoryStorePostCommitError",
    "MemoryStoreRevisionConflict",
    "MemoryStoreSnapshot",
    "memory_dedup_key",
    "memory_entries_revision",
]
