from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from my_agent.memory.types import MemoryEntry, MemoryScope, content_fingerprint
from my_agent.text_safety import sanitize_json_value


TraceSink = Callable[[str, dict[str, Any]], None]

STORAGE_FILE = "long_term_memory.jsonl"


class LongTermMemoryStore:
    """Persistent cross-session fact store backed by a JSONL file.

    Semantics (see plan §6):

    * Loaded fully at startup; malformed lines are skipped and traced.
    * ``add()`` deduplicates by ``scope`` + ``project_key`` + ``fingerprint``
      (global scope ignores ``project_key``). Duplicate writes return the
      original entry and ``False`` so the caller can report "already exists".
    * The original ``created_at`` is always preserved on a hit, so time-decay
      retrieval is not polluted by repeated saves.
    * Persistence uses an atomic temp-file + ``Path.replace``.
    """

    def __init__(self, path: Path, *, trace_sink: TraceSink | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_sink = trace_sink
        self._entries: list[MemoryEntry] = []
        self._loaded = False
        self._lock = threading.RLock()

    @classmethod
    def from_dir(cls, directory: str | Path, *, trace_sink: TraceSink | None = None) -> "LongTermMemoryStore":
        return cls(Path(directory) / STORAGE_FILE, trace_sink=trace_sink)

    def load(self) -> None:
        """Load entries from disk. Safe to call once at startup."""
        with self._lock:
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
            if skipped:
                self._trace("memory.load_skipped_summary", {
                    "file": str(self.path),
                    "skipped": skipped,
                    "loaded": len(self._entries),
                })

    def add(self, entry: MemoryEntry) -> tuple[MemoryEntry, bool]:
        """Add an entry, deduplicating by scope/project/fingerprint.

        Returns ``(stored_entry, created)``. On a duplicate the original entry
        (with its original ``created_at``) is returned and ``created`` is
        ``False``; nothing is persisted.

        The fingerprint is recomputed from ``entry.content`` at this boundary
        (plan §6: "add() 先计算 fingerprint") so callers that construct a
        ``MemoryEntry`` directly — with a blank or untrusted fingerprint —
        cannot collide. ``MemoryEntry.build()`` already fills a matching
        fingerprint, so well-formed entries pass through unchanged.
        """
        with self._lock:
            self._ensure_loaded()
            entry = _normalize_fingerprint(entry)
            existing = self._find_duplicate(entry)
            if existing is not None:
                upgraded = _manual_upgrade(existing, entry)
                if upgraded is not None:
                    snapshot = list(self._entries)
                    self._replace_entry(existing, upgraded)
                    try:
                        self._persist()
                    except Exception:
                        self._entries = snapshot
                        raise
                    return upgraded, False
                return existing, False
            snapshot = list(self._entries)
            self._entries.append(entry)
            try:
                self._persist()
            except Exception:
                self._entries = snapshot
                raise
            return entry, True

    def all(self, *, project_key: str | None = None) -> list[MemoryEntry]:
        with self._lock:
            self._ensure_loaded()
            if project_key is None:
                return list(self._entries)
            return [entry for entry in self._entries if _is_visible(entry, project_key)]

    def search_candidates(self, *, project_key: str | None = None) -> list[MemoryEntry]:
        """Entries visible to ``project_key`` for retrieval scoring."""
        return self.all(project_key=project_key)

    def clear(self, *, scope: MemoryScope | None = None, project_key: str | None = None) -> int:
        with self._lock:
            self._ensure_loaded()
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
                self._persist()
            return removed

    def __len__(self) -> int:
        with self._lock:
            self._ensure_loaded()
            return len(self._entries)

    def _ensure_loaded(self) -> None:
        """Lazily load disk state before the first read or mutation.

        Without this, ``add()`` on a fresh store (``load()`` not called) would
        ``_persist()`` an in-memory list containing only the new entry and
        silently overwrite the existing file — a data-loss risk. Loading here
        guarantees the on-disk entries are in memory before any write, so
        ``_persist()`` rewrites the file with the full set.
        """
        if not self._loaded:
            self.load()

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
        seen: dict[tuple[str, str, str], int] = {}
        unique: list[MemoryEntry] = []
        for entry in self._entries:
            key = _dedup_key(entry)
            if key not in seen:
                seen[key] = len(unique)
                unique.append(entry)
                continue
            first = unique[seen[key]]
            if entry.created_at < first.created_at:
                unique[seen[key]] = entry
        self._entries = unique

    def _persist(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as file:
            for entry in self._entries:
                payload = sanitize_json_value(entry.to_dict())
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        tmp.replace(self.path)

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_sink is not None:
            try:
                self._trace_sink(event, payload)
            except Exception:
                # Tracing must never break memory operations.
                pass


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
    if existing.scope != candidate.scope:
        return False
    if existing.scope == MemoryScope.GLOBAL:
        return existing.fingerprint == candidate.fingerprint
    if existing.project_key != candidate.project_key:
        return False
    return existing.fingerprint == candidate.fingerprint


def _dedup_key(entry: MemoryEntry) -> tuple[str, str, str]:
    if entry.scope == MemoryScope.GLOBAL:
        return (entry.scope.value, "", entry.fingerprint)
    return (entry.scope.value, entry.project_key, entry.fingerprint)


def _matches_filter(entry: MemoryEntry, *, scope: MemoryScope | None, project_key: str | None) -> bool:
    if scope is not None and entry.scope != scope:
        return False
    if project_key is not None and entry.scope != MemoryScope.GLOBAL and entry.project_key != project_key:
        return False
    return True


__all__ = ["LongTermMemoryStore", "STORAGE_FILE"]
