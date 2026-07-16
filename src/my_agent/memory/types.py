from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from my_agent.text_safety import sanitize_json_value

if TYPE_CHECKING:
    from my_agent.memory.evolver.types import ExperienceMemory


MemoryItemT = TypeVar("MemoryItemT")


class MemoryType(str, Enum):
    """Kind of memory entry."""

    CONVERSATION = "conversation"
    TOOL_RESULT = "tool_result"
    SUMMARY = "summary"
    FACT = "fact"


class MemoryScope(str, Enum):
    """Visibility scope of a long-term entry.

    ``global`` entries are visible to every repo; ``project`` entries are only
    visible when the active repo key matches. ``session`` is reserved for
    short-term, run-scoped entries that never reach the long-term store.
    """

    GLOBAL = "global"
    PROJECT = "project"
    SESSION = "session"


@dataclass(frozen=True)
class MemoryEntry:
    """A single immutable memory record.

    ``created_at`` is the authoritative birth time of an entry. When a
    long-term entry is re-saved with identical content, the original
    ``created_at`` is preserved so time-decay retrieval is not skewed by
    duplicate writes.
    """

    id: str
    content: str
    type: MemoryType
    scope: MemoryScope
    source: str
    created_at: datetime
    token_count: int
    project_key: str = ""
    run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,
            "scope": self.scope.value,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
            "project_key": self.project_key,
            "run_id": self.run_id,
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
        }
        return sanitize_json_value(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryEntry":
        content = str(payload.get("content", ""))
        fingerprint = str(payload.get("fingerprint") or "")
        if not fingerprint:
            # Older JSONL lines predate the fingerprint field, or a hand-edited
            # file left it blank. Recompute from content so the dedup key is
            # stable across reloads (plan §6).
            fingerprint = content_fingerprint(content)
        return cls(
            id=str(payload.get("id", "")),
            content=content,
            type=MemoryType(payload.get("type", MemoryType.CONVERSATION.value)),
            scope=MemoryScope(payload.get("scope", MemoryScope.PROJECT.value)),
            source=str(payload.get("source", "")),
            created_at=_parse_datetime(payload.get("created_at")),
            token_count=int(payload.get("token_count") or 0),
            project_key=str(payload.get("project_key", "")),
            run_id=str(payload.get("run_id", "")),
            metadata=dict(payload.get("metadata") or {}),
            fingerprint=fingerprint,
        )

    @classmethod
    def build(
        cls,
        *,
        id: str,
        content: str,
        type: MemoryType,
        scope: MemoryScope,
        source: str,
        token_count: int,
        created_at: datetime | None = None,
        project_key: str = "",
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "MemoryEntry":
        """Create an entry, filling ``fingerprint`` and a default timestamp."""
        return cls(
            id=id,
            content=content,
            type=type,
            scope=scope,
            source=source,
            created_at=created_at or _now_utc(),
            token_count=token_count,
            project_key=project_key,
            run_id=run_id,
            metadata=dict(metadata or {}),
            fingerprint=content_fingerprint(content),
        )


@dataclass(frozen=True)
class RetrievalHit(Generic[MemoryItemT]):
    """A scored retrieval result."""

    entry: MemoryItemT
    score: float
    matched_terms: tuple[str, ...]
    source_weight: float
    time_decay: float


@dataclass(frozen=True)
class MemoryContext(Generic[MemoryItemT]):
    """Rendered memory context ready to inject into the LLM prompt."""

    injected_text: str
    hits: list[RetrievalHit[MemoryItemT]]
    estimated_tokens: int


@dataclass(frozen=True)
class CompressionResult:
    """Outcome of a map-reduce compression attempt."""

    compacted: bool
    before_tokens: int
    after_tokens: int
    map_count: int = 0
    reduce_used: bool = False
    extracted_facts: int = 0
    fallback: bool = False


@dataclass(frozen=True)
class MemoryStatus:
    """Snapshot of the memory system for ``/memory`` and debugging."""

    project_key: str
    storage_path: str
    short_term_entries: int
    short_term_tokens: int
    short_term_storage_token_limit: int
    long_term_entries: int
    long_term_tokens: int
    compression_trigger_ratio: float
    retain_recent_turns: int
    map_chunk_size: int
    long_term_entries_detail: tuple["ExperienceMemory", ...] = ()


def normalize_content(text: str) -> str:
    """Normalize text for fingerprinting: casefold, collapse whitespace."""
    return " ".join(text.casefold().strip().split())


def content_fingerprint(text: str) -> str:
    """Stable content fingerprint used for long-term deduplication."""
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        dt = datetime.fromisoformat(value)
    else:
        dt = _now_utc()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _now_utc() -> datetime:
    # Centralized so tests can monkeypatch if needed.
    return datetime.now(timezone.utc)


__all__ = [
    "CompressionResult",
    "MemoryContext",
    "MemoryEntry",
    "MemoryScope",
    "MemoryType",
    "RetrievalHit",
    "content_fingerprint",
    "normalize_content",
]
