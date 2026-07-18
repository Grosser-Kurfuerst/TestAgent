"""Shared repository persistence for legacy and formal Experience writers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from uuid import uuid4

from my_agent.memory.evolver.writing.contracts import (
    ExperienceWriteProposal,
    ExperienceWriteResult,
)
from my_agent.memory.experience.models import ExperienceCreatedBy, ExperienceMemory
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.experience.repository_rules import experience_dedup_key
from my_agent.memory.store_errors import (
    MemoryStorePostCommitError,
    MemoryStoreRevisionConflict,
)
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryScope, content_fingerprint

ExperienceIdFactory = Callable[[], str]


def legacy_experience_id() -> str:
    return f"exp_{uuid4().hex[:12]}"


def formal_experience_id() -> str:
    return f"exp-{uuid4().hex}"


class ExperienceRepositoryWriter:
    """Construct and atomically append validated proposals to one repository."""

    def __init__(
        self,
        *,
        store: ExperienceStore,
        project_key: str,
        id_factory: ExperienceIdFactory = legacy_experience_id,
    ) -> None:
        self.store = store
        self.project_key = project_key
        self.id_factory = id_factory

    def write(
        self,
        proposals: Sequence[ExperienceWriteProposal],
        *,
        source_task: str,
        run_id: str,
        stream_id: str,
        expected_revision: str | None = None,
    ) -> ExperienceWriteResult:
        normalized = tuple(proposals)
        entries = tuple(
            self._memory_from_proposal(
                proposal,
                source_task=source_task,
                run_id=run_id,
                stream_id=stream_id,
            )
            for proposal in normalized
        )
        try:
            appended = self.store.append_all_atomically(
                entries,
                expected_revision=expected_revision,
            )
        except MemoryStoreRevisionConflict:
            return ExperienceWriteResult(
                proposals=normalized,
                rejected=({"reason": "stale_repository_revision"},),
                error="stale_repository_revision",
            )
        except MemoryStorePostCommitError as exc:
            recovered = self._recover_post_commit(entries, exc)
            if recovered is not None:
                saved, duplicate_ids = recovered
                return ExperienceWriteResult(
                    proposals=normalized,
                    saved=saved,
                    duplicate_ids=duplicate_ids,
                    rejected=({"reason": "post_commit_audit_recovered"},),
                )
            raise
        except Exception as exc:  # noqa: BLE001 - repository writes are all-or-nothing
            return ExperienceWriteResult(
                proposals=normalized,
                rejected=({"reason": "atomic_repository_write_failed"},),
                error=f"{type(exc).__name__}: {exc}",
            )
        return ExperienceWriteResult(
            proposals=normalized,
            saved=appended.appended,
            duplicate_ids=appended.duplicate_ids,
        )

    def _memory_from_proposal(
        self,
        proposal: ExperienceWriteProposal,
        *,
        source_task: str,
        run_id: str,
        stream_id: str,
    ) -> ExperienceMemory:
        return ExperienceMemory(
            id=self.id_factory(),
            content=proposal.content,
            tier=proposal.tier,
            payload=proposal.payload,
            scope=MemoryScope.PROJECT,
            project_key=self.project_key,
            created_at=datetime.now(timezone.utc),
            token_count=estimate_tokens(proposal.content),
            fingerprint=content_fingerprint(proposal.content),
            source_task=source_task,
            run_id=run_id,
            stream_id=stream_id,
            created_by=ExperienceCreatedBy.WRITER,
            writer_confidence=proposal.confidence,
        )

    def _recover_post_commit(
        self,
        entries: tuple[ExperienceMemory, ...],
        error: MemoryStorePostCommitError,
    ) -> tuple[tuple[ExperienceMemory, ...], tuple[str, ...]] | None:
        try:
            snapshot = self.store.load_strict_snapshot()
        except Exception:
            return None
        if snapshot.revision != error.expected_revision:
            return None

        by_dedup = {
            experience_dedup_key(memory): memory
            for memory in snapshot.memories
        }
        saved: list[ExperienceMemory] = []
        duplicate_ids: list[str] = []
        for entry in entries:
            stored = by_dedup.get(experience_dedup_key(entry))
            if stored is None:
                return None
            if stored.id == entry.id:
                saved.append(entry)
            else:
                duplicate_ids.append(stored.id)
        return tuple(saved), tuple(duplicate_ids)


__all__ = [
    "ExperienceRepositoryWriter",
    "formal_experience_id",
    "legacy_experience_id",
]
