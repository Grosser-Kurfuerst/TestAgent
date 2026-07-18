"""Disabled implementation of the runtime-facing memory service."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.llm.types import ChatResponse, MessageLike
from my_agent.memory.evolver.writing.contracts import ExperienceWriteResult
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperiencePayload,
    ExperienceTier,
)
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import (
    MemoryContext,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    content_fingerprint,
)
from my_agent.policy.identity import PolicyIdentity


class DisabledMemoryManager:
    """No-op memory service used when memory is disabled."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        repo_path: str | Path,
        session_id: str = "",
        trace_sink: Any | None = None,
    ) -> None:
        self.config = config
        self.repo_path = Path(repo_path)
        self.project_key = (
            str(getattr(config, "memory_project_key", "") or "").strip()
            or str(self.repo_path.resolve())
        )
        self.session_id = session_id
        self.context_profile = ContextProfile.resolve(
            config,
            getattr(config, "model", ""),
        )
        self._trace_sink = trace_sink
        self.last_evolver_selection = None

    @property
    def evolver_coordinator(self) -> None:
        return None

    def set_trace_sink(self, trace_sink: Any | None) -> tuple[Any | None, Any | None]:
        previous = (self._trace_sink, None)
        self._trace_sink = trace_sink
        return previous

    def restore_trace_sink(self, snapshot: tuple[Any | None, Any | None]) -> None:
        self._trace_sink = snapshot[0]

    def append_user_message(self, content: str, *, run_id: str = "") -> MemoryEntry:
        return _entry(
            content,
            source="user",
            run_id=run_id,
            project_key=self.project_key,
        )

    def append_task_goal(self, goal: str, *, run_id: str = "") -> MemoryEntry:
        return _entry(
            goal,
            source="task_goal",
            run_id=run_id,
            project_key=self.project_key,
            metadata={"kind": "task_goal"},
        )

    def append_assistant_response(
        self,
        response: ChatResponse,
        *,
        run_id: str = "",
    ) -> MemoryEntry:
        return _entry(
            response.content or "",
            source="assistant",
            run_id=run_id,
            project_key=self.project_key,
        )

    def append_tool_result(self, result: Any, *, run_id: str = "") -> MemoryEntry:
        return _entry(
            str(getattr(result, "content", "")),
            source=f"tool:{getattr(result, 'name', '')}",
            run_id=run_id,
            project_key=self.project_key,
        )

    def save_experience(
        self,
        *,
        tier: ExperienceTier,
        content: str,
        payload: ExperiencePayload,
        scope: MemoryScope = MemoryScope.PROJECT,
        source_task: str = "",
        run_id: str = "",
        stream_id: str = "",
        created_by: ExperienceCreatedBy = ExperienceCreatedBy.MANUAL,
        writer_confidence: float = 1.0,
    ) -> tuple[ExperienceMemory, bool]:
        project_key = "" if scope == MemoryScope.GLOBAL else self.project_key
        entry = ExperienceMemory(
            id=f"noop_exp_{uuid4().hex[:12]}",
            content=content,
            tier=tier,
            payload=payload,
            scope=scope,
            project_key=project_key,
            created_at=datetime.now(timezone.utc),
            token_count=estimate_tokens(content),
            fingerprint=content_fingerprint(content),
            source_task=source_task,
            run_id=run_id,
            stream_id=stream_id,
            created_by=created_by,
            writer_confidence=writer_confidence,
        )
        return entry, False

    def write_experiences_from_run(self, **_: Any) -> ExperienceWriteResult:
        return ExperienceWriteResult()

    def append_summary(
        self,
        content: str,
        *,
        source: str = "summary",
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        return _entry(
            content,
            type=MemoryType.SUMMARY,
            source=source,
            run_id=run_id,
            project_key=self.project_key,
            metadata=metadata,
        )

    def build_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        limit: int | None = None,
        include_short_term: bool = False,
    ) -> MemoryContext[ExperienceMemory]:
        del query, max_tokens, limit, include_short_term
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

    def build_evolver_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext[ExperienceMemory]:
        del query, max_tokens, top_k_per_tier, max_items
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

    def begin_formal_evolver_task(self, **kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("formal evolver task session is unavailable")

    def finalize_formal_evolver_task(self, episode: Any, outcome: Any) -> None:
        del episode, outcome
        return None

    def require_formal_runtime_binding(
        self,
        *,
        config: AgentConfig,
        policy_identity: PolicyIdentity,
        repo_path: Path | None,
    ) -> None:
        del config, policy_identity, repo_path
        raise ValueError("formal OPD runtime cannot use disabled memory")

    def render_short_term_messages(
        self,
        *,
        max_tokens: int | None = None,
    ) -> list[MessageLike]:
        del max_tokens
        return []

    def trace_context_event(self, event: str, payload: dict[str, Any]) -> None:
        del event, payload
        return None

    def compact_short_term(self, **kwargs: Any) -> None:
        del kwargs
        return None

    def status(self, *, include_entries: bool = True) -> MemoryStatus:
        del include_entries
        return MemoryStatus(
            project_key=self.project_key,
            storage_path="disabled",
            short_term_entries=0,
            short_term_tokens=0,
            short_term_storage_token_limit=(
                self.context_profile.short_term_storage_token_limit
            ),
            long_term_entries=0,
            long_term_tokens=0,
            compression_trigger_ratio=self.config.memory_compression_trigger_ratio,
            retain_recent_turns=self.config.memory_retain_recent_turns,
            map_chunk_size=self.config.memory_map_chunk_size,
            long_term_entries_detail=(),
        )

    def fork_for_task(
        self,
        *,
        session_id: str,
        run_id: str = "",
    ) -> "DisabledMemoryManager":
        del run_id
        return DisabledMemoryManager(
            config=self.config,
            repo_path=self.repo_path,
            session_id=session_id,
            trace_sink=self._trace_sink,
        )

    def clear_short_term(
        self,
        *,
        extract_first: bool = True,
        reason: str = "clear",
    ) -> tuple[int, list[MemoryEntry]]:
        del extract_first, reason
        return 0, []


NoopMemoryManager = DisabledMemoryManager


def _entry(
    content: str,
    *,
    type: MemoryType = MemoryType.CONVERSATION,
    scope: MemoryScope = MemoryScope.SESSION,
    source: str,
    run_id: str = "",
    project_key: str = "",
    metadata: dict[str, Any] | None = None,
) -> MemoryEntry:
    return MemoryEntry.build(
        id=f"noop_{uuid4().hex[:12]}",
        content=content,
        type=type,
        scope=scope,
        source=source,
        created_at=datetime.now(timezone.utc),
        token_count=0,
        project_key=project_key,
        run_id=run_id,
        metadata=metadata,
    )


__all__ = ["DisabledMemoryManager", "NoopMemoryManager"]
