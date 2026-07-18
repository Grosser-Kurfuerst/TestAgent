"""Stable runtime-facing memory service contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

from my_agent.config import AgentConfig
from my_agent.llm.types import ChatResponse, MessageLike
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperiencePayload,
    ExperienceTier,
)
from my_agent.memory.types import (
    CompressionResult,
    MemoryContext,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
)
from my_agent.policy.identity import PolicyIdentity

if TYPE_CHECKING:
    from my_agent.context import ContextProfile


@runtime_checkable
class MemoryService(Protocol):
    config: AgentConfig
    context_profile: ContextProfile
    project_key: str
    session_id: str

    @property
    def evolver_coordinator(self) -> Any | None: ...

    def set_trace_sink(self, trace_sink: Any | None) -> tuple[Any | None, Any | None]: ...

    def restore_trace_sink(self, snapshot: tuple[Any | None, Any | None]) -> None: ...

    def append_task_goal(self, goal: str, *, run_id: str = "") -> MemoryEntry: ...

    def append_user_message(self, content: str, *, run_id: str = "") -> MemoryEntry: ...

    def append_assistant_response(
        self,
        response: ChatResponse,
        *,
        run_id: str = "",
    ) -> MemoryEntry: ...

    def append_tool_result(self, result: Any, *, run_id: str = "") -> MemoryEntry: ...

    def append_summary(
        self,
        content: str,
        *,
        source: str = "summary",
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry: ...

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
    ) -> tuple[ExperienceMemory, bool]: ...

    def build_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        limit: int | None = None,
        include_short_term: bool = False,
    ) -> MemoryContext[ExperienceMemory]: ...

    def render_short_term_messages(
        self,
        *,
        max_tokens: int | None = None,
    ) -> list[MessageLike]: ...

    def compact_short_term(
        self,
        *,
        tools: list[dict[str, Any]],
        force: bool = False,
        focus: str = "",
        before_tokens: int | None = None,
        trace_completed: bool = True,
        trigger_tokens: int | None = None,
    ) -> CompressionResult | None: ...

    def clear_short_term(
        self,
        *,
        extract_first: bool = True,
        reason: str = "clear",
    ) -> tuple[int, list[MemoryEntry]]: ...

    def status(self, *, include_entries: bool = True) -> MemoryStatus: ...

    def fork_for_task(
        self,
        *,
        session_id: str,
        run_id: str = "",
    ) -> "MemoryService": ...

    def trace_context_event(self, event: str, payload: dict[str, Any]) -> None: ...

    def begin_formal_evolver_task(self, **kwargs: Any) -> Any: ...

    def write_experiences_from_run(self, **kwargs: Any) -> Any: ...

    def require_formal_runtime_binding(
        self,
        *,
        config: AgentConfig,
        policy_identity: PolicyIdentity,
        repo_path: Path | None,
    ) -> None: ...


__all__ = ["MemoryService"]
