from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.llm.types import ChatResponse, MessageLike
from my_agent.memory.evolver import ExperienceCreatedBy, ExperienceTier, ExperienceWriteResult, build_experience_entry
from my_agent.memory.types import MemoryContext, MemoryEntry, MemoryScope, MemoryStatus, MemoryType


class NoopMemoryManager:
    """No-op memory implementation used for no-memory evaluation groups."""

    last_fact_extraction_error = ""
    last_fact_save_errors: list[str] = []

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
        self.project_key = str(getattr(config, "memory_project_key", "") or "").strip() or str(self.repo_path.resolve())
        self.session_id = session_id
        self.context_profile = ContextProfile.resolve(config, getattr(config, "model", ""))
        self._trace_sink = trace_sink
        self.last_evolver_selection = None

    def set_trace_sink(self, trace_sink: Any | None) -> tuple[Any | None, Any | None]:
        previous = (self._trace_sink, None)
        self._trace_sink = trace_sink
        return previous

    def restore_trace_sink(self, snapshot: tuple[Any | None, Any | None]) -> None:
        self._trace_sink = snapshot[0]

    def append_user_message(self, content: str, *, run_id: str = "") -> MemoryEntry:
        return _entry(content, source="user", run_id=run_id, project_key=self.project_key)

    def append_task_goal(self, goal: str, *, run_id: str = "") -> MemoryEntry:
        return _entry(
            goal,
            source="task_goal",
            run_id=run_id,
            project_key=self.project_key,
            metadata={"kind": "task_goal"},
        )

    def append_assistant_response(self, response: ChatResponse, *, run_id: str = "") -> MemoryEntry:
        return _entry(response.content or "", source="assistant", run_id=run_id, project_key=self.project_key)

    def append_tool_result(self, result: Any, *, run_id: str = "") -> MemoryEntry:
        return _entry(
            str(getattr(result, "content", "")),
            source=f"tool:{getattr(result, 'name', '')}",
            run_id=run_id,
            project_key=self.project_key,
        )

    def save_fact(
        self,
        content: str,
        *,
        scope: MemoryScope = MemoryScope.PROJECT,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryEntry, bool]:
        entry = _entry(
            content,
            type=MemoryType.FACT,
            scope=scope,
            source=source,
            project_key=self.project_key,
            metadata=metadata,
        )
        return entry, False

    def save_experience(
        self,
        content: str,
        *,
        tier: ExperienceTier | str,
        scope: MemoryScope = MemoryScope.PROJECT,
        source_task: str = "",
        created_by: ExperienceCreatedBy | str = ExperienceCreatedBy.MANUAL,
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryEntry, bool]:
        project_key = "" if scope == MemoryScope.GLOBAL else self.project_key
        entry = build_experience_entry(
            id=f"noop_exp_{uuid4().hex[:12]}",
            content=content,
            tier=tier,
            project_key=project_key,
            scope=scope,
            run_id=run_id,
            source_task=source_task,
            created_by=created_by,
            extra_metadata=metadata,
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
    ) -> MemoryContext:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

    def build_evolver_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)

    def retrieve_hits(
        self,
        query: str,
        *,
        limit: int | None = None,
        include_short_term: bool = False,
    ) -> list[Any]:
        return []

    def render_short_term_messages(self, *, max_tokens: int | None = None) -> list[MessageLike]:
        return []

    def trace_context_event(self, event: str, payload: dict[str, Any]) -> None:
        return None

    def compact_short_term(
        self,
        *,
        tools: list[dict[str, Any]],
        force: bool = False,
        focus: str = "",
        before_tokens: int | None = None,
        trace_completed: bool = True,
        trigger_tokens: int | None = None,
    ) -> Any | None:
        return None

    def status(self, *, include_entries: bool = True) -> MemoryStatus:
        return MemoryStatus(
            project_key=self.project_key,
            storage_path="disabled",
            short_term_entries=0,
            short_term_tokens=0,
            short_term_storage_token_limit=self.context_profile.short_term_storage_token_limit,
            long_term_entries=0,
            long_term_tokens=0,
            compression_trigger_ratio=self.config.memory_compression_trigger_ratio,
            retain_recent_turns=self.config.memory_retain_recent_turns,
            map_chunk_size=self.config.memory_map_chunk_size,
            long_term_entries_detail=(),
        )

    def extract_facts(self, *, reason: str, run_id: str = "") -> list[MemoryEntry]:
        return []

    def fork_for_task(self, *, session_id: str, run_id: str = "") -> "NoopMemoryManager":
        return NoopMemoryManager(
            config=self.config,
            repo_path=self.repo_path,
            session_id=session_id,
            trace_sink=self._trace_sink,
        )

    def clear_short_term(self, *, extract_first: bool = True, reason: str = "clear") -> tuple[int, list[MemoryEntry]]:
        return 0, []


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
