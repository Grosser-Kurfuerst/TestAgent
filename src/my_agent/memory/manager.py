from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.llm import AgentLLM
from my_agent.llm.types import ChatResponse, MessageLike
from my_agent.memory.experience.models import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperiencePayload,
    ExperienceTier,
)
from my_agent.memory.evolver.selection.contracts import SelectionResult
from my_agent.memory.evolver.writing.contracts import ExperienceWriteResult
from my_agent.memory.evolver.runtime.contracts import EvolverRuntime
from my_agent.memory.evolver.task_session import (
    AgentEpisodeArtifact,
    EvolverFinalizeResult,
    TaskEvolverSession,
)
from my_agent.memory.experience.attribution import AttributionRecordLike
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.short_term import (
    MemoryCompressor,
    ShortTermMemory,
    render_short_term_messages,
)
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import (
    CompressionResult,
    MemoryContext,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RetrievalHit,
    content_fingerprint,
)
from my_agent.policy.identity import PolicyIdentity
from my_agent.tools import ToolExecutionResult
from my_agent.training.contracts import AuthoritativeTaskOutcome


class MemoryManager:
    """Memory storage, retrieval, and compression entry point.

    The manager owns short-term memory and typed long-term experiences:

    * :meth:`from_config` — build a manager wired to the config's memory dir.
    * :meth:`save_experience` — persist a typed four-tier experience.
    * :meth:`build_context_for_query` — retrieve long-term memory and return a
      token-bounded injection block (the 3.2 acceptance target).
    * :meth:`append_user_message` / :meth:`append_assistant_response` /
      :meth:`append_tool_result` — record short-term entries.
    * :meth:`status` — snapshot for ``/memory`` and debugging.

    Prompt assembly and context-window decisions live in
    :class:`my_agent.context.AgentContextManager`.
    """

    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: AgentLLM | None,
        repo_path: Path,
        short_term: ShortTermMemory,
        experience_store: ExperienceStore,
        compressor: MemoryCompressor,
        evolver_runtime: EvolverRuntime,
        project_key: str,
        session_id: str = "",
        trace_sink: Any | None = None,
        context_profile: ContextProfile | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.context_profile = context_profile or ContextProfile.resolve(config, _model_name(llm, config))
        self.repo_path = Path(repo_path)
        self.short_term = short_term
        self.experience_store = experience_store
        self.compressor = compressor
        self._evolver_runtime = evolver_runtime
        self.project_key = project_key
        self.session_id = session_id
        self._trace_sink = trace_sink

    @property
    def evolver_runtime(self) -> EvolverRuntime:
        return self._evolver_runtime

    @property
    def evolver_coordinator(self) -> Any | None:
        return self._evolver_runtime.coordinator

    @evolver_coordinator.setter
    def evolver_coordinator(self, value: Any | None) -> None:
        self._evolver_runtime.coordinator = value

    @property
    def embedding_retriever(self) -> Any | None:
        return self._evolver_runtime.embedding_retriever

    @embedding_retriever.setter
    def embedding_retriever(self, value: Any | None) -> None:
        self._evolver_runtime.embedding_retriever = value

    @property
    def experience_retriever(self) -> Any | None:
        return self._evolver_runtime.experience_retriever

    @experience_retriever.setter
    def experience_retriever(self, value: Any | None) -> None:
        self._evolver_runtime.experience_retriever = value

    @property
    def evolver_selector(self) -> Any | None:
        return self._evolver_runtime.selector

    @evolver_selector.setter
    def evolver_selector(self, value: Any | None) -> None:
        self._evolver_runtime.selector = value

    @property
    def evolver_writer(self) -> Any | None:
        return self._evolver_runtime.writer

    @evolver_writer.setter
    def evolver_writer(self, value: Any | None) -> None:
        self._evolver_runtime.writer = value

    @property
    def last_evolver_selection(self) -> SelectionResult | None:
        return self._evolver_runtime.last_selection

    @last_evolver_selection.setter
    def last_evolver_selection(self, value: SelectionResult | None) -> None:
        self._evolver_runtime.last_selection = value

    def set_trace_sink(self, trace_sink: Any | None) -> tuple[Any | None, Any | None]:
        previous = (
            self._trace_sink,
            getattr(self.experience_store, "_trace_sink", None),
        )
        self._trace_sink = trace_sink
        self._evolver_runtime.set_trace_sink(trace_sink)
        if hasattr(self.experience_store, "_trace_sink"):
            self.experience_store._trace_sink = trace_sink
        return previous

    def restore_trace_sink(self, snapshot: tuple[Any | None, Any | None]) -> None:
        self._trace_sink = snapshot[0]
        self._evolver_runtime.set_trace_sink(snapshot[0])
        if hasattr(self.experience_store, "_trace_sink"):
            self.experience_store._trace_sink = snapshot[1]

    @classmethod
    def from_config(
        cls,
        *,
        config: AgentConfig,
        llm: AgentLLM | None,
        repo_path: Path,
        session_id: str | None = None,
        trace_sink: Any | None = None,
        embedding_retriever: Any | None = None,
    ) -> "MemoryManager":
        from my_agent.memory.factory import build_memory_manager

        return build_memory_manager(
            cls,
            config=config,
            llm=llm,
            repo_path=repo_path,
            session_id=session_id,
            trace_sink=trace_sink,
            embedding_retriever=embedding_retriever,
        )

    def require_formal_runtime_binding(
        self,
        *,
        config: AgentConfig,
        policy_identity: PolicyIdentity,
        repo_path: Path | None,
    ) -> None:
        """Validate an injected manager against the active formal runtime."""

        self._evolver_runtime.require_formal_binding(
            config=config,
            policy_identity=policy_identity,
            repo_path=repo_path,
            manager_llm=self.llm,
            manager_store=self.experience_store,
            manager_project_key=self.project_key,
        )

    # ------------------------------------------------------------------ writes

    def append_task_goal(self, goal: str, *, run_id: str = "") -> MemoryEntry:
        entry = MemoryEntry.build(
            id=_new_id("goal"),
            content=goal,
            type=MemoryType.CONVERSATION,
            scope=MemoryScope.SESSION,
            source="task_goal",
            token_count=estimate_tokens(goal),
            project_key=self.project_key,
            run_id=run_id,
            metadata={"kind": "task_goal"},
        )
        self.short_term.append(entry)
        return entry

    def append_user_message(self, content: str, *, run_id: str = "") -> MemoryEntry:
        entry = MemoryEntry.build(
            id=_new_id("user"),
            content=content,
            type=MemoryType.CONVERSATION,
            scope=MemoryScope.SESSION,
            source="user",
            token_count=estimate_tokens(content),
            project_key=self.project_key,
            run_id=run_id,
        )
        self.short_term.append(entry)
        return entry

    def append_assistant_response(self, response: ChatResponse, *, run_id: str = "") -> MemoryEntry:
        tool_names = ",".join(call.name for call in response.tool_calls) if response.tool_calls else ""
        metadata: dict[str, Any] = {}
        if tool_names:
            metadata["tool_calls"] = tool_names
            metadata["tool_calls_payload"] = [call.to_openai() for call in response.tool_calls]
        entry = MemoryEntry.build(
            id=_new_id("assistant"),
            content=response.content or "",
            type=MemoryType.CONVERSATION,
            scope=MemoryScope.SESSION,
            source="assistant",
            token_count=estimate_tokens(response.content or ""),
            project_key=self.project_key,
            run_id=run_id,
            metadata=metadata,
        )
        self.short_term.append(entry)
        return entry

    def append_tool_result(self, result: ToolExecutionResult, *, run_id: str = "") -> MemoryEntry:
        truncated = _truncate_tool_result(result.content, self.context_profile.tool_result_char_limit)
        content = _tool_result_message_content(result, truncated)
        was_truncated = truncated != result.content
        entry = MemoryEntry.build(
            id=_new_id("tool"),
            content=content,
            type=MemoryType.TOOL_RESULT,
            scope=MemoryScope.SESSION,
            source=f"tool:{result.name}",
            token_count=estimate_tokens(content),
            project_key=self.project_key,
            run_id=run_id,
            metadata={
                "tool_call_id": result.id,
                "tool_name": result.name,
                "ok": result.ok,
                "error_code": result.error_code,
                "blocked": result.blocked,
                "timed_out": result.timed_out,
                "retryable": result.retryable,
                "elapsed_ms": result.elapsed_ms,
                "truncated": was_truncated,
                "original_content_chars": len(result.content),
            },
        )
        self.short_term.append(entry)
        return entry

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
            id=_new_id("exp"),
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
        stored, created = self.experience_store.add(entry)
        self._trace(
            "memory.evolver_saved",
            {
                "id": stored.id,
                "created": created,
                "tier": stored.tier.value,
                "scope": stored.scope.value,
                "tokens": stored.token_count,
                "source_task": stored.source_task,
                "created_by": stored.created_by.value,
            },
        )
        return stored, created

    def update_experience_attribution(self, record: AttributionRecordLike) -> bool:
        """Write one attribution record onto a visible experience memory."""
        try:
            expected_tier = ExperienceTier(record.tier)
        except ValueError:
            return False
        return self.experience_store.update_attribution(
            record,
            project_key=self.project_key,
            expected_tier=expected_tier,
        )

    def append_summary(
        self,
        content: str,
        *,
        source: str = "summary",
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry.build(
            id=_new_id("summary"),
            content=content,
            type=MemoryType.SUMMARY,
            scope=MemoryScope.SESSION,
            source=source,
            token_count=estimate_tokens(content),
            project_key=self.project_key,
            run_id=run_id,
            metadata=metadata,
        )
        self.short_term.append(entry)
        return entry

    # ------------------------------------------------------------------ retrieval

    def build_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        limit: int | None = None,
        include_short_term: bool = False,
    ) -> MemoryContext:
        del include_short_term
        return self._evolver_runtime.build_context(
            query,
            max_tokens=max_tokens,
            top_k_per_tier=limit,
        )

    def begin_formal_evolver_task(
        self,
        *,
        task: str,
        task_id: str,
        task_group: str,
        trajectory_id: str,
        stream_id: str,
    ) -> TaskEvolverSession:
        return self._evolver_runtime.begin_task(
            task=task,
            task_id=task_id,
            task_group=task_group,
            trajectory_id=trajectory_id,
            stream_id=stream_id,
        )

    def finalize_formal_evolver_task(
        self,
        episode: AgentEpisodeArtifact,
        outcome: AuthoritativeTaskOutcome,
    ) -> EvolverFinalizeResult | None:
        return self._evolver_runtime.finalize_task(episode, outcome)

    def retrieve_evolver_candidates(
        self,
        query: str,
        *,
        top_k_per_tier: int | None = None,
    ) -> list[RetrievalHit[ExperienceMemory]]:
        return self._evolver_runtime.retrieve_candidates(
            query,
            top_k_per_tier=top_k_per_tier,
        )

    def count_visible_experiences(self) -> int:
        return len(self.experience_store.all(project_key=self.project_key))

    def build_evolver_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext:
        return self._evolver_runtime.build_context(
            query=query,
            max_tokens=max_tokens,
            top_k_per_tier=top_k_per_tier,
            max_items=max_items,
        )

    # ------------------------------------------------------------------ status

    def status(self, *, include_entries: bool = True) -> MemoryStatus:
        long_entries = self.experience_store.all(project_key=self.project_key)
        long_tokens = sum(max(0, entry.token_count) for entry in long_entries)
        return MemoryStatus(
            project_key=self.project_key,
            storage_path=str(self.experience_store.path),
            short_term_entries=len(self.short_term),
            short_term_tokens=self.short_term.token_count(),
            short_term_storage_token_limit=self.context_profile.short_term_storage_token_limit,
            long_term_entries=len(long_entries),
            long_term_tokens=long_tokens,
            compression_trigger_ratio=self.config.memory_compression_trigger_ratio,
            retain_recent_turns=self.config.memory_retain_recent_turns,
            map_chunk_size=self.config.memory_map_chunk_size,
            long_term_entries_detail=tuple(long_entries) if include_entries else (),
        )

    # ------------------------------------------------------------------ memory operations

    def write_experiences_from_run(
        self,
        *,
        task: str,
        run_id: str,
        trace_path: str | Path | None = None,
        stop_reason: str = "",
        final_answer: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        outcome: str = "unknown",
        outcome_source: str = "runtime",
        source_task: str = "",
        stream_id: str = "",
        task_type: str = "",
        memory_mode: str = "",
    ) -> ExperienceWriteResult:
        return self._evolver_runtime.write_legacy_run(
            task=task,
            run_id=run_id,
            trace_path=trace_path,
            stop_reason=stop_reason,
            final_answer=final_answer,
            tool_history=tool_history,
            outcome=outcome,
            outcome_source=outcome_source,
            source_task=source_task,
            stream_id=stream_id,
            task_type=task_type,
            memory_mode=memory_mode,
        )

    def fork_for_task(self, *, session_id: str, run_id: str = "") -> "MemoryManager":
        del run_id
        return MemoryManager(
            config=self.config,
            llm=self.llm,
            repo_path=self.repo_path,
            short_term=self.short_term.fork(),
            experience_store=self.experience_store,
            compressor=self.compressor.fork(),
            evolver_runtime=self._evolver_runtime.fork(),
            project_key=self.project_key,
            session_id=session_id,
            trace_sink=self._trace_sink,
            context_profile=self.context_profile,
        )

    def clear_short_term(self, *, extract_first: bool = True, reason: str = "clear") -> tuple[int, list[MemoryEntry]]:
        """Clear session memory without creating long-term facts.

        ``extract_first`` remains a no-op compatibility argument until the
        legacy public API is removed.  Typed long-term experiences are only
        produced through ``save_experience`` or the Evolver writer.
        """
        del extract_first
        removed = self.short_term.clear()
        self._trace("memory.clear", {"removed": len(removed), "extracted_facts": 0, "reason": reason})
        return len(removed), []

    def render_short_term_messages(self, *, max_tokens: int | None = None) -> list[MessageLike]:
        return render_short_term_messages(
            self.short_term.all(),
            max_tokens=max_tokens,
        )

    def trace_context_event(self, event: str, payload: dict[str, Any]) -> None:
        self._trace(event, payload)

    def compact_short_term(
        self,
        *,
        tools: list[dict[str, Any]],
        force: bool = False,
        focus: str = "",
        before_tokens: int | None = None,
        trace_completed: bool = True,
        trigger_tokens: int | None = None,
    ) -> CompressionResult | None:
        return self._compact_if_needed(
            tools=tools,
            force=force,
            focus=focus,
            before_tokens=before_tokens,
            trace_completed=trace_completed,
            trigger_tokens=trigger_tokens,
        )

    def _compact_if_needed(
        self,
        *,
        tools: list[dict[str, Any]],
        force: bool = False,
        focus: str = "",
        before_tokens: int | None = None,
        trace_completed: bool = True,
        trigger_tokens: int | None = None,
    ) -> CompressionResult | None:
        tools_tokens = estimate_tokens(tools)
        before = before_tokens if before_tokens is not None else self.short_term.token_count() + tools_tokens
        threshold = trigger_tokens if trigger_tokens is not None else self._compression_threshold_tokens()
        if not force and before < threshold:
            return None

        old_entries = self.short_term.old_entries_for_compression(self.config.memory_retain_recent_turns)
        if not old_entries:
            return CompressionResult(False, before, before)

        self._trace(
            "memory.compaction_started",
            {"before_tokens": before, "old_entries": len(old_entries), "force": force},
        )
        summary, fallback, map_count, reduce_used = self.compressor.compact_entries(old_entries, focus=focus)
        if not summary.strip():
            result = CompressionResult(False, before, before, map_count=map_count, reduce_used=reduce_used, fallback=fallback)
            self._trace("memory.compaction_failed", {"reason": "empty_summary", "before_tokens": before})
            return result

        summary_content = "[Compressed memory summary]\n" + summary.strip()
        summary_entry = MemoryEntry.build(
            id=_new_id("summary"),
            content=summary_content,
            type=MemoryType.SUMMARY,
            scope=MemoryScope.SESSION,
            source="compressor",
            token_count=estimate_tokens(summary_content),
            project_key=self.project_key,
            metadata={"map_count": map_count, "reduce_used": reduce_used, "fallback": fallback},
        )
        self.short_term.replace_old_entries_with_summary({entry.id for entry in old_entries}, summary_entry)
        after = self.short_term.token_count() + tools_tokens

        result = CompressionResult(
            True,
            before,
            after,
            map_count=map_count,
            reduce_used=reduce_used,
            extracted_facts=0,
            fallback=fallback,
        )
        if trace_completed:
            self._trace(
                "memory.compacted",
                self._context_trace_payload(
                    {
                        "before_tokens": before,
                        "after_tokens": after,
                        "map_count": map_count,
                        "reduce_used": reduce_used,
                        "extracted_facts": 0,
                        "fallback": fallback,
                        "estimated_prompt_tokens": after,
                    }
                ),
            )
        return result

    def _compression_threshold_tokens(self) -> int:
        return self.context_profile.compression_trigger_tokens

    def _context_trace_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched.update(
            {
                "context_window": self.context_profile.max_context_tokens,
                "compression_trigger_tokens": self.context_profile.compression_trigger_tokens,
                "short_term_storage_token_limit": self.context_profile.short_term_storage_token_limit,
                "repo_context_budget_tokens": self.context_profile.repo_context_budget_tokens,
                "tool_schema_budget_tokens": self.context_profile.tool_schema_budget_tokens,
                "tool_result_char_limit": self.context_profile.tool_result_char_limit,
                "dynamic_profile_source": self.context_profile.dynamic_profile_source,
            }
        )
        return enriched

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(event, payload)
        except Exception:
            pass


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _model_name(llm: AgentLLM | None, config: AgentConfig) -> str:
    value = getattr(llm, "model", "") if llm is not None else ""
    if isinstance(value, str) and value.strip():
        return value
    return config.model


def _truncate_tool_result(content: str, limit: int) -> str:
    if limit < 1 or len(content) <= limit:
        return content
    return content[:limit] + "...(truncated)"


def _tool_result_message_content(result: ToolExecutionResult, truncated_content: str) -> str:
    payload: dict[str, Any] = {
        "tool": result.name,
        "ok": result.ok,
        "blocked": result.blocked,
        "timed_out": result.timed_out,
        "retryable": result.retryable,
        "error_code": result.error_code,
        "elapsed_ms": result.elapsed_ms,
        "content": truncated_content,
    }
    if truncated_content != result.content:
        payload["truncated"] = True
        payload["original_content_chars"] = len(result.content)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


__all__ = ["MemoryManager"]
