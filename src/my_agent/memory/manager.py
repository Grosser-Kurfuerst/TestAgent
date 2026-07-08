from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.llm import AgentLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message, MessageLike, messages_to_openai
from my_agent.memory.compression import MemoryCompressor
from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    ExperienceWriteRequest,
    ExperienceWriteResult,
    ExperienceWriter,
    ExperienceSelector,
    ExperienceTier,
    SelectionResult,
    build_experience_entry,
    build_write_steps_from_tool_history,
    candidate_tier,
    proposal_tier_counts,
    selection_candidate_summary,
    selection_tier_counts,
)
from my_agent.memory.long_term import LongTermMemoryStore, STORAGE_FILE
from my_agent.memory.retrieval import MemoryRetriever
from my_agent.memory.short_term import ShortTermMemory
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import (
    CompressionResult,
    MemoryContext,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)
from my_agent.tools import ToolExecutionResult

WRITER_METADATA_STRING_CHARS = 1_000


class MemoryManager:
    """Memory storage, retrieval, and compression entry point.

    The manager owns short-term and long-term memory primitives:

    * :meth:`from_config` — build a manager wired to the config's memory dir.
    * :meth:`save_fact` — persist a durable fact to long-term memory.
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
        long_term: LongTermMemoryStore,
        retriever: MemoryRetriever,
        compressor: MemoryCompressor,
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
        self.long_term = long_term
        self.retriever = retriever
        self.compressor = compressor
        self.project_key = project_key
        self.session_id = session_id
        self._trace_sink = trace_sink
        self.last_fact_extraction_error = ""
        self.last_fact_save_errors: list[str] = []
        self.evolver_selector = ExperienceSelector(
            tier_weights=self.config.memory_evolver_tier_weights,
            tier_caps=self.config.memory_evolver_tier_caps,
            selected_max_items=self.config.memory_evolver_selected_max_items,
            min_score=self.config.memory_evolver_min_score,
        )
        self.evolver_writer = ExperienceWriter(
            llm=self.llm,
            min_confidence=self.config.memory_evolver_writer_min_confidence,
            max_records=self.config.memory_evolver_writer_max_records,
            max_input_chars=self.config.memory_evolver_writer_max_input_chars,
            max_content_chars=self.config.memory_evolver_writer_max_content_chars,
        )
        self.last_evolver_selection: SelectionResult | None = None

    def set_trace_sink(self, trace_sink: Any | None) -> tuple[Any | None, Any | None]:
        previous = (self._trace_sink, getattr(self.long_term, "_trace_sink", None))
        self._trace_sink = trace_sink
        if hasattr(self.long_term, "_trace_sink"):
            self.long_term._trace_sink = trace_sink
        return previous

    def restore_trace_sink(self, snapshot: tuple[Any | None, Any | None]) -> None:
        self._trace_sink = snapshot[0]
        if hasattr(self.long_term, "_trace_sink"):
            self.long_term._trace_sink = snapshot[1]

    @classmethod
    def from_config(
        cls,
        *,
        config: AgentConfig,
        llm: AgentLLM | None,
        repo_path: Path,
        session_id: str | None = None,
        trace_sink: Any | None = None,
    ) -> "MemoryManager":
        memory_dir = Path(config.memory_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)
        long_term = LongTermMemoryStore(memory_dir / STORAGE_FILE, trace_sink=trace_sink)
        long_term.load()
        context_profile = ContextProfile.resolve(config, _model_name(llm, config))
        short_term = ShortTermMemory(
            max_tokens=context_profile.short_term_storage_token_limit,
            max_entries=config.memory_short_term_entries,
        )
        retriever = MemoryRetriever()
        compressor = MemoryCompressor(
            llm=llm,
            chunk_size=config.memory_map_chunk_size,
            retain_recent_turns=config.memory_retain_recent_turns,
            max_input_chars=config.max_summary_input_chars,
        )
        project_key = str(getattr(config, "memory_project_key", "") or "").strip()
        if not project_key:
            project_key = _normalize_project_key(repo_path)
        return cls(
            config=config,
            llm=llm,
            repo_path=Path(repo_path),
            short_term=short_term,
            long_term=long_term,
            retriever=retriever,
            compressor=compressor,
            project_key=project_key,
            session_id=session_id or "",
            trace_sink=trace_sink,
            context_profile=context_profile,
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

    def save_fact(
        self,
        content: str,
        *,
        scope: MemoryScope = MemoryScope.PROJECT,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryEntry, bool]:
        project_key = "" if scope == MemoryScope.GLOBAL else self.project_key
        fact_metadata = {"source": source}
        fact_metadata.update(metadata or {})
        entry = MemoryEntry.build(
            id=_new_id("fact"),
            content=content,
            type=MemoryType.FACT,
            scope=scope,
            source=source,
            token_count=estimate_tokens(content),
            project_key=project_key,
            metadata=fact_metadata,
        )
        stored, created = self.long_term.add(entry)
        self._trace(
            "memory.saved",
            {
                "id": stored.id,
                "created": created,
                "scope": stored.scope.value,
                "source": stored.source,
                "tokens": stored.token_count,
            },
        )
        return stored, created

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
            id=_new_id("exp"),
            content=content,
            tier=tier,
            project_key=project_key,
            scope=scope,
            run_id=run_id,
            source_task=source_task,
            created_by=created_by,
            extra_metadata=metadata,
        )
        stored, created = self.long_term.add(entry)
        self._trace(
            "memory.evolver_saved",
            {
                "id": stored.id,
                "created": created,
                "tier": stored.metadata.get("evolver_tier", ""),
                "scope": stored.scope.value,
                "tokens": stored.token_count,
                "source_task": str(stored.metadata.get("source_task") or ""),
                "created_by": str(stored.metadata.get("created_by") or ""),
            },
        )
        return stored, created

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
        if self.config.memory_evolver_mode in {"retrieve_select", "full"}:
            return self.build_evolver_context_for_query(
                query,
                max_tokens=max_tokens,
                top_k_per_tier=limit,
            )
        return self._build_legacy_context_for_query(
            query,
            max_tokens=max_tokens,
            limit=limit,
            include_short_term=include_short_term,
        )

    def _build_legacy_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        limit: int | None = None,
        include_short_term: bool = False,
    ) -> MemoryContext:
        """Retrieve memory and return a token-bounded injection block.

        This is the 3.2 acceptance target (plan §15). By default only
        long-term facts are injected so the current turn is not re-injected as
        "history"; ``include_short_term=True`` is for the ``/memory`` debug
        view.
        """
        resolved_limit = limit if limit is not None else self.config.memory_retrieval_limit
        resolved_tokens = max_tokens if max_tokens is not None else self.context_profile.memory_context_tokens
        hits = self.retriever.retrieve(
            query,
            short_term=self.short_term,
            long_term=self.long_term,
            project_key=self.project_key,
            limit=resolved_limit,
            include_short_term=include_short_term,
        )
        context = self.retriever.build_context(hits, max_tokens=resolved_tokens)
        self._trace(
            "memory.retrieved",
            {
                "query_chars": len(query),
                "hits": len(context.hits),
                "tokens": context.estimated_tokens,
                "include_short_term": include_short_term,
            },
        )
        return context

    def retrieve_evolver_candidates(
        self,
        query: str,
        *,
        top_k_per_tier: int | None = None,
    ) -> list:
        resolved_top_k = top_k_per_tier if top_k_per_tier is not None else self.config.memory_evolver_top_k_per_tier
        resolved_top_k = max(1, int(resolved_top_k))
        hits = self.retriever.retrieve(
            query,
            short_term=self.short_term,
            long_term=self.long_term,
            project_key=self.project_key,
            limit=0,
            include_short_term=False,
        )
        groups: dict[ExperienceTier, list] = {tier: [] for tier in ExperienceTier}
        for hit in hits:
            tier = candidate_tier(hit.entry)
            if tier is None:
                continue
            groups[tier].append(hit)

        selected: list = []
        for tier in ExperienceTier:
            selected.extend(groups[tier][:resolved_top_k])
        selected.sort(key=lambda hit: hit.score, reverse=True)
        return selected

    def count_visible_experiences(self) -> int:
        return sum(1 for entry in self.long_term.all(project_key=self.project_key) if candidate_tier(entry) is not None)

    def build_evolver_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext:
        resolved_tokens = max_tokens if max_tokens is not None else self.context_profile.memory_context_tokens
        resolved_top_k = top_k_per_tier if top_k_per_tier is not None else self.config.memory_evolver_top_k_per_tier
        resolved_top_k = max(1, int(resolved_top_k))
        visible_count = self.count_visible_experiences()
        if resolved_tokens <= 0 or visible_count < self.config.memory_evolver_min_experience_entries:
            result = SelectionResult(
                candidates=(),
                selected=(),
                context=MemoryContext(injected_text="", hits=[], estimated_tokens=0),
                policy="rule_tier_weighted_v1",
                estimated_tokens=0,
                metadata={
                    "query_chars": len(query),
                    "candidate_count": 0,
                    "selected_count": 0,
                    "candidate_tier_counts": {},
                    "selected_tier_counts": {},
                    "insufficient_experience_entries": visible_count < self.config.memory_evolver_min_experience_entries,
                },
            )
            self.last_evolver_selection = result
            self._trace_evolver_selection(
                query=query,
                result=result,
                candidates=[],
                max_tokens=resolved_tokens,
                top_k_per_tier=resolved_top_k,
                max_items=max_items,
                visible_experience_count=visible_count,
                insufficient_experience_entries=visible_count < self.config.memory_evolver_min_experience_entries,
            )
            return result.context

        candidates = self.retrieve_evolver_candidates(query, top_k_per_tier=resolved_top_k)
        try:
            result = self.evolver_selector.select(
                query=query,
                hits=candidates,
                max_tokens=resolved_tokens,
                max_items=max_items,
            )
        except Exception as exc:  # noqa: BLE001 - selector failures must not fall back to unselected legacy memory
            error = f"{type(exc).__name__}: {exc}"
            result = SelectionResult(
                candidates=(),
                selected=(),
                context=MemoryContext(injected_text="", hits=[], estimated_tokens=0),
                policy="rule_tier_weighted_v1",
                estimated_tokens=0,
                metadata={
                    "query_chars": len(query),
                    "candidate_count": 0,
                    "selected_count": 0,
                    "candidate_tier_counts": {},
                    "selected_tier_counts": {},
                    "fallback": True,
                    "error": error,
                },
            )
            self.last_evolver_selection = result
            self._trace(
                "memory.evolver_selection_failed",
                {
                    "query_chars": len(query),
                    "candidate_count": len(candidates),
                    "error": error,
                    "fallback": "empty_context",
                    "mode": self.config.memory_evolver_mode,
                    "memory_project_key": self.project_key,
                },
            )
            self._trace_evolver_selection(
                query=query,
                result=result,
                candidates=candidates,
                max_tokens=resolved_tokens,
                top_k_per_tier=resolved_top_k,
                max_items=max_items,
                visible_experience_count=visible_count,
                insufficient_experience_entries=False,
                fallback=True,
            )
            return result.context
        self.last_evolver_selection = result
        self._trace_evolver_selection(
            query=query,
            result=result,
            candidates=candidates,
            max_tokens=resolved_tokens,
            top_k_per_tier=resolved_top_k,
            max_items=max_items,
            visible_experience_count=visible_count,
            insufficient_experience_entries=False,
        )
        return result.context

    def _trace_evolver_selection(
        self,
        *,
        query: str,
        result: SelectionResult,
        candidates: list,
        max_tokens: int,
        top_k_per_tier: int,
        max_items: int | None,
        visible_experience_count: int,
        insufficient_experience_entries: bool,
        fallback: bool = False,
    ) -> None:
        retrieved_tiers = _hit_tier_counts(candidates)
        candidate_tiers = selection_tier_counts(result.candidates)
        selected_candidates = [item.candidate for item in result.selected]
        selected_tiers = selection_tier_counts(selected_candidates)
        candidate_payload = {
            "query_chars": len(query),
            "candidate_count": len(result.candidates),
            "top_k_per_tier": top_k_per_tier,
            "tiers": candidate_tiers,
            "retrieved_candidate_count": len(candidates),
            "retrieved_tiers": retrieved_tiers,
            "candidate_ids": [candidate.id for candidate in result.candidates],
            "candidate_summaries": [selection_candidate_summary(candidate) for candidate in result.candidates],
            "selection_policy": result.policy,
            "mode": self.config.memory_evolver_mode,
            "insufficient_experience_entries": insufficient_experience_entries,
            "visible_experience_count": visible_experience_count,
            "memory_project_key": self.project_key,
        }
        selected_payload = {
            "selected_count": len(result.selected),
            "selected_ids": [item.candidate.id for item in result.selected],
            "omitted_ids": list(result.omitted_ids),
            "tiers": selected_tiers,
            "estimated_tokens": result.context.estimated_tokens,
            "max_tokens": max_tokens,
            "max_items": max_items if max_items is not None else self.config.memory_evolver_selected_max_items,
            "selection_policy": result.policy,
            "selection_reasons": [
                {
                    "id": item.candidate.id,
                    "reason": f"selected: score={item.candidate.selection_score:.2f} "
                    f"tier={item.candidate.tier.value if item.candidate.tier is not None else ''} "
                    f"rank={item.rank}",
                }
                for item in result.selected
            ],
            "fallback": fallback,
            "memory_project_key": self.project_key,
        }
        self._trace("memory.evolver_candidates", candidate_payload)
        self._trace("memory.evolver_selected", selected_payload)
        self._trace(
            "memory.retrieved",
            {
                "query_chars": len(query),
                "hits": len(result.context.hits),
                "tokens": result.context.estimated_tokens,
                "include_short_term": False,
                "mode": self.config.memory_evolver_mode,
                "selection_policy": result.policy,
            },
        )

    def retrieve_hits(
        self,
        query: str,
        *,
        limit: int | None = None,
        include_short_term: bool = False,
    ) -> list:
        """Raw scored hits, for ``/memory`` search and tests."""
        resolved_limit = limit if limit is not None else self.config.memory_retrieval_limit
        return self.retriever.retrieve(
            query,
            short_term=self.short_term,
            long_term=self.long_term,
            project_key=self.project_key,
            limit=resolved_limit,
            include_short_term=include_short_term,
        )

    # ------------------------------------------------------------------ status

    def status(self, *, include_entries: bool = True) -> MemoryStatus:
        long_entries = self.long_term.all(project_key=self.project_key)
        long_tokens = sum(max(0, entry.token_count) for entry in long_entries)
        return MemoryStatus(
            project_key=self.project_key,
            storage_path=str(self.long_term.path),
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

    def extract_facts(self, *, reason: str, run_id: str = "") -> list[MemoryEntry]:
        entries = self.short_term.all()
        if run_id:
            entries = [entry for entry in entries if entry.run_id == run_id]
        return self._extract_and_store_facts(entries, reason=reason, run_id=run_id)

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
    ) -> ExperienceWriteResult:
        if not self.config.memory_evolver_writer_enabled:
            return ExperienceWriteResult()

        steps = build_write_steps_from_tool_history(
            tool_history,
            max_output_chars=min(1_000, max(0, self.config.memory_evolver_writer_max_input_chars)),
        )
        resolved_source_task = str(source_task or "").strip() or _task_ref(task)
        selected_ids = _selection_selected_ids(self.last_evolver_selection)
        candidate_ids = _selection_candidate_ids(self.last_evolver_selection)
        request = ExperienceWriteRequest(
            task=str(task or ""),
            run_id=str(run_id or ""),
            trace_path=Path(trace_path) if trace_path is not None else None,
            stop_reason=str(stop_reason or ""),
            outcome=str(outcome or "unknown"),
            outcome_source=str(outcome_source or "runtime"),
            final_answer=str(final_answer or ""),
            selected_memory_ids=selected_ids,
            candidate_memory_ids=candidate_ids,
            steps=steps,
            source_task=resolved_source_task,
            stream_id=str(stream_id or ""),
            task_type=str(task_type or ""),
            project_key=self.project_key,
        )
        context_payload = _writer_context_payload(request)
        try:
            self._trace(
                "memory.evolver_writer_started",
                {
                    **context_payload,
                    "mode": self.config.memory_evolver_writer_mode,
                    "task_chars": len(request.task),
                    "tool_steps": len(request.steps),
                    "outcome": request.outcome,
                    "outcome_source": request.outcome_source,
                    "selected_count": len(request.selected_memory_ids),
                    "candidate_count": len(request.candidate_memory_ids),
                },
            )
            proposed = self.evolver_writer.propose(request, mode=self.config.memory_evolver_writer_mode)
            self._trace(
                "memory.evolver_writer_proposed",
                {
                    "proposal_count": len(proposed.proposals),
                    "tiers": proposal_tier_counts(proposed.proposals),
                    "llm_used": proposed.llm_used,
                    "fallback_used": proposed.fallback_used,
                    "rejected_count": len(proposed.rejected),
                    "rejected_reasons": _rejected_reason_counts(proposed.rejected),
                },
            )
            if not proposed.proposals:
                self._trace(
                    "memory.evolver_writer_skipped",
                    {
                        **context_payload,
                        "reason": "no_valid_proposals",
                        "outcome": request.outcome,
                        "tool_steps": len(request.steps),
                    },
                )
                return proposed

            saved: list[MemoryEntry] = []
            duplicate_ids: list[str] = []
            for proposal in proposed.proposals:
                proposal_metadata = _sanitize_writer_metadata(proposal.metadata)
                metadata = {
                    **proposal_metadata,
                    "confidence": proposal.confidence,
                    "outcome": request.outcome,
                    "outcome_source": request.outcome_source,
                    "stop_reason": request.stop_reason,
                    "source_trace": str(request.trace_path or ""),
                    "source_task": request.source_task,
                    "task_type": request.task_type,
                    "selected_memory_ids": list(request.selected_memory_ids),
                    "candidate_memory_ids": list(request.candidate_memory_ids),
                    "stream_id": request.stream_id,
                    "memory_project_key": request.project_key,
                    "writer_policy": "fallback_runtime_v1",
                    "writer_reason": proposal.reason,
                }
                entry, created = self.save_experience(
                    proposal.content,
                    tier=proposal.tier,
                    source_task=request.source_task,
                    created_by=ExperienceCreatedBy.WRITER,
                    run_id=request.run_id,
                    metadata=metadata,
                )
                if created:
                    saved.append(entry)
                else:
                    duplicate_ids.append(entry.id)

            result = ExperienceWriteResult(
                proposals=proposed.proposals,
                saved=tuple(saved),
                duplicate_ids=tuple(duplicate_ids),
                rejected=proposed.rejected,
                llm_used=proposed.llm_used,
                fallback_used=proposed.fallback_used,
            )
            saved_records = _saved_records(saved)
            self._trace(
                "memory.evolver_writer_saved",
                {
                    **context_payload,
                    "saved_count": len(saved),
                    "duplicate_count": len(duplicate_ids),
                    "saved_ids": [entry.id for entry in saved],
                    "saved_records": saved_records,
                    "tiers": _saved_tier_counts(saved),
                    "writer_policy": "fallback_runtime_v1",
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001 - writer must not affect the agent loop
            error = f"{type(exc).__name__}: {exc}"
            self._trace(
                "memory.evolver_writer_failed",
                {
                    **context_payload,
                    "error": error,
                    "phase": "unknown",
                    "fallback_attempted": True,
                },
            )
            return ExperienceWriteResult(error=error)

    def fork_for_task(self, *, session_id: str, run_id: str = "") -> "MemoryManager":
        return MemoryManager(
            config=self.config,
            llm=self.llm,
            repo_path=self.repo_path,
            short_term=ShortTermMemory(
                max_tokens=self.context_profile.short_term_storage_token_limit,
                max_entries=self.config.memory_short_term_entries,
            ),
            long_term=self.long_term,
            retriever=self.retriever,
            compressor=MemoryCompressor(
                llm=self.llm,
                chunk_size=self.config.memory_map_chunk_size,
                retain_recent_turns=self.config.memory_retain_recent_turns,
                max_input_chars=self.config.max_summary_input_chars,
            ),
            project_key=self.project_key,
            session_id=session_id,
            trace_sink=self._trace_sink,
            context_profile=self.context_profile,
        )

    def clear_short_term(self, *, extract_first: bool = True, reason: str = "clear") -> tuple[int, list[MemoryEntry]]:
        extracted: list[MemoryEntry] = []
        if extract_first:
            extracted = self.extract_facts(reason=reason)
        removed = self.short_term.clear()
        self._trace("memory.clear", {"removed": len(removed), "extracted_facts": len(extracted), "reason": reason})
        return len(removed), extracted

    def render_short_term_messages(self, *, max_tokens: int | None = None) -> list[MessageLike]:
        entries = self.short_term.all()
        if max_tokens is not None:
            entries = _entries_within_token_budget(entries, max_tokens)
        return _render_short_term_entries(entries)

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
        extracted = self._extract_and_store_facts(old_entries, reason="compression")
        after = self.short_term.token_count() + tools_tokens

        result = CompressionResult(
            True,
            before,
            after,
            map_count=map_count,
            reduce_used=reduce_used,
            extracted_facts=len(extracted),
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
                        "extracted_facts": len(extracted),
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

    def _extract_and_store_facts(
        self,
        entries: list[MemoryEntry],
        *,
        reason: str,
        run_id: str = "",
    ) -> list[MemoryEntry]:
        self.last_fact_extraction_error = ""
        self.last_fact_save_errors = []
        if not self.config.memory_auto_extract:
            return []
        candidates = self.compressor.extract_facts(entries, reason=reason, project_key=self.project_key, run_id=run_id)
        if self.compressor.last_fact_error:
            self.last_fact_extraction_error = self.compressor.last_fact_error
            self._trace(
                "memory.fact_extraction_failed",
                {"reason": reason, "run_id": run_id, "error": self.last_fact_extraction_error},
            )
            return []
        stored: list[MemoryEntry] = []
        for candidate in candidates:
            try:
                entry, created = self.long_term.add(candidate)
            except Exception as exc:  # noqa: BLE001 - automatic extraction must not break the agent loop
                error = f"{type(exc).__name__}: {exc}"
                self.last_fact_save_errors.append(error)
                self._trace(
                    "memory.save_failed",
                    {
                        "reason": reason,
                        "run_id": run_id,
                        "candidate_id": candidate.id,
                        "error": error,
                    },
                )
                continue
            if created:
                stored.append(entry)
        if stored:
            self._trace(
                "memory.fact_extracted",
                {"count": len(stored), "reason": reason, "run_id": run_id},
            )
        return stored

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(event, payload)
        except Exception:
            pass


def _task_ref(task: str) -> str:
    return f"task_ref_{hashlib.sha256(str(task or '').encode('utf-8')).hexdigest()[:12]}"


def _selection_candidate_ids(selection: SelectionResult | None) -> tuple[str, ...]:
    if selection is None:
        return ()
    return tuple(candidate.id for candidate in selection.candidates)


def _selection_selected_ids(selection: SelectionResult | None) -> tuple[str, ...]:
    if selection is None:
        return ()
    return tuple(item.candidate.id for item in selection.selected)


def _writer_context_payload(request: ExperienceWriteRequest) -> dict[str, Any]:
    return {
        "source_task": request.source_task,
        "stream_id": request.stream_id,
        "task_type": request.task_type,
        "memory_project_key": request.project_key,
    }


def _sanitize_writer_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _truncate_writer_metadata_string(str(key)): _sanitize_writer_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_writer_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_writer_metadata(item) for item in value]
    if isinstance(value, str):
        return _truncate_writer_metadata_string(value)
    return value


def _truncate_writer_metadata_string(value: str) -> str:
    if len(value) <= WRITER_METADATA_STRING_CHARS:
        return value
    return value[: WRITER_METADATA_STRING_CHARS - 3].rstrip() + "..."


def _rejected_reason_counts(rejected: tuple[dict[str, Any], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rejected:
        reason = str(item.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _saved_records(entries: list[MemoryEntry]) -> list[dict[str, str]]:
    return [
        {
            "id": entry.id,
            "tier": str(entry.metadata.get("evolver_tier") or ""),
        }
        for entry in entries
    ]


def _saved_tier_counts(entries: list[MemoryEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        tier = str(entry.metadata.get("evolver_tier") or "")
        if not tier:
            continue
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _hit_tier_counts(hits: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        tier = candidate_tier(hit.entry)
        if tier is None:
            continue
        counts[tier.value] = counts.get(tier.value, 0) + 1
    return counts


def _model_name(llm: AgentLLM | None, config: AgentConfig) -> str:
    value = getattr(llm, "model", "") if llm is not None else ""
    if isinstance(value, str) and value.strip():
        return value
    return config.model


def _render_short_term_entries(entries: list[MemoryEntry]) -> list[MessageLike]:
    rendered: list[MessageLike] = []
    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        if entry.type == MemoryType.SUMMARY:
            rendered.append(Message(role="user", content=entry.content))
            idx += 1
        elif entry.source == "task_goal":
            rendered.append(Message(role="user", content=f"[Task goal]\n{entry.content}"))
            idx += 1
        elif entry.source == "user":
            rendered.append(Message(role="user", content=entry.content))
            idx += 1
        elif entry.source == "assistant":
            tool_calls = _tool_calls_from_metadata(entry.metadata)
            if not tool_calls:
                rendered.append(Message(role="assistant", content=entry.content or ""))
                idx += 1
                continue

            tool_entries, next_idx = _contiguous_tool_entries(entries, idx + 1)
            if _tool_entries_match_calls(tool_entries, tool_calls):
                rendered.append(
                    Message(
                        role="assistant",
                        content=entry.content or "",
                        tool_calls=tool_calls,
                    )
                )
                rendered.extend(_tool_message_from_entry(tool_entry) for tool_entry in tool_entries)
            else:
                rendered.append(Message(role="user", content=_incomplete_tool_call_memory(entry, tool_entries)))
            idx = next_idx
        elif entry.source.startswith("tool:"):
            rendered.append(Message(role="user", content=f"[Tool result memory]\n{entry.content}"))
            idx += 1
        else:
            rendered.append(Message(role="user", content=entry.content))
            idx += 1
    return rendered


def _entries_within_token_budget(entries: list[MemoryEntry], max_tokens: int) -> list[MemoryEntry]:
    if max_tokens <= 0 or not entries:
        return []
    groups = _entry_groups(entries)
    selected: list[list[MemoryEntry]] = []
    start_idx = 0
    if groups and groups[0] and groups[0][0].source == "task_goal":
        if _rendered_groups_tokens([groups[0]]) <= max_tokens:
            selected.append(groups[0])
        start_idx = 1
    tail: list[list[MemoryEntry]] = []
    for group in reversed(groups[start_idx:]):
        candidate_tail = [group, *tail]
        if _rendered_groups_tokens([*selected, *candidate_tail]) > max_tokens:
            continue
        tail = candidate_tail
    selected.extend(tail)
    return [entry for group in selected for entry in group]


def _entry_groups(entries: list[MemoryEntry]) -> list[list[MemoryEntry]]:
    groups: list[list[MemoryEntry]] = []
    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        if entry.source == "assistant":
            tool_entries, next_idx = _contiguous_tool_entries(entries, idx + 1)
            groups.append([entry, *tool_entries])
            idx = next_idx
            continue
        groups.append([entry])
        idx += 1
    return groups


def _rendered_groups_tokens(groups: list[list[MemoryEntry]]) -> int:
    entries = [entry for group in groups for entry in group]
    if not entries:
        return 0
    return estimate_tokens(messages_to_openai(_render_short_term_entries(entries)))


def _contiguous_tool_entries(entries: list[MemoryEntry], start: int) -> tuple[list[MemoryEntry], int]:
    tool_entries: list[MemoryEntry] = []
    idx = start
    while idx < len(entries) and entries[idx].source.startswith("tool:"):
        tool_entries.append(entries[idx])
        idx += 1
    return tool_entries, idx


def _tool_entries_match_calls(tool_entries: list[MemoryEntry], tool_calls: list[LLMToolCall]) -> bool:
    if len(tool_entries) != len(tool_calls):
        return False
    entry_ids = [str(entry.metadata.get("tool_call_id") or entry.id) for entry in tool_entries]
    call_ids = [call.id for call in tool_calls]
    return entry_ids == call_ids


def _tool_message_from_entry(entry: MemoryEntry) -> Message:
    return Message(
        role="tool",
        content=entry.content,
        tool_call_id=str(entry.metadata.get("tool_call_id") or entry.id),
        name=str(entry.metadata.get("tool_name") or entry.source.removeprefix("tool:")),
    )


def _incomplete_tool_call_memory(assistant: MemoryEntry, tool_entries: list[MemoryEntry]) -> str:
    parts = ["[Incomplete tool-call memory]"]
    if assistant.content:
        parts.append(f"assistant: {assistant.content}")
    tool_names = assistant.metadata.get("tool_calls")
    if tool_names:
        parts.append(f"tool_calls: {tool_names}")
    for tool_entry in tool_entries:
        parts.append(f"{tool_entry.source}: {tool_entry.content}")
    return "\n".join(parts)


def _tool_calls_from_metadata(metadata: dict[str, Any]) -> list[LLMToolCall]:
    raw_calls = metadata.get("tool_calls_payload")
    if not isinstance(raw_calls, list):
        return []
    calls: list[LLMToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        try:
            calls.append(LLMToolCall.from_openai(raw))
        except ValueError:
            continue
    return calls


def _normalize_project_key(repo_path: Path) -> str:
    try:
        resolved = Path(repo_path).expanduser().resolve()
    except (OSError, RuntimeError):
        resolved = Path(repo_path).expanduser().absolute()
    return str(resolved)


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
