from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.llm import AgentLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message, MessageLike, messages_to_openai
from my_agent.memory.compression import MemoryCompressor
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


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


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
