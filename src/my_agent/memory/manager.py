from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from my_agent.config import AgentConfig
from my_agent.llm import AgentLLM
from my_agent.llm.types import ChatResponse
from my_agent.memory.long_term import LongTermMemoryStore, STORAGE_FILE
from my_agent.memory.retrieval import MemoryRetriever
from my_agent.memory.short_term import ShortTermMemory
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import (
    MemoryContext,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)
from my_agent.tools import ToolExecutionResult


class MemoryManager:
    """The single memory entry point the Agent depends on (plan §10).

    Phase 3.1-3.2 ships the parts that do not require map-reduce compression
    or LLM fact extraction:

    * :meth:`from_config` — build a manager wired to the config's memory dir.
    * :meth:`save_fact` — persist a durable fact to long-term memory.
    * :meth:`build_context_for_query` — retrieve long-term memory and return a
      token-bounded injection block (the 3.2 acceptance target).
    * :meth:`append_user_message` / :meth:`append_assistant_response` /
      :meth:`append_tool_result` — record short-term entries.
    * :meth:`status` — snapshot for ``/memory`` and debugging.

    Methods that need the compressor or fact extractor (``prepare_messages``,
    ``extract_facts``, ``fork_for_task``, ``clear_short_term`) are declared to
    match the plan's facade contract and land in phases 3.3-3.5.
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
        project_key: str,
        session_id: str = "",
    ) -> None:
        self.config = config
        self.llm = llm
        self.repo_path = Path(repo_path)
        self.short_term = short_term
        self.long_term = long_term
        self.retriever = retriever
        self.project_key = project_key
        self.session_id = session_id

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
        short_term = ShortTermMemory(
            max_tokens=config.memory_short_term_tokens,
            max_entries=config.memory_short_term_entries,
        )
        retriever = MemoryRetriever()
        project_key = _normalize_project_key(repo_path)
        return cls(
            config=config,
            llm=llm,
            repo_path=Path(repo_path),
            short_term=short_term,
            long_term=long_term,
            retriever=retriever,
            project_key=project_key,
            session_id=session_id or "",
        )

    # ------------------------------------------------------------------ writes

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
        truncated = _truncate_tool_result(result.content, self.config.memory_tool_result_chars)
        content = f"[{result.name}] {truncated}"
        entry = MemoryEntry.build(
            id=_new_id("tool"),
            content=content,
            type=MemoryType.TOOL_RESULT,
            scope=MemoryScope.SESSION,
            source=f"tool:{result.name}",
            token_count=estimate_tokens(content),
            project_key=self.project_key,
            run_id=run_id,
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
        entry = MemoryEntry.build(
            id=_new_id("fact"),
            content=content,
            type=MemoryType.FACT,
            scope=scope,
            source=source,
            token_count=estimate_tokens(content),
            project_key=project_key,
            metadata=metadata,
        )
        return self.long_term.add(entry)

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
        resolved_tokens = max_tokens if max_tokens is not None else self.config.memory_context_tokens
        hits = self.retriever.retrieve(
            query,
            short_term=self.short_term,
            long_term=self.long_term,
            project_key=self.project_key,
            limit=resolved_limit,
            include_short_term=include_short_term,
        )
        return self.retriever.build_context(hits, max_tokens=resolved_tokens)

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
            short_term_token_limit=self.config.memory_short_term_tokens,
            long_term_entries=len(long_entries),
            long_term_tokens=long_tokens,
            compression_trigger_ratio=self.config.memory_compression_trigger_ratio,
            retain_recent_turns=self.config.memory_retain_recent_turns,
            map_chunk_size=self.config.memory_map_chunk_size,
            long_term_entries_detail=tuple(long_entries) if include_entries else (),
        )

    # ------------------------------------------------------------------ deferred (phases 3.3-3.5)

    def prepare_messages(
        self,
        *,
        base_messages: list,
        query: str,
        tools: list[dict[str, Any]],
        force_compact: bool = False,
        focus: str = "",
    ) -> tuple[list, MemoryContext, Any | None]:
        raise NotImplementedError("prepare_messages() lands in phase 3.3-3.4 with the compressor.")

    def extract_facts(self, *, reason: str, run_id: str = "") -> list[MemoryEntry]:
        raise NotImplementedError("extract_facts() lands in phase 3.3 with the fact extractor.")

    def fork_for_task(self, *, session_id: str, run_id: str = "") -> "MemoryManager":
        raise NotImplementedError("fork_for_task() lands in phase 3.5 with Plan runtime wiring.")

    def clear_short_term(self, *, extract_first: bool = True, reason: str = "clear") -> tuple[int, list[MemoryEntry]]:
        if extract_first:
            raise NotImplementedError("clear_short_term(extract_first=True) needs the fact extractor (phase 3.3).")
        removed = self.short_term.clear()
        return len(removed), removed


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


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


__all__ = ["MemoryManager"]
