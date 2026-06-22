from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from my_agent.config import AgentConfig
from my_agent.indexer import RepoIndexer
from my_agent.llm import AgentLLM
from my_agent.memory import MemoryManager
from my_agent.schema import AgentState, TraceEvent
from my_agent.tools import should_skip_path
from my_agent.tracing import TraceWriter

EventSink = Callable[[Any], None]
MemoryTraceSnapshot = tuple[Any | None, Any | None]


@dataclass(frozen=True)
class RunContext:
    state: AgentState
    writer: TraceWriter
    memory: MemoryManager
    repo_snapshot: Any | None = None


class AgentRunBase(ABC):
    """Shared runtime plumbing for ReAct, plan, and team modes."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: AgentLLM,
        trace_dir: str | Path,
        command_timeout: int,
        event_sink: EventSink | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.event_sink = event_sink
        self.memory_manager = memory_manager

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        ...

    @contextmanager
    def open_run_context(
        self,
        state: AgentState,
        *,
        repo: str | Path | None = None,
        query: str | None = None,
        index_repo: bool = True,
        emit_memory_loaded: bool = True,
    ) -> Iterator[RunContext]:
        repo_path = Path(repo if repo is not None else state.repo_path).resolve()
        state.repo_path = repo_path
        writer = self._create_writer(state)
        memory, memory_trace_snapshot = self._bind_memory(repo_path, state, writer)
        try:
            if emit_memory_loaded:
                self._emit_memory_loaded(writer, state, memory)
            repo_snapshot = self._repo_snapshot(repo_path, query or state.task, writer, state) if index_repo else None
            yield RunContext(state=state, writer=writer, memory=memory, repo_snapshot=repo_snapshot)
        finally:
            self._restore_memory(memory, memory_trace_snapshot)

    def _create_writer(self, state: AgentState) -> TraceWriter:
        writer = TraceWriter.create(self.trace_dir, state.run_id)
        state.trace_path = writer.path
        return writer

    def _emit_trace(self, writer: TraceWriter, state: AgentState, event: str, payload: dict[str, object]) -> None:
        writer.append(TraceEvent(event=event, payload=payload, run_id=state.run_id))

    def _emit_event(self, event: object) -> None:
        if self.event_sink is not None:
            self.event_sink(event)

    def _bind_memory(
        self,
        repo: str | Path,
        state: AgentState,
        writer: TraceWriter,
        *,
        memory_manager: MemoryManager | None = None,
    ) -> tuple[MemoryManager, MemoryTraceSnapshot | None]:
        trace_sink = lambda event, payload: self._emit_trace(writer, state, event, payload)
        active_memory = memory_manager if memory_manager is not None else self.memory_manager
        if active_memory is not None:
            snapshot = active_memory.set_trace_sink(trace_sink)
            return active_memory, snapshot
        return (
            MemoryManager.from_config(
                config=self.config,
                llm=self.llm,
                repo_path=Path(repo).resolve(),
                session_id=state.run_id,
                trace_sink=trace_sink,
            ),
            None,
        )

    def _restore_memory(self, memory: MemoryManager, snapshot: MemoryTraceSnapshot | None) -> None:
        if snapshot is not None:
            memory.restore_trace_sink(snapshot)

    def _emit_memory_loaded(self, writer: TraceWriter, state: AgentState, memory: MemoryManager) -> None:
        status = memory.status(include_entries=False)
        self._emit_trace(
            writer,
            state,
            "memory.loaded",
            {
                "storage_path": status.storage_path,
                "short_term_entries": status.short_term_entries,
                "long_term_entries": status.long_term_entries,
                "long_term_tokens": status.long_term_tokens,
            },
        )

    def _repo_snapshot(self, repo: str | Path, query: str, writer: TraceWriter, state: AgentState) -> Any:
        repo_path = Path(repo).resolve()
        snapshot = RepoIndexer(repo_path, skip_predicate=lambda path: should_skip_path(repo_path, path)).snapshot(query=query)
        self._emit_trace(
            writer,
            state,
            "repo.indexed",
            {"repo_path": str(repo_path), "task": query, "tree": snapshot.tree, "symbols": snapshot.symbols},
        )
        return snapshot
