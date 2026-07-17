from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from my_agent.config import AgentConfig
from my_agent.context import ContextProfile
from my_agent.hitl.handler import HitlHandler
from my_agent.llm import AgentLLM
from my_agent.memory import MemoryManager, NoopMemoryManager
from my_agent.observability.tracing import TraceWriter
from my_agent.policy.runtime_validation import require_formal_policy
from my_agent.repo import RepoContextRender, RepoIndexer
from my_agent.runtime.cancellation import CancellationToken
from my_agent.schema import AgentState, TraceEvent
from my_agent.tools import should_skip_path

EventSink = Callable[[Any], None]
MemoryTraceSnapshot = tuple[Any | None, Any | None]


@dataclass(frozen=True)
class RunContext:
    state: AgentState
    writer: TraceWriter
    memory: MemoryManager
    repo_snapshot: Any | None = None
    repo_context: str = ""
    repo_context_render: RepoContextRender | None = None


class AgentBase(ABC):
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
        hitl_handler: HitlHandler | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.command_timeout = command_timeout
        self.event_sink = event_sink
        self.memory_manager = memory_manager
        self.hitl_handler = hitl_handler

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
        if state.cancellation_token is None:
            state.cancellation_token = CancellationToken()
        repo_path = Path(repo if repo is not None else state.repo_path).resolve()
        state.repo_path = repo_path
        writer = self._create_writer(state)
        memory, memory_trace_snapshot = self._bind_memory(repo_path, state, writer)
        try:
            if emit_memory_loaded:
                self._emit_memory_loaded(writer, state, memory)
            repo_snapshot = self._repo_snapshot(
                repo_path,
                query or state.task,
                writer,
                state,
                getattr(memory, "context_profile", ContextProfile.resolve(self.config, getattr(self.llm, "model", ""))),
            ) if index_repo else None
            repo_context_render = (
                repo_snapshot.render_context(max_tokens=memory.context_profile.repo_context_budget_tokens)
                if repo_snapshot is not None
                else None
            )
            yield RunContext(
                state=state,
                writer=writer,
                memory=memory,
                repo_snapshot=repo_snapshot,
                repo_context=repo_context_render.text if repo_context_render is not None else "",
                repo_context_render=repo_context_render,
            )
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
        def trace_sink(event: str, payload: dict[str, object]) -> None:
            self._emit_trace(writer, state, event, payload)

        if not self.config.memory_enabled:
            return (
                NoopMemoryManager(
                    config=self.config,
                    repo_path=Path(repo).resolve(),
                    session_id=state.run_id,
                    trace_sink=trace_sink,
                ),
                None,
            )
        active_memory = memory_manager if memory_manager is not None else self.memory_manager
        if active_memory is not None:
            if self.config.memory_evolver_mode == "formal":
                policy_identity = require_formal_policy(self.config, self.llm)
                if policy_identity is None:
                    raise ValueError("formal OPD runtime requires a validated policy identity")
                active_memory.require_formal_runtime_binding(
                    config=self.config,
                    policy_identity=policy_identity,
                    repo_path=Path(repo).resolve(),
                )
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

    def _repo_snapshot(
        self,
        repo: str | Path,
        query: str,
        writer: TraceWriter,
        state: AgentState,
        context_profile: ContextProfile,
    ) -> Any:
        repo_path = Path(repo).resolve()
        snapshot = RepoIndexer(
            repo_path,
            skip_predicate=lambda path: should_skip_path(repo_path, path),
            cancellation_token=state.cancellation_token,
        ).snapshot(query=query)
        render = snapshot.render_context(max_tokens=context_profile.repo_context_budget_tokens)
        payload = {"repo_path": str(repo_path), "task": query, "tree": snapshot.tree, "symbols": snapshot.symbols}
        payload.update(render.to_trace_payload())
        self._emit_trace(
            writer,
            state,
            "repo.indexed",
            payload,
        )
        return snapshot
