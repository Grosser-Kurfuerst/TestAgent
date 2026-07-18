"""Disabled Evolver runtime strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from my_agent.config import AgentConfig
from my_agent.memory.evolver.writing.contracts import ExperienceWriteResult
from my_agent.memory.experience.models import ExperienceMemory
from my_agent.memory.types import MemoryContext
from my_agent.policy.identity import PolicyIdentity


class DisabledEvolverRuntime:
    mode = "off"
    last_selection = None

    def __init__(self, *, trace_sink: Any | None = None) -> None:
        self._trace_sink = trace_sink
        self._coordinator: Any | None = None
        self._experience_retriever: Any | None = None
        self._embedding_retriever: Any | None = None
        self._selector: Any | None = None
        self._writer: Any | None = None

    @property
    def coordinator(self) -> Any | None:
        return self._coordinator

    @coordinator.setter
    def coordinator(self, value: Any | None) -> None:
        self._coordinator = value

    @property
    def candidate_retriever(self) -> None:
        return None

    @property
    def experience_retriever(self) -> Any | None:
        return self._experience_retriever

    @experience_retriever.setter
    def experience_retriever(self, value: Any | None) -> None:
        self._experience_retriever = value

    @property
    def embedding_retriever(self) -> Any | None:
        return self._embedding_retriever

    @embedding_retriever.setter
    def embedding_retriever(self, value: Any | None) -> None:
        self._embedding_retriever = value

    @property
    def selector(self) -> Any | None:
        return self._selector

    @selector.setter
    def selector(self, value: Any | None) -> None:
        self._selector = value

    @property
    def writer(self) -> Any | None:
        return self._writer

    @writer.setter
    def writer(self, value: Any | None) -> None:
        self._writer = value

    def set_trace_sink(self, trace_sink: Any | None) -> None:
        self._trace_sink = trace_sink

    def build_context(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext[ExperienceMemory]:
        del max_tokens, top_k_per_tier, max_items
        context: MemoryContext[ExperienceMemory] = MemoryContext(
            injected_text="",
            hits=[],
            estimated_tokens=0,
        )
        self._trace("memory.retrieved", {
            "query_chars": len(query),
            "hits": 0,
            "tokens": 0,
            "include_short_term": False,
            "mode": "off",
        })
        return context

    def retrieve_candidates(self, query: str, **kwargs: Any) -> list[Any]:
        del query, kwargs
        return []

    def begin_task(self, **kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("formal evolver task session is unavailable")

    def finalize_task(self, episode: Any, outcome: Any) -> None:
        del episode, outcome
        return None

    def write_legacy_run(self, **kwargs: Any) -> ExperienceWriteResult:
        del kwargs
        return ExperienceWriteResult()

    def require_formal_binding(
        self,
        *,
        config: AgentConfig,
        policy_identity: PolicyIdentity,
        repo_path: Path | None,
        manager_llm: Any | None,
        manager_store: Any,
        manager_project_key: str,
    ) -> None:
        del config, policy_identity, repo_path, manager_llm, manager_store, manager_project_key
        raise ValueError("formal OPD runtime cannot use a non-formal MemoryManager")

    def fork(self) -> "DisabledEvolverRuntime":
        return DisabledEvolverRuntime(trace_sink=self._trace_sink)

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(event, payload)
        except Exception:
            pass


__all__ = ["DisabledEvolverRuntime"]
