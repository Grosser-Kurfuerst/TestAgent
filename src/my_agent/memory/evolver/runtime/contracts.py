"""Runtime strategy contract for disabled, legacy, and formal Evolver modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, TYPE_CHECKING

from my_agent.config import AgentConfig
from my_agent.memory.experience.models import ExperienceMemory
from my_agent.memory.types import MemoryContext, RetrievalHit
from my_agent.policy.identity import PolicyIdentity

if TYPE_CHECKING:
    from my_agent.memory.evolver.task_session import (
        AgentEpisodeArtifact,
        EvolverFinalizeResult,
        TaskEvolverSession,
    )
    from my_agent.training.contracts import AuthoritativeTaskOutcome

TraceSink = Callable[[str, dict[str, Any]], None]
SaveExperience = Callable[..., tuple[ExperienceMemory, bool]]


class EvolverRuntime(Protocol):
    mode: str
    last_selection: Any | None

    @property
    def coordinator(self) -> Any | None: ...

    @coordinator.setter
    def coordinator(self, value: Any | None) -> None: ...

    @property
    def candidate_retriever(self) -> Any | None: ...

    @property
    def experience_retriever(self) -> Any | None: ...

    @experience_retriever.setter
    def experience_retriever(self, value: Any | None) -> None: ...

    @property
    def embedding_retriever(self) -> Any | None: ...

    @embedding_retriever.setter
    def embedding_retriever(self, value: Any | None) -> None: ...

    @property
    def selector(self) -> Any | None: ...

    @selector.setter
    def selector(self, value: Any | None) -> None: ...

    @property
    def writer(self) -> Any | None: ...

    @writer.setter
    def writer(self, value: Any | None) -> None: ...

    def set_trace_sink(self, trace_sink: TraceSink | None) -> None: ...

    def build_context(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext[ExperienceMemory]: ...

    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k_per_tier: int | None = None,
    ) -> list[RetrievalHit[ExperienceMemory]]: ...

    def begin_task(self, **kwargs: Any) -> "TaskEvolverSession": ...

    def finalize_task(
        self,
        episode: "AgentEpisodeArtifact",
        outcome: "AuthoritativeTaskOutcome",
    ) -> "EvolverFinalizeResult | None": ...

    def write_legacy_run(
        self,
        *,
        save_experience: SaveExperience,
        **kwargs: Any,
    ) -> Any: ...

    def require_formal_binding(
        self,
        *,
        config: AgentConfig,
        policy_identity: PolicyIdentity,
        repo_path: Path | None,
        manager_llm: Any | None,
        manager_store: Any,
        manager_project_key: str,
    ) -> None: ...

    def fork(self) -> "EvolverRuntime": ...


__all__ = ["EvolverRuntime", "SaveExperience", "TraceSink"]
