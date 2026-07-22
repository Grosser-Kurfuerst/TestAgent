"""Paper-faithful formal Evolver runtime strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from my_agent.config import AgentConfig
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.task_session import (
    AgentEpisodeArtifact,
    EvolverFinalizeResult,
    TaskEvolverSession,
)
from my_agent.memory.evolver.writing.contracts import ExperienceWriteResult
from my_agent.memory.experience.models import ExperienceMemory
from my_agent.memory.types import MemoryContext, RetrievalHit
from my_agent.policy.identity import PolicyIdentity, require_matching_policy_identity
from my_agent.training.contracts import AuthoritativeTaskOutcome


class FormalEvolverRuntime:
    mode = "formal"
    last_selection = None

    def __init__(
        self,
        *,
        coordinator: EvolverCoordinator,
        candidate_retriever: Any,
    ) -> None:
        self._coordinator = coordinator
        self._candidate_retriever = candidate_retriever
        self._experience_retriever: Any | None = None
        self._selector: Any | None = None
        self._writer: Any | None = None
        self._session: TaskEvolverSession | None = None
        self._context: MemoryContext[ExperienceMemory] | None = None

    @property
    def coordinator(self) -> EvolverCoordinator:
        return self._coordinator

    @coordinator.setter
    def coordinator(self, value: EvolverCoordinator) -> None:
        self._coordinator = value
        self._candidate_retriever = value.retriever

    @property
    def candidate_retriever(self) -> Any:
        return self._candidate_retriever

    @property
    def experience_retriever(self) -> Any | None:
        return self._experience_retriever

    @experience_retriever.setter
    def experience_retriever(self, value: Any | None) -> None:
        self._experience_retriever = value

    @property
    def embedding_retriever(self) -> Any:
        return self._candidate_retriever

    @embedding_retriever.setter
    def embedding_retriever(self, value: Any) -> None:
        self._candidate_retriever = value
        self._coordinator.retriever = value

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
        self._coordinator.set_trace_sink(trace_sink)

    def build_context(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext[ExperienceMemory]:
        del query, max_tokens, top_k_per_tier, max_items
        if self._context is None:
            return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
        return self._context

    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k_per_tier: int | None = None,
    ) -> list[RetrievalHit[ExperienceMemory]]:
        resolved_top_k = (
            self._coordinator.top_k_per_tier
            if top_k_per_tier is None
            else max(1, int(top_k_per_tier))
        )
        return list(self._candidate_retriever.retrieve_candidates(
            query,
            store=self._coordinator.store,
            project_key=self._coordinator.project_key,
            top_k_per_tier=resolved_top_k,
        ))

    def begin_task(
        self,
        *,
        task: str,
        task_id: str,
        task_group: str,
        trajectory_id: str,
        stream_id: str,
    ) -> TaskEvolverSession:
        if self._session is not None:
            raise RuntimeError("formal evolver selection already ran for this task manager")
        session = self._coordinator.begin_task(
            task=task,
            task_id=task_id,
            task_group=task_group,
            trajectory_id=trajectory_id,
            stream_id=stream_id,
        )
        self._session = session
        self._context = self._coordinator.context_for_session(session)
        return session

    def finalize_task(
        self,
        episode: AgentEpisodeArtifact,
        outcome: AuthoritativeTaskOutcome,
    ) -> EvolverFinalizeResult:
        return self._coordinator.finalize_task(episode, outcome)

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
        coordinator = self._coordinator
        require_matching_policy_identity(policy_identity, coordinator.policy_identity)
        if manager_llm is None:
            raise ValueError("formal MemoryManager requires the shared runtime policy")
        coordinator.require_formal_role_bindings(manager_llm)
        if coordinator.store is not manager_store:
            raise ValueError(
                "formal MemoryManager coordinator must use the manager experience store"
            )
        if coordinator.project_key != manager_project_key:
            raise ValueError("formal MemoryManager coordinator project_key mismatch")
        if coordinator.retriever is not self._candidate_retriever:
            raise ValueError("formal MemoryManager candidate retriever binding mismatch")
        expected_limits = (
            config.memory_evolver_candidate_top_k_per_tier,
            config.memory_evolver_selected_max_items,
            config.memory_evolver_selection_prompt_tokens,
            config.memory_evolver_maintenance_max_turns,
            config.memory_evolver_generation_temperature,
            config.memory_evolver_generation_top_p,
        )
        actual_limits = (
            coordinator.top_k_per_tier,
            coordinator.selected_max_items,
            coordinator.selection_token_budget,
            coordinator.maintenance_max_turns,
            coordinator.generation_temperature,
            coordinator.generation_top_p,
        )
        if actual_limits != expected_limits:
            raise ValueError(
                "formal MemoryManager coordinator limits do not match runtime config"
            )
        if coordinator.collection_round != config.memory_evolver_collection_round:
            raise ValueError(
                "formal MemoryManager collection round does not match runtime config"
            )
        if coordinator.dataset_split != config.memory_evolver_dataset_split:
            raise ValueError(
                "formal MemoryManager dataset split does not match runtime config"
            )
        expected_dataset_dir = (
            Path(config.memory_evolver_dataset_dir).expanduser().resolve()
            if config.memory_evolver_dataset_dir is not None
            else None
        )
        actual_dataset_dir = (
            coordinator.dataset_dir.expanduser().resolve()
            if coordinator.dataset_dir is not None
            else None
        )
        if actual_dataset_dir != expected_dataset_dir:
            raise ValueError(
                "formal MemoryManager dataset directory does not match runtime config"
            )
        expected_memory_dir = Path(config.memory_dir).expanduser().resolve()
        actual_memory_dir = manager_store.path.parent.expanduser().resolve()
        if actual_memory_dir != expected_memory_dir:
            raise ValueError("formal MemoryManager memory_dir does not match runtime config")
        if repo_path is not None:
            expected_project_key = str(config.memory_project_key or "").strip()
            if not expected_project_key:
                expected_project_key = _normalize_project_key(repo_path)
            if manager_project_key != expected_project_key:
                raise ValueError(
                    "formal MemoryManager project_key does not match runtime repository"
                )

    def fork(self) -> "FormalEvolverRuntime":
        return FormalEvolverRuntime(
            coordinator=self._coordinator,
            candidate_retriever=self._candidate_retriever,
        )


def _normalize_project_key(repo_path: Path) -> str:
    try:
        resolved = Path(repo_path).expanduser().resolve()
    except (OSError, RuntimeError):
        resolved = Path(repo_path).expanduser().absolute()
    return str(resolved)


__all__ = ["FormalEvolverRuntime"]
