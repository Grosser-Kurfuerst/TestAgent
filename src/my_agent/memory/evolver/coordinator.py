"""Task lifecycle coordinator for retrieve-once and outcome-finalized writes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from my_agent.memory.embedding_retrieval import EmbeddingRetriever
from my_agent.memory.evolver.task_session import (
    AgentEpisodeArtifact,
    EvolverFinalizeResult,
    TaskEvolverSession,
)
from my_agent.memory.evolver.types import ExperienceMemory, ExperienceTier
from my_agent.memory.evolver.writer import ExperienceWriteResult
from my_agent.memory.experience_store import ExperienceStore
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryContext, RetrievalHit
from my_agent.policy.identity import PolicyIdentity, canonical_sha256, require_matching_policy_identity
from my_agent.training.contracts import AuthoritativeTaskOutcome
from my_agent.training.role_views import CandidateSnapshotEntry


TraceSink = Callable[[str, dict[str, Any]], None]
WriterCallback = Callable[[AgentEpisodeArtifact, AuthoritativeTaskOutcome], ExperienceWriteResult]


class TaskSelectionPolicy(Protocol):
    def select(
        self,
        *,
        task: str,
        candidates: tuple[CandidateSnapshotEntry, ...],
        token_budget: int,
        max_items: int,
    ) -> tuple[str, ...]: ...


class EmptyTaskSelectionPolicy:
    """Iteration-2 fail-closed selector; replaced by the LLM selector in 3B."""

    def select(
        self,
        *,
        task: str,
        candidates: tuple[CandidateSnapshotEntry, ...],
        token_budget: int,
        max_items: int,
    ) -> tuple[str, ...]:
        del task, candidates, token_budget, max_items
        return ()


class EvolverCoordinator:
    def __init__(
        self,
        *,
        store: ExperienceStore,
        project_key: str,
        policy_identity: PolicyIdentity,
        retriever: EmbeddingRetriever | None = None,
        selector: TaskSelectionPolicy | None = None,
        writer: WriterCallback | None = None,
        trace_sink: TraceSink | None = None,
        top_k_per_tier: int = 50,
        selected_max_items: int = 20,
        selection_token_budget: int = 1_800,
    ) -> None:
        if not project_key:
            raise ValueError("evolver coordinator requires project_key")
        if not isinstance(policy_identity, PolicyIdentity):
            raise ValueError("evolver coordinator requires PolicyIdentity")
        self.store = store
        self.project_key = project_key
        self.policy_identity = policy_identity
        self.retriever = retriever
        self.selector = selector or EmptyTaskSelectionPolicy()
        self.writer = writer
        self.trace_sink = trace_sink
        self.top_k_per_tier = top_k_per_tier
        self.selected_max_items = selected_max_items
        self.selection_token_budget = selection_token_budget
        self._finalized_trajectories: set[str] = set()

    def begin_task(
        self,
        *,
        task: str,
        task_id: str,
        task_group: str,
        trajectory_id: str,
        stream_id: str,
    ) -> TaskEvolverSession:
        if self.retriever is None:
            raise RuntimeError("evolver coordinator begin_task requires an embedding retriever")
        hits = self.retriever.retrieve_candidates(
            task,
            store=self.store,
            project_key=self.project_key,
            top_k_per_tier=self.top_k_per_tier,
        )
        repository_revision = self.retriever.last_metrics.repository_revision
        candidates = _candidate_snapshot(hits)
        candidate_snapshot_hash = canonical_sha256([item.to_dict() for item in candidates])
        selected_ids = self.selector.select(
            task=task,
            candidates=candidates,
            token_budget=self.selection_token_budget,
            max_items=self.selected_max_items,
        )
        selected_ids = _validate_and_clip_selection(
            selected_ids,
            candidates=candidates,
            token_budget=self.selection_token_budget,
            max_items=self.selected_max_items,
        )
        context = _render_selected_context(selected_ids, hits)
        session = TaskEvolverSession(
            task_id=task_id,
            task_group=task_group,
            trajectory_id=trajectory_id,
            stream_id=stream_id,
            memory_project_key=self.project_key,
            policy_identity=self.policy_identity,
            repository_revision=repository_revision,
            candidate_snapshot_hash=candidate_snapshot_hash,
            selected_memory_ids=selected_ids,
            rendered_memory_context=context.injected_text,
            candidate_snapshot=candidates,
        )
        self._trace("memory.evolver_session_started", {
            "task_id": task_id,
            "task_group": task_group,
            "trajectory_id": trajectory_id,
            "repository_revision": repository_revision,
            "candidate_snapshot_hash": candidate_snapshot_hash,
            "candidate_count": len(candidates),
            "candidates": [item.to_dict() for item in candidates],
            "selected_count": len(selected_ids),
            "selected_memory_ids": list(selected_ids),
            "selection_calls": 1,
            **self.retriever.last_metrics.to_trace_payload(),
        })
        return session

    def context_for_session(self, session: TaskEvolverSession) -> MemoryContext[ExperienceMemory]:
        require_matching_policy_identity(self.policy_identity, session.policy_identity)
        return MemoryContext(
            injected_text=session.rendered_memory_context,
            hits=[],
            estimated_tokens=estimate_tokens(session.rendered_memory_context),
        )

    def finalize_task(
        self,
        episode: AgentEpisodeArtifact,
        outcome: AuthoritativeTaskOutcome,
    ) -> EvolverFinalizeResult:
        outcome.require_formal()
        if episode.session.task_id != outcome.task_id or episode.session.task_group != outcome.task_group:
            raise ValueError("authoritative outcome does not match the evolver session")
        require_matching_policy_identity(self.policy_identity, episode.session.policy_identity)
        trajectory_id = episode.session.trajectory_id
        if trajectory_id in self._finalized_trajectories:
            raise ValueError(f"evolver episode already finalized: {trajectory_id}")
        self._finalized_trajectories.add(trajectory_id)

        if self.writer is None:
            writer_result = ExperienceWriteResult()
            writer_status = "no_write"
        else:
            current_revision = self.store.revision()
            if current_revision != episode.session.repository_revision:
                self._trace("memory.evolver_task_finalized", {
                    "task_id": outcome.task_id,
                    "task_group": outcome.task_group,
                    "trajectory_id": trajectory_id,
                    "outcome_finalized": outcome.outcome_finalized,
                    "evaluator_name": outcome.evaluator.name,
                    "evaluator_version": outcome.evaluator.version,
                    "evaluator_hash": outcome.evaluator.evaluator_hash,
                    "resolved": outcome.resolved,
                    "reward": outcome.reward,
                    "writer_status": "failed_no_write",
                    "writer_failure_reason": "stale_repository_revision",
                    "repository_revision_expected": episode.session.repository_revision,
                    "repository_revision_after": current_revision,
                    "written_memory_ids": [],
                })
                return EvolverFinalizeResult(
                    writer_status="failed_no_write",
                    written_memory_ids=(),
                    repository_revision_after=current_revision,
                )
            try:
                writer_result = self.writer(episode, outcome)
            except Exception as exc:  # noqa: BLE001 - formal writer failures are audited no-write outcomes
                writer_result = ExperienceWriteResult(error=f"{type(exc).__name__}: {exc}")
            if writer_result.error:
                writer_status = "failed_no_write"
            elif writer_result.saved:
                writer_status = "committed"
            else:
                writer_status = "no_write"
        revision_after = self.store.revision()
        written_ids = tuple(item.id for item in writer_result.saved)
        self._trace("memory.evolver_task_finalized", {
            "task_id": outcome.task_id,
            "task_group": outcome.task_group,
            "trajectory_id": trajectory_id,
            "outcome_finalized": outcome.outcome_finalized,
            "evaluator_name": outcome.evaluator.name,
            "evaluator_version": outcome.evaluator.version,
            "evaluator_hash": outcome.evaluator.evaluator_hash,
            "resolved": outcome.resolved,
            "reward": outcome.reward,
            "writer_status": writer_status,
            "written_memory_ids": list(written_ids),
            "repository_revision_after": revision_after,
        })
        return EvolverFinalizeResult(
            writer_status=writer_status,
            written_memory_ids=written_ids,
            repository_revision_after=revision_after,
        )

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self.trace_sink is not None:
            self.trace_sink(event, payload)


def _candidate_snapshot(
    hits: tuple[RetrievalHit[ExperienceMemory], ...],
) -> tuple[CandidateSnapshotEntry, ...]:
    ranks = {tier: 0 for tier in ExperienceTier}
    candidates: list[CandidateSnapshotEntry] = []
    for hit in hits:
        tier = hit.entry.tier
        ranks[tier] += 1
        candidates.append(CandidateSnapshotEntry(
            label=f"RETRIEVED_{tier.value.upper()}_{ranks[tier]:02d}",
            memory_id=hit.entry.id,
            tier=tier.value,
            content=hit.entry.content,
            retrieval_score=float(hit.score),
            rank=ranks[tier],
            token_count=hit.entry.token_count,
        ))
    return tuple(candidates)


def _validate_and_clip_selection(
    selected_ids: tuple[str, ...],
    *,
    candidates: tuple[CandidateSnapshotEntry, ...],
    token_budget: int,
    max_items: int,
) -> tuple[str, ...]:
    if not isinstance(selected_ids, tuple) or any(not isinstance(item, str) for item in selected_ids):
        raise ValueError("selector must return a tuple of memory IDs")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selector returned duplicate memory IDs")
    by_id = {item.memory_id: item for item in candidates}
    if any(memory_id not in by_id for memory_id in selected_ids):
        raise ValueError("selector referenced memory outside the frozen candidate snapshot")
    kept: list[str] = []
    used_tokens = 0
    for memory_id in selected_ids:
        if len(kept) >= max_items:
            break
        candidate = by_id[memory_id]
        if used_tokens + candidate.token_count > token_budget:
            break
        kept.append(memory_id)
        used_tokens += candidate.token_count
    return tuple(kept)


def _render_selected_context(
    selected_ids: tuple[str, ...],
    hits: tuple[RetrievalHit[ExperienceMemory], ...],
) -> MemoryContext[ExperienceMemory]:
    by_id = {hit.entry.id: hit for hit in hits}
    selected_hits = [by_id[memory_id] for memory_id in selected_ids]
    if not selected_hits:
        return MemoryContext(injected_text="", hits=[], estimated_tokens=0)
    blocks = ["[Selected evolver memory - frozen for this task]"]
    for hit in selected_hits:
        blocks.append(f"[{hit.entry.id} | {hit.entry.tier.value}]\n{hit.entry.content}")
    rendered = "\n\n".join(blocks)
    return MemoryContext(rendered, selected_hits, estimate_tokens(rendered))


__all__ = [
    "EmptyTaskSelectionPolicy",
    "EvolverCoordinator",
    "TaskSelectionPolicy",
    "WriterCallback",
]
