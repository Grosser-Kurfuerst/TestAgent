"""Task lifecycle coordinator for retrieve-once and outcome-finalized writes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from my_agent.memory.embedding_retrieval import EmbeddingRetriever
from my_agent.memory.evolver.task_session import (
    AgentEpisodeArtifact,
    EvolverFinalizeResult,
    TaskEvolverSession,
)
from my_agent.memory.evolver.cadence_ledger import (
    MAINTENANCE_HISTORY_FILENAME,
    CadenceLedger,
    CadenceRecord,
    load_formal_maintenance_history,
)
from my_agent.memory.evolver.cadence_schema import EVOLVER_STATE_FILENAME
from my_agent.memory.evolver.contracts import MaintenanceOperation
from my_agent.memory.evolver.formal_writer import FormalExperienceWriter
from my_agent.memory.evolver.maintenance_agent import FormalMaintenanceAgent, FormalMaintenanceResult
from my_agent.memory.evolver.selector_prompt import LLMTaskSelectionPolicy
from my_agent.memory.evolver.types import ExperienceMemory, ExperienceTier
from my_agent.memory.evolver.writer import ExperienceWriteResult
from my_agent.memory.evolver.transaction import apply_formal_maintenance_operations
from my_agent.memory.experience_store import ExperienceStore
from my_agent.memory.store_errors import MemoryStorePostCommitError
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryContext, RetrievalHit
from my_agent.opd_data.runtime_recorder import RuntimeEvidenceRecorder
from my_agent.policy.identity import PolicyIdentity, canonical_sha256, require_matching_policy_identity
from my_agent.policy.contracts import GenerationPolicy
from my_agent.training.contracts import AuthoritativeTaskOutcome
from my_agent.training.decision_log import DecisionEventContext, DecisionEventRecorder
from my_agent.training.role_views import CandidateSnapshotEntry, SELECTED_MEMORY_CONTEXT_HEADER
from my_agent.training.role_views import TaskOutcomeRef


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
        context: DecisionEventContext,
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
        context: DecisionEventContext,
    ) -> tuple[str, ...]:
        del task, candidates, token_budget, max_items, context
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
        policy: GenerationPolicy | None = None,
        decision_recorder: DecisionEventRecorder | None = None,
        dataset_dir: str | Path | None = None,
        trace_sink: TraceSink | None = None,
        top_k_per_tier: int = 50,
        selected_max_items: int = 20,
        selection_token_budget: int = 1_800,
        maintenance_max_turns: int = 8,
        maintenance_interval_tasks: int = 30,
        ledger_path: str | Path | None = None,
        collection_round: int = 0,
        dataset_split: str = "train",
    ) -> None:
        if not project_key:
            raise ValueError("evolver coordinator requires project_key")
        if not isinstance(policy_identity, PolicyIdentity):
            raise ValueError("evolver coordinator requires PolicyIdentity")
        if policy is not None:
            require_matching_policy_identity(policy_identity, policy.identity())
        if decision_recorder is not None:
            require_matching_policy_identity(policy_identity, decision_recorder.policy.identity())
        self.store = store
        self.project_key = project_key
        self.policy_identity = policy_identity
        self.policy = policy
        self.retriever = retriever
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else None
        self.collection_round = collection_round
        self.dataset_split = dataset_split
        self.decision_recorder = decision_recorder
        dataset_path = (
            self.dataset_dir / "decision_events.jsonl"
            if self.dataset_dir is not None
            else None
        )
        if self.decision_recorder is None and policy is not None:
            self.decision_recorder = DecisionEventRecorder(
                policy=policy,
                dataset_path=dataset_path,
                trace_sink=trace_sink,
            )
        elif self.decision_recorder is not None and dataset_path is not None:
            self.decision_recorder.bind_dataset_path(dataset_path)
        if dataset_dir is not None and self.decision_recorder is None:
            raise ValueError("formal dataset recording requires a decision recorder")
        self.runtime_evidence_recorder = (
            RuntimeEvidenceRecorder(
                dataset_dir=self.dataset_dir,
                store=store,
                decision_recorder=self.decision_recorder,
                collection_round=collection_round,
                split=dataset_split,
            )
            if self.dataset_dir is not None and self.decision_recorder is not None
            else None
        )
        if selector is not None:
            self.selector = selector
        elif policy is not None and self.decision_recorder is not None:
            self.selector = LLMTaskSelectionPolicy(policy=policy, recorder=self.decision_recorder)
        else:
            self.selector = EmptyTaskSelectionPolicy()
        if writer is not None:
            self.writer = writer
        elif policy is not None and self.decision_recorder is not None:
            self.writer = FormalExperienceWriter(
                policy=policy,
                recorder=self.decision_recorder,
                store=store,
                project_key=project_key,
            )
        else:
            self.writer = None
        self.maintainer = (
            FormalMaintenanceAgent(
                policy=policy,
                recorder=self.decision_recorder,
                store=store,
                project_key=project_key,
                max_turns=maintenance_max_turns,
            )
            if policy is not None and self.decision_recorder is not None
            else None
        )
        self.trace_sink = trace_sink
        self.top_k_per_tier = top_k_per_tier
        self.selected_max_items = selected_max_items
        self.selection_token_budget = selection_token_budget
        self.maintenance_max_turns = maintenance_max_turns
        self.maintenance_interval_tasks = maintenance_interval_tasks
        self.ledger_path = (
            Path(ledger_path)
            if ledger_path is not None
            else self.store.path.parent / EVOLVER_STATE_FILENAME
        )
        self.maintenance_history_path = self.store.path.parent / MAINTENANCE_HISTORY_FILENAME
        self.cadence_ledger = CadenceLedger(
            self.ledger_path,
            interval_tasks=maintenance_interval_tasks,
            process_lock=self.store.exclusive_process_lock,
        )
        self._finalized_trajectories: set[str] = set()
        self._reconcile_persisted_cadences()

    def set_trace_sink(self, trace_sink: TraceSink | None) -> None:
        self.trace_sink = trace_sink
        if self.decision_recorder is not None:
            self.decision_recorder.trace_sink = trace_sink

    def require_formal_role_bindings(self, policy: GenerationPolicy) -> None:
        if self.policy is not policy:
            raise ValueError("formal coordinator must retain the shared runtime policy handle")
        recorder = self.decision_recorder
        if recorder is None or recorder.policy is not policy:
            raise ValueError("formal coordinator decision recorder must use the shared policy")
        if not isinstance(self.selector, LLMTaskSelectionPolicy):
            raise ValueError("formal coordinator requires LLMTaskSelectionPolicy")
        if self.selector.policy is not policy or self.selector.recorder is not recorder:
            raise ValueError("formal selector must share the runtime policy and decision recorder")
        if not isinstance(self.writer, FormalExperienceWriter):
            raise ValueError("formal coordinator requires FormalExperienceWriter")
        if self.writer.policy is not policy or self.writer.recorder is not recorder:
            raise ValueError("formal writer must share the runtime policy and decision recorder")
        if self.writer.store is not self.store or self.writer.project_key != self.project_key:
            raise ValueError("formal writer repository binding mismatch")
        if not isinstance(self.maintainer, FormalMaintenanceAgent):
            raise ValueError("formal coordinator requires FormalMaintenanceAgent")
        if self.maintainer.policy is not policy or self.maintainer.recorder is not recorder:
            raise ValueError("formal maintainer must share the runtime policy and decision recorder")
        if self.maintainer.store is not self.store or self.maintainer.project_key != self.project_key:
            raise ValueError("formal maintainer repository binding mismatch")
        if self.maintainer.max_turns != self.maintenance_max_turns:
            raise ValueError("formal maintainer max-turn binding mismatch")

    def run_maintenance(
        self,
        *,
        maintenance_id: str,
        attempt_id: str,
        stream_id: str,
        task_group: str,
        history_window: tuple[TaskOutcomeRef, ...] = (),
    ) -> FormalMaintenanceResult:
        if self.maintainer is None:
            raise RuntimeError("formal maintenance agent is unavailable")
        return self.maintainer.run(
            maintenance_id=maintenance_id,
            attempt_id=attempt_id,
            stream_id=stream_id,
            task_group=task_group,
            history_window=history_window,
        )

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
        decision_context = DecisionEventContext(
            trajectory_id=trajectory_id,
            turn_index=0,
            step_index=0,
            task_id=task_id,
            task_group=task_group,
            stream_id=stream_id,
            memory_project_key=self.project_key,
            run_id=trajectory_id,
            repository_revision=repository_revision,
            candidate_snapshot_hash=candidate_snapshot_hash,
        )
        selected_ids = self.selector.select(
            task=task,
            candidates=candidates,
            token_budget=self.selection_token_budget,
            max_items=self.selected_max_items,
            context=decision_context,
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
        if self.runtime_evidence_recorder is not None:
            self.runtime_evidence_recorder.begin_task(
                task=task,
                session=session,
                selection_token_budget=self.selection_token_budget,
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
        outcome_event_id = canonical_sha256({
            "schema_version": "opd-task-outcome-event-v1",
            "trajectory_id": trajectory_id,
            **outcome.to_dict(),
        })
        self._trace("memory.task_outcome_finalized", {
            "outcome_event_id": outcome_event_id,
            "trajectory_id": trajectory_id,
            **outcome.to_dict(),
        })

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
                    "outcome_event_id": outcome_event_id,
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
                cadence_id, maintenance_status = self._advance_cadence(
                    episode=episode,
                    outcome=outcome,
                    writer_status="failed_no_write",
                    repository_revision_after=current_revision,
                    written_memory_ids=(),
                )
                return EvolverFinalizeResult(
                    writer_status="failed_no_write",
                    written_memory_ids=(),
                    repository_revision_after=current_revision,
                    cadence_id=cadence_id,
                    maintenance_status=maintenance_status,
                )
            try:
                writer_result = self.writer(episode, outcome)
            except MemoryStorePostCommitError:
                raise
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
            "outcome_event_id": outcome_event_id,
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
        cadence_id, maintenance_status = self._advance_cadence(
            episode=episode,
            outcome=outcome,
            writer_status=writer_status,
            repository_revision_after=revision_after,
            written_memory_ids=written_ids,
        )
        return EvolverFinalizeResult(
            writer_status=writer_status,
            written_memory_ids=written_ids,
            repository_revision_after=revision_after,
            cadence_id=cadence_id,
            maintenance_status=maintenance_status,
        )

    def _advance_cadence(
        self,
        *,
        episode: AgentEpisodeArtifact,
        outcome: AuthoritativeTaskOutcome,
        writer_status: str,
        repository_revision_after: str,
        written_memory_ids: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        session = episode.session
        advance = self.cadence_ledger.record_task_completion(
            stream_id=session.stream_id,
            memory_project_key=session.memory_project_key,
            task_id=session.task_id,
            task_valid=outcome.task_valid,
            outcome_finalized=outcome.outcome_finalized,
            writer_terminal_status=writer_status,
            repository_revision_after_writer=repository_revision_after,
            outcome=outcome.to_ref(),
        )
        self._trace("memory.evolver_cadence_advanced", {
            "task_id": session.task_id,
            "stream_id": session.stream_id,
            "memory_project_key": session.memory_project_key,
            "counted": advance.counted,
            "task_ordinal": advance.task_ordinal,
            "cadence_id": advance.cadence.cadence_id if advance.cadence else None,
        })
        if (
            self.runtime_evidence_recorder is not None
            and advance.task_ordinal is not None
            and outcome.task_valid
            and outcome.outcome_finalized
        ):
            self.runtime_evidence_recorder.finish_task(
                episode=episode,
                outcome=outcome,
                task_ordinal=advance.task_ordinal,
                written_memory_ids=written_memory_ids,
            )
        due = self.cadence_ledger.oldest_open_cadence(
            stream_id=session.stream_id,
            memory_project_key=session.memory_project_key,
        )
        if due is None:
            if advance.cadence is None:
                return None, None
            return advance.cadence.cadence_id, advance.cadence.status
        return due.cadence_id, self._run_or_reconcile_cadence(due)

    def _run_or_reconcile_cadence(
        self,
        cadence: CadenceRecord,
    ) -> str:
        with self.store.exclusive_process_lock():
            return self._run_or_reconcile_cadence_locked(cadence)

    def _run_or_reconcile_cadence_locked(
        self,
        cadence: CadenceRecord,
    ) -> str:
        history = load_formal_maintenance_history(
            self.maintenance_history_path,
            cadence_id=cadence.cadence_id,
        )
        if history.completion is not None:
            if self.runtime_evidence_recorder is not None:
                self.runtime_evidence_recorder.recover_maintenance(
                    cadence_id=cadence.cadence_id,
                    project_key=cadence.memory_project_key,
                    status=str(history.completion["status"]),
                )
            committed = self.cadence_ledger.mark_committed(
                cadence.cadence_id,
                maintenance_plan_id=str(history.completion["plan_id"]),
                repository_revision_after=str(history.completion["after_revision"]),
            )
            return committed.status
        if history.intent is not None:
            operations = tuple(
                MaintenanceOperation.from_dict(item)
                for item in history.intent["operations"]
            )
            applied = apply_formal_maintenance_operations(
                store=self.store,
                cadence_id=cadence.cadence_id,
                stream_id=cadence.stream_id,
                expected_revision=str(history.intent["before_revision"]),
                project_key=cadence.memory_project_key,
                operations=operations,
                history_path=self.maintenance_history_path,
            )
            if applied.status in {"committed", "noop"}:
                if self.runtime_evidence_recorder is not None:
                    self.runtime_evidence_recorder.recover_maintenance(
                        cadence_id=cadence.cadence_id,
                        project_key=cadence.memory_project_key,
                        status=applied.status,
                    )
                self.cadence_ledger.mark_committed(
                    cadence.cadence_id,
                    maintenance_plan_id=applied.plan_id,
                    repository_revision_after=applied.after_revision,
                )
            return applied.status
        if self.maintainer is None:
            return cadence.status
        task_group, history_window = self.cadence_ledger.cadence_context(cadence)
        attempt_id = canonical_sha256({
            "schema_version": "opd-maintenance-attempt-runtime-v1",
            "cadence_id": cadence.cadence_id,
            "event_count": len(
                self.decision_recorder.events_for(cadence.cadence_id)
                if self.decision_recorder is not None
                else ()
            ),
        })
        if self.runtime_evidence_recorder is not None:
            attempt_id = self.runtime_evidence_recorder.begin_maintenance(
                cadence_id=cadence.cadence_id,
                task_group=task_group,
                stream_id=cadence.stream_id,
                project_key=cadence.memory_project_key,
                as_of_task_ordinal=cadence.boundary_ordinal,
                history_window=history_window,
            )
        self.cadence_ledger.mark_started(cadence.cadence_id)
        result = self.run_maintenance(
            maintenance_id=cadence.cadence_id,
            attempt_id=attempt_id,
            stream_id=cadence.stream_id,
            task_group=task_group,
            history_window=history_window,
        )
        if self.runtime_evidence_recorder is not None:
            self.runtime_evidence_recorder.finish_maintenance(
                cadence_id=cadence.cadence_id,
                attempt_id=attempt_id,
                project_key=cadence.memory_project_key,
                status=result.status,
            )
        if result.status in {"committed", "noop"}:
            self.cadence_ledger.mark_committed(
                cadence.cadence_id,
                maintenance_plan_id=result.plan_id,
                repository_revision_after=result.after_revision,
            )
        self._trace("memory.evolver_maintenance_cadence", {
            "cadence_id": cadence.cadence_id,
            "cadence_index": cadence.cadence_index,
            "boundary_ordinal": cadence.boundary_ordinal,
            "status": result.status,
            "maintenance_plan_id": result.plan_id or None,
            "maintenance_transaction_id": result.transaction_id or None,
            "repository_revision_after": result.after_revision,
        })
        return result.status

    def _reconcile_persisted_cadences(self) -> None:
        """Recover every open cadence with its persisted boundary context."""

        for cadence in self.cadence_ledger.open_cadences(
            memory_project_key=self.project_key,
        ):
            self._run_or_reconcile_cadence(cadence)

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
    blocks = [SELECTED_MEMORY_CONTEXT_HEADER]
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
