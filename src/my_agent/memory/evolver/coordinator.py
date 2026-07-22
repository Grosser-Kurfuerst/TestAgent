"""Task lifecycle coordinator for retrieve-once and outcome-finalized writes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from my_agent.memory.evolver.task_session import (
    AgentEpisodeArtifact,
    EvolverFinalizeResult,
    TaskEvolverSession,
)
from my_agent.memory.evolver.maintenance.cadence.ledger import (
    MAINTENANCE_HISTORY_FILENAME,
    CadenceLedger,
    CadenceRecord,
)
from my_agent.memory.evolver.maintenance.cadence.scheduler import (
    MaintenanceCadenceScheduler,
)
from my_agent.memory.evolver.maintenance.cadence.schema import EVOLVER_STATE_FILENAME
from my_agent.memory.evolver.selection.contracts import TaskSelectionPolicy
from my_agent.memory.evolver.selection.formal import (
    EmptyTaskSelectionPolicy,
    LLMTaskSelectionPolicy,
    SimilarityTaskSelectionPolicy,
)
from my_agent.memory.evolver.selection.rendering import render_formal_selected_context
from my_agent.memory.evolver.selection.service import (
    SelectionService,
    candidate_snapshot,
    limit_selected_ids,
)
from my_agent.memory.evolver.maintenance.formal.agent import (
    FormalMaintenanceAgent,
    FormalMaintenanceResult,
)
from my_agent.memory.evolver.writing.contracts import ExperienceWriteResult
from my_agent.memory.evolver.writing.formal import FormalExperienceWriter
from my_agent.memory.experience.models import ExperienceMemory
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.memory.store_errors import MemoryStorePostCommitError
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryContext
from my_agent.opd_data.runtime_recorder import RuntimeEvidenceRecorder
from my_agent.policy.identity import PolicyIdentity, canonical_sha256, require_matching_policy_identity
from my_agent.policy.contracts import GenerationPolicy
from my_agent.training.contracts import AuthoritativeTaskOutcome
from my_agent.training.decision_log import DecisionEventContext, DecisionEventRecorder
from my_agent.training.role_views import TaskOutcomeRef


TraceSink = Callable[[str, dict[str, Any]], None]
WriterCallback = Callable[[AgentEpisodeArtifact, AuthoritativeTaskOutcome], ExperienceWriteResult]


class EvolverCoordinator:
    def __init__(
        self,
        *,
        store: ExperienceStore,
        project_key: str,
        policy_identity: PolicyIdentity,
        retriever: Any | None = None,
        selector: TaskSelectionPolicy | None = None,
        writer: WriterCallback | None = None,
        policy: GenerationPolicy | None = None,
        decision_recorder: DecisionEventRecorder | None = None,
        dataset_dir: str | Path | None = None,
        trace_sink: TraceSink | None = None,
        top_k_per_tier: int = 50,
        selected_max_items: int = 20,
        selection_token_budget: int = 1_800,
        generation_temperature: float = 1.0,
        generation_top_p: float = 0.95,
        maintenance_max_turns: int = 8,
        maintenance_interval_tasks: int = 30,
        ledger_path: str | Path | None = None,
        collection_round: int = 0,
        dataset_split: str = "train",
        maintenance_enabled: bool = True,
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
            self.selector = LLMTaskSelectionPolicy(
                policy=policy,
                recorder=self.decision_recorder,
                temperature=generation_temperature,
                top_p=generation_top_p,
            )
        else:
            self.selector = EmptyTaskSelectionPolicy()
        self.selection_service = SelectionService(self.selector)
        if writer is not None:
            self.writer = writer
        elif policy is not None and self.decision_recorder is not None:
            self.writer = FormalExperienceWriter(
                policy=policy,
                recorder=self.decision_recorder,
                store=store,
                project_key=project_key,
                temperature=generation_temperature,
                top_p=generation_top_p,
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
                temperature=generation_temperature,
                top_p=generation_top_p,
            )
            if policy is not None and self.decision_recorder is not None
            else None
        )
        self.trace_sink = trace_sink
        self.top_k_per_tier = top_k_per_tier
        self.selected_max_items = selected_max_items
        self.selection_token_budget = selection_token_budget
        self.generation_temperature = float(generation_temperature)
        self.generation_top_p = float(generation_top_p)
        self.maintenance_max_turns = maintenance_max_turns
        self.maintenance_interval_tasks = maintenance_interval_tasks
        self.maintenance_enabled = maintenance_enabled
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
        self.cadence_scheduler = MaintenanceCadenceScheduler(
            store=self.store,
            ledger=self.cadence_ledger,
            history_path=self.maintenance_history_path,
            project_key=self.project_key,
            maintenance_enabled=self.maintenance_enabled,
            run_maintenance=self.run_maintenance if self.maintainer is not None else None,
            decision_recorder=self.decision_recorder,
            runtime_evidence_recorder=self.runtime_evidence_recorder,
            trace_sink=self.trace_sink,
        )
        self._finalized_trajectories: set[str] = set()
        self.cadence_scheduler.reconcile_persisted()

    def set_trace_sink(self, trace_sink: TraceSink | None) -> None:
        self.trace_sink = trace_sink
        self.cadence_scheduler.set_trace_sink(trace_sink)
        if self.decision_recorder is not None:
            self.decision_recorder.trace_sink = trace_sink

    def require_formal_role_bindings(self, policy: GenerationPolicy) -> None:
        if self.policy is not policy:
            raise ValueError("formal coordinator must retain the shared runtime policy handle")
        recorder = self.decision_recorder
        if recorder is None or recorder.policy is not policy:
            raise ValueError("formal coordinator decision recorder must use the shared policy")
        if isinstance(self.selector, LLMTaskSelectionPolicy):
            if self.selector.policy is not policy or self.selector.recorder is not recorder:
                raise ValueError("formal selector must share the runtime policy and decision recorder")
        elif not isinstance(self.selector, SimilarityTaskSelectionPolicy):
            raise ValueError(
                "formal coordinator selector must be LLMTaskSelectionPolicy "
                "or the SimilarityTaskSelectionPolicy ablation"
            )
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
        candidates = candidate_snapshot(hits)
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
        selected_ids = self.selection_service.select(
            task=task,
            candidates=candidates,
            token_budget=self.selection_token_budget,
            max_items=self.selected_max_items,
            context=decision_context,
        )
        selected_ids = limit_selected_ids(
            selected_ids,
            candidates=candidates,
            token_budget=self.selection_token_budget,
            max_items=self.selected_max_items,
        )
        context = render_formal_selected_context(selected_ids, hits)
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
            "selection_calls": 1 if candidates else 0,
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
        self.cadence_scheduler.run_maintenance = (
            self.run_maintenance if self.maintainer is not None else None
        )
        return self.cadence_scheduler.advance(
            episode=episode,
            outcome=outcome,
            writer_status=writer_status,
            repository_revision_after=repository_revision_after,
            written_memory_ids=written_memory_ids,
        )

    def _run_or_reconcile_cadence(self, cadence: CadenceRecord) -> str:
        self.cadence_scheduler.run_maintenance = (
            self.run_maintenance if self.maintainer is not None else None
        )
        return self.cadence_scheduler.run_or_reconcile(cadence)

    def _reconcile_persisted_cadences(self) -> None:
        self.cadence_scheduler.reconcile_persisted()
    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self.trace_sink is not None:
            self.trace_sink(event, payload)


__all__ = [
    "EmptyTaskSelectionPolicy",
    "SimilarityTaskSelectionPolicy",
    "EvolverCoordinator",
    "TaskSelectionPolicy",
    "WriterCallback",
]
