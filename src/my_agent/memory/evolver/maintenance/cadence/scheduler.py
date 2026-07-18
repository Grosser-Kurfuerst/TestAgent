"""Cadence advancement and formal maintenance recovery orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from my_agent.memory.evolver.maintenance.cadence.ledger import (
    CadenceLedger,
    CadenceRecord,
)
from my_agent.memory.evolver.maintenance.contracts import MaintenanceOperation
from my_agent.memory.evolver.maintenance.formal.agent import FormalMaintenanceResult
from my_agent.memory.evolver.maintenance.formal.history import (
    load_formal_maintenance_history,
)
from my_agent.memory.evolver.maintenance.formal.transaction import (
    apply_formal_maintenance_operations,
)
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.policy.identity import canonical_sha256
from my_agent.training.contracts import AuthoritativeTaskOutcome

TraceSink = Callable[[str, dict[str, Any]], None]
RunMaintenance = Callable[..., FormalMaintenanceResult]


class MaintenanceCadenceScheduler:
    def __init__(
        self,
        *,
        store: ExperienceStore,
        ledger: CadenceLedger,
        history_path: str | Path,
        project_key: str,
        maintenance_enabled: bool,
        run_maintenance: RunMaintenance | None,
        decision_recorder: Any | None = None,
        runtime_evidence_recorder: Any | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.history_path = Path(history_path)
        self.project_key = project_key
        self.maintenance_enabled = maintenance_enabled
        self.run_maintenance = run_maintenance
        self.decision_recorder = decision_recorder
        self.runtime_evidence_recorder = runtime_evidence_recorder
        self.trace_sink = trace_sink

    def set_trace_sink(self, trace_sink: TraceSink | None) -> None:
        self.trace_sink = trace_sink

    def advance(
        self,
        *,
        episode: AgentEpisodeArtifact,
        outcome: AuthoritativeTaskOutcome,
        writer_status: str,
        repository_revision_after: str,
        written_memory_ids: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        session = episode.session
        advance = self.ledger.record_task_completion(
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
        due = self.ledger.oldest_open_cadence(
            stream_id=session.stream_id,
            memory_project_key=session.memory_project_key,
        )
        if due is None:
            if advance.cadence is None:
                return None, None
            return advance.cadence.cadence_id, advance.cadence.status
        if not self.maintenance_enabled:
            return due.cadence_id, "disabled_ablation"
        return due.cadence_id, self.run_or_reconcile(due)

    def run_or_reconcile(self, cadence: CadenceRecord) -> str:
        with self.store.exclusive_process_lock():
            return self._run_or_reconcile_locked(cadence)

    def _run_or_reconcile_locked(self, cadence: CadenceRecord) -> str:
        history = load_formal_maintenance_history(
            self.history_path,
            cadence_id=cadence.cadence_id,
        )
        if history.completion is not None:
            if self.runtime_evidence_recorder is not None:
                self.runtime_evidence_recorder.recover_maintenance(
                    cadence_id=cadence.cadence_id,
                    project_key=cadence.memory_project_key,
                    status=str(history.completion["status"]),
                )
            return self.ledger.mark_committed(
                cadence.cadence_id,
                maintenance_plan_id=str(history.completion["plan_id"]),
                repository_revision_after=str(history.completion["after_revision"]),
            ).status
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
                history_path=self.history_path,
            )
            if applied.status in {"committed", "noop"}:
                if self.runtime_evidence_recorder is not None:
                    self.runtime_evidence_recorder.recover_maintenance(
                        cadence_id=cadence.cadence_id,
                        project_key=cadence.memory_project_key,
                        status=applied.status,
                    )
                self.ledger.mark_committed(
                    cadence.cadence_id,
                    maintenance_plan_id=applied.plan_id,
                    repository_revision_after=applied.after_revision,
                )
            return applied.status
        if self.run_maintenance is None:
            return cadence.status
        task_group, history_window = self.ledger.cadence_context(cadence)
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
        self.ledger.mark_started(cadence.cadence_id)
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
            self.ledger.mark_committed(
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

    def reconcile_persisted(self) -> None:
        for cadence in self.ledger.open_cadences(
            memory_project_key=self.project_key,
        ):
            self.run_or_reconcile(cadence)

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        if self.trace_sink is not None:
            self.trace_sink(event, payload)


__all__ = ["MaintenanceCadenceScheduler"]
