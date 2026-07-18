"""Append strict OPD evidence produced by the formal runtime."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from threading import Lock
from typing import Any
import json

from my_agent.memory.evolver.maintenance.formal.prompt import repository_snapshot_ref
from my_agent.memory.evolver.maintenance.lookup import redundancy_score
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact, TaskEvolverSession
from my_agent.memory.experience.repository import ExperienceStore
from my_agent.opd_data.schema import (
    ActionDecisionEvidence,
    ActionExecutionEvidence,
    MaintenanceAttemptEvidence,
    MaintenanceEvidence,
    RepositoryEvidence,
    RepositoryMemoryEvidence,
    RuntimeExclusionEvidence,
    TaskEvidence,
    TaskOutcomeEvidence,
)
from my_agent.policy.identity import (
    canonical_json_bytes,
    canonical_sha256,
    require_matching_policy_identity,
)
from my_agent.training.contracts import AuthoritativeTaskOutcome, DecisionEvent
from my_agent.training.decision_log import DecisionEventRecorder
from my_agent.training.role_views import (
    CanonicalMessage,
    CanonicalTrajectoryStep,
    CanonicalToolCall,
    RedundancyDiagnostic,
    TaskOutcomeRef,
    TrajectoryEvidence,
    without_selected_memory_context,
)


@dataclass(frozen=True)
class _PendingTask:
    task: str
    session: TaskEvolverSession
    repository_snapshot_hash: str
    selection_token_budget: int


@dataclass(frozen=True)
class _PendingMaintenance:
    cadence_id: str
    attempt_id: str
    attempt_index: int
    task_group: str
    stream_id: str
    repository_snapshot_hash: str
    as_of_task_ordinal: int
    outcome_ids: tuple[str, ...]
    redundancy_diagnostics: tuple[RedundancyDiagnostic, ...]


@dataclass(frozen=True)
class _PendingActionExecution:
    session: TaskEvolverSession
    decision_id: str
    turn_index: int
    step_index: int
    call_index: int
    call_id: str
    tool_name: str
    run_id: str
    arguments_hash: str
    ok: bool
    blocked: bool
    error_code: str
    output_hash: str

    @property
    def idempotency_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.session.stream_id,
            self.session.memory_project_key,
            self.session.task_id,
            self.decision_id,
            self.call_id,
        )


class _EvidenceStream:
    def __init__(
        self,
        path: Path,
        *,
        loader: Callable[[str | Path], tuple[Any, ...]],
        key: Callable[[Any], Hashable],
    ) -> None:
        self.path = path
        self._key = key
        self._lock = Lock()
        self.records = list(loader(path) if path.exists() else ())
        self._keys = {key(record) for record in self.records}
        self._by_key = {key(record): record for record in self.records}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: Any) -> bool:
        record_key = self._key(record)
        payload = canonical_json_bytes(record.to_dict()).decode("utf-8")
        with self._lock:
            if record_key in self._keys:
                existing = self._by_key[record_key]
                if existing != record:
                    raise ValueError("runtime evidence idempotency key has conflicting payloads")
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            self.records.append(record)
            self._keys.add(record_key)
            self._by_key[record_key] = record
            return True


class RuntimeEvidenceRecorder:
    """Join formal task lifecycle state into the four exporter evidence streams."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        store: ExperienceStore,
        decision_recorder: DecisionEventRecorder,
        collection_round: int = 0,
        split: str = "train",
    ) -> None:
        if collection_round < 0:
            raise ValueError("collection_round must be non-negative")
        if split not in {"train", "validation", "test"}:
            raise ValueError("dataset split must be train, validation, or test")
        root = Path(dataset_dir)
        self.store = store
        self.decision_recorder = decision_recorder
        self.collection_round = collection_round
        self.split = split
        self.tasks = _EvidenceStream(
            root / "task_evidence.jsonl",
            loader=lambda path: _load_records(path, TaskEvidence.from_dict),
            key=lambda item: _task_key(
                item.stream_id, item.memory_project_key, item.task_id
            ),
        )
        self.outcomes = _EvidenceStream(
            root / "task_outcomes.jsonl",
            loader=lambda path: _load_records(path, TaskOutcomeEvidence.from_dict),
            key=lambda item: _task_key(
                item.stream_id, item.memory_project_key, item.outcome.task_id
            ),
        )
        self.executions = _EvidenceStream(
            root / "tool_execution_evidence.jsonl",
            loader=lambda path: _load_records(path, ActionExecutionEvidence.from_dict),
            key=lambda item: item.idempotency_key,
        )
        self.repositories = _EvidenceStream(
            root / "repository_events.jsonl",
            loader=lambda path: _load_records(path, RepositoryEvidence.from_dict),
            key=lambda item: item.snapshot.snapshot_hash,
        )
        self.maintenance = _EvidenceStream(
            root / "maintenance_evidence.jsonl",
            loader=lambda path: _load_records(path, MaintenanceEvidence.from_dict),
            key=lambda item: item.cadence_id,
        )
        self.maintenance_attempts = _EvidenceStream(
            root / "maintenance_attempts.jsonl",
            loader=lambda path: _load_records(path, MaintenanceAttemptEvidence.from_dict),
            key=lambda item: item.attempt_event_id,
        )
        self.exclusions = _EvidenceStream(
            root / "runtime_exclusions.jsonl",
            loader=lambda path: _load_records(path, RuntimeExclusionEvidence.from_dict),
            key=lambda item: item.exclusion_id,
        )
        self._pending_tasks: dict[str, _PendingTask] = {}
        self._pending_action_executions: dict[
            tuple[str, str, str, str, str], _PendingActionExecution
        ] = {}
        self._pending_maintenance: dict[str, _PendingMaintenance] = {}
        self._outcome_by_task = {
            _task_key(item.stream_id, item.memory_project_key, item.outcome.task_id): item
            for item in self.outcomes.records
        }
        self._repository_state: dict[tuple[str, str], tuple[int, str]] = {}
        for record in (
            *self.tasks.records,
            *self.outcomes.records,
            *self.executions.records,
            *self.repositories.records,
            *self.maintenance.records,
            *self.maintenance_attempts.records,
            *self.exclusions.records,
        ):
            if record.collection_round != collection_round:
                raise ValueError("runtime evidence directory crosses collection rounds")
        for record in (
            *self.tasks.records,
            *self.executions.records,
            *self.maintenance.records,
            *self.maintenance_attempts.records,
            *self.exclusions.records,
        ):
            if record.split != split:
                raise ValueError("runtime evidence directory crosses dataset splits")
        recorder_identity = self.decision_recorder.policy.identity()
        for record in self.executions.records:
            require_matching_policy_identity(recorder_identity, record.policy_identity)
            self._validate_execution_join(record)
        for record in sorted(
            self.repositories.records,
            key=lambda item: (item.stream_id, item.snapshot.memory_project_key, item.event_ordinal),
        ):
            self._repository_state[
                (record.stream_id, record.snapshot.memory_project_key)
            ] = (record.event_ordinal + 1, record.snapshot.repository_revision)

    def begin_task(
        self,
        *,
        task: str,
        session: TaskEvolverSession,
        selection_token_budget: int,
    ) -> None:
        if session.trajectory_id in self._pending_tasks:
            raise ValueError("task evidence is already pending for this trajectory")
        repository = self.record_repository(
            stream_id=session.stream_id,
            project_key=session.memory_project_key,
            expected_revision=session.repository_revision,
        )
        self._pending_tasks[session.trajectory_id] = _PendingTask(
            task=task,
            session=session,
            repository_snapshot_hash=repository.snapshot.snapshot_hash,
            selection_token_budget=selection_token_budget,
        )

    def finish_task(
        self,
        *,
        episode: AgentEpisodeArtifact,
        outcome: AuthoritativeTaskOutcome,
        task_ordinal: int,
        written_memory_ids: tuple[str, ...],
    ) -> None:
        pending = self._pending_tasks.get(episode.session.trajectory_id)
        if pending is None or pending.session != episode.session:
            raise ValueError("missing frozen task evidence for finalized trajectory")
        self._materialize_action_executions(
            session=episode.session,
            task_ordinal=task_ordinal,
        )
        all_events = self._events(episode.session.trajectory_id)
        successful = tuple(event for event in all_events if event.status == "success")
        selections = _role_events(successful, "selection")
        actions = _role_events(successful, "action")
        writing = _role_events(successful, "writing")

        outcome_record = TaskOutcomeEvidence(
            collection_round=self.collection_round,
            task_ordinal=task_ordinal,
            trajectory_id=episode.session.trajectory_id,
            stream_id=episode.session.stream_id,
            memory_project_key=episode.session.memory_project_key,
            outcome=outcome.to_ref(),
            task_valid=outcome.task_valid,
            outcome_finalized=outcome.outcome_finalized,
        )
        invalid_roles: list[tuple[str, str]] = []
        if len(selections) != 1:
            invalid_roles.append(("selection", "missing_successful_selection_decision"))
        if not actions:
            invalid_roles.append(("action", "missing_successful_action_decision"))
        if len(writing) > 1 or (written_memory_ids and not writing):
            invalid_roles.append(("writing", "inconsistent_writing_decision_evidence"))
        if invalid_roles:
            for role, reason in invalid_roles:
                self._record_exclusion(
                    role=role,
                    reason=reason,
                    task_id=episode.session.task_id,
                    trajectory_id=episode.session.trajectory_id,
                    stream_id=episode.session.stream_id,
                    project_key=episode.session.memory_project_key,
                    task_ordinal=task_ordinal,
                    decision_ids=tuple(
                        event.decision_id for event in _role_events(all_events, role)
                    ),
                )
            self._finish_task_streams(
                episode=episode,
                outcome_record=outcome_record,
            )
            del self._pending_tasks[episode.session.trajectory_id]
            return

        selection = selections[0]

        trajectory = TrajectoryEvidence(
            trajectory_id=episode.session.trajectory_id,
            task_group=episode.session.task_group,
            outcome="success" if outcome.resolved else "failure",
            reward=outcome.reward,
            steps=tuple(
                CanonicalTrajectoryStep(
                    step_index=index,
                    observation="",
                    action=step.tool,
                    arguments_json=canonical_json_bytes(step.arguments).decode("utf-8"),
                    result=step.output,
                    reward=None,
                )
                for index, step in enumerate(episode.tool_history)
            ),
        )
        task_record = TaskEvidence(
            collection_round=self.collection_round,
            task_ordinal=task_ordinal,
            split=self.split,
            task=pending.task,
            task_id=episode.session.task_id,
            task_group=episode.session.task_group,
            trajectory_id=episode.session.trajectory_id,
            stream_id=episode.session.stream_id,
            memory_project_key=episode.session.memory_project_key,
            policy_identity=episode.session.policy_identity,
            repository_snapshot_hash=pending.repository_snapshot_hash,
            candidate_snapshot_hash=episode.session.candidate_snapshot_hash,
            candidates=episode.session.candidate_snapshot,
            selected_memory_ids=episode.session.selected_memory_ids,
            trajectory=trajectory,
            written_memory_ids=written_memory_ids,
            selection_decision_id=selection.decision_id,
            action_decisions=tuple(
                ActionDecisionEvidence(
                    decision_id=event.decision_id,
                    turn_index=event.turn_index,
                    step_index=event.step_index,
                    prefix_messages=without_selected_memory_context(event.canonical_messages),
                    tools=event.canonical_tools,
                    expected_tool_calls=_event_tool_calls(event),
                    observation_messages=_observations_between(
                        event,
                        actions[index + 1] if index + 1 < len(actions) else None,
                    ),
                )
                for index, event in enumerate(actions)
            ),
            writing_decision_id=writing[0].decision_id if writing else None,
            selection_token_budget=pending.selection_token_budget,
        )
        self.tasks.append(task_record)
        self._finish_task_streams(episode=episode, outcome_record=outcome_record)
        del self._pending_tasks[episode.session.trajectory_id]

    def record_action_execution(
        self,
        *,
        session: TaskEvolverSession,
        decision_id: str,
        turn_index: int,
        step_index: int,
        call_index: int,
        call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        ok: bool,
        blocked: bool,
        error_code: str,
        output: str,
    ) -> None:
        pending_task = self._pending_tasks.get(session.trajectory_id)
        if pending_task is None or pending_task.session != session:
            raise ValueError("action execution requires a pending formal task session")
        if not isinstance(arguments, Mapping):
            raise ValueError("action execution arguments must be an object")
        if not isinstance(output, str):
            raise ValueError("action execution output must be a string")
        event = self._decision_event(session.trajectory_id, decision_id)
        if event.role != "action" or event.status != "success":
            raise ValueError("action execution requires a successful action decision")
        if event.turn_index != turn_index or event.step_index != step_index:
            raise ValueError("action execution indexes do not match the decision event")
        require_matching_policy_identity(session.policy_identity, event.policy_identity)
        expected_calls = _event_tool_calls(event)
        if call_index < 0 or call_index >= len(expected_calls):
            raise ValueError("action execution call_index is absent from the decision")
        expected_call = expected_calls[call_index]
        arguments_hash = canonical_sha256(dict(arguments))
        if (
            expected_call.call_id != call_id
            or expected_call.name != tool_name
            or canonical_sha256(json.loads(expected_call.arguments_json)) != arguments_hash
        ):
            raise ValueError("action execution call does not match the decision event")
        pending = _PendingActionExecution(
            session=session,
            decision_id=decision_id,
            turn_index=turn_index,
            step_index=step_index,
            call_index=call_index,
            call_id=call_id,
            tool_name=tool_name,
            run_id=event.run_id,
            arguments_hash=arguments_hash,
            ok=ok,
            blocked=blocked,
            error_code=error_code,
            output_hash=canonical_sha256(output),
        )
        existing = self._pending_action_executions.get(pending.idempotency_key)
        if existing is not None and existing != pending:
            raise ValueError("pending action execution idempotency key conflicts")
        self._pending_action_executions[pending.idempotency_key] = pending

    def _materialize_action_executions(
        self,
        *,
        session: TaskEvolverSession,
        task_ordinal: int,
    ) -> None:
        pending = sorted(
            (
                item
                for item in self._pending_action_executions.values()
                if item.session.trajectory_id == session.trajectory_id
            ),
            key=lambda item: (item.turn_index, item.step_index, item.call_index),
        )
        for item in pending:
            record = ActionExecutionEvidence(
                collection_round=self.collection_round,
                task_ordinal=task_ordinal,
                split=self.split,
                task_id=session.task_id,
                task_group=session.task_group,
                decision_id=item.decision_id,
                trajectory_id=session.trajectory_id,
                stream_id=session.stream_id,
                memory_project_key=session.memory_project_key,
                run_id=item.run_id,
                policy_identity=session.policy_identity,
                turn_index=item.turn_index,
                step_index=item.step_index,
                call_index=item.call_index,
                call_id=item.call_id,
                tool_name=item.tool_name,
                arguments_hash=item.arguments_hash,
                ok=item.ok,
                blocked=item.blocked,
                error_code=item.error_code,
                output_hash=item.output_hash,
            )
            self.executions.append(record)
            del self._pending_action_executions[item.idempotency_key]

    def _decision_event(self, trajectory_id: str, decision_id: str) -> DecisionEvent:
        matches = tuple(
            event
            for event in self._events(trajectory_id)
            if event.decision_id == decision_id
        )
        if len(matches) != 1:
            raise ValueError("action execution decision_id is absent or ambiguous")
        return matches[0]

    def _validate_execution_join(self, record: ActionExecutionEvidence) -> None:
        event = self._decision_event(record.trajectory_id, record.decision_id)
        if (
            event.task_id != record.task_id
            or event.task_group != record.task_group
            or event.stream_id != record.stream_id
            or event.memory_project_key != record.memory_project_key
            or event.run_id != record.run_id
            or event.turn_index != record.turn_index
            or event.step_index != record.step_index
        ):
            raise ValueError("persisted action execution does not join its decision event")
        require_matching_policy_identity(event.policy_identity, record.policy_identity)
        calls = _event_tool_calls(event)
        if record.call_index >= len(calls):
            raise ValueError("persisted action execution call_index is absent")
        call = calls[record.call_index]
        if (
            call.call_id != record.call_id
            or call.name != record.tool_name
            or canonical_sha256(json.loads(call.arguments_json)) != record.arguments_hash
        ):
            raise ValueError("persisted action execution call does not match its decision")

    def begin_maintenance(
        self,
        *,
        cadence_id: str,
        task_group: str,
        stream_id: str,
        project_key: str,
        as_of_task_ordinal: int,
        history_window: Sequence[TaskOutcomeRef],
    ) -> str:
        attempt_states = self._maintenance_attempt_states(cadence_id)
        for started, terminal in attempt_states:
            if terminal is not None:
                continue
            decision_ids = tuple(
                event.decision_id
                for event in self._maintenance_attempt_events(cadence_id, started.attempt_id)
            )
            self._record_exclusion(
                role="maintenance",
                reason="maintenance_attempt_abandoned_after_interruption",
                task_id=cadence_id,
                trajectory_id=cadence_id,
                stream_id=started.stream_id,
                project_key=started.memory_project_key,
                task_ordinal=started.as_of_task_ordinal,
                decision_ids=decision_ids,
            )
            self._append_maintenance_attempt_terminal(
                started,
                status="abandoned",
                decision_ids=decision_ids,
                reason="superseded_after_interrupted_attempt",
            )

        repository = self.record_repository(stream_id=stream_id, project_key=project_key)
        snapshot = self.store.load_strict_snapshot()
        if snapshot.revision != repository.snapshot.repository_revision:
            raise ValueError("repository changed while freezing maintenance diagnostics")
        outcome_ids: list[str] = []
        for outcome in history_window:
            stored = self._outcome_by_task.get(
                _task_key(stream_id, project_key, outcome.task_id)
            )
            if stored is None or stored.outcome != outcome:
                raise ValueError("maintenance history outcome is absent from runtime evidence")
            outcome_ids.append(stored.outcome_id)
        attempt_index = 1 + max(
            (started.attempt_index for started, _terminal in attempt_states),
            default=0,
        )
        started = MaintenanceAttemptEvidence(
            collection_round=self.collection_round,
            split=self.split,
            cadence_id=cadence_id,
            attempt_index=attempt_index,
            status="started",
            task_group=task_group,
            stream_id=stream_id,
            memory_project_key=project_key,
            repository_snapshot_hash=repository.snapshot.snapshot_hash,
            as_of_task_ordinal=as_of_task_ordinal,
            outcome_ids=tuple(outcome_ids),
            redundancy_diagnostics=_redundancy_diagnostics(
                snapshot.memories,
                project_key=project_key,
            ),
        )
        self.maintenance_attempts.append(started)
        self._pending_maintenance[started.attempt_id] = _pending_maintenance(started)
        return started.attempt_id

    def finish_maintenance(
        self,
        *,
        cadence_id: str,
        attempt_id: str,
        project_key: str,
        status: str,
    ) -> None:
        started, terminal = self._maintenance_attempt(attempt_id, cadence_id=cadence_id)
        if terminal is not None:
            if terminal.status != status:
                raise ValueError("maintenance attempt terminal status conflicts with recovery")
            return
        pending = self._pending_maintenance.get(attempt_id) or _pending_maintenance(started)
        self._pending_maintenance[attempt_id] = pending
        all_events = self._maintenance_attempt_events(cadence_id, attempt_id)
        events = tuple(event for event in all_events if event.status == "success")
        existing = next(
            (item for item in self.maintenance.records if item.cadence_id == cadence_id),
            None,
        )
        if existing is not None:
            if existing.attempt_id != attempt_id:
                raise ValueError("maintenance evidence belongs to another attempt")
            self._append_maintenance_attempt_terminal(
                started,
                status=status,
                decision_ids=tuple(event.decision_id for event in all_events),
            )
            self._pending_maintenance.pop(attempt_id, None)
            return
        if status not in {"committed", "noop"} or not events:
            self._record_exclusion(
                role="maintenance",
                reason=f"maintenance_terminal_status:{status}",
                task_id=cadence_id,
                trajectory_id=cadence_id,
                stream_id=pending.stream_id,
                project_key=project_key,
                task_ordinal=pending.as_of_task_ordinal,
                decision_ids=tuple(event.decision_id for event in all_events),
            )
            self.record_repository(stream_id=pending.stream_id, project_key=project_key)
            self._append_maintenance_attempt_terminal(
                started,
                status=status,
                decision_ids=tuple(event.decision_id for event in all_events),
                reason=f"maintenance_terminal_status:{status}",
            )
            self._pending_maintenance.pop(attempt_id, None)
            return
        tools = events[0].canonical_tools
        if any(event.canonical_tools != tools for event in events[1:]):
            raise ValueError("formal maintenance tool schema changed within one cadence")
        record = MaintenanceEvidence(
            collection_round=self.collection_round,
            as_of_task_ordinal=pending.as_of_task_ordinal,
            split=self.split,
            cadence_id=cadence_id,
            attempt_id=attempt_id,
            task_group=pending.task_group,
            stream_id=pending.stream_id,
            memory_project_key=project_key,
            policy_identity=self.decision_recorder.policy.identity(),
            repository_snapshot_hash=pending.repository_snapshot_hash,
            outcome_ids=pending.outcome_ids,
            tools=tools,
            redundancy_diagnostics=pending.redundancy_diagnostics,
            decision_ids=tuple(event.decision_id for event in events),
        )
        self.maintenance.append(record)
        self.record_repository(stream_id=pending.stream_id, project_key=project_key)
        self._append_maintenance_attempt_terminal(
            started,
            status=status,
            decision_ids=tuple(event.decision_id for event in all_events),
        )
        self._pending_maintenance.pop(attempt_id, None)

    def recover_maintenance(
        self,
        *,
        cadence_id: str,
        project_key: str,
        status: str,
    ) -> None:
        active = tuple(
            started
            for started, terminal in self._maintenance_attempt_states(cadence_id)
            if terminal is None
        )
        if len(active) != 1:
            existing = any(item.cadence_id == cadence_id for item in self.maintenance.records)
            if existing and not active:
                return
            raise ValueError("maintenance recovery requires exactly one active attempt")
        self.finish_maintenance(
            cadence_id=cadence_id,
            attempt_id=active[0].attempt_id,
            project_key=project_key,
            status=status,
        )

    def _maintenance_attempt_states(
        self,
        cadence_id: str,
    ) -> tuple[tuple[MaintenanceAttemptEvidence, MaintenanceAttemptEvidence | None], ...]:
        grouped: dict[str, list[MaintenanceAttemptEvidence]] = {}
        for record in self.maintenance_attempts.records:
            if record.cadence_id == cadence_id:
                grouped.setdefault(record.attempt_id, []).append(record)
        states: list[tuple[MaintenanceAttemptEvidence, MaintenanceAttemptEvidence | None]] = []
        for records in grouped.values():
            starts = [record for record in records if record.status == "started"]
            terminals = [record for record in records if record.status != "started"]
            if len(starts) != 1 or len(terminals) > 1:
                raise ValueError("maintenance attempt history is inconsistent")
            started = starts[0]
            terminal = terminals[0] if terminals else None
            if terminal is not None and _maintenance_attempt_context(terminal) != (
                _maintenance_attempt_context(started)
            ):
                raise ValueError("maintenance attempt terminal context mismatch")
            states.append((started, terminal))
        states.sort(key=lambda item: item[0].attempt_index)
        indexes = tuple(started.attempt_index for started, _terminal in states)
        if indexes != tuple(range(1, len(states) + 1)):
            raise ValueError("maintenance attempt indexes must be contiguous from one")
        return tuple(states)

    def _maintenance_attempt(
        self,
        attempt_id: str,
        *,
        cadence_id: str,
    ) -> tuple[MaintenanceAttemptEvidence, MaintenanceAttemptEvidence | None]:
        matches = tuple(
            state
            for state in self._maintenance_attempt_states(cadence_id)
            if state[0].attempt_id == attempt_id
        )
        if len(matches) != 1:
            raise ValueError("unknown maintenance attempt")
        return matches[0]

    def _append_maintenance_attempt_terminal(
        self,
        started: MaintenanceAttemptEvidence,
        *,
        status: str,
        decision_ids: tuple[str, ...],
        reason: str = "",
    ) -> None:
        terminal = MaintenanceAttemptEvidence(
            collection_round=started.collection_round,
            split=started.split,
            cadence_id=started.cadence_id,
            attempt_index=started.attempt_index,
            status=status,
            task_group=started.task_group,
            stream_id=started.stream_id,
            memory_project_key=started.memory_project_key,
            repository_snapshot_hash=started.repository_snapshot_hash,
            as_of_task_ordinal=started.as_of_task_ordinal,
            outcome_ids=started.outcome_ids,
            redundancy_diagnostics=started.redundancy_diagnostics,
            decision_ids=decision_ids,
            reason=reason,
        )
        self.maintenance_attempts.append(terminal)

    def _maintenance_attempt_events(
        self,
        cadence_id: str,
        attempt_id: str,
    ) -> tuple[DecisionEvent, ...]:
        return tuple(
            event
            for event in _role_events(self._events(cadence_id), "maintenance")
            if event.run_id == attempt_id
        )

    def record_repository(
        self,
        *,
        stream_id: str,
        project_key: str,
        expected_revision: str | None = None,
    ) -> RepositoryEvidence:
        snapshot = self.store.load_strict_snapshot()
        if expected_revision is not None and snapshot.revision != expected_revision:
            raise ValueError("repository changed while freezing runtime evidence")
        entries = tuple(sorted(snapshot.memories, key=lambda item: item.id))
        snapshot_ref = repository_snapshot_ref(
            entries,
            repository_revision=snapshot.revision,
            project_key=project_key,
            stream_id=stream_id,
        )
        existing = next(
            (
                item
                for item in self.repositories.records
                if item.snapshot.snapshot_hash == snapshot_ref.snapshot_hash
            ),
            None,
        )
        state_key = (stream_id, project_key)
        next_ordinal, previous_revision = self._repository_state.get(state_key, (0, ""))
        if existing is not None:
            self._repository_state[state_key] = (next_ordinal, snapshot.revision)
            return existing
        record = RepositoryEvidence(
            collection_round=self.collection_round,
            event_ordinal=next_ordinal,
            stream_id=stream_id,
            previous_revision=previous_revision or None,
            snapshot=snapshot_ref,
            memories=tuple(
                RepositoryMemoryEvidence(
                    memory_id=entry.id,
                    tier=entry.tier.value,
                    content=entry.content,
                    candidate_count=entry.candidate_count,
                    selected_count=entry.selected_count,
                    last_used=entry.last_used.isoformat() if entry.last_used is not None else "",
                )
                for entry in entries
            ),
        )
        self.repositories.append(record)
        self._repository_state[state_key] = (next_ordinal + 1, snapshot.revision)
        return record

    def _finish_task_streams(
        self,
        *,
        episode: AgentEpisodeArtifact,
        outcome_record: TaskOutcomeEvidence,
    ) -> None:
        self.outcomes.append(outcome_record)
        self._outcome_by_task[
            _task_key(
                episode.session.stream_id,
                episode.session.memory_project_key,
                outcome_record.outcome.task_id,
            )
        ] = outcome_record
        self.record_repository(
            stream_id=episode.session.stream_id,
            project_key=episode.session.memory_project_key,
        )

    def _record_exclusion(
        self,
        *,
        role: str,
        reason: str,
        task_id: str,
        trajectory_id: str,
        stream_id: str,
        project_key: str,
        task_ordinal: int,
        decision_ids: tuple[str, ...],
    ) -> None:
        self.exclusions.append(RuntimeExclusionEvidence(
            collection_round=self.collection_round,
            task_ordinal=task_ordinal,
            split=self.split,
            role=role,
            reason=reason,
            task_id=task_id,
            trajectory_id=trajectory_id,
            stream_id=stream_id,
            memory_project_key=project_key,
            decision_ids=decision_ids,
        ))

    def _events(self, trajectory_id: str) -> tuple[DecisionEvent, ...]:
        events = self.decision_recorder.events_for(
            trajectory_id,
            purpose="fast_loop_evidence",
        )
        return tuple(sorted(events, key=lambda item: (item.turn_index, item.step_index, item.decision_id)))


def _role_events(events: Sequence[DecisionEvent], role: str) -> tuple[DecisionEvent, ...]:
    return tuple(event for event in events if event.role == role)


def _event_tool_calls(event: DecisionEvent) -> tuple[CanonicalToolCall, ...]:
    raw = event.parsed_output.get("tool_calls", ())
    if raw in (None, ()):
        return ()
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise ValueError("action decision parsed tool_calls must be an array of objects")
    return tuple(CanonicalToolCall.from_dict(item) for item in raw)


def _observations_between(
    current: DecisionEvent,
    following: DecisionEvent | None,
) -> tuple[CanonicalMessage, ...]:
    if following is None:
        return ()
    current_prefix = without_selected_memory_context(current.canonical_messages)
    following_prefix = without_selected_memory_context(following.canonical_messages)
    if following_prefix[: len(current_prefix)] != current_prefix:
        return ()
    return tuple(
        message
        for message in following_prefix[len(current_prefix):]
        if message.role == "tool"
    )


def _pending_maintenance(started: MaintenanceAttemptEvidence) -> _PendingMaintenance:
    return _PendingMaintenance(
        cadence_id=started.cadence_id,
        attempt_id=started.attempt_id,
        attempt_index=started.attempt_index,
        task_group=started.task_group,
        stream_id=started.stream_id,
        repository_snapshot_hash=started.repository_snapshot_hash,
        as_of_task_ordinal=started.as_of_task_ordinal,
        outcome_ids=started.outcome_ids,
        redundancy_diagnostics=started.redundancy_diagnostics,
    )


def _maintenance_attempt_context(record: MaintenanceAttemptEvidence) -> tuple[Any, ...]:
    return (
        record.collection_round,
        record.split,
        record.cadence_id,
        record.attempt_index,
        record.task_group,
        record.stream_id,
        record.memory_project_key,
        record.repository_snapshot_hash,
        record.as_of_task_ordinal,
        record.outcome_ids,
        record.redundancy_diagnostics,
    )


def _task_key(stream_id: str, project_key: str, task_id: str) -> tuple[str, str, str]:
    return stream_id, project_key, task_id


def _redundancy_diagnostics(
    entries: Sequence[Any],
    *,
    project_key: str,
) -> tuple[RedundancyDiagnostic, ...]:
    visible = tuple(sorted(
        (
            entry
            for entry in entries
            if entry.scope.value == "global" or entry.project_key == project_key
        ),
        key=lambda entry: entry.id,
    ))
    return tuple(
        RedundancyDiagnostic(left.id, right.id, redundancy_score(left, right))
        for left, right in combinations(visible, 2)
    )


def _load_records(
    path: str | Path,
    loader: Callable[[Mapping[str, Any]], Any],
) -> tuple[Any, ...]:
    records: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid runtime evidence JSON at line {line_number}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError("runtime evidence lines must be JSON objects")
            records.append(loader(payload))
    return tuple(records)


__all__ = ["RuntimeEvidenceRecorder"]
