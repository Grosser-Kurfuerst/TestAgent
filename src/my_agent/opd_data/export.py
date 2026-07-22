"""Strict evidence joins and typed role-view construction for OPD rounds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
import json

from my_agent.opd_data.schema import (
    ActionExecutionEvidence,
    LearnerSample,
    MaintenanceAttemptEvidence,
    MaintenanceEvidence,
    RepositoryEvidence,
    RuntimeExclusionEvidence,
    TaskEvidence,
    TaskOutcomeEvidence,
)
from my_agent.opd_data.attribution.equations import (
    positive_selected_memory_ids,
    teacher_memory_records,
    writing_top_fraction,
)
from my_agent.opd_data.attribution.schema import (
    PaperAttributionRecord,
    WritingScoreDecision,
)
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.training.contracts import DecisionEvent
from my_agent.training.role_views import (
    ActionHindsight,
    ActionPublic,
    CanonicalMessage,
    CanonicalToolCall,
    MaintenanceHindsight,
    MaintenancePublic,
    MemoryDiagnostic,
    MemoryValueEvidence,
    SelectionHindsight,
    SelectionPublic,
    WritingHindsight,
    WritingPublic,
    without_selected_memory_context,
)


T = TypeVar("T")
PublicView = SelectionPublic | ActionPublic | WritingPublic | MaintenancePublic
HindsightView = SelectionHindsight | ActionHindsight | WritingHindsight | MaintenanceHindsight


@dataclass(frozen=True)
class PreparedLearnerDecision:
    role: str
    collection_round: int
    split: str
    task_group: str
    stream_id: str
    memory_project_key: str
    source_evidence_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    public_view: PublicView
    hindsight_view: HindsightView
    forbidden_action_memories: tuple[tuple[str, str], ...] = ()
    action_rollout_id: str = ""
    action_turn_index: int = 0
    action_expected_tool_calls: tuple[CanonicalToolCall, ...] = ()
    action_observation_messages: tuple[CanonicalMessage, ...] = ()
    maintenance_rollout_id: str = ""
    maintenance_turn_index: int = 0
    maintenance_expected_tool_calls: tuple[CanonicalToolCall, ...] = ()
    maintenance_observation_messages: tuple[CanonicalMessage, ...] = ()


@dataclass(frozen=True)
class PreparedRound:
    decisions: tuple[PreparedLearnerDecision, ...]
    writing_score_decisions: tuple[WritingScoreDecision, ...]
    exclusions: tuple[Mapping[str, Any], ...]


def prepare_round_decisions(
    *,
    collection_round: int,
    trainer_identity: PolicyIdentity,
    tasks: Sequence[TaskEvidence],
    outcomes: Sequence[TaskOutcomeEvidence],
    repositories: Sequence[RepositoryEvidence],
    maintenance: Sequence[MaintenanceEvidence],
    decision_events: Sequence[DecisionEvent],
    attribution: Sequence[PaperAttributionRecord],
    writing_top_fraction_value: float = 0.30,
    teacher_minimum_score: float = 0.01,
    teacher_max_items: int = 20,
) -> PreparedRound:
    if collection_round < 0:
        raise ValueError("collection_round must be non-negative")
    if not isinstance(trainer_identity, PolicyIdentity):
        raise ValueError("round preparation requires trainer PolicyIdentity")
    task_by_id = _unique_map(tasks, _task_key, "task evidence")
    outcome_by_task = _unique_map(outcomes, _outcome_task_key, "task outcome")
    outcome_by_id = _unique_map(outcomes, lambda item: item.outcome_id, "task outcome")
    repository_by_hash = _unique_map(
        repositories,
        lambda item: item.snapshot.snapshot_hash,
        "repository snapshot",
    )
    decision_by_id = _unique_map(decision_events, lambda item: item.decision_id, "decision event")
    attribution_by_id = _unique_map(attribution, lambda item: item.memory_id, "attribution")
    _validate_round_inputs(
        collection_round=collection_round,
        trainer_identity=trainer_identity,
        tasks=tasks,
        outcomes=outcomes,
        repositories=repositories,
        maintenance=maintenance,
        attribution=attribution_by_id,
    )
    _validate_attribution_joins(
        attribution_by_id,
        task_by_id,
        outcome_by_task,
        repository_by_hash,
    )
    _validate_repository_continuity(repositories)
    _validate_split_isolation(tasks, maintenance)

    all_written_ids = tuple(
        memory_id
        for task in sorted(tasks, key=lambda item: item.task_ordinal)
        if (
            _task_key(task) in outcome_by_task
            and outcome_by_task[_task_key(task)].task_valid
            and outcome_by_task[_task_key(task)].outcome_finalized
        )
        for memory_id in task.written_memory_ids
    )
    writing_decisions = writing_top_fraction(
        all_written_ids,
        attribution_by_id,
        collection_round=collection_round,
        fraction=writing_top_fraction_value,
    )
    selected_writing_ids = {
        item.memory_id for item in writing_decisions if item.selected
    }
    successful = _successful_trajectories(tasks, outcome_by_task)
    prepared: list[PreparedLearnerDecision] = []
    exclusions: list[Mapping[str, Any]] = []

    for task in sorted(tasks, key=lambda item: (item.task_ordinal, item.task_id)):
        outcome = outcome_by_task.get(_task_key(task))
        if outcome is None:
            raise ValueError(f"missing authoritative outcome for task {task.task_id}")
        repository = repository_by_hash.get(task.repository_snapshot_hash)
        if repository is None:
            raise ValueError(f"missing repository snapshot for task {task.task_id}")
        _validate_task_join(task, outcome, repository, decision_by_id)
        if not outcome.task_valid or not outcome.outcome_finalized:
            exclusions.append(_exclusion(task.evidence_id, "task", "invalid_or_unknown_outcome"))
            continue

        if task.candidates:
            candidate_values = _ready_values(task.candidates, attribution_by_id)
            if candidate_values is None:
                exclusions.append(_exclusion(
                    task.evidence_id,
                    "selection",
                    "missing_candidate_attribution",
                ))
            else:
                assert task.selection_decision_id is not None
                prepared.append(PreparedLearnerDecision(
                    role="selection",
                    collection_round=collection_round,
                    split=task.split,
                    task_group=task.task_group,
                    stream_id=task.stream_id,
                    memory_project_key=task.memory_project_key,
                    source_evidence_ids=(task.evidence_id, outcome.outcome_id),
                    evidence_refs=(
                        task.selection_decision_id,
                        *(
                            _attribution_ref(item.memory_id, attribution_by_id)
                            for item in task.candidates
                        ),
                    ),
                    public_view=SelectionPublic(
                        task=task.task,
                        candidates=task.candidates,
                        token_budget=task.selection_token_budget,
                    ),
                    hindsight_view=SelectionHindsight(candidate_values),
                ))

        positive_ids = positive_selected_memory_ids(
            task.selected_memory_ids,
            attribution_by_id,
        )
        successful_task = successful.get((task.split, task.task_group))
        if not positive_ids or successful_task is None:
            exclusions.append(_exclusion(task.evidence_id, "action", "missing_positive_hindsight"))
        else:
            positive_values = tuple(
                _memory_value(attribution_by_id[memory_id]) for memory_id in positive_ids
            )
            tau_outcome = outcome_by_task[_task_key(successful_task)]
            by_candidate = {item.memory_id: item for item in task.candidates}
            initial_prefix = task.action_decisions[0].prefix_messages
            for action_index, action in enumerate(task.action_decisions):
                prepared.append(PreparedLearnerDecision(
                    role="action",
                    collection_round=collection_round,
                    split=task.split,
                    task_group=task.task_group,
                    stream_id=task.stream_id,
                    memory_project_key=task.memory_project_key,
                    source_evidence_ids=_unique_refs((
                        task.evidence_id,
                        outcome.outcome_id,
                        successful_task.evidence_id,
                        tau_outcome.outcome_id,
                    )),
                    evidence_refs=(
                        action.decision_id,
                        *(
                            _attribution_ref(memory_id, attribution_by_id)
                            for memory_id in positive_ids
                        ),
                    ),
                    public_view=ActionPublic(
                        task=task.task,
                        tools=action.tools,
                        prefix_messages=initial_prefix,
                    ),
                    hindsight_view=ActionHindsight(
                        positive_values,
                        successful_task.trajectory,
                    ),
                    forbidden_action_memories=tuple(
                        (memory_id, by_candidate[memory_id].content)
                        for memory_id in task.selected_memory_ids
                    ),
                    action_rollout_id=task.evidence_id,
                    action_turn_index=action_index,
                    action_expected_tool_calls=action.expected_tool_calls,
                    action_observation_messages=action.observation_messages,
                ))

        kept_written = tuple(
            memory_id
            for memory_id in task.written_memory_ids
            if memory_id in selected_writing_ids
        )
        if not kept_written:
            if task.written_memory_ids:
                exclusions.append(_exclusion(task.evidence_id, "writing", "below_round_top_fraction"))
        elif task.writing_decision_id is None:
            raise ValueError("selected writing evidence requires writing_decision_id")
        else:
            prepared.append(PreparedLearnerDecision(
                role="writing",
                collection_round=collection_round,
                split=task.split,
                task_group=task.task_group,
                stream_id=task.stream_id,
                memory_project_key=task.memory_project_key,
                source_evidence_ids=(task.evidence_id, outcome.outcome_id),
                evidence_refs=(
                    task.writing_decision_id,
                    *(_attribution_ref(memory_id, attribution_by_id) for memory_id in kept_written),
                ),
                public_view=WritingPublic(
                    task=task.task,
                    trajectory=task.trajectory,
                    reward=outcome.outcome.reward,
                    selected_memory_ids=task.selected_memory_ids,
                ),
                hindsight_view=WritingHindsight(tuple(
                    _memory_value(attribution_by_id[memory_id]) for memory_id in kept_written
                )),
            ))

    for item in sorted(maintenance, key=lambda value: (value.stream_id, value.cadence_id)):
        repository = repository_by_hash.get(item.repository_snapshot_hash)
        if repository is None:
            raise ValueError(f"missing repository snapshot for cadence {item.cadence_id}")
        joined_outcomes = tuple(_required(outcome_by_id, key, "maintenance outcome") for key in item.outcome_ids)
        _validate_maintenance_join(item, repository, joined_outcomes, decision_by_id)
        memory_by_id = {memory.memory_id: memory for memory in repository.memories}
        project_attribution = {
            memory_id: record
            for memory_id, record in attribution_by_id.items()
            if record.memory_project_key == item.memory_project_key and memory_id in memory_by_id
        }
        teacher_records = teacher_memory_records(
            project_attribution,
            minimum_memory_score=teacher_minimum_score,
            max_items=teacher_max_items,
        )
        diagnostics = tuple(
            MemoryDiagnostic(
                memory_id=record.memory_id,
                tier=record.tier,
                memory_score=float(record.memory_score),
                gamma=record.gamma,
                candidate_count=memory_by_id[record.memory_id].candidate_count,
                selected_count=memory_by_id[record.memory_id].selected_count,
                last_used=memory_by_id[record.memory_id].last_used,
            )
            for record in teacher_records
        )
        events = tuple(
            _required(decision_by_id, decision_id, "maintenance decision")
            for decision_id in item.decision_ids
        )
        indexes = tuple((event.turn_index, event.step_index) for event in events)
        if indexes != tuple((index, index) for index in range(len(events))):
            raise ValueError("maintenance decision indexes must be contiguous from zero")
        public_view = MaintenancePublic(
            repository_snapshot=repository.snapshot,
            history_window=tuple(outcome.outcome for outcome in joined_outcomes),
            tools=item.tools,
        )
        hindsight_view = MaintenanceHindsight(
            memory_diagnostics=diagnostics,
            redundancy_diagnostics=item.redundancy_diagnostics,
        )
        source_evidence_ids = (
            item.evidence_id,
            repository.repository_event_id,
            *(outcome.outcome_id for outcome in joined_outcomes),
        )
        attribution_refs = tuple(
            _attribution_ref(record.memory_id, attribution_by_id)
            for record in teacher_records
        )
        for turn_index, event in enumerate(events):
            following = events[turn_index + 1] if turn_index + 1 < len(events) else None
            prepared.append(PreparedLearnerDecision(
                role="maintenance",
                collection_round=collection_round,
                split=item.split,
                task_group=item.task_group,
                stream_id=item.stream_id,
                memory_project_key=item.memory_project_key,
                source_evidence_ids=source_evidence_ids,
                evidence_refs=(event.decision_id, *attribution_refs),
                public_view=public_view,
                hindsight_view=hindsight_view,
                maintenance_rollout_id=item.cadence_id,
                maintenance_turn_index=turn_index,
                maintenance_expected_tool_calls=_maintenance_event_tool_calls(event),
                maintenance_observation_messages=_observations_between(event, following),
            ))

    return PreparedRound(
        decisions=tuple(prepared),
        writing_score_decisions=writing_decisions,
        exclusions=tuple(exclusions),
    )


def write_learner_samples(samples: Sequence[LearnerSample], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(samples, key=lambda item: (item.role, item.task_group, item.sample_id))
    with output.open("w", encoding="utf-8") as handle:
        for sample in ordered:
            handle.write(canonical_json_bytes(sample.to_dict()).decode("utf-8") + "\n")
    return output


def load_learner_samples(path: str | Path) -> tuple[LearnerSample, ...]:
    return _load_jsonl(path, LearnerSample.from_dict)


def write_evidence_jsonl(records: Iterable[Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payloads = sorted(
        (record.to_dict() for record in records),
        key=lambda payload: canonical_sha256(payload),
    )
    with output.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(canonical_json_bytes(payload).decode("utf-8") + "\n")
    return output


def load_task_evidence(path: str | Path) -> tuple[TaskEvidence, ...]:
    return _load_jsonl(path, TaskEvidence.from_dict)


def load_task_outcomes(path: str | Path) -> tuple[TaskOutcomeEvidence, ...]:
    return _load_jsonl(path, TaskOutcomeEvidence.from_dict)


def load_action_execution_evidence(
    path: str | Path,
) -> tuple[ActionExecutionEvidence, ...]:
    return _load_jsonl(path, ActionExecutionEvidence.from_dict)


def load_repository_evidence(path: str | Path) -> tuple[RepositoryEvidence, ...]:
    return _load_jsonl(path, RepositoryEvidence.from_dict)


def load_maintenance_evidence(path: str | Path) -> tuple[MaintenanceEvidence, ...]:
    return _load_jsonl(path, MaintenanceEvidence.from_dict)


def load_maintenance_attempts(path: str | Path) -> tuple[MaintenanceAttemptEvidence, ...]:
    return _load_jsonl(path, MaintenanceAttemptEvidence.from_dict)


def load_runtime_exclusions(path: str | Path) -> tuple[RuntimeExclusionEvidence, ...]:
    return _load_jsonl(path, RuntimeExclusionEvidence.from_dict)


def sample_statistics(samples: Sequence[LearnerSample]) -> dict[str, Mapping[str, int]]:
    return {
        "role_counts": dict(Counter(sample.role for sample in samples)),
        "split_counts": dict(Counter(sample.split for sample in samples)),
        "task_group_counts": dict(Counter(sample.task_group for sample in samples)),
    }


def _validate_round_inputs(
    *,
    collection_round: int,
    trainer_identity: PolicyIdentity,
    tasks: Sequence[TaskEvidence],
    outcomes: Sequence[TaskOutcomeEvidence],
    repositories: Sequence[RepositoryEvidence],
    maintenance: Sequence[MaintenanceEvidence],
    attribution: Mapping[str, PaperAttributionRecord],
) -> None:
    for record in (*tasks, *outcomes, *repositories, *maintenance):
        if record.collection_round != collection_round:
            raise ValueError("evidence crosses collection rounds")
    for record in (*tasks, *maintenance):
        if record.policy_identity != trainer_identity:
            raise ValueError("evidence policy identity does not match trainer initialization")
    for memory_id, record in attribution.items():
        if record.memory_id != memory_id:
            raise ValueError("attribution mapping key mismatch")
        if record.collection_round != collection_round:
            raise ValueError("attribution crosses collection rounds")
        if record.policy_identity_hash != trainer_identity.identity_hash:
            raise ValueError("attribution identity does not match trainer initialization")


def _validate_split_isolation(
    tasks: Sequence[TaskEvidence],
    maintenance: Sequence[MaintenanceEvidence],
) -> None:
    splits: dict[tuple[str, str], str] = {}
    for record in (*tasks, *maintenance):
        key = (record.stream_id, record.task_group)
        previous = splits.setdefault(key, record.split)
        if previous != record.split:
            raise ValueError("one stream/task_group cannot cross dataset splits")


def _validate_repository_continuity(repositories: Sequence[RepositoryEvidence]) -> None:
    grouped: dict[tuple[str, str], list[RepositoryEvidence]] = {}
    for record in repositories:
        key = (record.stream_id, record.snapshot.memory_project_key)
        grouped.setdefault(key, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item.event_ordinal)
        if len({item.event_ordinal for item in records}) != len(records):
            raise ValueError("repository event ordinals must be unique per stream/project")
        previous_revision: str | None = None
        for index, record in enumerate(records):
            if index == 0:
                if record.previous_revision is not None:
                    raise ValueError("first repository event must not declare previous_revision")
            elif record.previous_revision != previous_revision:
                raise ValueError("repository revision history is not continuous")
            previous_revision = record.snapshot.repository_revision


def _validate_attribution_joins(
    attribution: Mapping[str, PaperAttributionRecord],
    tasks: Mapping[Hashable, TaskEvidence],
    outcomes: Mapping[Hashable, TaskOutcomeEvidence],
    repositories: Mapping[str, RepositoryEvidence],
) -> None:
    for record in attribution.values():
        for evidence_ref in record.evidence_refs:
            exposure = evidence_ref.exposure
            join_key = _task_key_values(
                exposure.stream_id,
                exposure.memory_project_key,
                exposure.task_id,
            )
            task = _required(tasks, join_key, "attribution task evidence")
            outcome = _required(outcomes, join_key, "attribution outcome evidence")
            if not outcome.task_valid or not outcome.outcome_finalized:
                raise ValueError(
                    "attribution exposure references an invalid or unfinalized outcome"
                )
            repository = _required(
                repositories,
                task.repository_snapshot_hash,
                "attribution repository evidence",
            )
            candidates = {item.memory_id: item for item in task.candidates}
            candidate = candidates.get(record.memory_id)
            if candidate is None:
                raise ValueError("attribution exposure memory is absent from task candidates")
            if (
                task.task_group != exposure.task_group
                or task.stream_id != exposure.stream_id
                or task.memory_project_key != exposure.memory_project_key
                or task.task_ordinal != exposure.task_ordinal
                or task.candidate_snapshot_hash != exposure.candidate_snapshot_hash
                or task.policy_identity != exposure.policy_identity
                or repository.snapshot.repository_revision != exposure.repository_revision
                or candidate.tier != exposure.tier
                or (record.memory_id in task.selected_memory_ids) != exposure.selected
                or outcome.outcome.reward != exposure.reward
                or outcome.outcome.evaluator_name != exposure.evaluator_name
                or outcome.outcome.evaluator_version != exposure.evaluator_version
                or outcome.outcome.evaluator_hash != exposure.evaluator_hash
            ):
                raise ValueError("attribution exposure does not match joined task/outcome evidence")


def _validate_task_join(
    task: TaskEvidence,
    outcome: TaskOutcomeEvidence,
    repository: RepositoryEvidence,
    decisions: Mapping[str, DecisionEvent],
) -> None:
    if (
        outcome.task_ordinal != task.task_ordinal
        or outcome.trajectory_id != task.trajectory_id
        or outcome.stream_id != task.stream_id
        or outcome.memory_project_key != task.memory_project_key
        or outcome.outcome.task_group != task.task_group
    ):
        raise ValueError(f"task/outcome join mismatch: {task.task_id}")
    if task.trajectory.reward != outcome.outcome.reward:
        raise ValueError("trajectory reward does not match authoritative outcome")
    trajectory_resolved = _trajectory_resolved(task.trajectory.outcome)
    if trajectory_resolved != outcome.outcome.resolved:
        raise ValueError("trajectory outcome does not match authoritative outcome")
    if repository.stream_id != task.stream_id:
        raise ValueError("task repository stream mismatch")
    if repository.snapshot.memory_project_key != task.memory_project_key:
        raise ValueError("task repository project mismatch")
    if any(
        candidate.memory_id not in set(repository.snapshot.memory_ids)
        for candidate in task.candidates
    ):
        raise ValueError("task candidate is absent from the joined repository snapshot")
    expected_roles: dict[str, str] = {}
    if task.selection_decision_id is not None:
        expected_roles[task.selection_decision_id] = "selection"
    if task.writing_decision_id is not None:
        expected_roles[task.writing_decision_id] = "writing"
    for decision_id, role in expected_roles.items():
        event = _required(decisions, decision_id, "decision event")
        _validate_decision_event(
            event,
            role=role,
            trajectory_id=task.trajectory_id,
            task_group=task.task_group,
            stream_id=task.stream_id,
            project_key=task.memory_project_key,
            identity=task.policy_identity,
            repository_revision=repository.snapshot.repository_revision,
            candidate_snapshot_hash=task.candidate_snapshot_hash,
        )
        if role in {"selection", "writing"} and event.canonical_tools:
            raise ValueError(f"{role} evidence unexpectedly contains tool schemas")
    for action_index, action in enumerate(task.action_decisions):
        event = _required(decisions, action.decision_id, "action decision event")
        _validate_decision_event(
            event,
            role="action",
            trajectory_id=task.trajectory_id,
            task_group=task.task_group,
            stream_id=task.stream_id,
            project_key=task.memory_project_key,
            identity=task.policy_identity,
            repository_revision=repository.snapshot.repository_revision,
            candidate_snapshot_hash=task.candidate_snapshot_hash,
        )
        if event.canonical_tools != action.tools:
            raise ValueError("action tool schema does not match decision evidence")
        if (event.turn_index, event.step_index) != (action.turn_index, action.step_index):
            raise ValueError("action decision indexes do not match decision evidence")
        if without_selected_memory_context(event.canonical_messages) != action.prefix_messages:
            raise ValueError("action public prefix does not match memory-free decision evidence")
        if _event_tool_calls(event) != action.expected_tool_calls:
            raise ValueError("action tool calls do not match decision evidence")
        following = (
            _required(
                decisions,
                task.action_decisions[action_index + 1].decision_id,
                "following action decision event",
            )
            if action_index + 1 < len(task.action_decisions)
            else None
        )
        if _observations_between(event, following) != action.observation_messages:
            raise ValueError("action observations do not match decision evidence")


def _validate_maintenance_join(
    maintenance: MaintenanceEvidence,
    repository: RepositoryEvidence,
    outcomes: Sequence[TaskOutcomeEvidence],
    decisions: Mapping[str, DecisionEvent],
) -> None:
    if (
        repository.stream_id != maintenance.stream_id
        or repository.snapshot.memory_project_key != maintenance.memory_project_key
    ):
        raise ValueError("maintenance repository join mismatch")
    repository_ids = set(repository.snapshot.memory_ids)
    for diagnostic in maintenance.redundancy_diagnostics:
        if (
            diagnostic.left_memory_id not in repository_ids
            or diagnostic.right_memory_id not in repository_ids
        ):
            raise ValueError("maintenance redundancy diagnostic references unknown memory")
    for outcome in outcomes:
        if (
            outcome.stream_id != maintenance.stream_id
            or outcome.memory_project_key != maintenance.memory_project_key
            or not outcome.task_valid
            or not outcome.outcome_finalized
            or outcome.task_ordinal > maintenance.as_of_task_ordinal
        ):
            raise ValueError("maintenance outcome window is invalid")
    for decision_id in maintenance.decision_ids:
        event = _required(decisions, decision_id, "maintenance decision")
        _validate_decision_event(
            event,
            role="maintenance",
            trajectory_id=maintenance.cadence_id,
            task_group=maintenance.task_group,
            stream_id=maintenance.stream_id,
            project_key=maintenance.memory_project_key,
            identity=maintenance.policy_identity,
            repository_revision=repository.snapshot.repository_revision,
            candidate_snapshot_hash=repository.snapshot.snapshot_hash,
        )
        if event.run_id != maintenance.attempt_id:
            raise ValueError("maintenance decision attempt does not match evidence")
        if event.canonical_tools != maintenance.tools:
            raise ValueError("maintenance tool schema does not match decision evidence")


def _validate_decision_event(
    event: DecisionEvent,
    *,
    role: str,
    trajectory_id: str,
    task_group: str,
    stream_id: str,
    project_key: str,
    identity: PolicyIdentity,
    repository_revision: str,
    candidate_snapshot_hash: str,
) -> None:
    if (
        event.status != "success"
        or event.purpose != "fast_loop_evidence"
        or event.role != role
        or event.trajectory_id != trajectory_id
        or event.task_group != task_group
        or event.stream_id != stream_id
        or event.memory_project_key != project_key
        or event.policy_identity != identity
        or event.repository_revision != repository_revision
        or event.candidate_snapshot_hash != candidate_snapshot_hash
    ):
        raise ValueError(f"decision event join mismatch: {event.decision_id}")


def _successful_trajectories(
    tasks: Sequence[TaskEvidence],
    outcomes: Mapping[Hashable, TaskOutcomeEvidence],
) -> dict[tuple[str, str], TaskEvidence]:
    grouped: dict[tuple[str, str], list[tuple[TaskEvidence, TaskOutcomeEvidence]]] = {}
    for task in tasks:
        outcome = outcomes.get(_task_key(task))
        if (
            outcome is not None
            and outcome.task_valid
            and outcome.outcome_finalized
            and outcome.outcome.resolved
            and outcome.outcome.reward > 0.0
        ):
            grouped.setdefault((task.split, task.task_group), []).append((task, outcome))
    selected: dict[tuple[str, str], TaskEvidence] = {}
    for key, values in grouped.items():
        values.sort(key=lambda item: (
            -item[1].outcome.reward,
            len(item[0].trajectory.steps),
            item[0].trajectory_id,
        ))
        selected[key] = values[0][0]
    return selected


def _trajectory_resolved(outcome: str) -> bool:
    normalized = outcome.strip().casefold()
    if normalized in {"resolved", "success"}:
        return True
    if normalized in {"failed", "failure"}:
        return False
    raise ValueError("trajectory outcome must identify success or failure")


def _event_tool_calls(event: DecisionEvent) -> tuple[CanonicalToolCall, ...]:
    raw = event.parsed_output.get("tool_calls", ())
    if raw in (None, ()):
        return ()
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise ValueError("action decision parsed tool_calls must be an array of objects")
    return tuple(CanonicalToolCall.from_dict(item) for item in raw)


def _maintenance_event_tool_calls(event: DecisionEvent) -> tuple[CanonicalToolCall, ...]:
    raw = event.parsed_output.get("tool_call")
    if not isinstance(raw, Mapping):
        raise ValueError("maintenance decision requires one parsed tool_call")
    arguments = raw.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("maintenance tool_call arguments must be an object")
    call_id = raw.get("call_id")
    name = raw.get("name")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("maintenance tool_call requires call_id")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("maintenance tool_call requires name")
    return (
        CanonicalToolCall(
            call_id,
            name,
            canonical_json_bytes(arguments).decode("utf-8"),
        ),
    )


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


def _ready_values(
    candidates: Sequence[Any],
    attribution: Mapping[str, PaperAttributionRecord],
) -> tuple[MemoryValueEvidence, ...] | None:
    values: list[MemoryValueEvidence] = []
    for candidate in candidates:
        record = attribution.get(candidate.memory_id)
        if record is None or record.status != "ready" or record.memory_score is None:
            return None
        values.append(_memory_value(record))
    return tuple(values)


def _memory_value(record: PaperAttributionRecord) -> MemoryValueEvidence:
    if record.status != "ready" or record.attribution is None or record.memory_score is None:
        raise ValueError(f"memory attribution is not ready: {record.memory_id}")
    return MemoryValueEvidence(
        memory_id=record.memory_id,
        tier=record.tier,
        attribution=record.attribution,
        gamma=record.gamma,
        memory_score=record.memory_score,
        status=record.status,
    )


def _attribution_ref(
    memory_id: str,
    attribution: Mapping[str, PaperAttributionRecord],
) -> str:
    record = _required(attribution, memory_id, "attribution")
    return canonical_sha256(record.to_dict())


def _exclusion(evidence_id: str, role: str, reason: str) -> Mapping[str, Any]:
    return {"evidence_id": evidence_id, "role": role, "reason": reason}


def _unique_refs(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _task_key(task: TaskEvidence) -> tuple[str, str, str]:
    return _task_key_values(task.stream_id, task.memory_project_key, task.task_id)


def _outcome_task_key(outcome: TaskOutcomeEvidence) -> tuple[str, str, str]:
    return _task_key_values(
        outcome.stream_id,
        outcome.memory_project_key,
        outcome.outcome.task_id,
    )


def _task_key_values(stream_id: str, project_key: str, task_id: str) -> tuple[str, str, str]:
    return stream_id, project_key, task_id


def _required(mapping: Mapping[Hashable, T], key: Hashable, name: str) -> T:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"missing {name}: {key}")
    return value


def _unique_map(
    values: Sequence[T],
    key: Callable[[T], Hashable],
    name: str,
) -> dict[Hashable, T]:
    result: dict[Hashable, T] = {}
    for value in values:
        item_key = key(value)
        if item_key in result:
            raise ValueError(f"duplicate {name}: {item_key}")
        result[item_key] = value
    return result


def _load_jsonl(path: str | Path, loader: Callable[[Mapping[str, Any]], T]) -> tuple[T, ...]:
    values: list[T] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid OPD JSON at line {line_no}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"OPD JSON line {line_no} must be an object")
            values.append(loader(payload))
    return tuple(values)


__all__ = [
    "PreparedLearnerDecision",
    "PreparedRound",
    "load_learner_samples",
    "load_maintenance_evidence",
    "load_maintenance_attempts",
    "load_repository_evidence",
    "load_runtime_exclusions",
    "load_task_evidence",
    "load_task_outcomes",
    "prepare_round_decisions",
    "sample_statistics",
    "write_evidence_jsonl",
    "write_learner_samples",
]
