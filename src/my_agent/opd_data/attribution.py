"""Build authoritative paper-attribution artifacts for one collection round."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable, Sequence, TypeVar

from my_agent.memory.evolver.attribution_export import (
    write_attribution_events,
    write_candidate_exposures,
)
from my_agent.memory.evolver.attribution_schema import (
    CandidateExposure,
    PaperAttributionRecord,
)
from my_agent.memory.evolver.paper_attribution import compute_round_attribution
from my_agent.opd_data.schema import RepositoryEvidence, TaskEvidence, TaskOutcomeEvidence


CANDIDATE_EXPOSURES_FILENAME = "candidate_exposures.jsonl"
ATTRIBUTION_EVENTS_FILENAME = "attribution_events.jsonl"

T = TypeVar("T")


@dataclass(frozen=True)
class RoundAttributionResult:
    candidate_exposures_path: Path
    attribution_events_path: Path
    as_of_ordinal: int
    exposure_count: int
    attribution_count: int
    records: tuple[PaperAttributionRecord, ...]


def build_round_attribution(
    *,
    collection_round: int,
    tasks: Sequence[TaskEvidence],
    outcomes: Sequence[TaskOutcomeEvidence],
    repositories: Sequence[RepositoryEvidence],
    output_dir: str | Path,
) -> RoundAttributionResult:
    """Strictly join runtime evidence and persist Eq. 11-12 inputs and outputs."""

    if collection_round < 0:
        raise ValueError("collection_round must be non-negative")
    for record in (*tasks, *outcomes, *repositories):
        if record.collection_round != collection_round:
            raise ValueError("attribution evidence crosses collection rounds")

    outcome_by_task = _unique_map(outcomes, _outcome_key, "task outcome")
    repository_by_hash = _unique_map(
        repositories,
        lambda item: item.snapshot.snapshot_hash,
        "repository snapshot",
    )
    valid_outcomes = tuple(
        outcome
        for outcome in outcomes
        if outcome.task_valid and outcome.outcome_finalized
    )
    if not valid_outcomes:
        raise ValueError("round attribution requires at least one valid finalized task outcome")
    as_of_ordinal = max(outcome.task_ordinal for outcome in valid_outcomes)

    exposures: list[CandidateExposure] = []
    for task in sorted(
        tasks,
        key=lambda item: (
            item.task_ordinal,
            item.stream_id,
            item.memory_project_key,
            item.task_id,
        ),
    ):
        outcome = outcome_by_task.get(_task_key(task))
        if outcome is None:
            raise ValueError(f"missing authoritative outcome for task {task.task_id}")
        if not outcome.task_valid or not outcome.outcome_finalized:
            continue
        repository = repository_by_hash.get(task.repository_snapshot_hash)
        if repository is None:
            raise ValueError(f"missing repository snapshot for task {task.task_id}")
        _validate_task_join(task, outcome, repository)
        selected_ids = set(task.selected_memory_ids)
        for candidate in task.candidates:
            exposures.append(CandidateExposure(
                task_id=task.task_id,
                task_group=task.task_group,
                stream_id=task.stream_id,
                memory_project_key=task.memory_project_key,
                memory_id=candidate.memory_id,
                tier=candidate.tier,
                selected=candidate.memory_id in selected_ids,
                reward=outcome.outcome.reward,
                collection_round=collection_round,
                task_ordinal=task.task_ordinal,
                candidate_snapshot_hash=task.candidate_snapshot_hash,
                policy_identity=task.policy_identity,
                repository_revision=repository.snapshot.repository_revision,
                evaluator_name=outcome.outcome.evaluator_name,
                evaluator_version=outcome.outcome.evaluator_version,
                evaluator_hash=outcome.outcome.evaluator_hash,
            ))

    records = compute_round_attribution(
        tuple(exposures),
        collection_round=collection_round,
        valid_task_ordinals=tuple(outcome.task_ordinal for outcome in valid_outcomes),
    )
    root = Path(output_dir)
    exposure_path = write_candidate_exposures(
        exposures,
        root / CANDIDATE_EXPOSURES_FILENAME,
    )
    attribution_path = write_attribution_events(
        records,
        root / ATTRIBUTION_EVENTS_FILENAME,
    )
    return RoundAttributionResult(
        candidate_exposures_path=exposure_path,
        attribution_events_path=attribution_path,
        as_of_ordinal=as_of_ordinal,
        exposure_count=len(exposures),
        attribution_count=len(records),
        records=records,
    )


def _validate_task_join(
    task: TaskEvidence,
    outcome: TaskOutcomeEvidence,
    repository: RepositoryEvidence,
) -> None:
    if (
        outcome.task_ordinal != task.task_ordinal
        or outcome.trajectory_id != task.trajectory_id
        or outcome.stream_id != task.stream_id
        or outcome.memory_project_key != task.memory_project_key
        or outcome.outcome.task_id != task.task_id
        or outcome.outcome.task_group != task.task_group
    ):
        raise ValueError(f"task/outcome attribution join mismatch: {task.task_id}")
    if task.trajectory.reward != outcome.outcome.reward:
        raise ValueError("trajectory reward does not match authoritative outcome")
    if _trajectory_resolved(task.trajectory.outcome) != outcome.outcome.resolved:
        raise ValueError("trajectory outcome does not match authoritative outcome")
    if repository.stream_id != task.stream_id:
        raise ValueError("task repository stream mismatch")
    if repository.snapshot.memory_project_key != task.memory_project_key:
        raise ValueError("task repository project mismatch")
    repository_ids = set(repository.snapshot.memory_ids)
    if any(candidate.memory_id not in repository_ids for candidate in task.candidates):
        raise ValueError("task candidate is absent from the joined repository snapshot")


def _task_key(task: TaskEvidence) -> tuple[str, str, str]:
    return task.stream_id, task.memory_project_key, task.task_id


def _trajectory_resolved(outcome: str) -> bool:
    normalized = outcome.strip().casefold()
    if normalized in {"resolved", "success"}:
        return True
    if normalized in {"failed", "failure"}:
        return False
    raise ValueError("trajectory outcome must identify success or failure")


def _outcome_key(outcome: TaskOutcomeEvidence) -> tuple[str, str, str]:
    return outcome.stream_id, outcome.memory_project_key, outcome.outcome.task_id


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


__all__ = [
    "ATTRIBUTION_EVENTS_FILENAME",
    "CANDIDATE_EXPOSURES_FILENAME",
    "RoundAttributionResult",
    "build_round_attribution",
]
