"""Build one strict current-checkpoint OPD learner collection round."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from my_agent.opd_ablation import (
    ablation_excluded_roles,
    ablation_recipe_hash,
)
from my_agent.opd_data.export import (
    load_learner_samples,
    prepare_round_decisions,
    sample_statistics,
    write_learner_samples,
)
from my_agent.opd_data.schema import (
    ExportManifest,
    MaintenanceAttemptEvidence,
    MaintenanceEvidence,
    RepositoryEvidence,
    RuntimeExclusionEvidence,
    TaskEvidence,
    TaskOutcomeEvidence,
)
from my_agent.memory.evolver.attribution_schema import PaperAttributionRecord
from my_agent.policy.contracts import TrainablePolicy
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.training.contracts import DecisionEvent
from my_agent.training.ablation import effective_attribution
from my_agent.training.opd_rollout import (
    generate_action_rollout_samples,
    generate_learner_sample,
    generate_maintenance_rollout,
)


LEARNER_EVENTS_FILENAME = "learner_events.jsonl"
EXPORT_MANIFEST_FILENAME = "export_manifest.json"


@dataclass(frozen=True)
class CollectionRoundResult:
    learner_path: Path
    manifest_path: Path
    manifest: ExportManifest


def build_collection_round(
    *,
    collection_round: int,
    policy: TrainablePolicy,
    tasks: Sequence[TaskEvidence],
    outcomes: Sequence[TaskOutcomeEvidence],
    repositories: Sequence[RepositoryEvidence],
    maintenance: Sequence[MaintenanceEvidence],
    decision_events: Sequence[DecisionEvent],
    attribution: Sequence[PaperAttributionRecord],
    output_dir: str | Path,
    runtime_exclusions: Sequence[RuntimeExclusionEvidence] = (),
    maintenance_attempts: Sequence[MaintenanceAttemptEvidence] = (),
    writing_top_fraction: float = 0.30,
    teacher_minimum_score: float = 0.01,
    teacher_max_items: int = 20,
    max_new_tokens: int = 1_024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int | None = None,
    ablation: str = "",
) -> CollectionRoundResult:
    normalized_ablation = str(ablation).strip().lower()
    recipe_hash = ablation_recipe_hash(normalized_ablation)
    identity = policy.identity()
    applied_attribution = effective_attribution(normalized_ablation, attribution, tasks)
    prepared = prepare_round_decisions(
        collection_round=collection_round,
        trainer_identity=identity,
        tasks=tasks,
        outcomes=outcomes,
        repositories=repositories,
        maintenance=maintenance,
        decision_events=decision_events,
        attribution=applied_attribution,
        writing_top_fraction_value=writing_top_fraction,
        teacher_minimum_score=teacher_minimum_score,
        teacher_max_items=teacher_max_items,
    )
    excluded_roles = ablation_excluded_roles(normalized_ablation)
    decisions = sorted(
        (item for item in prepared.decisions if item.role not in excluded_roles),
        key=lambda item: (
            item.role,
            item.stream_id,
            item.task_group,
            item.source_evidence_ids,
        ),
    )
    samples_list: list[Any] = []
    rollout_exclusions: list[Mapping[str, Any]] = []
    generation_index = 0
    action_groups: dict[str, list[Any]] = {}
    maintenance_groups: dict[str, list[Any]] = {}
    for decision in decisions:
        if decision.role == "action":
            action_groups.setdefault(decision.action_rollout_id, []).append(decision)
            continue
        if decision.role == "maintenance":
            maintenance_groups.setdefault(decision.maintenance_rollout_id, []).append(decision)
            continue
        samples_list.append(generate_learner_sample(
            decision,
            policy=policy,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=None if seed is None else seed + generation_index,
        ))
        generation_index += 1
    for rollout_id in sorted(action_groups):
        rollout = action_groups[rollout_id]
        generated = generate_action_rollout_samples(
            rollout,
            policy=policy,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=None if seed is None else seed + generation_index,
        )
        samples_list.extend(generated)
        if len(generated) < len(rollout):
            rollout_exclusions.append({
                "evidence_id": rollout_id,
                "role": "action",
                "reason": "student_tool_call_diverged_from_replayable_observation",
                "generated_turns": len(generated),
                "available_turns": len(rollout),
            })
        generation_index += len(rollout)
    for rollout_id in sorted(maintenance_groups):
        rollout = maintenance_groups[rollout_id]
        generated = generate_maintenance_rollout(
            rollout,
            policy=policy,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=None if seed is None else seed + generation_index,
        )
        samples_list.extend(generated.samples)
        if generated.diverged:
            rollout_exclusions.append({
                "evidence_id": rollout_id,
                "role": "maintenance",
                "reason": "student_tool_call_diverged_from_replayable_observation",
                "generated_turns": len(generated.samples),
                "available_turns": len(rollout),
            })
        generation_index += len(rollout)
    samples = tuple(samples_list)
    if policy.identity() != identity:
        raise ValueError("collection policy identity changed during learner regeneration")
    validate_learner_samples(
        samples,
        collection_round=collection_round,
        trainer_identity_hash=identity.identity_hash,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    learner_path = write_learner_samples(samples, root / LEARNER_EVENTS_FILENAME)
    loaded = load_learner_samples(learner_path)
    if loaded != tuple(sorted(samples, key=lambda item: (item.role, item.task_group, item.sample_id))):
        raise ValueError("learner dataset round-trip mismatch")
    dataset_hash = canonical_sha256([sample.to_dict() for sample in loaded])
    stats = sample_statistics(loaded)
    outcome_counts = Counter(
        "resolved" if item.outcome.resolved else "failed"
        for item in outcomes
        if item.task_valid and item.outcome_finalized
    )
    attribution_input_hash = _sequence_hash(attribution)
    attribution_effective_hash = _sequence_hash(applied_attribution)
    manifest = ExportManifest(
        collection_round=collection_round,
        trainer_initialization_identity=identity,
        learner_dataset_hash=dataset_hash,
        sample_count=len(loaded),
        role_counts=stats["role_counts"],
        split_counts=stats["split_counts"],
        task_group_counts=stats["task_group_counts"],
        outcome_counts=dict(outcome_counts),
        source_hashes={
            "task_evidence": _sequence_hash(tasks),
            "task_outcomes": _sequence_hash(outcomes),
            "repository_evidence": _sequence_hash(repositories),
            "maintenance_evidence": _sequence_hash(maintenance),
            "decision_events": _sequence_hash(decision_events),
            "attribution": attribution_effective_hash,
            "attribution_input": attribution_input_hash,
            "attribution_effective": attribution_effective_hash,
            "runtime_exclusions": _sequence_hash(runtime_exclusions),
            "maintenance_attempts": _sequence_hash(maintenance_attempts),
        },
        writing_score_decisions=tuple(
            item.to_dict() for item in prepared.writing_score_decisions
        ),
        exclusions=tuple((
            *prepared.exclusions,
            *rollout_exclusions,
            *(
                {
                    "evidence_id": canonical_sha256({
                        "ablation": normalized_ablation,
                        "role": role,
                    }),
                    "role": role,
                    "reason": "disabled_by_ablation",
                }
                for role in sorted(excluded_roles)
            ),
            *(item.to_dict() for item in runtime_exclusions),
        )),
        ablation=normalized_ablation,
        ablation_recipe_hash=recipe_hash,
    )
    manifest_path = root / EXPORT_MANIFEST_FILENAME
    manifest_path.write_bytes(canonical_json_bytes(manifest.to_dict()) + b"\n")
    return CollectionRoundResult(learner_path, manifest_path, manifest)


def build_replay_ablation_dataset(
    *,
    d0_dir: str | Path,
    d1_dir: str | Path,
    output_dir: str | Path,
) -> CollectionRoundResult:
    sources = tuple(
        _load_round_source(path, expected_round=index)
        for index, path in enumerate((d0_dir, d1_dir))
    )
    d0_manifest, d0_samples, d0_manifest_hash = sources[0]
    d1_manifest, d1_samples, d1_manifest_hash = sources[1]
    if d0_manifest.ablation or d1_manifest.ablation:
        raise ValueError("replay D0+D1 inputs must be main-experiment datasets")
    combined = tuple((*d0_samples, *d1_samples))
    if len({sample.sample_id for sample in combined}) != len(combined):
        raise ValueError("replay D0+D1 dataset contains duplicate sample IDs")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    learner_path = write_learner_samples(combined, root / LEARNER_EVENTS_FILENAME)
    loaded = load_learner_samples(learner_path)
    dataset_hash = canonical_sha256([sample.to_dict() for sample in loaded])
    stats = sample_statistics(loaded)
    outcome_counts = Counter(d0_manifest.outcome_counts)
    outcome_counts.update(d1_manifest.outcome_counts)
    manifest = ExportManifest(
        collection_round=1,
        trainer_initialization_identity=d1_manifest.trainer_initialization_identity,
        learner_dataset_hash=dataset_hash,
        sample_count=len(loaded),
        role_counts=stats["role_counts"],
        split_counts=stats["split_counts"],
        task_group_counts=stats["task_group_counts"],
        outcome_counts=dict(outcome_counts),
        source_hashes={
            "d0_learner_dataset": d0_manifest.learner_dataset_hash,
            "d0_export_manifest": d0_manifest_hash,
            "d1_learner_dataset": d1_manifest.learner_dataset_hash,
            "d1_export_manifest": d1_manifest_hash,
        },
        writing_score_decisions=tuple((
            *d0_manifest.writing_score_decisions,
            *d1_manifest.writing_score_decisions,
        )),
        exclusions=tuple((*d0_manifest.exclusions, *d1_manifest.exclusions)),
        ablation="replay_d0_d1",
        ablation_recipe_hash=ablation_recipe_hash("replay_d0_d1"),
        sample_policy_identity_hashes=tuple(sorted({
            sample.policy_identity.identity_hash for sample in loaded
        })),
        sample_collection_rounds=tuple(sorted({
            sample.collection_round for sample in loaded
        })),
        current_checkpoint_only=False,
        replay_enabled=True,
    )
    manifest_path = root / EXPORT_MANIFEST_FILENAME
    manifest_path.write_bytes(canonical_json_bytes(manifest.to_dict()) + b"\n")
    return CollectionRoundResult(learner_path, manifest_path, manifest)


def validate_learner_samples(
    samples: Sequence[Any],
    *,
    collection_round: int,
    trainer_identity_hash: str,
    sample_policy_identity_hashes: Sequence[str] = (),
    sample_collection_rounds: Sequence[int] = (),
) -> None:
    allowed_identity_hashes = frozenset(
        sample_policy_identity_hashes or (trainer_identity_hash,)
    )
    allowed_rounds = frozenset(sample_collection_rounds or (collection_round,))
    sample_ids: set[str] = set()
    expected_views = {
        "selection": ("selection_public", "selection_hindsight"),
        "action": ("action_public", "action_hindsight"),
        "writing": ("writing_public", "writing_hindsight"),
        "maintenance": ("maintenance_public", "maintenance_hindsight"),
    }
    for sample in samples:
        if sample.sample_id in sample_ids:
            raise ValueError(f"duplicate learner sample_id: {sample.sample_id}")
        sample_ids.add(sample.sample_id)
        if sample.collection_round not in allowed_rounds:
            raise ValueError("learner sample collection round is not declared")
        if sample.policy_identity.identity_hash not in allowed_identity_hashes:
            raise ValueError("learner sample policy identity is not declared")
        public_type, hindsight_type = expected_views[sample.role]
        if sample.student_public_view.get("view_type") != public_type:
            raise ValueError("learner public view does not match role")
        if sample.teacher_hindsight_view.get("view_type") != hindsight_type:
            raise ValueError("learner hindsight view does not match role")
        if not sample.student_completion_token_ids or not any(sample.assistant_loss_mask):
            raise ValueError("learner sample requires a non-empty trainable completion")


def _sequence_hash(values: Sequence[Any]) -> str:
    payloads: list[Mapping[str, Any]] = []
    for value in values:
        payload = value.to_dict()
        if not isinstance(payload, Mapping):
            raise ValueError("round source to_dict() must return an object")
        payloads.append(payload)
    payloads.sort(key=canonical_sha256)
    return canonical_sha256(payloads)


def _load_round_source(
    path: str | Path,
    *,
    expected_round: int,
) -> tuple[ExportManifest, tuple[Any, ...], str]:
    root = Path(path).expanduser().resolve()
    manifest_payload = json.loads((root / EXPORT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("replay source export manifest must be an object")
    manifest = ExportManifest.from_dict(manifest_payload)
    if manifest.collection_round != expected_round:
        raise ValueError(f"replay source D{expected_round} has the wrong collection round")
    samples = load_learner_samples(root / LEARNER_EVENTS_FILENAME)
    if len(samples) != manifest.sample_count:
        raise ValueError("replay source learner count does not match its manifest")
    if canonical_sha256([sample.to_dict() for sample in samples]) != manifest.learner_dataset_hash:
        raise ValueError("replay source learner hash does not match its manifest")
    return manifest, samples, canonical_sha256(manifest_payload)


__all__ = [
    "EXPORT_MANIFEST_FILENAME",
    "LEARNER_EVENTS_FILENAME",
    "CollectionRoundResult",
    "build_replay_ablation_dataset",
    "build_collection_round",
    "validate_learner_samples",
]
