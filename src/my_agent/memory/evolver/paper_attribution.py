"""Paper-faithful OPD-Evolver attribution (Eq. 11-12)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, isfinite, sqrt
from statistics import mean

from my_agent.memory.evolver.attribution_schema import (
    AttributionEvidenceRef,
    CandidateExposure,
    GroupAttribution,
    PaperAttributionRecord,
    WritingScoreDecision,
)


DEFAULT_PAPER_TIER_PRIORS: dict[str, float] = {
    "trajectory": 1.0,
    "tip": 1.0,
    "skill": 1.0,
    "tool": 1.0,
}


def rho_g(n_plus: int, n_minus: int) -> float:
    if n_plus < 1 or n_minus < 1:
        raise ValueError("rho_g requires selected and not-selected counterfactual evidence")
    return n_plus / (n_plus + n_minus)


def confidence_gamma(n_plus: int) -> float:
    if isinstance(n_plus, bool) or not isinstance(n_plus, int) or n_plus < 0:
        raise ValueError("n_plus must be a non-negative integer")
    return 1.0 - 1.0 / sqrt(1.0 + n_plus)


def memory_score(*, tier_prior: float, gamma: float, attribution: float) -> float:
    if not all(isfinite(value) for value in (tier_prior, gamma, attribution)):
        raise ValueError("memory score inputs must be finite")
    return tier_prior * gamma * attribution


def compute_memory_attribution(
    *,
    memory_id: str,
    tier: str,
    memory_project_key: str,
    exposures: Sequence[CandidateExposure],
    collection_round: int,
    as_of_ordinal: int,
    tier_prior: float = 1.0,
) -> PaperAttributionRecord:
    relevant = [
        exposure
        for exposure in exposures
        if exposure.memory_id == memory_id
        and exposure.tier == tier
        and exposure.memory_project_key == memory_project_key
        and exposure.collection_round == collection_round
        and exposure.task_ordinal <= as_of_ordinal
    ]
    if not relevant:
        raise ValueError(f"no candidate exposures found for memory: {memory_id}")
    policy_hashes = {exposure.policy_identity.identity_hash for exposure in relevant}
    if len(policy_hashes) != 1:
        raise ValueError("one attribution round must use exactly one policy identity")
    duplicate_keys: set[tuple[str, str, str]] = set()
    for exposure in relevant:
        key = (exposure.stream_id, exposure.task_id, exposure.memory_id)
        if key in duplicate_keys:
            raise ValueError(f"duplicate candidate exposure for task/memory: {exposure.task_id}")
        duplicate_keys.add(key)

    grouped: dict[str, list[CandidateExposure]] = {}
    for exposure in relevant:
        grouped.setdefault(exposure.task_group, []).append(exposure)
    group_records: list[GroupAttribution] = []
    contributions: list[float] = []
    for task_group, group_exposures in sorted(grouped.items()):
        selected = [item.reward for item in group_exposures if item.selected]
        not_selected = [item.reward for item in group_exposures if not item.selected]
        if not selected or not not_selected:
            group_records.append(GroupAttribution(
                task_group=task_group,
                n_plus=len(selected),
                n_minus=len(not_selected),
                mean_plus=mean(selected) if selected else None,
                mean_minus=mean(not_selected) if not_selected else None,
                rho_g=None,
                delta=None,
                status="insufficient_counterfactual_evidence",
            ))
            continue
        selected_mean = mean(selected)
        not_selected_mean = mean(not_selected)
        weight = rho_g(len(selected), len(not_selected))
        delta = selected_mean - not_selected_mean
        contributions.append(weight * delta)
        group_records.append(GroupAttribution(
            task_group=task_group,
            n_plus=len(selected),
            n_minus=len(not_selected),
            mean_plus=selected_mean,
            mean_minus=not_selected_mean,
            rho_g=weight,
            delta=delta,
            status="ready",
        ))

    n_plus_total = sum(1 for exposure in relevant if exposure.selected)
    gamma = confidence_gamma(n_plus_total)
    evidence_refs = tuple(
        AttributionEvidenceRef(
            exposure=exposure,
        )
        for exposure in sorted(
            relevant,
            key=lambda item: (item.task_ordinal, item.stream_id, item.task_id),
        )
    )
    if not contributions:
        return PaperAttributionRecord(
            memory_id=memory_id,
            tier=tier,
            memory_project_key=memory_project_key,
            collection_round=collection_round,
            groups=tuple(group_records),
            evidence_refs=evidence_refs,
            attribution=None,
            n_plus_total=n_plus_total,
            gamma=gamma,
            tier_prior=tier_prior,
            memory_score=None,
            as_of_ordinal=as_of_ordinal,
            policy_identity_hash=next(iter(policy_hashes)),
            status="insufficient_counterfactual_evidence",
        )
    attribution = sum(contributions)
    return PaperAttributionRecord(
        memory_id=memory_id,
        tier=tier,
        memory_project_key=memory_project_key,
        collection_round=collection_round,
        groups=tuple(group_records),
        evidence_refs=evidence_refs,
        attribution=attribution,
        n_plus_total=n_plus_total,
        gamma=gamma,
        tier_prior=tier_prior,
        memory_score=memory_score(
            tier_prior=tier_prior,
            gamma=gamma,
            attribution=attribution,
        ),
        as_of_ordinal=as_of_ordinal,
        policy_identity_hash=next(iter(policy_hashes)),
        status="ready",
    )


def compute_round_attribution(
    exposures: Sequence[CandidateExposure],
    *,
    collection_round: int,
    valid_task_ordinals: Sequence[int],
    tier_priors: Mapping[str, float] | None = None,
) -> tuple[PaperAttributionRecord, ...]:
    ordinals = tuple(valid_task_ordinals)
    if any(
        isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1
        for ordinal in ordinals
    ):
        raise ValueError("valid task ordinals must be positive integers")
    if not exposures:
        return ()
    if not ordinals:
        raise ValueError("candidate exposures require authoritative valid task ordinals")
    valid_ordinal_set = set(ordinals)
    if any(exposure.task_ordinal not in valid_ordinal_set for exposure in exposures):
        raise ValueError("candidate exposure does not reference an authoritative valid task")
    as_of_ordinal = max(ordinals)
    priors = dict(DEFAULT_PAPER_TIER_PRIORS)
    if tier_priors is not None:
        priors.update({str(key): float(value) for key, value in tier_priors.items()})
    identities = {exposure.policy_identity.identity_hash for exposure in exposures}
    if len(identities) != 1:
        raise ValueError("collection round exposures must share one policy identity")
    rounds = {exposure.collection_round for exposure in exposures}
    if rounds != {collection_round}:
        raise ValueError("collection round exposures must match collection_round")
    keys = sorted({
        (exposure.memory_project_key, exposure.tier, exposure.memory_id)
        for exposure in exposures
    })
    return tuple(
        compute_memory_attribution(
            memory_id=memory_id,
            tier=tier,
            memory_project_key=project_key,
            exposures=exposures,
            collection_round=collection_round,
            as_of_ordinal=as_of_ordinal,
            tier_prior=priors[tier],
        )
        for project_key, tier, memory_id in keys
    )


def positive_selected_memory_ids(
    selected_memory_ids: Sequence[str],
    attribution: Mapping[str, PaperAttributionRecord],
) -> tuple[str, ...]:
    positive: list[str] = []
    for memory_id in selected_memory_ids:
        record = _record_for_id(attribution, memory_id)
        if (
            record is not None
            and record.status == "ready"
            and record.memory_score is not None
            and record.memory_score > 0.0
        ):
            positive.append(memory_id)
    return tuple(positive)


def teacher_memory_records(
    attribution: Mapping[str, PaperAttributionRecord],
    *,
    minimum_memory_score: float = 0.01,
    max_items: int = 20,
) -> tuple[PaperAttributionRecord, ...]:
    if not isfinite(minimum_memory_score):
        raise ValueError("minimum_memory_score must be finite")
    eligible: list[PaperAttributionRecord] = []
    for memory_id, record in attribution.items():
        if record.memory_id != memory_id:
            raise ValueError("attribution mapping key does not match record.memory_id")
        if (
            record.status == "ready"
            and record.memory_score is not None
            and record.memory_score >= minimum_memory_score
        ):
            eligible.append(record)
    eligible.sort(key=lambda record: (-float(record.memory_score), record.memory_id))
    return tuple(eligible[:max(0, int(max_items))])


def writing_top_fraction(
    written_memory_ids: Sequence[str],
    attribution: Mapping[str, PaperAttributionRecord],
    *,
    collection_round: int,
    fraction: float = 0.30,
) -> tuple[WritingScoreDecision, ...]:
    if not isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("writing top fraction must be in (0, 1]")
    ready: list[PaperAttributionRecord] = []
    missing: list[str] = []
    seen: set[str] = set()
    for memory_id in written_memory_ids:
        if memory_id in seen:
            raise ValueError(f"duplicate written memory id: {memory_id}")
        seen.add(memory_id)
        record = _record_for_id(attribution, memory_id)
        if record is not None and record.collection_round != collection_round:
            raise ValueError("writing attribution records must match collection_round")
        if record is not None and record.status == "ready" and record.memory_score is not None:
            ready.append(record)
        else:
            missing.append(memory_id)
    ready.sort(key=lambda record: (-float(record.memory_score), record.memory_id))
    keep_count = max(1, ceil(len(ready) * fraction)) if ready else 0
    boundary_score = float(ready[keep_count - 1].memory_score) if keep_count else None
    ranked = tuple(
        WritingScoreDecision(
            memory_id=record.memory_id,
            memory_score=float(record.memory_score),
            rank=index,
            selected=index <= keep_count,
            reason="round_top_fraction" if index <= keep_count else "below_round_top_fraction",
            collection_round=collection_round,
            top_fraction=fraction,
            cutoff_rank=keep_count or None,
            boundary_score=boundary_score,
        )
        for index, record in enumerate(ready, 1)
    )
    unscored = tuple(
        WritingScoreDecision(
            memory_id=memory_id,
            memory_score=None,
            rank=None,
            selected=False,
            reason="no_ready_attribution",
            collection_round=collection_round,
            top_fraction=fraction,
            cutoff_rank=keep_count or None,
            boundary_score=boundary_score,
        )
        for memory_id in sorted(missing)
    )
    return ranked + unscored


def _record_for_id(
    attribution: Mapping[str, PaperAttributionRecord],
    memory_id: str,
) -> PaperAttributionRecord | None:
    record = attribution.get(memory_id)
    if record is not None and record.memory_id != memory_id:
        raise ValueError("attribution mapping key does not match record.memory_id")
    return record


__all__ = [
    "DEFAULT_PAPER_TIER_PRIORS",
    "compute_memory_attribution",
    "compute_round_attribution",
    "confidence_gamma",
    "memory_score",
    "positive_selected_memory_ids",
    "rho_g",
    "teacher_memory_records",
    "writing_top_fraction",
]
