"""Apply paper-ablation semantics while constructing learner datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from my_agent.memory.evolver.attribution_schema import (
    AttributionEvidenceRef,
    PaperAttributionRecord,
)
from my_agent.opd_ablation import ablation_recipe_hash
from my_agent.opd_data.schema import TaskEvidence
from my_agent.policy.identity import canonical_sha256


ABLATION_ATTRIBUTION_SCHEMA_VERSION = "opd-ablation-attribution-v1"


@dataclass(frozen=True)
class AblationAttributionRecord:
    memory_id: str
    tier: str
    memory_project_key: str
    collection_round: int
    evidence_refs: tuple[AttributionEvidenceRef, ...]
    attribution: float
    memory_score: float
    as_of_ordinal: int
    policy_identity_hash: str
    mode: str
    source_attribution_hash: str
    n_plus_total: int = 0
    gamma: float = 1.0
    tier_prior: float = 1.0
    status: str = "ready"
    schema_version: str = ABLATION_ATTRIBUTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "memory_id": self.memory_id,
            "tier": self.tier,
            "memory_project_key": self.memory_project_key,
            "collection_round": self.collection_round,
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "attribution": self.attribution,
            "n_plus_total": self.n_plus_total,
            "gamma": self.gamma,
            "tier_prior": self.tier_prior,
            "memory_score": self.memory_score,
            "as_of_ordinal": self.as_of_ordinal,
            "policy_identity_hash": self.policy_identity_hash,
            "status": self.status,
            "source_attribution_hash": self.source_attribution_hash,
        }


def effective_attribution(
    ablation: str,
    records: Sequence[PaperAttributionRecord],
    tasks: Sequence[TaskEvidence],
) -> tuple[PaperAttributionRecord | AblationAttributionRecord, ...]:
    normalized = str(ablation).strip().lower()
    if normalized not in {"no_attribution", "similarity_only"}:
        return tuple(records)
    scores_by_memory: dict[str, list[float]] = {}
    if normalized == "similarity_only":
        for task in tasks:
            for candidate in task.candidates:
                scores_by_memory.setdefault(candidate.memory_id, []).append(
                    float(candidate.retrieval_score)
                )
    result: list[AblationAttributionRecord] = []
    for record in records:
        if normalized == "no_attribution":
            score = 1.0
            mode = "uniform_unscored"
        else:
            values = scores_by_memory.get(record.memory_id, ())
            if not values:
                raise ValueError(
                    f"similarity-only attribution lacks retrieval scores: {record.memory_id}"
                )
            score = sum(values) / len(values)
            mode = "retrieval_score_mean"
        result.append(AblationAttributionRecord(
            memory_id=record.memory_id,
            tier=record.tier,
            memory_project_key=record.memory_project_key,
            collection_round=record.collection_round,
            evidence_refs=record.evidence_refs,
            attribution=score,
            memory_score=score,
            as_of_ordinal=record.as_of_ordinal,
            policy_identity_hash=record.policy_identity_hash,
            mode=mode,
            source_attribution_hash=canonical_sha256(record.to_dict()),
            n_plus_total=record.n_plus_total,
        ))
    if canonical_sha256([item.to_dict() for item in result]) == canonical_sha256(
        [item.to_dict() for item in records]
    ):
        raise ValueError("attribution ablation did not change the effective evidence")
    if ablation_recipe_hash(normalized) == ablation_recipe_hash(""):
        raise ValueError("attribution ablation recipe is not versioned")
    return tuple(result)


__all__ = [
    "ABLATION_ATTRIBUTION_SCHEMA_VERSION",
    "AblationAttributionRecord",
    "effective_attribution",
]
