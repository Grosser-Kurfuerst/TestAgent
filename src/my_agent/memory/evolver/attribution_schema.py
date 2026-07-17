"""Versioned schemas for paper-faithful attribution evidence and results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, sqrt
from typing import Any, Mapping

from my_agent.memory.evolver.types import ExperienceTier
from my_agent.policy.identity import PolicyIdentity, canonical_sha256, require_sha256


PAPER_ATTRIBUTION_SCHEMA_VERSION = "opd-paper-attribution-v1"


@dataclass(frozen=True)
class CandidateExposure:
    task_id: str
    task_group: str
    stream_id: str
    memory_project_key: str
    memory_id: str
    tier: str
    selected: bool
    reward: float
    collection_round: int
    task_ordinal: int
    candidate_snapshot_hash: str
    policy_identity: PolicyIdentity
    repository_revision: str
    evaluator_name: str
    evaluator_version: str
    evaluator_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "task_id", "task_group", "stream_id", "memory_project_key", "memory_id",
            "repository_revision", "evaluator_name", "evaluator_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"candidate exposure {field_name} must not be empty")
        ExperienceTier(self.tier)
        if not isinstance(self.selected, bool):
            raise ValueError("candidate exposure selected must be bool")
        if not isfinite(self.reward):
            raise ValueError("candidate exposure reward must be finite")
        if (
            isinstance(self.collection_round, bool)
            or not isinstance(self.collection_round, int)
            or self.collection_round < 0
        ):
            raise ValueError("candidate exposure collection_round must be non-negative")
        if isinstance(self.task_ordinal, bool) or not isinstance(self.task_ordinal, int) or self.task_ordinal < 1:
            raise ValueError("candidate exposure task_ordinal must be a positive integer")
        require_sha256(self.candidate_snapshot_hash, field_name="candidate_snapshot_hash")
        if not isinstance(self.policy_identity, PolicyIdentity):
            raise ValueError("candidate exposure requires PolicyIdentity")
        require_sha256(self.evaluator_hash, field_name="evaluator_hash")

    @property
    def exposure_id(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_group": self.task_group,
            "stream_id": self.stream_id,
            "memory_project_key": self.memory_project_key,
            "memory_id": self.memory_id,
            "tier": self.tier,
            "selected": self.selected,
            "reward": self.reward,
            "collection_round": self.collection_round,
            "task_ordinal": self.task_ordinal,
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "policy_identity": self.policy_identity.to_dict(),
            "policy_identity_hash": self.policy_identity.identity_hash,
            "repository_revision": self.repository_revision,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "evaluator_hash": self.evaluator_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"exposure_id": self.exposure_id, **self._payload()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateExposure":
        expected = {
            "task_id", "task_group", "stream_id", "memory_project_key", "memory_id", "tier",
            "selected", "reward", "collection_round", "task_ordinal", "candidate_snapshot_hash", "policy_identity",
            "policy_identity_hash", "repository_revision", "evaluator_name", "evaluator_version",
            "evaluator_hash", "exposure_id",
        }
        if set(data) != expected:
            raise ValueError("candidate exposure fields do not match the versioned schema")
        raw_identity = data["policy_identity"]
        if not isinstance(raw_identity, Mapping):
            raise ValueError("candidate exposure policy_identity must be an object")
        identity = PolicyIdentity.from_dict(raw_identity)
        if data["policy_identity_hash"] != identity.identity_hash:
            raise ValueError("candidate exposure policy identity hash mismatch")
        exposure = cls(
            task_id=_string(data["task_id"], "task_id"),
            task_group=_string(data["task_group"], "task_group"),
            stream_id=_string(data["stream_id"], "stream_id"),
            memory_project_key=_string(data["memory_project_key"], "memory_project_key"),
            memory_id=_string(data["memory_id"], "memory_id"),
            tier=_string(data["tier"], "tier"),
            selected=_bool(data["selected"], "selected"),
            reward=_number(data["reward"], "reward"),
            collection_round=_int(data["collection_round"], "collection_round"),
            task_ordinal=_int(data["task_ordinal"], "task_ordinal"),
            candidate_snapshot_hash=_string(data["candidate_snapshot_hash"], "candidate_snapshot_hash"),
            policy_identity=identity,
            repository_revision=_string(data["repository_revision"], "repository_revision"),
            evaluator_name=_string(data["evaluator_name"], "evaluator_name"),
            evaluator_version=_string(data["evaluator_version"], "evaluator_version"),
            evaluator_hash=_string(data["evaluator_hash"], "evaluator_hash"),
        )
        if data["exposure_id"] != exposure.exposure_id:
            raise ValueError("candidate exposure id does not match its canonical payload")
        return exposure


@dataclass(frozen=True)
class GroupAttribution:
    task_group: str
    n_plus: int
    n_minus: int
    mean_plus: float | None
    mean_minus: float | None
    rho_g: float | None
    delta: float | None
    status: str

    def __post_init__(self) -> None:
        if not self.task_group:
            raise ValueError("group attribution task_group must not be empty")
        if self.n_plus < 0 or self.n_minus < 0:
            raise ValueError("group attribution counts must be non-negative")
        if self.mean_plus is None and self.n_plus > 0:
            raise ValueError("group attribution with selected evidence requires mean_plus")
        if self.mean_minus is None and self.n_minus > 0:
            raise ValueError("group attribution with not-selected evidence requires mean_minus")
        if self.mean_plus is not None and self.n_plus == 0:
            raise ValueError("group attribution without selected evidence must preserve missing mean_plus")
        if self.mean_minus is not None and self.n_minus == 0:
            raise ValueError("group attribution without not-selected evidence must preserve missing mean_minus")
        for value in (self.mean_plus, self.mean_minus, self.rho_g, self.delta):
            if value is not None and not isfinite(value):
                raise ValueError("group attribution numeric values must be finite")
        if self.status == "ready":
            if self.n_plus < 1 or self.n_minus < 1:
                raise ValueError("ready group attribution requires both counterfactual sides")
            if None in (self.mean_plus, self.mean_minus, self.rho_g, self.delta):
                raise ValueError("ready group attribution requires complete formula values")
            expected_rho = self.n_plus / (self.n_plus + self.n_minus)
            expected_delta = float(self.mean_plus) - float(self.mean_minus)
            if not isclose(float(self.rho_g), expected_rho, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("group attribution rho_g does not match its counts")
            if not isclose(float(self.delta), expected_delta, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("group attribution delta does not match its means")
        elif self.status == "insufficient_counterfactual_evidence":
            if self.n_plus > 0 and self.n_minus > 0:
                raise ValueError("insufficient group cannot contain both counterfactual sides")
            if self.rho_g is not None or self.delta is not None:
                raise ValueError("insufficient group must preserve missing rho_g/delta")
        else:
            raise ValueError("unsupported group attribution status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_group": self.task_group,
            "n_plus": self.n_plus,
            "n_minus": self.n_minus,
            "mean_plus": self.mean_plus,
            "mean_minus": self.mean_minus,
            "rho_g": self.rho_g,
            "delta": self.delta,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, task_group: str, data: Mapping[str, Any]) -> "GroupAttribution":
        expected = {"n_plus", "n_minus", "mean_plus", "mean_minus", "rho_g", "delta", "status"}
        if set(data) != expected:
            raise ValueError("group attribution fields do not match the versioned schema")
        return cls(
            task_group=task_group,
            n_plus=_int(data["n_plus"], "n_plus"),
            n_minus=_int(data["n_minus"], "n_minus"),
            mean_plus=_optional_number(data["mean_plus"], "mean_plus"),
            mean_minus=_optional_number(data["mean_minus"], "mean_minus"),
            rho_g=_optional_number(data["rho_g"], "rho_g"),
            delta=_optional_number(data["delta"], "delta"),
            status=_string(data["status"], "status"),
        )


@dataclass(frozen=True)
class AttributionEvidenceRef:
    exposure: CandidateExposure

    def __post_init__(self) -> None:
        if not isinstance(self.exposure, CandidateExposure):
            raise ValueError("attribution evidence ref requires CandidateExposure")

    @property
    def exposure_id(self) -> str:
        return self.exposure.exposure_id

    @property
    def task_id(self) -> str:
        return self.exposure.task_id

    @property
    def task_group(self) -> str:
        return self.exposure.task_group

    @property
    def selected(self) -> bool:
        return self.exposure.selected

    @property
    def reward(self) -> float:
        return self.exposure.reward

    @property
    def collection_round(self) -> int:
        return self.exposure.collection_round

    @property
    def task_ordinal(self) -> int:
        return self.exposure.task_ordinal

    @property
    def candidate_snapshot_hash(self) -> str:
        return self.exposure.candidate_snapshot_hash

    @property
    def repository_revision(self) -> str:
        return self.exposure.repository_revision

    @property
    def evaluator_hash(self) -> str:
        return self.exposure.evaluator_hash

    def to_dict(self) -> dict[str, Any]:
        return self.exposure.to_dict()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttributionEvidenceRef":
        return cls(CandidateExposure.from_dict(data))


@dataclass(frozen=True)
class PaperAttributionRecord:
    memory_id: str
    tier: str
    memory_project_key: str
    collection_round: int
    groups: tuple[GroupAttribution, ...]
    evidence_refs: tuple[AttributionEvidenceRef, ...]
    attribution: float | None
    n_plus_total: int
    gamma: float
    tier_prior: float
    memory_score: float | None
    as_of_ordinal: int
    policy_identity_hash: str
    status: str
    schema_version: str = PAPER_ATTRIBUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PAPER_ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError("unsupported paper attribution schema")
        ExperienceTier(self.tier)
        if self.collection_round < 0 or self.n_plus_total < 0 or self.as_of_ordinal < 1:
            raise ValueError("paper attribution counts/ordinal are invalid")
        if not self.groups or not self.evidence_refs:
            raise ValueError("paper attribution requires groups and evidence refs")
        if len({ref.exposure_id for ref in self.evidence_refs}) != len(self.evidence_refs):
            raise ValueError("paper attribution evidence refs must be unique")
        if any(ref.collection_round != self.collection_round for ref in self.evidence_refs):
            raise ValueError("paper attribution evidence refs cross collection rounds")
        if any(ref.task_ordinal > self.as_of_ordinal for ref in self.evidence_refs):
            raise ValueError("paper attribution evidence exceeds as_of_ordinal")
        for ref in self.evidence_refs:
            exposure = ref.exposure
            if (
                exposure.memory_id != self.memory_id
                or exposure.tier != self.tier
                or exposure.memory_project_key != self.memory_project_key
            ):
                raise ValueError("paper attribution evidence identity mismatch")
            if exposure.policy_identity.identity_hash != self.policy_identity_hash:
                raise ValueError("paper attribution evidence policy identity mismatch")
        refs_by_group: dict[str, list[AttributionEvidenceRef]] = {}
        for ref in self.evidence_refs:
            refs_by_group.setdefault(ref.task_group, []).append(ref)
        if set(refs_by_group) != {group.task_group for group in self.groups}:
            raise ValueError("paper attribution groups do not match evidence refs")
        for group in self.groups:
            refs = refs_by_group[group.task_group]
            selected_rewards = [ref.reward for ref in refs if ref.selected]
            not_selected_rewards = [ref.reward for ref in refs if not ref.selected]
            if group.n_plus != len(selected_rewards) or group.n_minus != len(not_selected_rewards):
                raise ValueError("group attribution counts do not match evidence refs")
            if selected_rewards and not isclose(
                float(group.mean_plus), sum(selected_rewards) / len(selected_rewards),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("group mean_plus does not match evidence refs")
            if not_selected_rewards and not isclose(
                float(group.mean_minus), sum(not_selected_rewards) / len(not_selected_rewards),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("group mean_minus does not match evidence refs")
        for field_name in ("gamma", "tier_prior"):
            if not isfinite(getattr(self, field_name)):
                raise ValueError(f"paper attribution {field_name} must be finite")
        for field_name in ("attribution", "memory_score"):
            value = getattr(self, field_name)
            if value is not None and not isfinite(value):
                raise ValueError(f"paper attribution {field_name} must be finite when present")
        require_sha256(self.policy_identity_hash, field_name="policy_identity_hash")
        expected_n_plus = sum(group.n_plus for group in self.groups)
        if self.n_plus_total != expected_n_plus:
            raise ValueError("paper attribution n_plus_total does not match groups")
        expected_gamma = 1.0 - 1.0 / sqrt(1.0 + self.n_plus_total)
        if not isclose(self.gamma, expected_gamma, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("paper attribution gamma does not match n_plus_total")
        ready_groups = [group for group in self.groups if group.status == "ready"]
        if self.status == "ready" and (self.attribution is None or self.memory_score is None):
            raise ValueError("ready paper attribution requires attribution and memory_score")
        if self.status != "ready" and (self.attribution is not None or self.memory_score is not None):
            raise ValueError("immature paper attribution must preserve missing values")
        if self.status == "ready":
            if not ready_groups:
                raise ValueError("ready paper attribution requires a ready group")
            expected_attribution = sum(float(group.rho_g) * float(group.delta) for group in ready_groups)
            expected_score = self.tier_prior * self.gamma * expected_attribution
            if not isclose(float(self.attribution), expected_attribution, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("paper attribution does not match group contributions")
            if not isclose(float(self.memory_score), expected_score, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("paper memory_score does not match Eq. 12")
        elif self.status == "insufficient_counterfactual_evidence":
            if ready_groups:
                raise ValueError("immature paper attribution cannot contain ready groups")
        else:
            raise ValueError("unsupported paper attribution status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "tier": self.tier,
            "memory_project_key": self.memory_project_key,
            "collection_round": self.collection_round,
            "groups": {group.task_group: {
                key: value
                for key, value in group.to_dict().items()
                if key != "task_group"
            } for group in self.groups},
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "attribution": self.attribution,
            "n_plus_total": self.n_plus_total,
            "gamma": self.gamma,
            "tier_prior": self.tier_prior,
            "memory_score": self.memory_score,
            "as_of_ordinal": self.as_of_ordinal,
            "policy_identity_hash": self.policy_identity_hash,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaperAttributionRecord":
        expected = {
            "schema_version", "memory_id", "tier", "memory_project_key", "collection_round", "groups",
            "evidence_refs",
            "attribution", "n_plus_total", "gamma", "tier_prior", "memory_score",
            "as_of_ordinal", "policy_identity_hash", "status",
        }
        if set(data) != expected:
            raise ValueError("paper attribution fields do not match the versioned schema")
        raw_groups = data["groups"]
        if not isinstance(raw_groups, Mapping):
            raise ValueError("paper attribution groups must be an object")
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            memory_id=_string(data["memory_id"], "memory_id"),
            tier=_string(data["tier"], "tier"),
            memory_project_key=_string(data["memory_project_key"], "memory_project_key"),
            collection_round=_int(data["collection_round"], "collection_round"),
            groups=tuple(
                GroupAttribution.from_dict(
                    _string(task_group, "task_group"),
                    group_data if isinstance(group_data, Mapping) else {},
                )
                for task_group, group_data in sorted(raw_groups.items())
            ),
            evidence_refs=_evidence_refs(data["evidence_refs"]),
            attribution=_optional_number(data["attribution"], "attribution"),
            n_plus_total=_int(data["n_plus_total"], "n_plus_total"),
            gamma=_number(data["gamma"], "gamma"),
            tier_prior=_number(data["tier_prior"], "tier_prior"),
            memory_score=_optional_number(data["memory_score"], "memory_score"),
            as_of_ordinal=_int(data["as_of_ordinal"], "as_of_ordinal"),
            policy_identity_hash=_string(data["policy_identity_hash"], "policy_identity_hash"),
            status=_string(data["status"], "status"),
        )


@dataclass(frozen=True)
class WritingScoreDecision:
    memory_id: str
    memory_score: float | None
    rank: int | None
    selected: bool
    reason: str
    collection_round: int
    top_fraction: float
    cutoff_rank: int | None
    boundary_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_score": self.memory_score,
            "rank": self.rank,
            "selected": self.selected,
            "reason": self.reason,
            "collection_round": self.collection_round,
            "top_fraction": self.top_fraction,
            "cutoff_rank": self.cutoff_rank,
            "boundary_score": self.boundary_score,
        }


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool")
    return value


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _optional_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _number(value, field_name)


def _int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be int")
    return value


def _evidence_refs(value: Any) -> tuple[AttributionEvidenceRef, ...]:
    if not isinstance(value, list):
        raise ValueError("evidence_refs must be an array")
    return tuple(
        AttributionEvidenceRef.from_dict(item if isinstance(item, Mapping) else {})
        for item in value
    )


__all__ = [
    "PAPER_ATTRIBUTION_SCHEMA_VERSION",
    "CandidateExposure",
    "AttributionEvidenceRef",
    "GroupAttribution",
    "PaperAttributionRecord",
    "WritingScoreDecision",
]
