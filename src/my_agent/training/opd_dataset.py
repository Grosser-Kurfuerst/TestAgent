"""Strict current-checkpoint datasets and role-balanced sampling for OPD."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any
import json

from my_agent.opd_data.export import load_learner_samples
from my_agent.opd_data.schema import ExportManifest, LearnerSample
from my_agent.policy.contracts import EVOLVER_ROLES
from my_agent.policy.identity import PolicyIdentity, canonical_sha256


@dataclass(frozen=True)
class DatasetStatistics:
    sample_count: int
    role_counts: Mapping[str, int]
    task_group_counts: Mapping[str, int]
    token_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "role_counts": dict(sorted(self.role_counts.items())),
            "task_group_counts": dict(sorted(self.task_group_counts.items())),
            "token_counts": dict(sorted(self.token_counts.items())),
        }


class OPDLearnerDataset(Sequence[LearnerSample]):
    """Validated samples from exactly one collection checkpoint and split."""

    def __init__(
        self,
        samples: Sequence[LearnerSample],
        *,
        initialization_identity: PolicyIdentity,
        collection_round: int,
        split: str = "train",
        require_all_roles: bool = True,
        learner_dataset_hash: str | None = None,
        export_manifest_hash: str | None = None,
    ) -> None:
        selected = tuple(sample for sample in samples if sample.split == split)
        if not selected:
            raise ValueError(f"OPD dataset split {split!r} is empty")
        for sample in selected:
            if sample.collection_round != collection_round:
                raise ValueError("OPD dataset crosses collection rounds")
            if sample.policy_identity != initialization_identity:
                raise ValueError("learner sample identity does not match trainer initialization")
            if not sample.student_completion_token_ids or not any(sample.assistant_loss_mask):
                raise ValueError("OPD dataset sample has no trainable assistant tokens")
        roles = frozenset(sample.role for sample in selected)
        if require_all_roles and roles != EVOLVER_ROLES:
            missing = sorted(EVOLVER_ROLES - roles)
            raise ValueError(f"formal OPD dataset is missing roles: {missing}")
        self._samples = selected
        self.initialization_identity = initialization_identity
        self.collection_round = collection_round
        self.split = split
        self.learner_dataset_hash = (
            learner_dataset_hash
            if learner_dataset_hash is not None
            else canonical_sha256([sample.to_dict() for sample in samples])
        )
        self.export_manifest_hash = export_manifest_hash

    @classmethod
    def from_files(
        cls,
        learner_path: str | Path,
        export_manifest_path: str | Path,
        *,
        split: str = "train",
        require_all_roles: bool = True,
    ) -> "OPDLearnerDataset":
        manifest_path = Path(export_manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("export manifest must be a JSON object")
        manifest = ExportManifest.from_dict(payload)
        samples = load_learner_samples(learner_path)
        if len(samples) != manifest.sample_count:
            raise ValueError("learner dataset count does not match export manifest")
        dataset_hash = canonical_sha256([sample.to_dict() for sample in samples])
        if dataset_hash != manifest.learner_dataset_hash:
            raise ValueError("learner dataset hash does not match export manifest")
        return cls(
            samples,
            initialization_identity=manifest.trainer_initialization_identity,
            collection_round=manifest.collection_round,
            split=split,
            require_all_roles=require_all_roles,
            learner_dataset_hash=manifest.learner_dataset_hash,
            export_manifest_hash=canonical_sha256(payload),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> LearnerSample:
        return self._samples[index]

    @property
    def statistics(self) -> DatasetStatistics:
        return DatasetStatistics(
            sample_count=len(self),
            role_counts=dict(Counter(sample.role for sample in self)),
            task_group_counts=dict(Counter(sample.task_group for sample in self)),
            token_counts=dict(Counter({
                role: sum(
                    sum(sample.assistant_loss_mask)
                    for sample in self
                    if sample.role == role
                )
                for role in EVOLVER_ROLES
            })),
        )


class RoleSampler:
    """Deterministically allocate samples by role weight, then sample within roles."""

    def __init__(
        self,
        dataset: Sequence[LearnerSample],
        *,
        role_weights: Mapping[str, float],
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if not dataset:
            raise ValueError("role sampler requires a non-empty dataset")
        unknown = sorted(set(role_weights) - EVOLVER_ROLES)
        if unknown:
            raise ValueError(f"unknown OPD role weights: {unknown}")
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(dataset):
            buckets[sample.role].append(index)
        missing_weights = sorted(set(buckets) - set(role_weights))
        if missing_weights:
            raise ValueError(f"missing role sampling weights: {missing_weights}")
        active_weights = {
            role: float(role_weights[role])
            for role in sorted(buckets)
        }
        if any(weight <= 0.0 for weight in active_weights.values()):
            raise ValueError("role sampling weights must be positive")
        self._buckets = {role: tuple(values) for role, values in buckets.items()}
        self.role_weights = active_weights
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        if self.num_samples < len(self._buckets):
            raise ValueError("num_samples must be large enough to sample every available role")
        self.seed = int(seed)
        self.epoch = 0
        self._last_counts: Mapping[str, int] = {}

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = epoch

    @property
    def sampled_role_counts(self) -> Mapping[str, int]:
        return dict(self._last_counts)

    def __iter__(self) -> Iterator[int]:
        random = Random(self.seed + self.epoch)
        quotas = _weighted_quotas(self.role_weights, self.num_samples)
        schedule = [role for role, count in quotas.items() for _ in range(count)]
        random.shuffle(schedule)
        pools: dict[str, list[int]] = {}
        offsets: Counter[str] = Counter()
        for role, indexes in self._buckets.items():
            pool = list(indexes)
            random.shuffle(pool)
            pools[role] = pool
        sampled: list[int] = []
        for role in schedule:
            pool = pools[role]
            offset = offsets[role]
            if offset and offset % len(pool) == 0:
                random.shuffle(pool)
            sampled.append(pool[offset % len(pool)])
            offsets[role] += 1
        self._last_counts = dict(Counter(schedule))
        return iter(sampled)


def _weighted_quotas(weights: Mapping[str, float], total: int) -> dict[str, int]:
    roles = tuple(sorted(weights))
    remaining = total - len(roles)
    weight_sum = sum(weights.values())
    exact = {role: remaining * weights[role] / weight_sum for role in roles}
    quotas = {role: 1 + int(exact[role]) for role in roles}
    leftover = total - sum(quotas.values())
    order = sorted(
        roles,
        key=lambda role: (exact[role] - int(exact[role]), role),
        reverse=True,
    )
    for role in order[:leftover]:
        quotas[role] += 1
    return quotas


__all__ = [
    "DatasetStatistics",
    "OPDLearnerDataset",
    "RoleSampler",
]
