"""Versioned manifest contracts for the SFT warm-start pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from my_agent.policy.identity import canonical_sha256, require_sha256
from my_agent.sft.contracts import DATASET_MANIFEST_SCHEMA_VERSION
from my_agent.sft.rendered import RenderedSFTManifest
from my_agent.sft.semantic import SemanticSFTSample


_FIELDS = {
    "schema_version",
    "canonical_dataset_hash",
    "split_sample_ids",
    "split_counts",
    "role_counts",
    "tool_counts",
    "source_counts",
    "quality_counts",
    "filter_reason_counts",
    "source_evidence_hashes",
    "group_split_hash",
    "dataset_manifest_hash",
}


@dataclass(frozen=True)
class SFTDatasetManifest:
    canonical_dataset_hash: str
    split_sample_ids: Mapping[str, tuple[str, ...]]
    split_counts: Mapping[str, int]
    role_counts: Mapping[str, Mapping[str, int]]
    tool_counts: Mapping[str, Mapping[str, int]]
    source_counts: Mapping[str, Mapping[str, int]]
    quality_counts: Mapping[str, int]
    filter_reason_counts: Mapping[str, int]
    source_evidence_hashes: Mapping[str, str]
    group_split_hash: str
    dataset_manifest_hash: str
    schema_version: str = DATASET_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported canonical SFT dataset manifest schema")
        for field_name in (
            "canonical_dataset_hash",
            "group_split_hash",
            "dataset_manifest_hash",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if set(self.split_sample_ids) != set(self.split_counts):
            raise ValueError("SFT manifest split IDs and counts must cover the same splits")
        if set(self.role_counts) != set(self.split_counts):
            raise ValueError("SFT manifest role counts must cover every split")
        if set(self.tool_counts) != set(self.split_counts):
            raise ValueError("SFT manifest tool counts must cover every split")
        if set(self.source_counts) != set(self.split_counts):
            raise ValueError("SFT manifest source counts must cover every split")
        for split, sample_ids in self.split_sample_ids.items():
            if not split or len(sample_ids) != len(set(sample_ids)):
                raise ValueError("SFT manifest split sample IDs are invalid")
            if self.split_counts[split] != len(sample_ids):
                raise ValueError("SFT manifest split count does not match sample IDs")
            for sample_id in sample_ids:
                require_sha256(sample_id, field_name="SFT sample ID")
        for mapping in (
            self.split_counts,
            self.quality_counts,
            self.filter_reason_counts,
        ):
            _validate_counts(mapping)
        for nested in (self.role_counts, self.tool_counts, self.source_counts):
            for counts in nested.values():
                _validate_counts(counts)
        for value in self.source_evidence_hashes.values():
            require_sha256(value, field_name="source evidence hash")
        if self.dataset_manifest_hash != canonical_sha256(self.payload_without_hash()):
            raise ValueError("canonical SFT dataset manifest hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        samples_by_split: Mapping[str, Sequence[SemanticSFTSample]],
        quality_counts: Mapping[str, int],
        filter_reason_counts: Mapping[str, int],
        source_evidence_hashes: Mapping[str, str],
        group_splits: Mapping[str, str],
    ) -> "SFTDatasetManifest":
        split_sample_ids = {
            split: tuple(sample.sample_id for sample in samples)
            for split, samples in sorted(samples_by_split.items())
        }
        split_counts = {
            split: len(samples) for split, samples in sorted(samples_by_split.items())
        }
        role_counts = {
            split: _counts(sample.role for sample in samples)
            for split, samples in sorted(samples_by_split.items())
        }
        tool_counts = {
            split: _counts(
                call.name
                for sample in samples
                for call in sample.target.tool_calls
            )
            for split, samples in sorted(samples_by_split.items())
        }
        source_counts = {
            split: _counts(str(sample.metadata["source"]) for sample in samples)
            for split, samples in sorted(samples_by_split.items())
        }
        ordered_samples = [
            sample.to_dict()
            for split in sorted(samples_by_split)
            for sample in samples_by_split[split]
        ]
        values = {
            "canonical_dataset_hash": canonical_sha256(ordered_samples),
            "split_sample_ids": split_sample_ids,
            "split_counts": split_counts,
            "role_counts": role_counts,
            "tool_counts": tool_counts,
            "source_counts": source_counts,
            "quality_counts": dict(quality_counts),
            "filter_reason_counts": dict(filter_reason_counts),
            "source_evidence_hashes": dict(source_evidence_hashes),
            "group_split_hash": canonical_sha256(dict(sorted(group_splits.items()))),
        }
        payload = _manifest_payload(**values)
        return cls(
            **values,
            dataset_manifest_hash=canonical_sha256(payload),
        )

    def payload_without_hash(self) -> dict[str, Any]:
        return _manifest_payload(
            canonical_dataset_hash=self.canonical_dataset_hash,
            split_sample_ids=self.split_sample_ids,
            split_counts=self.split_counts,
            role_counts=self.role_counts,
            tool_counts=self.tool_counts,
            source_counts=self.source_counts,
            quality_counts=self.quality_counts,
            filter_reason_counts=self.filter_reason_counts,
            source_evidence_hashes=self.source_evidence_hashes,
            group_split_hash=self.group_split_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_manifest_hash": self.dataset_manifest_hash, **self.payload_without_hash()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SFTDatasetManifest":
        if set(data) != _FIELDS:
            raise ValueError("canonical SFT dataset manifest fields do not match schema")
        split_ids = _mapping(data["split_sample_ids"], "split_sample_ids")
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            canonical_dataset_hash=_string(
                data["canonical_dataset_hash"], "canonical_dataset_hash"
            ),
            split_sample_ids={
                _string(split, "split"): _string_array(ids, "sample IDs")
                for split, ids in split_ids.items()
            },
            split_counts=_count_mapping(data["split_counts"], "split_counts"),
            role_counts=_nested_counts(data["role_counts"], "role_counts"),
            tool_counts=_nested_counts(data["tool_counts"], "tool_counts"),
            source_counts=_nested_counts(data["source_counts"], "source_counts"),
            quality_counts=_count_mapping(data["quality_counts"], "quality_counts"),
            filter_reason_counts=_count_mapping(
                data["filter_reason_counts"], "filter_reason_counts"
            ),
            source_evidence_hashes=_string_mapping(
                data["source_evidence_hashes"], "source_evidence_hashes"
            ),
            group_split_hash=_string(data["group_split_hash"], "group_split_hash"),
            dataset_manifest_hash=_string(
                data["dataset_manifest_hash"], "dataset_manifest_hash"
            ),
        )


def _manifest_payload(**values: Any) -> dict[str, Any]:
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "canonical_dataset_hash": values["canonical_dataset_hash"],
        "split_sample_ids": {
            split: list(ids) for split, ids in sorted(values["split_sample_ids"].items())
        },
        "split_counts": dict(sorted(values["split_counts"].items())),
        "role_counts": _sorted_nested(values["role_counts"]),
        "tool_counts": _sorted_nested(values["tool_counts"]),
        "source_counts": _sorted_nested(values["source_counts"]),
        "quality_counts": dict(sorted(values["quality_counts"].items())),
        "filter_reason_counts": dict(sorted(values["filter_reason_counts"].items())),
        "source_evidence_hashes": dict(sorted(values["source_evidence_hashes"].items())),
        "group_split_hash": values["group_split_hash"],
    }


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _validate_counts(counts: Mapping[str, int]) -> None:
    if any(
        not isinstance(key, str)
        or not key
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in counts.items()
    ):
        raise ValueError("SFT manifest counts must use non-empty keys and non-negative integers")


def _sorted_nested(value: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(counts.items()))
        for split, counts in sorted(value.items())
    }


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"SFT manifest {field_name} must be an object")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"SFT manifest {field_name} must be a non-empty string")
    return value


def _string_array(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"SFT manifest {field_name} must be a string array")
    return tuple(value)


def _count_mapping(value: Any, field_name: str) -> dict[str, int]:
    payload = _mapping(value, field_name)
    result = dict(payload)
    _validate_counts(result)
    return result


def _nested_counts(value: Any, field_name: str) -> dict[str, dict[str, int]]:
    payload = _mapping(value, field_name)
    return {
        _string(split, "split"): _count_mapping(counts, field_name)
        for split, counts in payload.items()
    }


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    payload = _mapping(value, field_name)
    return {
        _string(key, field_name): _string(item, field_name)
        for key, item in payload.items()
    }


__all__ = ["RenderedSFTManifest", "SFTDatasetManifest"]
