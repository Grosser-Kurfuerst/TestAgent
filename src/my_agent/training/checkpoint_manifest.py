"""Versioned artifacts for one shared-adapter OPD checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping
import json

from my_agent.opd_data.attribution.schema import PAPER_ATTRIBUTION_SCHEMA_VERSION
from my_agent.opd_ablation import MAIN_ABLATION_RECIPE_HASH, ablation_recipe_hash
from my_agent.opd_data.schema import OPD_LEARNER_SCHEMA_VERSION
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    policy_identity_manifest_payload,
    require_sha256,
)
from my_agent.policy.transformers_policy import hash_adapter_artifacts
from my_agent.training.role_views import ROLE_VIEW_SCHEMA_VERSION


OPD_CHECKPOINT_MANIFEST_SCHEMA_VERSION = "opd-checkpoint-manifest-v2"
OPD_PROMPT_VERSION = "opd-role-prompts-v1"
CHECKPOINT_MANIFEST_FILENAME = "opd_checkpoint_manifest.json"
POLICY_IDENTITY_MANIFEST_FILENAME = "policy_identity_manifest.json"


@dataclass(frozen=True)
class CheckpointManifest:
    collection_round: int
    initialization_identity: PolicyIdentity
    output_identity: PolicyIdentity
    learner_dataset_hash: str
    export_manifest_hash: str
    role_sampling_weights: Mapping[str, float]
    raw_role_counts: Mapping[str, int]
    valid_role_counts: Mapping[str, int]
    sampled_role_counts: Mapping[str, int]
    task_group_counts: Mapping[str, int]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    state_artifacts: Mapping[str, str]
    train_role_kl: Mapping[str, float]
    train_role_tokens: Mapping[str, int]
    validation_role_kl: Mapping[str, float]
    validation_role_tokens: Mapping[str, int]
    gradient_norm: Mapping[str, float]
    mixed_step_gradient_norm_by_role: Mapping[str, float]
    shared_adapter_name: str
    reload_identity_verified: bool
    ablation: str = ""
    ablation_recipe_hash: str = MAIN_ABLATION_RECIPE_HASH
    dataset_source_hashes: Mapping[str, str] = field(default_factory=dict)
    learner_schema_version: str = OPD_LEARNER_SCHEMA_VERSION
    role_view_schema_version: str = ROLE_VIEW_SCHEMA_VERSION
    attribution_schema_version: str = PAPER_ATTRIBUTION_SCHEMA_VERSION
    prompt_version: str = OPD_PROMPT_VERSION
    schema_version: str = OPD_CHECKPOINT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPD_CHECKPOINT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported OPD checkpoint manifest schema")
        if self.collection_round < 0:
            raise ValueError("checkpoint collection_round must be non-negative")
        for field_name in ("learner_dataset_hash", "export_manifest_hash"):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if not self.shared_adapter_name.strip():
            raise ValueError("checkpoint requires one shared adapter name")
        if self.output_identity.adapter_hash is None:
            raise ValueError("OPD checkpoint output identity requires a shared adapter hash")
        if self.ablation.strip().lower() != self.ablation:
            raise ValueError("checkpoint ablation must be normalized")
        if self.ablation_recipe_hash != ablation_recipe_hash(self.ablation):
            raise ValueError("checkpoint ablation recipe hash mismatch")
        if self.ablation and not self.dataset_source_hashes:
            raise ValueError("ablation checkpoint requires dataset source hashes")
        for source_hash in self.dataset_source_hashes.values():
            require_sha256(source_hash, field_name="checkpoint dataset source hash")
        if set(self.role_sampling_weights) != set(self.raw_role_counts):
            raise ValueError("role sampling weights and raw counts must cover the same roles")
        if any(value <= 0.0 for value in self.role_sampling_weights.values()):
            raise ValueError("role sampling weights must be positive")
        for artifact_hash in self.state_artifacts.values():
            require_sha256(artifact_hash, field_name="state artifact hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collection_round": self.collection_round,
            "initialization_identity": self.initialization_identity.to_dict(),
            "initialization_identity_hash": self.initialization_identity.identity_hash,
            "output_identity": self.output_identity.to_dict(),
            "output_identity_hash": self.output_identity.identity_hash,
            "learner_dataset_hash": self.learner_dataset_hash,
            "export_manifest_hash": self.export_manifest_hash,
            "versions": {
                "learner_schema": self.learner_schema_version,
                "role_view_schema": self.role_view_schema_version,
                "attribution_schema": self.attribution_schema_version,
                "prompt": self.prompt_version,
            },
            "role_sampling_weights": dict(sorted(self.role_sampling_weights.items())),
            "raw_role_counts": dict(sorted(self.raw_role_counts.items())),
            "valid_role_counts": dict(sorted(self.valid_role_counts.items())),
            "sampled_role_counts": dict(sorted(self.sampled_role_counts.items())),
            "task_group_counts": dict(sorted(self.task_group_counts.items())),
            "optimizer": dict(self.optimizer),
            "scheduler": dict(self.scheduler),
            "state_artifacts": dict(sorted(self.state_artifacts.items())),
            "train_role_kl": dict(sorted(self.train_role_kl.items())),
            "train_role_tokens": dict(sorted(self.train_role_tokens.items())),
            "validation_role_kl": dict(sorted(self.validation_role_kl.items())),
            "validation_role_tokens": dict(sorted(self.validation_role_tokens.items())),
            "gradient_norm": dict(self.gradient_norm),
            "mixed_step_gradient_norm_by_role": dict(sorted(
                self.mixed_step_gradient_norm_by_role.items()
            )),
            "shared_adapter_name": self.shared_adapter_name,
            "reload_identity_verified": self.reload_identity_verified,
            "ablation": self.ablation,
            "ablation_recipe_hash": self.ablation_recipe_hash,
            "dataset_source_hashes": dict(sorted(self.dataset_source_hashes.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointManifest":
        versions = _mapping(data.get("versions"), "versions")
        initialization = PolicyIdentity.from_dict(_mapping(
            data.get("initialization_identity"), "initialization_identity"
        ))
        output = PolicyIdentity.from_dict(_mapping(data.get("output_identity"), "output_identity"))
        if data.get("initialization_identity_hash") != initialization.identity_hash:
            raise ValueError("checkpoint initialization identity hash mismatch")
        if data.get("output_identity_hash") != output.identity_hash:
            raise ValueError("checkpoint output identity hash mismatch")
        return cls(
            schema_version=str(data.get("schema_version", "")),
            collection_round=int(data.get("collection_round", -1)),
            initialization_identity=initialization,
            output_identity=output,
            learner_dataset_hash=str(data.get("learner_dataset_hash", "")),
            export_manifest_hash=str(data.get("export_manifest_hash", "")),
            learner_schema_version=str(versions.get("learner_schema", "")),
            role_view_schema_version=str(versions.get("role_view_schema", "")),
            attribution_schema_version=str(versions.get("attribution_schema", "")),
            prompt_version=str(versions.get("prompt", "")),
            role_sampling_weights=_float_mapping(
                data.get("role_sampling_weights"), "role_sampling_weights"
            ),
            raw_role_counts=_int_mapping(data.get("raw_role_counts"), "raw_role_counts"),
            valid_role_counts=_int_mapping(data.get("valid_role_counts"), "valid_role_counts"),
            sampled_role_counts=_int_mapping(
                data.get("sampled_role_counts"), "sampled_role_counts"
            ),
            task_group_counts=_int_mapping(
                data.get("task_group_counts"), "task_group_counts"
            ),
            optimizer=dict(_mapping(data.get("optimizer"), "optimizer")),
            scheduler=dict(_mapping(data.get("scheduler"), "scheduler")),
            state_artifacts=_string_mapping(
                data.get("state_artifacts"), "state_artifacts"
            ),
            train_role_kl=_float_mapping(data.get("train_role_kl"), "train_role_kl"),
            train_role_tokens=_int_mapping(
                data.get("train_role_tokens"), "train_role_tokens"
            ),
            validation_role_kl=_float_mapping(
                data.get("validation_role_kl"), "validation_role_kl"
            ),
            validation_role_tokens=_int_mapping(
                data.get("validation_role_tokens"), "validation_role_tokens"
            ),
            gradient_norm=_float_mapping(data.get("gradient_norm"), "gradient_norm"),
            mixed_step_gradient_norm_by_role=_float_mapping(
                data.get("mixed_step_gradient_norm_by_role"),
                "mixed_step_gradient_norm_by_role",
            ),
            shared_adapter_name=str(data.get("shared_adapter_name", "")),
            reload_identity_verified=bool(data.get("reload_identity_verified", False)),
            ablation=str(data.get("ablation", "")),
            ablation_recipe_hash=str(data.get("ablation_recipe_hash", "")),
            dataset_source_hashes=_string_mapping(
                data.get("dataset_source_hashes"), "dataset_source_hashes"
            ),
        )


def output_identity_for_adapter(
    initialization_identity: PolicyIdentity,
    checkpoint_dir: str | Path,
) -> PolicyIdentity:
    return replace(
        initialization_identity,
        adapter_hash=hash_adapter_artifacts(checkpoint_dir),
    )


def write_checkpoint_manifests(
    checkpoint_dir: str | Path,
    manifest: CheckpointManifest,
) -> tuple[Path, Path]:
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / CHECKPOINT_MANIFEST_FILENAME
    identity_path = root / POLICY_IDENTITY_MANIFEST_FILENAME
    checkpoint_path.write_bytes(canonical_json_bytes(manifest.to_dict()) + b"\n")
    identity_path.write_bytes(
        canonical_json_bytes(policy_identity_manifest_payload(manifest.output_identity)) + b"\n"
    )
    return checkpoint_path, identity_path


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint manifest must be a JSON object")
    if payload.get("schema_version") != OPD_CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported OPD checkpoint manifest schema")
    return CheckpointManifest.from_dict(payload)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {field_name} must be an object")
    return value


def _int_mapping(value: Any, field_name: str) -> dict[str, int]:
    return {str(key): int(item) for key, item in _mapping(value, field_name).items()}


def _float_mapping(value: Any, field_name: str) -> dict[str, float]:
    return {str(key): float(item) for key, item in _mapping(value, field_name).items()}


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    return {str(key): str(item) for key, item in _mapping(value, field_name).items()}


__all__ = [
    "CHECKPOINT_MANIFEST_FILENAME",
    "OPD_CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "OPD_PROMPT_VERSION",
    "POLICY_IDENTITY_MANIFEST_FILENAME",
    "CheckpointManifest",
    "load_checkpoint_manifest",
    "output_identity_for_adapter",
    "write_checkpoint_manifests",
]
