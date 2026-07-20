"""Register one legacy LLaMA-Factory SFT adapter as OPD checkpoint M0."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import json
import shutil

import yaml

from my_agent.config import AgentConfig
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    canonical_sha256,
    policy_identity_manifest_payload,
)
from my_agent.policy.transformers_policy import TransformersPolicy, hash_adapter_artifacts
from my_agent.training.opd_trainer import (
    OPDTrainerConfig,
    canonical_adapter_payload,
    validate_shared_adapter_config,
)


SFT_TRAINING_MANIFEST_FILENAME = "sft_training_manifest.json"
SFT_TRAINING_MANIFEST_SCHEMA_VERSION = "agentcli-legacy-sft-training-v1"
SFT_TRAINING_TEMPLATE = "qwen3_nothink"
SFT_LLAMAFACTORY_VERSION = "0.9.4"
_TRAINING_MANIFEST_FIELDS = {
    "schema_version",
    "base_model",
    "model_revision",
    "tokenizer_revision",
    "template",
    "llamafactory_version",
    "adapter_config",
    "adapter_config_hash",
}


@dataclass(frozen=True)
class SFTRegistrationResult:
    adapter_dir: Path
    identity_manifest_path: Path
    training_manifest_path: Path
    identity: PolicyIdentity


def register_sft_checkpoint(
    *,
    trainer_output: str | Path,
    output: str | Path,
    base_model: str,
    base_revision: str,
    tokenizer_revision: str,
    opd_config: str | Path,
    chat_template: str = "model_default",
    dtype: str = "bfloat16",
    device: str = "auto",
    policy_loader: Callable[[AgentConfig], Any] = TransformersPolicy.from_config,
) -> SFTRegistrationResult:
    source = Path(trainer_output).expanduser().resolve()
    adapter_config_path, adapter_model_path = _source_adapter_files(source)
    expected_adapter = _load_expected_adapter(opd_config)
    base_model_value = _required_text(base_model, "base model")
    base_revision_value = _required_text(base_revision, "base revision")
    tokenizer_revision_value = _required_text(
        tokenizer_revision, "tokenizer revision"
    )
    training_manifest_path = source / SFT_TRAINING_MANIFEST_FILENAME
    training_manifest = _load_training_manifest(
        training_manifest_path,
        expected_adapter=expected_adapter.canonical_payload,
        base_model=base_model_value,
        base_revision=base_revision_value,
        tokenizer_revision=tokenizer_revision_value,
    )
    adapter_payload = _load_mapping(adapter_config_path, "SFT adapter config")
    adapter_base_model = adapter_payload.get("base_model_name_or_path")
    if (
        not isinstance(adapter_base_model, str)
        or adapter_base_model.strip() != base_model_value
    ):
        raise ValueError("SFT adapter base_model_name_or_path does not match training manifest")
    actual_adapter = canonical_adapter_payload(adapter_payload)
    if actual_adapter != expected_adapter.canonical_payload:
        raise ValueError(
            "SFT adapter config does not match OPD shared adapter: "
            f"expected={expected_adapter.canonical_payload}, actual={actual_adapter}"
        )

    target_root = Path(output).expanduser().resolve()
    _require_empty_output(target_root)
    config = replace(
        AgentConfig.from_env(env={}, require_env_file=False),
        policy_backend="transformers",
        policy_base_model=base_model_value,
        policy_base_revision=base_revision_value,
        policy_tokenizer_revision=tokenizer_revision_value,
        policy_adapter_path=source,
        policy_identity_manifest=None,
        policy_chat_template=_required_text(chat_template, "chat template"),
        policy_dtype=_required_text(dtype, "policy dtype"),
        policy_device=_required_text(device, "policy device"),
    )
    policy = policy_loader(config)
    validate_shared_adapter_config(policy.model, expected_adapter)
    identity = policy.identity()
    if not isinstance(identity, PolicyIdentity):
        raise ValueError("SFT policy reload did not return PolicyIdentity")
    if identity.adapter_hash != hash_adapter_artifacts(source):
        raise ValueError("SFT policy identity adapter hash does not match trainer output")
    if (
        identity.base_model != config.policy_base_model
        or identity.base_revision != config.policy_base_revision
        or identity.tokenizer_revision != config.policy_tokenizer_revision
    ):
        raise ValueError("SFT policy identity does not match requested base identity")

    adapter_dir = target_root / "adapter"
    adapter_dir.mkdir(parents=True)
    shutil.copy2(adapter_config_path, adapter_dir / adapter_config_path.name)
    shutil.copy2(adapter_model_path, adapter_dir / adapter_model_path.name)
    if hash_adapter_artifacts(adapter_dir) != identity.adapter_hash:
        raise ValueError("registered M0 adapter hash changed during copy")

    registered_training_manifest = target_root / SFT_TRAINING_MANIFEST_FILENAME
    registered_training_manifest.write_bytes(
        canonical_json_bytes(training_manifest) + b"\n"
    )
    identity_path = target_root / "policy_identity_manifest.json"
    identity_path.write_bytes(
        canonical_json_bytes(policy_identity_manifest_payload(identity)) + b"\n"
    )
    return SFTRegistrationResult(
        adapter_dir=adapter_dir,
        identity_manifest_path=identity_path,
        training_manifest_path=registered_training_manifest,
        identity=identity,
    )


def _source_adapter_files(root: Path) -> tuple[Path, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"trainer output directory not found: {root}")
    config = root / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError(
            "trainer output root must contain the final adapter_config.json"
        )
    models = tuple(
        path
        for path in (
            root / "adapter_model.safetensors",
            root / "adapter_model.bin",
        )
        if path.is_file()
    )
    if len(models) != 1:
        raise ValueError("trainer output root must contain exactly one adapter model file")
    return config, models[0]


def _load_expected_adapter(path: str | Path):
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("OPD train config must be a YAML object")
    trainer = payload.get("trainer")
    if not isinstance(trainer, Mapping):
        raise ValueError("OPD train config requires trainer settings")
    return OPDTrainerConfig.from_mapping(trainer).shared_adapter


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_training_manifest(
    path: Path,
    *,
    expected_adapter: Mapping[str, Any],
    base_model: str,
    base_revision: str,
    tokenizer_revision: str,
) -> Mapping[str, Any]:
    manifest = _load_mapping(path, "SFT training manifest")
    if set(manifest) != _TRAINING_MANIFEST_FIELDS:
        raise ValueError("SFT training manifest fields do not match schema")
    if manifest["schema_version"] != SFT_TRAINING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported SFT training manifest schema")
    if manifest["base_model"] != base_model:
        raise ValueError("SFT training manifest base model does not match registration")
    if manifest["model_revision"] != base_revision:
        raise ValueError("SFT training manifest model revision does not match registration")
    if manifest["tokenizer_revision"] != tokenizer_revision:
        raise ValueError("SFT training manifest tokenizer revision does not match registration")
    if manifest["template"] != SFT_TRAINING_TEMPLATE:
        raise ValueError("SFT training manifest must use qwen3_nothink")
    if manifest["llamafactory_version"] != SFT_LLAMAFACTORY_VERSION:
        raise ValueError("SFT training manifest LLaMA-Factory version is unsupported")
    adapter = manifest["adapter_config"]
    if not isinstance(adapter, Mapping):
        raise ValueError("SFT training manifest adapter_config must be an object")
    normalized_adapter = canonical_adapter_payload(adapter)
    if normalized_adapter != dict(expected_adapter):
        raise ValueError("SFT training manifest adapter does not match OPD shared adapter")
    if manifest["adapter_config_hash"] != canonical_sha256(normalized_adapter):
        raise ValueError("SFT training manifest adapter config hash does not match")
    return manifest


def _require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"M0 output directory must be empty: {path}")


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


__all__ = [
    "SFTRegistrationResult",
    "SFT_TRAINING_MANIFEST_FILENAME",
    "register_sft_checkpoint",
]
