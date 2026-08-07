#!/usr/bin/env python3
"""Train one strict current-round OPD checkpoint with a shared LoRA adapter."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
import json

import yaml

from my_agent.config import AgentConfig
from my_agent.opd_ablation import PAPER_ABLATIONS
from my_agent.policy.contracts import DecisionRequest
from my_agent.policy.identity import (
    PolicyIdentity,
    load_policy_identity_manifest,
    require_matching_policy_identity,
)
from my_agent.policy.transformers_policy import TransformersPolicy
from my_agent.training.opd_dataset import OPDLearnerDataset
from my_agent.training.opd_trainer import (
    OPDTrainer,
    OPDTrainerConfig,
    attach_or_validate_shared_adapter,
    build_training_accelerator,
)


RUN_CONFIG_SCHEMA_VERSION = "opd-train-run-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--identity-manifest")
    parser.add_argument("--learner-dataset")
    parser.add_argument("--export-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--ablation", choices=PAPER_ABLATIONS)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("OPD train config must be a YAML object")
    if payload.get("schema_version") != RUN_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported OPD train run config schema")
    base = config_path.parent
    learner_path = _resolve_override(base, args.learner_dataset, payload, "learner_dataset")
    export_manifest_path = _resolve_override(
        base, args.export_manifest, payload, "export_manifest"
    )
    identity_manifest_path = _resolve_override(
        base, args.identity_manifest, payload, "identity_manifest"
    )
    output_dir = _resolve_override(base, args.output_dir, payload, "output_dir")
    checkpoint_value = args.checkpoint if args.checkpoint is not None else payload.get("checkpoint")
    checkpoint = (
        _resolve_path_value(base, checkpoint_value, "checkpoint")
        if checkpoint_value is not None
        else None
    )
    expected_identity = load_policy_identity_manifest(identity_manifest_path)
    if expected_identity.adapter_hash is not None and checkpoint is None:
        raise ValueError("adapter-backed trainer initialization requires checkpoint path")
    policy_data = payload.get("policy", {})
    if not isinstance(policy_data, Mapping):
        raise ValueError("policy config must be an object")
    trainer_data = payload.get("trainer", {})
    if not isinstance(trainer_data, Mapping):
        raise ValueError("trainer config must be an object")
    trainer_config = OPDTrainerConfig.from_mapping(trainer_data)
    requested_device = str(policy_data.get("device", "auto")).strip().lower()
    accelerator = build_training_accelerator(
        trainer_config,
        cpu=requested_device == "cpu",
    )
    if accelerator.num_processes > 1 and requested_device == "cpu":
        raise ValueError("multi-GPU OPD training cannot use policy device=cpu")
    policy_device = (
        str(accelerator.device)
        if requested_device in {"auto", "cuda"} or accelerator.num_processes > 1
        else requested_device
    )
    agent_config = replace(
        AgentConfig.from_env(),
        policy_backend="transformers",
        policy_base_model=expected_identity.base_model,
        policy_base_revision=expected_identity.base_revision,
        policy_tokenizer_revision=expected_identity.tokenizer_revision,
        policy_adapter_path=checkpoint if expected_identity.adapter_hash is not None else None,
        policy_identity_manifest=identity_manifest_path,
        policy_chat_template=str(policy_data.get("chat_template", "model_default")),
        policy_dtype=str(policy_data.get("dtype", "bfloat16")),
        policy_device=policy_device,
    )
    # 加载模型运行对象
    policy = TransformersPolicy.from_config(agent_config)
    require_matching_policy_identity(expected_identity, policy.identity())
    policy.model = attach_or_validate_shared_adapter(
        policy.model,
        trainer_config.shared_adapter,
    )
    _configure_gradient_checkpointing(
        policy.model,
        enabled=_required_bool(
            policy_data.get("gradient_checkpointing", False),
            "policy.gradient_checkpointing",
        ),
    )
    dataset = OPDLearnerDataset.from_files(
        learner_path,
        export_manifest_path,
        split="train",
        require_all_roles=True,
    )
    requested_ablation = (
        args.ablation
        if args.ablation is not None
        else str(payload.get("ablation", "")).strip().lower()
    )
    if requested_ablation != dataset.ablation:
        raise ValueError("trainer ablation does not match the learner export manifest")
    try:
        validation_dataset = OPDLearnerDataset.from_files(
            learner_path,
            export_manifest_path,
            split="validation",
            require_all_roles=False,
        )
    except ValueError as exc:
        if "split 'validation' is empty" not in str(exc):
            raise
        validation_dataset = None
    require_matching_policy_identity(expected_identity, dataset.initialization_identity)
    trainer = OPDTrainer(
        policy=policy,
        dataset=dataset,
        validation_dataset=validation_dataset,
        config=trainer_config,
        accelerator=accelerator,
    )

    def verify_reload(path: Path, expected: PolicyIdentity) -> bool:
        reload_config = replace(agent_config, policy_adapter_path=path)
        reloaded = TransformersPolicy.from_config(reload_config)
        require_matching_policy_identity(expected, reloaded.identity())
        sample = dataset[0]
        request = DecisionRequest(
            role=sample.role,
            purpose="opd_learner",
            messages=sample.canonical_student_messages,
            tools=sample.canonical_tools,
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
        )
        batch = reloaded.tokenize(request)
        logits = reloaded.forward_logits(batch)
        if tuple(logits.shape[:2]) != tuple(batch.input_ids.shape):
            raise ValueError("reloaded policy forward shape does not match tokenized prompt")
        return True

    result = trainer.train(output_dir, reload_identity_verifier=verify_reload)
    print(json.dumps({
        "checkpoint_dir": str(result.checkpoint_dir),
        "checkpoint_manifest": str(result.checkpoint_manifest_path),
        "identity_manifest": str(result.identity_manifest_path),
        "initialization_identity_hash": expected_identity.identity_hash,
        "output_identity_hash": result.output_identity.identity_hash,
        "collection_round": result.manifest.collection_round,
        "role_kl": dict(result.manifest.train_role_kl),
        "reload_identity_verified": result.manifest.reload_identity_verified,
        "ablation": result.manifest.ablation,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _required_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _configure_gradient_checkpointing(model: Any, *, enabled: bool) -> None:
    if not enabled:
        return
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    enable_checkpointing = getattr(base_model, "gradient_checkpointing_enable", None)
    if not callable(enable_checkpointing):
        raise ValueError("policy model does not support gradient checkpointing")
    enable_input_grads = getattr(base_model, "enable_input_require_grads", None)
    if not callable(enable_input_grads):
        raise ValueError("policy model does not support input gradient hooks")
    config = getattr(base_model, "config", None)
    if config is None:
        raise ValueError("policy model does not expose a config")
    enable_checkpointing()
    enable_input_grads()
    config.use_cache = False


def _resolve_path(base: Path, payload: Mapping[str, Any], key: str) -> Path:
    if key not in payload:
        raise ValueError(f"OPD train run config requires {key}")
    return _resolve_path_value(base, payload[key], key)


def _resolve_override(
    base: Path,
    override: str | None,
    payload: Mapping[str, Any],
    key: str,
) -> Path:
    return (
        _resolve_path_value(Path.cwd(), override, key)
        if override is not None
        else _resolve_path(base, payload, key)
    )


def _resolve_path_value(base: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"OPD train run config {key} must be a path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
