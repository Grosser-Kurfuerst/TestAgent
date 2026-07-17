#!/usr/bin/env python3
"""Create isolated held-out OPD method and paper-ablation evaluation arms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping
import json

import yaml

from my_agent.evaluation.opd_evaluation import (
    CommandEvaluationBackend,
    HeldOutProtocol,
    build_evaluation_matrix,
    check_numerical_reproduction_readiness,
    execute_evaluation_matrix,
)
from my_agent.policy.identity import load_policy_identity_manifest
from my_agent.training.recollection import CheckpointArtifact, load_trained_checkpoint


RUN_CONFIG_SCHEMA_VERSION = "opd-evaluation-plan-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation plan config must be a YAML object")
    if payload.get("schema_version") != RUN_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation plan config schema")
    base = config_path.parent
    protocol_data = _mapping(payload.get("protocol"), "protocol")
    protocol = HeldOutProtocol.from_manifest(
        _path(base, protocol_data.get("tasks"), "protocol.tasks"),
        max_steps=int(protocol_data.get("max_steps", 0)),
        token_budget=int(protocol_data.get("token_budget", 0)),
        command_timeout=int(protocol_data.get("command_timeout", 0)),
        tools_hash=_string(protocol_data.get("tools_hash"), "protocol.tools_hash"),
        evaluator_name=_string(
            protocol_data.get("evaluator_name"), "protocol.evaluator_name"
        ),
        evaluator_version=_string(
            protocol_data.get("evaluator_version"), "protocol.evaluator_version"
        ),
        evaluator_hash=_string(
            protocol_data.get("evaluator_hash"), "protocol.evaluator_hash"
        ),
        temperature=float(protocol_data.get("temperature", 1.0)),
        top_p=float(protocol_data.get("top_p", 0.95)),
    )
    m0_data = _mapping(payload.get("m0"), "m0")
    m0_identity_path = _path(base, m0_data.get("identity_manifest"), "m0.identity_manifest")
    m0 = CheckpointArtifact(
        "m0",
        _path(base, m0_data.get("checkpoint"), "m0.checkpoint"),
        m0_identity_path,
        load_policy_identity_manifest(m0_identity_path),
    )
    trained = load_trained_checkpoint(
        _path(base, payload.get("trained_checkpoint"), "trained_checkpoint"),
        label="m2",
    )
    ablation_checkpoints = {
        str(name): load_trained_checkpoint(
            _path(base, path, f"ablations.{name}"),
            label=str(name),
        )
        for name, path in _mapping(payload.get("ablations", {}), "ablations").items()
        if str(path).strip()
    }
    numerical = _mapping(payload.get("numerical_reproduction", {}), "numerical_reproduction")
    readiness = check_numerical_reproduction_readiness(
        project_root=Path(__file__).resolve().parents[1],
        source_revisions=_string_mapping(
            numerical.get("source_revisions", {}), "source_revisions"
        ),
        model_revisions=_string_mapping(
            numerical.get("model_revisions", {}), "model_revisions"
        ),
        training_tasks_total=int(numerical.get("training_tasks_total", 0)),
        source_manifests=_string_mapping(
            numerical.get("source_manifests", {}), "source_manifests"
        ),
        source_manifest_hashes=_string_mapping(
            numerical.get("source_manifest_hashes", {}), "source_manifest_hashes"
        ),
        model_artifacts=_string_mapping(
            numerical.get("model_artifacts", {}), "model_artifacts"
        ),
    )
    if bool(numerical.get("enabled", False)) and not readiness.ready:
        raise ValueError(
            "numerical reproduction resources are incomplete: "
            + ", ".join(readiness.missing_requirements)
        )
    matrix = build_evaluation_matrix(
        root=_path(base, payload.get("root"), "root"),
        baseline_commit=_string(payload.get("baseline_commit"), "baseline_commit"),
        protocol=protocol,
        m0=m0,
        trained=trained,
        ablation_checkpoints=ablation_checkpoints,
        lockfile_path=(
            _path(base, payload["lockfile"], "lockfile")
            if payload.get("lockfile") is not None
            else None
        ),
    )
    executed = {}
    if args.execute:
        executed = execute_evaluation_matrix(
            matrix,
            backend=CommandEvaluationBackend(
                environment=_string_mapping(
                    payload.get("environment", {}), "environment"
                )
            ),
        )
    print(json.dumps({
        "evaluation_matrix": str(matrix.manifest_path),
        "protocol_hash": protocol.protocol_hash,
        "ready_arms": [arm.label for arm in matrix.arms if arm.ready],
        "planned_ablations": [arm.ablation for arm in matrix.arms if arm.ablation],
        "executed_arms": sorted(executed),
        "numerical_readiness": readiness.to_dict(),
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _path(base: Path, value: Any, field_name: str) -> Path:
    text = _string(value, field_name)
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _string_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    return {str(key): str(item) for key, item in _mapping(value, field_name).items()}


if __name__ == "__main__":
    raise SystemExit(main())
