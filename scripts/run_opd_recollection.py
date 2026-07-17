#!/usr/bin/env python3
"""Execute the strict two-round OPD recollection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping
import json

import yaml

from my_agent.evaluation.opd_evaluation import check_numerical_reproduction_readiness
from my_agent.training.recollection import (
    CommandRecollectionBackend,
    RoundCommandSpec,
    run_recollection,
)


RUN_CONFIG_SCHEMA_VERSION = "opd-recollection-run-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("recollection config must be a YAML object")
    if payload.get("schema_version") != RUN_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported recollection config schema")
    base = config_path.parent
    rounds = payload.get("rounds")
    if not isinstance(rounds, Mapping):
        raise ValueError("recollection config requires rounds mapping")
    specs = {
        index: _round_spec(rounds.get(index, rounds.get(str(index))), index)
        for index in (0, 1)
    }
    environment = payload.get("environment", {})
    if not isinstance(environment, Mapping):
        raise ValueError("recollection environment must be an object")
    numerical = payload.get("numerical_reproduction", {})
    if not isinstance(numerical, Mapping):
        raise ValueError("numerical_reproduction must be an object")
    readiness = check_numerical_reproduction_readiness(
        project_root=Path(__file__).resolve().parents[1],
        source_revisions=_mapping(numerical.get("source_revisions", {}), "source_revisions"),
        model_revisions=_mapping(numerical.get("model_revisions", {}), "model_revisions"),
        training_tasks_total=int(numerical.get("training_tasks_total", 0)),
        source_manifests=_mapping(
            numerical.get("source_manifests", {}), "source_manifests"
        ),
        source_manifest_hashes=_mapping(
            numerical.get("source_manifest_hashes", {}), "source_manifest_hashes"
        ),
        model_artifacts=_mapping(
            numerical.get("model_artifacts", {}), "model_artifacts"
        ),
    )
    if bool(numerical.get("enabled", False)) and not readiness.ready:
        raise ValueError(
            "numerical reproduction resources are incomplete: "
            + ", ".join(readiness.missing_requirements)
        )
    result = run_recollection(
        root=_path(base, payload.get("root"), "root"),
        baseline_commit=_string(payload.get("baseline_commit"), "baseline_commit"),
        m0_checkpoint=_path(base, payload.get("m0_checkpoint"), "m0_checkpoint"),
        m0_identity_manifest=_path(
            base, payload.get("m0_identity_manifest"), "m0_identity_manifest"
        ),
        backend=CommandRecollectionBackend(
            specs,
            environment={str(key): str(value) for key, value in environment.items()},
        ),
        lockfile_path=(
            _path(base, payload["lockfile"], "lockfile")
            if payload.get("lockfile") is not None
            else None
        ),
    )
    print(json.dumps({
        "recollection_manifest": str(result.manifest_path),
        "m1_identity_hash": result.stages[0].output_checkpoint.identity.identity_hash,
        "m2_identity_hash": result.stages[1].output_checkpoint.identity.identity_hash,
        "replay_enabled": False,
        "numerical_readiness": readiness.to_dict(),
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _round_spec(value: Any, index: int) -> RoundCommandSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"recollection round {index} must be an object")
    collection = value.get("collection_commands")
    training = value.get("training_command")
    environment = value.get("environment", {})
    if not isinstance(collection, list) or not collection:
        raise ValueError(f"recollection round {index} requires collection_commands")
    return RoundCommandSpec(
        collection_commands=tuple(_command(item, f"round {index} collection") for item in collection),
        training_command=_command(training, f"round {index} training"),
        environment=_mapping(environment, f"round {index} environment"),
    )


def _command(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} command must be a non-empty argv array")
    command = tuple(str(item) for item in value)
    if any(not item.strip() for item in command):
        raise ValueError(f"{field_name} command contains an empty argv item")
    return command


def _path(base: Path, value: Any, field_name: str) -> Path:
    text = _string(value, field_name)
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): str(item) for key, item in value.items()}


if __name__ == "__main__":
    raise SystemExit(main())
