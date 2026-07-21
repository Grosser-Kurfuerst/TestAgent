from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from my_agent.cli.common import CliContext
from my_agent.config import AgentConfig
from my_agent.opd_ablation import PAPER_ABLATIONS
from my_agent.opd_data.export import (
    load_learner_samples,
    load_maintenance_attempts,
    load_maintenance_evidence,
    load_repository_evidence,
    load_runtime_exclusions,
    load_task_evidence,
    load_task_outcomes,
    sample_statistics,
)
from my_agent.opd_data.schema import ExportManifest
from my_agent.opd_data.attribution.io import load_attribution_events
from my_agent.policy.contracts import TrainablePolicy
from my_agent.policy.identity import (
    canonical_sha256,
    load_policy_identity_manifest,
    require_matching_policy_identity,
)
from my_agent.policy.transformers_policy import TransformersPolicy
from my_agent.training.collection_round import (
    EXPORT_MANIFEST_FILENAME,
    LEARNER_EVENTS_FILENAME,
    build_collection_round,
    build_replay_ablation_dataset,
    validate_learner_samples,
)
from my_agent.training.decision_log import load_decision_events
from my_agent.training.recollection import verify_recollection


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("opd", help="Build and verify strict OPD learner rounds.")
    commands = parser.add_subparsers(dest="opd_command", required=True)
    build = commands.add_parser(
        "build-round",
        help="Join round evidence and regenerate all learner completions with one checkpoint.",
    )
    build.add_argument("--run-dir", required=True)
    build.add_argument("--checkpoint", required=True)
    build.add_argument("--identity-manifest")
    build.add_argument("--output", required=True)
    build.add_argument("--collection-round", type=int, required=True)
    build.add_argument(
        "--max-new-tokens",
        type=_positive_max_new_tokens,
        default=1_024,
        help="Maximum learner completion tokens (default: 1024).",
    )
    build.add_argument("--seed", type=int)
    build.add_argument("--ablation", choices=PAPER_ABLATIONS)
    build.set_defaults(_handler=handle)

    replay = commands.add_parser(
        "build-replay-ablation",
        help="Build the explicit off-policy D0+D1 replay dataset.",
    )
    replay.add_argument("--d0", required=True)
    replay.add_argument("--d1", required=True)
    replay.add_argument("--output", required=True)
    replay.set_defaults(_handler=handle)

    verify = commands.add_parser("verify-run", help="Verify a generated learner dataset manifest.")
    verify.add_argument("--run-dir", required=True)
    verify.set_defaults(_handler=handle)

    recollection = commands.add_parser(
        "verify-recollection",
        help="Verify the isolated M0 -> D0 -> M1 -> D1 -> M2 chain.",
    )
    recollection.add_argument("--run-dir", required=True)
    recollection.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    del ctx
    try:
        if args.opd_command == "build-round":
            payload = _build_round(args)
        elif args.opd_command == "build-replay-ablation":
            result = build_replay_ablation_dataset(
                d0_dir=args.d0,
                d1_dir=args.d1,
                output_dir=args.output,
            )
            payload = {
                "learner_path": str(result.learner_path),
                "manifest_path": str(result.manifest_path),
                "sample_count": result.manifest.sample_count,
                "ablation": result.manifest.ablation,
            }
        elif args.opd_command == "verify-recollection":
            payload = verify_recollection(Path(args.run_dir))
        else:
            payload = _verify_run(Path(args.run_dir))
    except Exception as exc:  # noqa: BLE001 - CLI reports fail-closed evidence errors
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _build_round(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    identity_manifest = (
        Path(args.identity_manifest).expanduser().resolve()
        if args.identity_manifest
        else checkpoint / "policy_identity_manifest.json"
    )
    expected_identity = load_policy_identity_manifest(identity_manifest)
    config = replace(
        AgentConfig.from_env(),
        policy_backend="transformers",
        policy_base_model=expected_identity.base_model,
        policy_base_revision=expected_identity.base_revision,
        policy_tokenizer_revision=expected_identity.tokenizer_revision,
        policy_adapter_path=checkpoint if expected_identity.adapter_hash is not None else None,
        policy_identity_manifest=identity_manifest,
    )
    requested_ablation = args.ablation or config.opd_ablation
    if args.ablation is not None and (
        config.memory_evolver_mode != "formal"
        or config.opd_ablation != args.ablation
    ):
        raise ValueError(
            "opd build-round ablation must match the formal collection runtime"
        )
    policy = TransformersPolicy.from_config(config)
    require_matching_policy_identity(expected_identity, policy.identity())
    if not isinstance(policy, TrainablePolicy):
        raise ValueError("opd build-round requires a local TrainablePolicy")
    source = _evidence_dir(Path(args.run_dir))
    result = build_collection_round(
        collection_round=args.collection_round,
        policy=policy,
        tasks=load_task_evidence(source / "task_evidence.jsonl"),
        outcomes=load_task_outcomes(source / "task_outcomes.jsonl"),
        repositories=load_repository_evidence(source / "repository_events.jsonl"),
        maintenance=load_maintenance_evidence(source / "maintenance_evidence.jsonl"),
        decision_events=load_decision_events(source / "decision_events.jsonl"),
        attribution=load_attribution_events(source / "attribution_events.jsonl"),
        output_dir=args.output,
        runtime_exclusions=(
            load_runtime_exclusions(source / "runtime_exclusions.jsonl")
            if (source / "runtime_exclusions.jsonl").exists()
            else ()
        ),
        maintenance_attempts=(
            load_maintenance_attempts(source / "maintenance_attempts.jsonl")
            if (source / "maintenance_attempts.jsonl").exists()
            else ()
        ),
        writing_top_fraction=config.memory_evolver_writing_top_fraction,
        teacher_minimum_score=config.memory_evolver_teacher_min_score,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        ablation=requested_ablation,
    )
    return {
        "learner_path": str(result.learner_path),
        "manifest_path": str(result.manifest_path),
        "sample_count": result.manifest.sample_count,
        "policy_identity_hash": result.manifest.trainer_initialization_identity.identity_hash,
        "ablation": result.manifest.ablation,
    }


def _positive_max_new_tokens(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-new-tokens must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("max-new-tokens must be >= 1")
    return parsed


def _verify_run(run_dir: Path) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    manifest_payload = json.loads((root / EXPORT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError("export manifest must be an object")
    manifest = ExportManifest.from_dict(manifest_payload)
    samples = load_learner_samples(root / LEARNER_EVENTS_FILENAME)
    validate_learner_samples(
        samples,
        collection_round=manifest.collection_round,
        trainer_identity_hash=manifest.trainer_initialization_identity.identity_hash,
        sample_policy_identity_hashes=manifest.sample_policy_identity_hashes,
        sample_collection_rounds=manifest.sample_collection_rounds,
    )
    dataset_hash = canonical_sha256([sample.to_dict() for sample in samples])
    if dataset_hash != manifest.learner_dataset_hash:
        raise ValueError("learner dataset hash does not match export manifest")
    if len(samples) != manifest.sample_count:
        raise ValueError("learner sample count does not match export manifest")
    if tuple(sorted({sample.policy_identity.identity_hash for sample in samples})) != tuple(
        sorted(manifest.sample_policy_identity_hashes)
    ):
        raise ValueError("learner sample policy provenance does not match export manifest")
    if tuple(sorted({sample.collection_round for sample in samples})) != tuple(
        sorted(manifest.sample_collection_rounds)
    ):
        raise ValueError("learner sample round provenance does not match export manifest")
    stats = sample_statistics(samples)
    if (
        dict(manifest.role_counts) != dict(stats["role_counts"])
        or dict(manifest.split_counts) != dict(stats["split_counts"])
        or dict(manifest.task_group_counts) != dict(stats["task_group_counts"])
    ):
        raise ValueError("learner dataset statistics do not match export manifest")
    return {
        "status": "ok",
        "sample_count": len(samples),
        "collection_round": manifest.collection_round,
        "policy_identity_hash": manifest.trainer_initialization_identity.identity_hash,
        "ablation": manifest.ablation,
    }


def _evidence_dir(run_dir: Path) -> Path:
    root = run_dir.expanduser().resolve()
    nested = root / "evolver_datasets"
    return nested if nested.is_dir() else root


__all__ = ["add_parser", "handle"]
