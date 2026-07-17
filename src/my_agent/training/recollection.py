"""Strict M0 -> D0 -> M1 -> D1 -> M2 recollection orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import json
import os
import subprocess

from my_agent.opd_data.schema import ExportManifest
from my_agent.opd_data.export import load_learner_samples
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    canonical_sha256,
    load_policy_identity_manifest,
    require_matching_policy_identity,
)
from my_agent.training.checkpoint_manifest import (
    CHECKPOINT_MANIFEST_FILENAME,
    POLICY_IDENTITY_MANIFEST_FILENAME,
    CheckpointManifest,
    load_checkpoint_manifest,
)
from my_agent.training.collection_round import EXPORT_MANIFEST_FILENAME, LEARNER_EVENTS_FILENAME
from my_agent.training.run_layout import ReproductionRunLayout, prepare_isolated_runs


RECOLLECTION_MANIFEST_SCHEMA_VERSION = "opd-recollection-manifest-v1"
RECOLLECTION_MANIFEST_FILENAME = "recollection_manifest.json"


@dataclass(frozen=True)
class CheckpointArtifact:
    label: str
    path: Path
    identity_manifest_path: Path
    identity: PolicyIdentity
    checkpoint_manifest: CheckpointManifest | None = None

    @property
    def manifest_hash(self) -> str:
        payload = (
            self.checkpoint_manifest.to_dict()
            if self.checkpoint_manifest is not None
            else {
                "label": self.label,
                "identity": self.identity.to_dict(),
            }
        )
        return canonical_sha256(payload)


@dataclass(frozen=True)
class RoundDatasetArtifact:
    label: str
    path: Path
    learner_path: Path
    manifest_path: Path
    manifest: ExportManifest

    @property
    def identity(self) -> PolicyIdentity:
        return self.manifest.trainer_initialization_identity


@dataclass(frozen=True)
class RecollectionStage:
    collection_round: int
    run_id: str
    input_checkpoint: CheckpointArtifact
    dataset: RoundDatasetArtifact
    output_checkpoint: CheckpointArtifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_round": self.collection_round,
            "run_id": self.run_id,
            "input_checkpoint": _checkpoint_payload(self.input_checkpoint),
            "dataset": {
                "label": self.dataset.label,
                "path": str(self.dataset.path),
                "learner_path": str(self.dataset.learner_path),
                "manifest_path": str(self.dataset.manifest_path),
                "dataset_hash": self.dataset.manifest.learner_dataset_hash,
                "identity_hash": self.dataset.identity.identity_hash,
                "sample_count": self.dataset.manifest.sample_count,
            },
            "output_checkpoint": _checkpoint_payload(self.output_checkpoint),
            "training_inputs": [self.dataset.label],
            "replay_enabled": False,
        }


@dataclass(frozen=True)
class RecollectionResult:
    layouts: tuple[ReproductionRunLayout, ...]
    stages: tuple[RecollectionStage, RecollectionStage]
    manifest_path: Path


class RecollectionBackend(Protocol):
    def collect(
        self,
        *,
        collection_round: int,
        layout: ReproductionRunLayout,
        checkpoint: CheckpointArtifact,
        dataset_dir: Path,
    ) -> RoundDatasetArtifact: ...

    def train(
        self,
        *,
        collection_round: int,
        checkpoint: CheckpointArtifact,
        dataset: RoundDatasetArtifact,
        checkpoint_dir: Path,
    ) -> CheckpointArtifact: ...


def run_recollection(
    *,
    root: str | Path,
    baseline_commit: str,
    m0_checkpoint: str | Path,
    m0_identity_manifest: str | Path,
    backend: RecollectionBackend,
    lockfile_path: str | Path | None = None,
) -> RecollectionResult:
    base = Path(root).expanduser().resolve()
    layouts = prepare_isolated_runs(
        base,
        baseline_commit=baseline_commit,
        run_ids=("m0", "m1", "m2"),
        lockfile_path=lockfile_path,
    )
    by_id = {layout.run_id: layout for layout in layouts}
    m0_identity_path = Path(m0_identity_manifest).expanduser().resolve()
    m0 = CheckpointArtifact(
        label="m0",
        path=Path(m0_checkpoint).expanduser().resolve(),
        identity_manifest_path=m0_identity_path,
        identity=load_policy_identity_manifest(m0_identity_path),
    )
    stage0 = _run_stage(
        collection_round=0,
        layout=by_id["m0"],
        input_checkpoint=m0,
        dataset_label="d0",
        output_label="m1",
        output_checkpoint_dir=by_id["m1"].root / "checkpoint",
        backend=backend,
    )
    stage1 = _run_stage(
        collection_round=1,
        layout=by_id["m1"],
        input_checkpoint=stage0.output_checkpoint,
        dataset_label="d1",
        output_label="m2",
        output_checkpoint_dir=by_id["m2"].root / "checkpoint",
        backend=backend,
    )
    _validate_isolation(layouts, (stage0, stage1))
    payload = {
        "schema_version": RECOLLECTION_MANIFEST_SCHEMA_VERSION,
        "baseline_commit": baseline_commit,
        "main_experiment": True,
        "replay_enabled": False,
        "stages": [stage0.to_dict(), stage1.to_dict()],
        "dataset_consumption": {
            "m1": ["d0"],
            "m2": ["d1"],
        },
        "runs": {
            layout.run_id: {
                "root": str(layout.root),
                "repository": str(layout.repository_path),
                "ledger": str(layout.ledger_path),
                "output": str(layout.output_dir),
                "checkpoint": str(layout.root / "checkpoint"),
            }
            for layout in layouts
        },
    }
    manifest_path = base / RECOLLECTION_MANIFEST_FILENAME
    manifest_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return RecollectionResult(layouts, (stage0, stage1), manifest_path)


def load_round_dataset(path: str | Path, *, label: str) -> RoundDatasetArtifact:
    root = Path(path).expanduser().resolve()
    manifest_path = root / EXPORT_MANIFEST_FILENAME
    learner_path = root / LEARNER_EVENTS_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("round export manifest must be an object")
    manifest = ExportManifest.from_dict(payload)
    if not learner_path.is_file():
        raise FileNotFoundError(f"round learner dataset not found: {learner_path}")
    samples = load_learner_samples(learner_path)
    if len(samples) != manifest.sample_count:
        raise ValueError("round learner sample count does not match export manifest")
    if canonical_sha256([sample.to_dict() for sample in samples]) != manifest.learner_dataset_hash:
        raise ValueError("round learner dataset hash does not match export manifest")
    return RoundDatasetArtifact(label, root, learner_path, manifest_path, manifest)


def load_trained_checkpoint(path: str | Path, *, label: str) -> CheckpointArtifact:
    root = Path(path).expanduser().resolve()
    manifest = load_checkpoint_manifest(root / CHECKPOINT_MANIFEST_FILENAME)
    identity_path = root / POLICY_IDENTITY_MANIFEST_FILENAME
    identity = load_policy_identity_manifest(identity_path)
    require_matching_policy_identity(manifest.output_identity, identity)
    return CheckpointArtifact(label, root, identity_path, identity, manifest)


def verify_recollection(path: str | Path) -> Mapping[str, Any]:
    root = Path(path).expanduser().resolve()
    manifest_path = root / RECOLLECTION_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("recollection manifest must be an object")
    if payload.get("schema_version") != RECOLLECTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported recollection manifest schema")
    if payload.get("replay_enabled") is not False:
        raise ValueError("main recollection manifest must disable replay")
    if payload.get("dataset_consumption") != {"m1": ["d0"], "m2": ["d1"]}:
        raise ValueError("recollection dataset consumption violates the main experiment")
    stages = payload.get("stages")
    if not isinstance(stages, list) or len(stages) != 2:
        raise ValueError("recollection manifest requires exactly two stages")
    datasets: list[RoundDatasetArtifact] = []
    outputs: list[CheckpointArtifact] = []
    previous_identity_hash: str | None = None
    for index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, Mapping):
            raise ValueError("recollection stage must be an object")
        if raw_stage.get("collection_round") != index:
            raise ValueError("recollection stage order is invalid")
        if raw_stage.get("training_inputs") != [f"d{index}"]:
            raise ValueError("recollection stage includes replay data")
        input_payload = raw_stage.get("input_checkpoint")
        dataset_payload = raw_stage.get("dataset")
        output_payload = raw_stage.get("output_checkpoint")
        if not all(isinstance(item, Mapping) for item in (
            input_payload, dataset_payload, output_payload
        )):
            raise ValueError("recollection stage artifact payload is invalid")
        input_identity_hash = str(input_payload.get("identity_hash"))
        if previous_identity_hash is not None and input_identity_hash != previous_identity_hash:
            raise ValueError("recollection checkpoint identity chain is broken")
        dataset = load_round_dataset(dataset_payload["path"], label=f"d{index}")
        output = load_trained_checkpoint(output_payload["path"], label=f"m{index + 1}")
        if dataset.identity.identity_hash != input_identity_hash:
            raise ValueError("recollection dataset identity does not match stage input")
        if dataset_payload.get("identity_hash") != dataset.identity.identity_hash:
            raise ValueError("recollection dataset manifest identity hash mismatch")
        if dataset_payload.get("dataset_hash") != dataset.manifest.learner_dataset_hash:
            raise ValueError("recollection dataset manifest hash mismatch")
        if output_payload.get("identity_hash") != output.identity.identity_hash:
            raise ValueError("recollection output checkpoint identity hash mismatch")
        manifest = output.checkpoint_manifest
        assert manifest is not None
        if manifest.initialization_identity.identity_hash != input_identity_hash:
            raise ValueError("recollection trainer initialization identity mismatch")
        if manifest.learner_dataset_hash != dataset.manifest.learner_dataset_hash:
            raise ValueError("recollection checkpoint dataset hash mismatch")
        datasets.append(dataset)
        outputs.append(output)
        previous_identity_hash = output.identity.identity_hash
    if outputs[1].checkpoint_manifest.learner_dataset_hash == datasets[0].manifest.learner_dataset_hash:
        raise ValueError("M2 main experiment replayed D0")
    runs = payload.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != {"m0", "m1", "m2"}:
        raise ValueError("recollection run isolation manifest is incomplete")
    isolated_paths: list[Path] = []
    for run_payload in runs.values():
        if not isinstance(run_payload, Mapping):
            raise ValueError("recollection run payload must be an object")
        for field_name in ("repository", "ledger", "output", "checkpoint"):
            isolated_paths.append(Path(str(run_payload[field_name])).resolve())
    if len(isolated_paths) != len(set(isolated_paths)):
        raise ValueError("recollection run paths are not fully isolated")
    return {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "m1_identity_hash": outputs[0].identity.identity_hash,
        "m2_identity_hash": outputs[1].identity.identity_hash,
        "d0_hash": datasets[0].manifest.learner_dataset_hash,
        "d1_hash": datasets[1].manifest.learner_dataset_hash,
        "replay_enabled": False,
    }


@dataclass(frozen=True)
class RoundCommandSpec:
    collection_commands: tuple[tuple[str, ...], ...]
    training_command: tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)


class CommandRecollectionBackend:
    """Execute configured argv commands, then verify their formal artifacts."""

    def __init__(
        self,
        specs: Mapping[int, RoundCommandSpec],
        *,
        environment: Mapping[str, str] | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        if set(specs) != {0, 1}:
            raise ValueError("recollection command backend requires round 0 and round 1")
        self.specs = dict(specs)
        self.environment = {str(key): str(value) for key, value in (environment or {}).items()}
        self.command_runner = command_runner

    def collect(
        self,
        *,
        collection_round: int,
        layout: ReproductionRunLayout,
        checkpoint: CheckpointArtifact,
        dataset_dir: Path,
    ) -> RoundDatasetArtifact:
        context = _command_context(
            collection_round=collection_round,
            layout=layout,
            checkpoint=checkpoint,
            dataset_dir=dataset_dir,
            checkpoint_dir=Path(""),
        )
        for command in self.specs[collection_round].collection_commands:
            self._run(
                _render_command(command, context),
                round_environment=self.specs[collection_round].environment,
                context=context,
            )
        return load_round_dataset(dataset_dir, label=f"d{collection_round}")

    def train(
        self,
        *,
        collection_round: int,
        checkpoint: CheckpointArtifact,
        dataset: RoundDatasetArtifact,
        checkpoint_dir: Path,
    ) -> CheckpointArtifact:
        layout_root = checkpoint_dir.parent
        layout = ReproductionRunLayout(
            run_id=f"m{collection_round + 1}",
            root=layout_root,
            baseline_commit="command-backend",
        )
        context = _command_context(
            collection_round=collection_round,
            layout=layout,
            checkpoint=checkpoint,
            dataset_dir=dataset.path,
            checkpoint_dir=checkpoint_dir,
        )
        self._run(
            _render_command(self.specs[collection_round].training_command, context),
            round_environment=self.specs[collection_round].environment,
            context=context,
        )
        return load_trained_checkpoint(checkpoint_dir, label=f"m{collection_round + 1}")

    def _run(
        self,
        command: Sequence[str],
        *,
        round_environment: Mapping[str, str],
        context: Mapping[str, str],
    ) -> None:
        if not command or any(not str(item).strip() for item in command):
            raise ValueError("recollection commands must be non-empty argv sequences")
        environment = dict(os.environ)
        environment.update(self.environment)
        environment.update({
            str(key): str(value).format_map(context)
            for key, value in round_environment.items()
        })
        self.command_runner(
            list(command),
            check=True,
            env=environment,
        )


def _run_stage(
    *,
    collection_round: int,
    layout: ReproductionRunLayout,
    input_checkpoint: CheckpointArtifact,
    dataset_label: str,
    output_label: str,
    output_checkpoint_dir: Path,
    backend: RecollectionBackend,
) -> RecollectionStage:
    if layout.repository_path.read_text(encoding="utf-8") != "":
        raise ValueError(f"recollection run {layout.run_id} repository is not empty")
    frozen_identity = load_policy_identity_manifest(input_checkpoint.identity_manifest_path)
    require_matching_policy_identity(input_checkpoint.identity, frozen_identity)
    dataset_dir = layout.output_dir / dataset_label
    dataset = backend.collect(
        collection_round=collection_round,
        layout=layout,
        checkpoint=input_checkpoint,
        dataset_dir=dataset_dir,
    )
    if dataset.path != dataset_dir.resolve():
        raise ValueError("recollection dataset escaped its isolated run output")
    if dataset.manifest.collection_round != collection_round:
        raise ValueError("recollection dataset round mismatch")
    require_matching_policy_identity(input_checkpoint.identity, dataset.identity)
    require_matching_policy_identity(
        frozen_identity,
        load_policy_identity_manifest(input_checkpoint.identity_manifest_path),
    )
    output_checkpoint = backend.train(
        collection_round=collection_round,
        checkpoint=input_checkpoint,
        dataset=dataset,
        checkpoint_dir=output_checkpoint_dir,
    )
    if output_checkpoint.path != output_checkpoint_dir.resolve():
        raise ValueError("trained checkpoint escaped its isolated run")
    manifest = output_checkpoint.checkpoint_manifest
    if manifest is None:
        raise ValueError("trained checkpoint requires a checkpoint manifest")
    require_matching_policy_identity(input_checkpoint.identity, manifest.initialization_identity)
    if manifest.learner_dataset_hash != dataset.manifest.learner_dataset_hash:
        raise ValueError("trained checkpoint consumed a different learner dataset")
    if manifest.collection_round != collection_round:
        raise ValueError("trained checkpoint collection round mismatch")
    if output_checkpoint.identity == input_checkpoint.identity:
        raise ValueError("recollection training did not produce a new policy identity")
    if output_checkpoint.label != output_label:
        raise ValueError("trained checkpoint label mismatch")
    return RecollectionStage(
        collection_round,
        layout.run_id,
        input_checkpoint,
        dataset,
        output_checkpoint,
    )


def _validate_isolation(
    layouts: Sequence[ReproductionRunLayout],
    stages: Sequence[RecollectionStage],
) -> None:
    paths = [
        *(layout.repository_path.resolve() for layout in layouts),
        *(layout.ledger_path.resolve() for layout in layouts),
        *(layout.output_dir.resolve() for layout in layouts),
        *(stage.output_checkpoint.path.resolve() for stage in stages),
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("recollection checkpoint/repository/ledger/output paths must be isolated")
    if stages[0].dataset.path == stages[1].dataset.path:
        raise ValueError("D0 and D1 must be isolated datasets")
    if stages[1].output_checkpoint.checkpoint_manifest is None:
        raise ValueError("M2 checkpoint manifest is missing")
    if (
        stages[1].output_checkpoint.checkpoint_manifest.learner_dataset_hash
        == stages[0].dataset.manifest.learner_dataset_hash
    ):
        raise ValueError("M2 main experiment must not replay D0")


def _checkpoint_payload(checkpoint: CheckpointArtifact) -> dict[str, Any]:
    return {
        "label": checkpoint.label,
        "path": str(checkpoint.path),
        "identity_manifest_path": str(checkpoint.identity_manifest_path),
        "identity_hash": checkpoint.identity.identity_hash,
        "manifest_hash": checkpoint.manifest_hash,
    }


def _command_context(
    *,
    collection_round: int,
    layout: ReproductionRunLayout,
    checkpoint: CheckpointArtifact,
    dataset_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, str]:
    return {
        "collection_round": str(collection_round),
        "run_id": layout.run_id,
        "run_dir": str(layout.root),
        "run_output": str(layout.output_dir),
        "memory_dir": str(layout.memory_dir),
        "repository": str(layout.repository_path),
        "ledger": str(layout.ledger_path),
        "checkpoint": str(checkpoint.path),
        "identity_manifest": str(checkpoint.identity_manifest_path),
        "dataset_dir": str(dataset_dir),
        "learner_dataset": str(dataset_dir / LEARNER_EVENTS_FILENAME),
        "export_manifest": str(dataset_dir / EXPORT_MANIFEST_FILENAME),
        "checkpoint_output": str(checkpoint_dir),
    }


def _render_command(command: Sequence[str], context: Mapping[str, str]) -> tuple[str, ...]:
    try:
        return tuple(str(item).format_map(context) for item in command)
    except KeyError as exc:
        raise ValueError(f"unknown recollection command placeholder: {exc.args[0]}") from exc


__all__ = [
    "RECOLLECTION_MANIFEST_FILENAME",
    "RECOLLECTION_MANIFEST_SCHEMA_VERSION",
    "CheckpointArtifact",
    "CommandRecollectionBackend",
    "RecollectionBackend",
    "RecollectionResult",
    "RecollectionStage",
    "RoundCommandSpec",
    "RoundDatasetArtifact",
    "load_round_dataset",
    "load_trained_checkpoint",
    "run_recollection",
    "verify_recollection",
]
