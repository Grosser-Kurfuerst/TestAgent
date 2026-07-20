"""Comparable held-out evaluation matrices and numerical-reproduction gating."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import json
import os
import subprocess

from my_agent.evaluation.manifest_benchmark import load_manifest_tasks
from my_agent.opd_ablation import (
    ABLATION_RECIPES,
    PAPER_ABLATIONS,
    ablation_recipe_hash,
)
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
)
from my_agent.training.recollection import CheckpointArtifact
from my_agent.training.run_layout import ReproductionRunLayout, prepare_isolated_runs


EVALUATION_MATRIX_SCHEMA_VERSION = "opd-evaluation-matrix-v1"
EVALUATION_MATRIX_FILENAME = "evaluation_matrix.json"
ABLATION_MANIFEST_SCHEMA_VERSION = "opd-ablation-manifest-v2"
ABLATION_MANIFEST_FILENAME = "opd_ablation_manifest.json"
PAPER_SOURCE_ADAPTERS = (
    "__init__.py",
    "agent_world_model.py",
    "nemotron_terminal.py",
    "envscaler.py",
)
@dataclass(frozen=True)
class HeldOutProtocol:
    tasks_path: Path
    ordered_task_ids: tuple[str, ...]
    task_manifest_hash: str
    max_steps: int
    token_budget: int
    command_timeout: int
    tools_hash: str
    evaluator_name: str
    evaluator_version: str
    evaluator_hash: str
    temperature: float
    top_p: float

    @classmethod
    def from_manifest(
        cls,
        tasks_path: str | Path,
        *,
        max_steps: int,
        token_budget: int,
        command_timeout: int,
        tools_hash: str,
        evaluator_name: str,
        evaluator_version: str,
        evaluator_hash: str,
        temperature: float,
        top_p: float,
    ) -> "HeldOutProtocol":
        path = Path(tasks_path).expanduser().resolve()
        tasks = load_manifest_tasks(path)
        task_ids = tuple(str(task.get("id") or "").strip() for task in tasks)
        if not task_ids or any(not task_id for task_id in task_ids):
            raise ValueError("held-out manifest tasks require stable non-empty IDs")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("held-out manifest task IDs must be unique")
        if min(max_steps, token_budget, command_timeout) < 1:
            raise ValueError("held-out budgets must be positive")
        if not 0.0 < top_p <= 1.0 or temperature < 0.0:
            raise ValueError("held-out decoding configuration is invalid")
        if top_p != 0.95:
            raise ValueError("current Transformers evaluation contract requires top_p=0.95")
        require_sha256(tools_hash, field_name="held-out tools_hash")
        require_sha256(evaluator_hash, field_name="held-out evaluator_hash")
        if not evaluator_name.strip() or not evaluator_version.strip():
            raise ValueError("held-out evaluator identity must be complete")
        return cls(
            path,
            task_ids,
            canonical_sha256(tasks),
            max_steps,
            token_budget,
            command_timeout,
            tools_hash,
            evaluator_name,
            evaluator_version,
            evaluator_hash,
            temperature,
            top_p,
        )

    @property
    def protocol_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks_path": str(self.tasks_path),
            "ordered_task_ids": list(self.ordered_task_ids),
            "task_manifest_hash": self.task_manifest_hash,
            "max_steps": self.max_steps,
            "token_budget": self.token_budget,
            "command_timeout": self.command_timeout,
            "tools_hash": self.tools_hash,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "evaluator_hash": self.evaluator_hash,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass(frozen=True)
class EvaluationArm:
    label: str
    policy_label: str
    policy_identity: PolicyIdentity | None
    policy_checkpoint: Path | None
    policy_identity_manifest: Path | None
    memory_enabled: bool
    ablation: str | None
    layout: ReproductionRunLayout
    ready: bool
    ablation_manifest_hash: str | None = None

    def to_dict(self, *, protocol_hash: str) -> dict[str, Any]:
        return {
            "label": self.label,
            "policy_label": self.policy_label,
            "policy_identity_hash": (
                self.policy_identity.identity_hash if self.policy_identity is not None else None
            ),
            "policy_checkpoint": (
                str(self.policy_checkpoint) if self.policy_checkpoint is not None else None
            ),
            "policy_identity_manifest": (
                str(self.policy_identity_manifest)
                if self.policy_identity_manifest is not None
                else None
            ),
            "memory_enabled": self.memory_enabled,
            "ablation": self.ablation,
            "ready": self.ready,
            "ablation_manifest_hash": self.ablation_manifest_hash,
            "protocol_hash": protocol_hash,
            "repository": str(self.layout.repository_path),
            "ledger": str(self.layout.ledger_path),
            "output": str(self.layout.output_dir),
        }


@dataclass(frozen=True)
class EvaluationMatrix:
    protocol: HeldOutProtocol
    arms: tuple[EvaluationArm, ...]
    manifest_path: Path


class EvaluationBackend(Protocol):
    def run(self, arm: EvaluationArm, protocol: HeldOutProtocol) -> Mapping[str, Any]: ...


def build_evaluation_matrix(
    *,
    root: str | Path,
    baseline_commit: str,
    protocol: HeldOutProtocol,
    m0: CheckpointArtifact,
    trained: CheckpointArtifact,
    ablation_checkpoints: Mapping[str, CheckpointArtifact] | None = None,
    lockfile_path: str | Path | None = None,
) -> EvaluationMatrix:
    variants = dict(ablation_checkpoints or {})
    unknown = sorted(set(variants) - set(PAPER_ABLATIONS))
    if unknown:
        raise ValueError(f"unknown paper ablation checkpoints: {unknown}")
    labels = ("a_m0_no_memory", "b_m0_memory", "c_trained_memory", "d_trained_no_memory", *(
        f"ablation_{name}" for name in PAPER_ABLATIONS
    ))
    layouts = prepare_isolated_runs(
        root,
        baseline_commit=baseline_commit,
        run_ids=labels,
        lockfile_path=lockfile_path,
    )
    by_label = {layout.run_id: layout for layout in layouts}
    arms: list[EvaluationArm] = [
        EvaluationArm(
            labels[0], "m0", m0.identity, m0.path, m0.identity_manifest_path,
            False, None, by_label[labels[0]], True,
        ),
        EvaluationArm(
            labels[1], "m0", m0.identity, m0.path, m0.identity_manifest_path,
            True, None, by_label[labels[1]], True,
        ),
        EvaluationArm(
            labels[2], trained.label, trained.identity, trained.path,
            trained.identity_manifest_path, True, None,
            by_label[labels[2]], True,
        ),
        EvaluationArm(
            labels[3], trained.label, trained.identity, trained.path,
            trained.identity_manifest_path, False, None,
            by_label[labels[3]], True,
        ),
    ]
    for name in PAPER_ABLATIONS:
        checkpoint = variants.get(name)
        ablation_manifest_hash = None
        if checkpoint is not None:
            ablation_manifest_hash = _load_ablation_manifest(checkpoint, expected_name=name)
        label = f"ablation_{name}"
        arms.append(EvaluationArm(
            label=label,
            policy_label=checkpoint.label if checkpoint is not None else name,
            policy_identity=checkpoint.identity if checkpoint is not None else None,
            policy_checkpoint=checkpoint.path if checkpoint is not None else None,
            policy_identity_manifest=(
                checkpoint.identity_manifest_path if checkpoint is not None else None
            ),
            memory_enabled=True,
            ablation=name,
            layout=by_label[label],
            ready=checkpoint is not None,
            ablation_manifest_hash=ablation_manifest_hash,
        ))
    _validate_matrix(protocol, arms)
    manifest_path = Path(root).expanduser().resolve() / EVALUATION_MATRIX_FILENAME
    manifest_path.write_bytes(canonical_json_bytes({
        "schema_version": EVALUATION_MATRIX_SCHEMA_VERSION,
        "protocol": protocol.to_dict(),
        "protocol_hash": protocol.protocol_hash,
        "arms": [arm.to_dict(protocol_hash=protocol.protocol_hash) for arm in arms],
    }) + b"\n")
    return EvaluationMatrix(protocol, tuple(arms), manifest_path)


def execute_evaluation_matrix(
    matrix: EvaluationMatrix,
    *,
    backend: EvaluationBackend,
) -> Mapping[str, Mapping[str, Any]]:
    results: dict[str, Mapping[str, Any]] = {}
    for arm in matrix.arms:
        if not arm.ready:
            continue
        payload = dict(backend.run(arm, matrix.protocol))
        if payload.get("protocol_hash") != matrix.protocol.protocol_hash:
            raise ValueError(f"evaluation arm {arm.label} used a different held-out protocol")
        if tuple(payload.get("ordered_task_ids", ())) != matrix.protocol.ordered_task_ids:
            raise ValueError(f"evaluation arm {arm.label} changed held-out task order")
        results[arm.label] = payload
    return results


class CommandEvaluationBackend:
    """Run each ready arm through the same eval-manifest command contract."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        command_runner: Any = subprocess.run,
        executable: Sequence[str] = ("uv", "run", "--extra", "opd-train", "my-agent"),
    ) -> None:
        self.environment = {str(key): str(value) for key, value in (environment or {}).items()}
        self.command_runner = command_runner
        self.executable = tuple(executable)

    def run(self, arm: EvaluationArm, protocol: HeldOutProtocol) -> Mapping[str, Any]:
        if arm.policy_checkpoint is None or arm.policy_identity_manifest is None:
            raise ValueError(f"evaluation arm {arm.label} lacks checkpoint artifacts")
        command = (
            *self.executable,
            "eval-manifest",
            "--tasks", str(protocol.tasks_path),
            "--output-dir", str(arm.layout.output_dir),
            "--mode", "react",
            "--max-steps", str(protocol.max_steps),
            "--command-timeout", str(protocol.command_timeout),
            "--checkpoint", str(arm.policy_checkpoint),
            "--identity-manifest", str(arm.policy_identity_manifest),
        )
        environment = dict(os.environ)
        environment.update(self.environment)
        environment.update({
            "MY_AGENT_MEMORY_DIR": str(arm.layout.memory_dir),
            "AGENTCLI_MEMORY_EVOLVER_MODE": "formal" if arm.memory_enabled else "off",
            "MY_AGENT_TOKEN_BUDGET": str(protocol.token_budget),
            "MY_AGENT_TEMPERATURE": str(protocol.temperature),
            "AGENTCLI_OPD_ABLATION": "",
            "AGENTCLI_MEMORY_EVOLVER_RETRIEVAL_BACKEND": "embedding_cosine",
            "AGENTCLI_MEMORY_EVOLVER_SELECTION_BACKEND": "llm",
            "AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_ENABLED": "1",
        })
        if arm.ablation is not None:
            environment["AGENTCLI_OPD_ABLATION"] = arm.ablation
        if arm.ablation == "no_maintenance":
            environment["AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_ENABLED"] = "0"
        elif arm.ablation == "lexical_retrieval":
            environment["AGENTCLI_MEMORY_EVOLVER_RETRIEVAL_BACKEND"] = "lexical_ablation"
        elif arm.ablation == "similarity_only":
            environment["AGENTCLI_MEMORY_EVOLVER_SELECTION_BACKEND"] = "similarity_ablation"
        self.command_runner(list(command), check=True, env=environment)
        results_path = arm.layout.output_dir / "results.jsonl"
        task_ids: list[str] = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("evaluation result row must be an object")
            task_ids.append(str(payload.get("task_id") or ""))
        if tuple(task_ids) != protocol.ordered_task_ids:
            raise ValueError(f"evaluation arm {arm.label} changed held-out task order")
        run_manifest = arm.layout.output_dir / "evaluation_protocol.json"
        run_manifest.write_bytes(canonical_json_bytes({
            "schema_version": EVALUATION_MATRIX_SCHEMA_VERSION,
            "arm": arm.to_dict(protocol_hash=protocol.protocol_hash),
            "protocol": protocol.to_dict(),
            "protocol_hash": protocol.protocol_hash,
            "ordered_task_ids": task_ids,
        }) + b"\n")
        return {
            "protocol_hash": protocol.protocol_hash,
            "ordered_task_ids": task_ids,
            "run_manifest": str(run_manifest),
        }


@dataclass(frozen=True)
class NumericalReadinessReport:
    ready: bool
    missing_requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "missing_requirements": list(self.missing_requirements),
        }


def check_numerical_reproduction_readiness(
    *,
    project_root: str | Path,
    source_revisions: Mapping[str, str],
    model_revisions: Mapping[str, str],
    training_tasks_total: int,
    source_manifests: Mapping[str, str | Path] | None = None,
    source_manifest_hashes: Mapping[str, str] | None = None,
    model_artifacts: Mapping[str, str | Path] | None = None,
) -> NumericalReadinessReport:
    root = Path(project_root).expanduser().resolve()
    adapter_root = root / "src" / "my_agent" / "data" / "opd_sources"
    missing: list[str] = []
    for filename in PAPER_SOURCE_ADAPTERS:
        if not (adapter_root / filename).is_file():
            missing.append(f"source_adapter:{filename}")
    expected_sources = {"agent_world_model", "nemotron_terminal_corpus", "envscaler"}
    expected_counts = {
        "agent_world_model": 3_000,
        "nemotron_terminal_corpus": 2_000,
        "envscaler": 2_000,
    }
    manifests = dict(source_manifests or {})
    manifest_hashes = dict(source_manifest_hashes or {})
    observed_total = 0
    for source in sorted(expected_sources):
        revision = str(source_revisions.get(source, "")).strip()
        if not revision or "REPLACE_WITH" in revision:
            missing.append(f"source_revision:{source}")
        manifest = manifests.get(source)
        if manifest is None or not Path(manifest).expanduser().is_file():
            missing.append(f"source_manifest:{source}")
            continue
        tasks = load_manifest_tasks(Path(manifest).expanduser())
        observed_total += len(tasks)
        if len(tasks) != expected_counts[source]:
            missing.append(f"source_count:{source}:{expected_counts[source]}")
        pinned_hash = str(manifest_hashes.get(source, ""))
        if not pinned_hash:
            missing.append(f"source_manifest_hash:{source}")
        else:
            try:
                require_sha256(pinned_hash, field_name="source manifest hash")
            except ValueError:
                missing.append(f"source_manifest_hash:{source}")
            else:
                if canonical_sha256(tasks) != pinned_hash:
                    missing.append(f"source_manifest_hash_mismatch:{source}")
        required_fields = {
            "source_name", "source_revision", "source_task_id", "task_group",
            "split", "license", "provenance", "content_hash",
        }
        if any(
            not required_fields.issubset(task)
            or str(task.get("source_name")) != source
            or str(task.get("source_revision")) != revision
            or str(task.get("split")) not in {"train", "validation", "test"}
            or not _valid_content_hash(task.get("content_hash"))
            for task in tasks
        ):
            missing.append(f"source_schema:{source}")
    artifacts = dict(model_artifacts or {})
    for model in ("qwen3_5_4b", "qwen3_5_9b", "embedding"):
        revision = str(model_revisions.get(model, "")).strip()
        if not revision or "REPLACE_WITH" in revision:
            missing.append(f"model_revision:{model}")
        artifact = artifacts.get(model)
        if artifact is None or not Path(artifact).expanduser().exists():
            missing.append(f"model_artifact:{model}")
    if training_tasks_total != 7_000 or observed_total != 7_000:
        missing.append("training_tasks_total:7000")
    return NumericalReadinessReport(not missing, tuple(missing))


def _valid_content_hash(value: Any) -> bool:
    try:
        require_sha256(value, field_name="source task content_hash")
    except ValueError:
        return False
    return True


def write_ablation_manifest(
    checkpoint: CheckpointArtifact,
    *,
    ablation: str,
) -> Path:
    recipe = ABLATION_RECIPES.get(ablation)
    if recipe is None:
        raise ValueError(f"unsupported paper ablation: {ablation}")
    checkpoint_manifest = checkpoint.checkpoint_manifest
    if checkpoint_manifest is None:
        raise ValueError("ablation registration requires a trained checkpoint manifest")
    if (
        checkpoint_manifest.ablation != ablation
        or checkpoint_manifest.ablation_recipe_hash != ablation_recipe_hash(ablation)
    ):
        raise ValueError("checkpoint was not trained with the requested ablation recipe")
    source_manifest_hashes = {
        "learner_dataset": checkpoint_manifest.learner_dataset_hash,
        "export_manifest": checkpoint_manifest.export_manifest_hash,
        **checkpoint_manifest.dataset_source_hashes,
    }
    path = checkpoint.path / ABLATION_MANIFEST_FILENAME
    path.write_bytes(canonical_json_bytes({
        "schema_version": ABLATION_MANIFEST_SCHEMA_VERSION,
        "ablation": ablation,
        "checkpoint_identity_hash": checkpoint.identity.identity_hash,
        "checkpoint_manifest_hash": checkpoint.manifest_hash,
        "recipe": dict(recipe),
        "ablation_recipe_hash": checkpoint_manifest.ablation_recipe_hash,
        "source_manifest_hashes": dict(sorted(source_manifest_hashes.items())),
    }) + b"\n")
    return path


def _load_ablation_manifest(
    checkpoint: CheckpointArtifact,
    *,
    expected_name: str,
) -> str:
    path = checkpoint.path / ABLATION_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ablation manifest must be an object")
    if payload.get("schema_version") != ABLATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported ablation manifest schema")
    if payload.get("ablation") != expected_name:
        raise ValueError("ablation checkpoint label does not match its manifest")
    if payload.get("checkpoint_identity_hash") != checkpoint.identity.identity_hash:
        raise ValueError("ablation manifest checkpoint identity mismatch")
    if payload.get("checkpoint_manifest_hash") != checkpoint.manifest_hash:
        raise ValueError("ablation manifest checkpoint hash mismatch")
    if payload.get("recipe") != ABLATION_RECIPES[expected_name]:
        raise ValueError("ablation manifest recipe does not match the executable contract")
    checkpoint_manifest = checkpoint.checkpoint_manifest
    if checkpoint_manifest is None:
        raise ValueError("ablation checkpoint manifest is missing")
    if (
        checkpoint_manifest.ablation != expected_name
        or checkpoint_manifest.ablation_recipe_hash != ablation_recipe_hash(expected_name)
        or payload.get("ablation_recipe_hash") != checkpoint_manifest.ablation_recipe_hash
    ):
        raise ValueError("ablation checkpoint training recipe is not bound")
    source_hashes = payload.get("source_manifest_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("ablation manifest source hashes are missing")
    for value in source_hashes.values():
        require_sha256(value, field_name="ablation source manifest hash")
    expected_hashes = {
        "learner_dataset": checkpoint_manifest.learner_dataset_hash,
        "export_manifest": checkpoint_manifest.export_manifest_hash,
        **checkpoint_manifest.dataset_source_hashes,
    }
    if dict(source_hashes) != expected_hashes:
        raise ValueError("ablation manifest source hashes do not match checkpoint evidence")
    return canonical_sha256(payload)


def _validate_matrix(protocol: HeldOutProtocol, arms: Sequence[EvaluationArm]) -> None:
    labels = tuple(arm.label for arm in arms)
    if len(labels) != len(set(labels)):
        raise ValueError("evaluation arm labels must be unique")
    paths = [
        *(arm.layout.repository_path.resolve() for arm in arms),
        *(arm.layout.ledger_path.resolve() for arm in arms),
        *(arm.layout.output_dir.resolve() for arm in arms),
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("evaluation arms must isolate repository, ledger, and output")
    for arm in arms:
        if arm.layout.repository_path.read_text(encoding="utf-8") != "":
            raise ValueError(f"evaluation arm {arm.label} did not start with an empty repository")
        if arm.ready and (arm.policy_identity is None or arm.policy_checkpoint is None):
            raise ValueError(f"ready evaluation arm {arm.label} lacks a policy checkpoint")
        if arm.ready and not arm.policy_checkpoint.exists():
            raise FileNotFoundError(
                f"evaluation arm {arm.label} checkpoint not found: {arm.policy_checkpoint}"
            )
        if arm.to_dict(protocol_hash=protocol.protocol_hash)["protocol_hash"] != protocol.protocol_hash:
            raise ValueError("evaluation arm protocol mismatch")


__all__ = [
    "EVALUATION_MATRIX_FILENAME",
    "EVALUATION_MATRIX_SCHEMA_VERSION",
    "PAPER_ABLATIONS",
    "ABLATION_MANIFEST_FILENAME",
    "ABLATION_MANIFEST_SCHEMA_VERSION",
    "ABLATION_RECIPES",
    "EvaluationArm",
    "EvaluationBackend",
    "EvaluationMatrix",
    "HeldOutProtocol",
    "CommandEvaluationBackend",
    "NumericalReadinessReport",
    "build_evaluation_matrix",
    "check_numerical_reproduction_readiness",
    "execute_evaluation_matrix",
    "write_ablation_manifest",
]
