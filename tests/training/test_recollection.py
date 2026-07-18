from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

from my_agent.opd_data.export import write_learner_samples
from my_agent.opd_data.schema import ExportManifest, LearnerSample
from my_agent.policy.identity import (
    canonical_json_bytes,
    canonical_sha256,
    policy_identity_manifest_payload,
)
from my_agent.training.checkpoint_manifest import (
    CheckpointManifest,
    output_identity_for_adapter,
    write_checkpoint_manifests,
)
from my_agent.training.recollection import (
    RECOLLECTION_MANIFEST_SCHEMA_VERSION,
    load_round_dataset,
    load_trained_checkpoint,
    run_recollection,
    verify_recollection,
)
from my_agent.training.collection_round import build_replay_ablation_dataset
from my_agent.training.opd_dataset import OPDLearnerDataset
from my_agent.training.opd_trainer import SharedAdapterConfig
from my_agent.training.role_views import CanonicalMessage
from my_agent.cli.main import build_parser
from tests.training.opd_round_fixtures import identity


class _FakeBackend:
    def collect(self, *, collection_round, layout, checkpoint, dataset_dir):
        samples = tuple(_sample(role, checkpoint.identity, collection_round) for role in (
            "selection", "action", "writing", "maintenance",
        ))
        learner_path = write_learner_samples(samples, dataset_dir / "learner_events.jsonl")
        ordered = sorted(samples, key=lambda item: (item.role, item.task_group, item.sample_id))
        dataset_hash = canonical_sha256([sample.to_dict() for sample in ordered])
        manifest = ExportManifest(
            collection_round=collection_round,
            trainer_initialization_identity=checkpoint.identity,
            learner_dataset_hash=dataset_hash,
            sample_count=4,
            role_counts={role: 1 for role in (
                "selection", "action", "writing", "maintenance",
            )},
            split_counts={"train": 4},
            task_group_counts={"group-a": 4},
            outcome_counts={"resolved": 4},
            source_hashes={"stub": canonical_sha256({"round": collection_round})},
            writing_score_decisions=(),
            exclusions=(),
        )
        (dataset_dir / "export_manifest.json").write_bytes(
            canonical_json_bytes(manifest.to_dict()) + b"\n"
        )
        self.assert_path = learner_path
        return load_round_dataset(dataset_dir, label=f"d{collection_round}")

    def train(self, *, collection_round, checkpoint, dataset, checkpoint_dir):
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "adapter_config.json").write_text(
            '{"adapter":"shared"}\n', encoding="utf-8"
        )
        (checkpoint_dir / "adapter_model.bin").write_bytes(
            f"round-{collection_round}".encode()
        )
        output_identity = output_identity_for_adapter(checkpoint.identity, checkpoint_dir)
        adapter = SharedAdapterConfig()
        manifest = CheckpointManifest(
            collection_round=collection_round,
            initialization_identity=checkpoint.identity,
            output_identity=output_identity,
            learner_dataset_hash=dataset.manifest.learner_dataset_hash,
            export_manifest_hash=canonical_sha256(dataset.manifest.to_dict()),
            role_sampling_weights={role: 1.0 for role in (
                "selection", "action", "writing", "maintenance",
            )},
            raw_role_counts=dict(dataset.manifest.role_counts),
            valid_role_counts=dict(dataset.manifest.role_counts),
            sampled_role_counts=dict(dataset.manifest.role_counts),
            task_group_counts={"group-a": 4},
            optimizer={"name": "fake"},
            scheduler={"name": "fake"},
            state_artifacts={},
            train_role_kl={role: 0.1 for role in dataset.manifest.role_counts},
            train_role_tokens={role: 1 for role in dataset.manifest.role_counts},
            validation_role_kl={},
            validation_role_tokens={},
            gradient_norm={"mean": 1.0, "max": 1.0},
            mixed_step_gradient_norm_by_role={
                role: 1.0 for role in dataset.manifest.role_counts
            },
            shared_adapter_name="shared",
            shared_adapter_config=adapter.canonical_payload,
            adapter_config_hash=adapter.adapter_config_hash,
            reload_identity_verified=True,
        )
        write_checkpoint_manifests(checkpoint_dir, manifest)
        return load_trained_checkpoint(checkpoint_dir, label=f"m{collection_round + 1}")


class RecollectionTests(unittest.TestCase):
    def test_cli_registers_recollection_verifier_and_frozen_eval_checkpoint(self) -> None:
        parser = build_parser()
        verify = parser.parse_args([
            "opd", "verify-recollection", "--run-dir", "/tmp/recollection",
        ])
        evaluation = parser.parse_args([
            "eval-manifest",
            "--tasks", "/tmp/tasks.json",
            "--output-dir", "/tmp/output",
            "--checkpoint", "/tmp/m1",
            "--identity-manifest", "/tmp/m1/policy_identity_manifest.json",
        ])

        self.assertEqual(verify.opd_command, "verify-recollection")
        self.assertEqual(evaluation.checkpoint, "/tmp/m1")

    def test_m0_d0_m1_d1_m2_isolated_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lockfile = root / "uv.lock"
            lockfile.write_text("version = 1\n", encoding="utf-8")
            m0 = root / "m0-checkpoint"
            m0.mkdir()
            identity_path = m0 / "policy_identity_manifest.json"
            identity_path.write_bytes(
                canonical_json_bytes(policy_identity_manifest_payload(identity())) + b"\n"
            )
            result = run_recollection(
                root=root / "runs",
                baseline_commit="baseline",
                m0_checkpoint=m0,
                m0_identity_manifest=identity_path,
                backend=_FakeBackend(),
                lockfile_path=lockfile,
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            verified = verify_recollection(root / "runs")

        self.assertEqual(manifest["schema_version"], RECOLLECTION_MANIFEST_SCHEMA_VERSION)
        self.assertFalse(manifest["replay_enabled"])
        self.assertEqual(manifest["dataset_consumption"], {"m1": ["d0"], "m2": ["d1"]})
        self.assertEqual(
            result.stages[1].output_checkpoint.checkpoint_manifest.learner_dataset_hash,
            result.stages[1].dataset.manifest.learner_dataset_hash,
        )
        self.assertNotEqual(
            result.stages[0].dataset.manifest.learner_dataset_hash,
            result.stages[1].dataset.manifest.learner_dataset_hash,
        )
        self.assertEqual(len({layout.repository_path for layout in result.layouts}), 3)
        self.assertEqual(verified["status"], "ok")

    def test_replay_ablation_materializes_d0_and_d1_with_m1_initialization(self) -> None:
        m0_identity = identity()
        m1_identity = replace(m0_identity, adapter_hash="sha256:" + "8" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for round_index, policy_identity in enumerate((m0_identity, m1_identity)):
                dataset_dir = root / f"d{round_index}"
                samples = tuple(
                    _sample(role, policy_identity, round_index)
                    for role in ("selection", "action", "writing", "maintenance")
                )
                learner_path = write_learner_samples(
                    samples, dataset_dir / "learner_events.jsonl"
                )
                loaded = tuple(sorted(
                    samples,
                    key=lambda item: (item.role, item.task_group, item.sample_id),
                ))
                manifest = ExportManifest(
                    collection_round=round_index,
                    trainer_initialization_identity=policy_identity,
                    learner_dataset_hash=canonical_sha256([
                        sample.to_dict() for sample in loaded
                    ]),
                    sample_count=len(samples),
                    role_counts={role: 1 for role in (
                        "selection", "action", "writing", "maintenance",
                    )},
                    split_counts={"train": len(samples)},
                    task_group_counts={"group-a": len(samples)},
                    outcome_counts={"resolved": len(samples)},
                    source_hashes={"stub": canonical_sha256({"round": round_index})},
                    writing_score_decisions=(),
                    exclusions=(),
                )
                (dataset_dir / "export_manifest.json").write_bytes(
                    canonical_json_bytes(manifest.to_dict()) + b"\n"
                )
                self.assertTrue(learner_path.is_file())

            result = build_replay_ablation_dataset(
                d0_dir=root / "d0",
                d1_dir=root / "d1",
                output_dir=root / "replay",
            )
            dataset = OPDLearnerDataset.from_files(
                result.learner_path,
                result.manifest_path,
            )

        self.assertEqual(result.manifest.ablation, "replay_d0_d1")
        self.assertTrue(result.manifest.replay_enabled)
        self.assertFalse(result.manifest.current_checkpoint_only)
        self.assertEqual(result.manifest.sample_collection_rounds, (0, 1))
        self.assertEqual(dataset.initialization_identity, m1_identity)
        self.assertEqual(len(dataset), 8)


def _sample(role: str, policy_identity, collection_round: int) -> LearnerSample:
    student = (CanonicalMessage("user", f"public-{role}-{collection_round}"),)
    teacher = (*student, CanonicalMessage("user", "hindsight"))
    public_hash = canonical_sha256({
        "messages": [item.to_dict() for item in student],
        "tools": [],
    })
    return LearnerSample(
        role=role,
        collection_round=collection_round,
        split="train",
        task_group="group-a",
        stream_id="stream-a",
        memory_project_key="project-a",
        source_evidence_ids=(f"evidence-{role}-{collection_round}",),
        evidence_refs=(f"decision-{role}-{collection_round}",),
        policy_identity=policy_identity,
        student_public_view={"view_type": f"{role}_public"},
        teacher_hindsight_view={"view_type": f"{role}_hindsight"},
        canonical_student_messages=student,
        canonical_teacher_messages=teacher,
        canonical_tools=(),
        student_raw_completion="done",
        student_prompt_token_ids=(1,),
        student_completion_token_ids=(2,),
        assistant_loss_mask=(1,),
        public_prefix_hash=public_hash,
        student_prompt_hash=canonical_sha256({"student": role, "round": collection_round}),
        teacher_prompt_hash=canonical_sha256({"teacher": role, "round": collection_round}),
    )


if __name__ == "__main__":
    unittest.main()
