from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from my_agent.evaluation.opd_evaluation import (
    CommandEvaluationBackend,
    PAPER_ABLATIONS,
    HeldOutProtocol,
    build_evaluation_matrix,
    check_numerical_reproduction_readiness,
    execute_evaluation_matrix,
    write_ablation_manifest,
)
from my_agent.policy.identity import canonical_sha256
from my_agent.opd_ablation import ablation_excluded_roles, ablation_recipe_hash
from my_agent.memory.evolver.coordinator import SimilarityTaskSelectionPolicy
from my_agent.training.decision_log import DecisionEventContext
from my_agent.training.role_views import CandidateSnapshotEntry
from my_agent.training.recollection import CheckpointArtifact
from my_agent.training.checkpoint_manifest import CheckpointManifest
from my_agent.training.opd_trainer import SharedAdapterConfig
from tests.training.opd_round_fixtures import identity


class _EvaluationBackend:
    def run(self, arm, protocol):
        return {
            "arm": arm.label,
            "protocol_hash": protocol.protocol_hash,
            "ordered_task_ids": list(protocol.ordered_task_ids),
        }


class OPDEvaluationTests(unittest.TestCase):
    def test_similarity_ablation_selects_by_retrieval_score_and_budget(self) -> None:
        candidates = (
            CandidateSnapshotEntry("A", "mem-a", "skill", "a", 0.4, 1, 2),
            CandidateSnapshotEntry("B", "mem-b", "tip", "b", 0.9, 1, 3),
            CandidateSnapshotEntry("C", "mem-c", "tool", "c", 0.7, 1, 2),
        )
        selected = SimilarityTaskSelectionPolicy().select(
            task="task",
            candidates=candidates,
            token_budget=4,
            max_items=2,
            context=DecisionEventContext(
                trajectory_id="traj",
                turn_index=0,
                step_index=0,
                task_id="task",
                task_group="group",
                stream_id="stream",
                memory_project_key="project",
                run_id="run",
                repository_revision="rev",
                candidate_snapshot_hash=canonical_sha256([]),
            ),
        )

        self.assertEqual(selected, ("mem-b",))

    def test_method_arms_share_protocol_and_isolate_all_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lockfile = root / "uv.lock"
            lockfile.write_text("version = 1\n", encoding="utf-8")
            tasks = root / "heldout.json"
            tasks.write_text(json.dumps({"tasks": [
                {"id": "task-1", "task_group": "group-a"},
                {"id": "task-2", "task_group": "group-b"},
            ]}), encoding="utf-8")
            protocol = HeldOutProtocol.from_manifest(
                tasks,
                max_steps=8,
                token_budget=4096,
                command_timeout=120,
                tools_hash=canonical_sha256({"tools": ["shell"]}),
                evaluator_name="manifest",
                evaluator_version="v1",
                evaluator_hash=canonical_sha256({"evaluator": "manifest"}),
                temperature=1.0,
                top_p=0.95,
            )
            m0_dir = root / "m0"
            m2_dir = root / "m2"
            m0_dir.mkdir()
            m2_dir.mkdir()
            m0 = CheckpointArtifact("m0", m0_dir, m0_dir / "identity.json", identity())
            m2_identity = _next_identity()
            m2 = CheckpointArtifact("m2", m2_dir, m2_dir / "identity.json", m2_identity)
            matrix = build_evaluation_matrix(
                root=root / "evaluation",
                baseline_commit="baseline",
                protocol=protocol,
                m0=m0,
                trained=m2,
                lockfile_path=lockfile,
            )
            results = execute_evaluation_matrix(matrix, backend=_EvaluationBackend())
            captured = {}

            def command_runner(command, *, check, env):
                captured.update({"command": command, "check": check, "env": env})
                output = matrix.arms[0].layout.output_dir
                output.mkdir(parents=True, exist_ok=True)
                (output / "results.jsonl").write_text(
                    "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in (
                        "task-1", "task-2",
                    )),
                    encoding="utf-8",
                )

            command_result = CommandEvaluationBackend(
                command_runner=command_runner,
            ).run(matrix.arms[0], protocol)

        self.assertEqual(len(matrix.arms), 4 + len(PAPER_ABLATIONS))
        self.assertEqual(set(results), {
            "a_m0_no_memory", "b_m0_memory", "c_trained_memory", "d_trained_no_memory",
        })
        self.assertEqual(
            len({arm.layout.repository_path for arm in matrix.arms}),
            len(matrix.arms),
        )
        self.assertTrue(all(
            arm.to_dict(protocol_hash=protocol.protocol_hash)["protocol_hash"]
            == protocol.protocol_hash
            for arm in matrix.arms
        ))
        self.assertEqual(
            {arm.ablation for arm in matrix.arms if arm.ablation},
            set(PAPER_ABLATIONS),
        )
        self.assertEqual(captured["env"]["AGENTCLI_MEMORY_EVOLVER_MODE"], "off")
        self.assertIn("--identity-manifest", captured["command"])
        self.assertEqual(command_result["ordered_task_ids"], ["task-1", "task-2"])

    def test_evaluation_backend_cannot_change_task_order(self) -> None:
        class ReorderedBackend(_EvaluationBackend):
            def run(self, arm, protocol):
                payload = dict(super().run(arm, protocol))
                payload["ordered_task_ids"] = list(reversed(protocol.ordered_task_ids))
                return payload

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lockfile = root / "uv.lock"
            lockfile.write_text("version = 1\n", encoding="utf-8")
            tasks = root / "heldout.json"
            tasks.write_text(json.dumps([{"id": "one"}, {"id": "two"}]), encoding="utf-8")
            protocol = HeldOutProtocol.from_manifest(
                tasks,
                max_steps=1,
                token_budget=1,
                command_timeout=1,
                tools_hash=canonical_sha256({"tools": []}),
                evaluator_name="manifest",
                evaluator_version="v1",
                evaluator_hash=canonical_sha256({"evaluator": 1}),
                temperature=0.0,
                top_p=0.95,
            )
            m0_dir = root / "m0"
            m2_dir = root / "m2"
            m0_dir.mkdir()
            m2_dir.mkdir()
            matrix = build_evaluation_matrix(
                root=root / "evaluation",
                baseline_commit="baseline",
                protocol=protocol,
                m0=CheckpointArtifact("m0", m0_dir, root / "m0-id", identity()),
                trained=CheckpointArtifact("m2", m2_dir, root / "m2-id", _next_identity()),
                lockfile_path=lockfile,
            )

            with self.assertRaisesRegex(ValueError, "task order"):
                execute_evaluation_matrix(matrix, backend=ReorderedBackend())

    def test_numerical_reproduction_stays_gated_until_resources_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = check_numerical_reproduction_readiness(
                project_root=tmp,
                source_revisions={},
                model_revisions={},
                training_tasks_total=0,
            )

        self.assertFalse(report.ready)
        self.assertIn("source_adapter:agent_world_model.py", report.missing_requirements)
        self.assertIn("training_tasks_total:7000", report.missing_requirements)

    def test_empty_manifests_and_single_policy_cannot_pass_numerical_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_root = root / "src" / "my_agent" / "data" / "opd_sources"
            adapter_root.mkdir(parents=True)
            for name in ("__init__.py", "agent_world_model.py", "nemotron_terminal.py", "envscaler.py"):
                (adapter_root / name).write_text("", encoding="utf-8")
            manifests = {}
            hashes = {}
            for source in ("agent_world_model", "nemotron_terminal_corpus", "envscaler"):
                path = root / f"{source}.json"
                path.write_text("[]", encoding="utf-8")
                manifests[source] = path
                hashes[source] = canonical_sha256([])
            policy = root / "policy"
            embedding = root / "embedding"
            policy.mkdir()
            embedding.mkdir()
            report = check_numerical_reproduction_readiness(
                project_root=root,
                source_revisions={source: "rev" for source in manifests},
                model_revisions={"qwen3_4b": "rev", "embedding": "rev"},
                training_tasks_total=7000,
                source_manifests=manifests,
                source_manifest_hashes=hashes,
                model_artifacts={"qwen3_4b": policy, "embedding": embedding},
            )

        self.assertFalse(report.ready)
        self.assertIn("source_count:agent_world_model:3000", report.missing_requirements)
        self.assertIn("model_revision:qwen3_5_9b", report.missing_requirements)

    def test_ablation_checkpoint_requires_exact_recipe_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lockfile = root / "uv.lock"
            lockfile.write_text("version = 1\n", encoding="utf-8")
            tasks = root / "heldout.json"
            tasks.write_text(json.dumps([{"id": "one"}]), encoding="utf-8")
            protocol = HeldOutProtocol.from_manifest(
                tasks,
                max_steps=1,
                token_budget=1,
                command_timeout=1,
                tools_hash=canonical_sha256({"tools": []}),
                evaluator_name="manifest",
                evaluator_version="v1",
                evaluator_hash=canonical_sha256({"evaluator": 1}),
                temperature=0.0,
                top_p=0.95,
            )
            m0_dir = root / "m0"
            m2_dir = root / "m2"
            ablation_dir = root / "no-maintenance"
            for path in (m0_dir, m2_dir, ablation_dir):
                path.mkdir()
            m0 = CheckpointArtifact("m0", m0_dir, root / "m0-id", identity())
            m2 = CheckpointArtifact("m2", m2_dir, root / "m2-id", _next_identity())
            ablation = CheckpointArtifact(
                "no_maintenance",
                ablation_dir,
                root / "ablation-id",
                _next_identity(),
                _checkpoint_manifest(_next_identity(), ablation="no_maintenance"),
            )
            write_ablation_manifest(
                ablation,
                ablation="no_maintenance",
            )
            matrix = build_evaluation_matrix(
                root=root / "evaluation",
                baseline_commit="baseline",
                protocol=protocol,
                m0=m0,
                trained=m2,
                ablation_checkpoints={"no_maintenance": ablation},
                lockfile_path=lockfile,
            )
            arm = next(item for item in matrix.arms if item.ablation == "no_maintenance")
            captured = {}

            def runner(command, *, check, env):
                del command, check
                captured.update(env)
                (arm.layout.output_dir / "results.jsonl").write_text(
                    json.dumps({"task_id": "one"}) + "\n",
                    encoding="utf-8",
                )

            CommandEvaluationBackend(command_runner=runner).run(arm, protocol)

        self.assertTrue(arm.ready)
        self.assertIsNotNone(arm.ablation_manifest_hash)
        self.assertEqual(captured["AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_ENABLED"], "0")

    def test_standard_checkpoint_cannot_be_relabelled_as_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_identity = _next_identity()
            checkpoint = CheckpointArtifact(
                "m2",
                Path(tmp),
                Path(tmp) / "identity.json",
                output_identity,
                _checkpoint_manifest(output_identity),
            )

            with self.assertRaisesRegex(ValueError, "not trained with"):
                write_ablation_manifest(checkpoint, ablation="no_attribution")


def _next_identity():
    return type(identity())(
        base_model=identity().base_model,
        base_revision=identity().base_revision,
        checkpoint_hash=identity().checkpoint_hash,
        adapter_hash="sha256:" + "8" * 64,
        tokenizer_revision=identity().tokenizer_revision,
        tokenizer_hash=identity().tokenizer_hash,
        chat_template_hash=identity().chat_template_hash,
    )


def _checkpoint_manifest(
    output_identity,
    *,
    ablation: str = "",
) -> CheckpointManifest:
    roles = {
        "selection", "action", "writing", "maintenance",
    } - set(ablation_excluded_roles(ablation))
    counts = {role: 1 for role in sorted(roles)}
    adapter = SharedAdapterConfig()
    return CheckpointManifest(
        collection_round=1,
        initialization_identity=identity(),
        output_identity=output_identity,
        learner_dataset_hash=canonical_sha256({"dataset": ablation or "main"}),
        export_manifest_hash=canonical_sha256({"export": ablation or "main"}),
        role_sampling_weights={role: 1.0 for role in sorted(roles)},
        raw_role_counts=counts,
        valid_role_counts=counts,
        sampled_role_counts=counts,
        task_group_counts={"group-a": len(roles)},
        optimizer={"name": "fake"},
        scheduler={"name": "fake"},
        state_artifacts={},
        train_role_kl={role: 0.1 for role in sorted(roles)},
        train_role_tokens=counts,
        validation_role_kl={},
        validation_role_tokens={},
        gradient_norm={"mean": 1.0, "max": 1.0},
        mixed_step_gradient_norm_by_role={role: 1.0 for role in sorted(roles)},
        shared_adapter_name="shared",
        shared_adapter_config=adapter.canonical_payload,
        adapter_config_hash=adapter.adapter_config_hash,
        reload_identity_verified=True,
        ablation=ablation,
        ablation_recipe_hash=ablation_recipe_hash(ablation),
        dataset_source_hashes=(
            {"evidence": canonical_sha256({"source": ablation})}
            if ablation
            else {}
        ),
    )


if __name__ == "__main__":
    unittest.main()
