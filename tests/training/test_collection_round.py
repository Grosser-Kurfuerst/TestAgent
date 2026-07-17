from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from my_agent.cli.commands.opd import _verify_run
from my_agent.cli.main import build_parser
from my_agent.opd_data.export import load_learner_samples
from my_agent.training.collection_round import build_collection_round
from tests.training.opd_round_fixtures import FakeTrainablePolicy, identity, round_fixture


class CollectionRoundTests(unittest.TestCase):
    def test_build_round_regenerates_all_roles_with_current_checkpoint(self) -> None:
        fixture = round_fixture()
        policy = FakeTrainablePolicy()
        with tempfile.TemporaryDirectory() as tmp:
            result = build_collection_round(
                collection_round=0,
                policy=policy,
                tasks=fixture.tasks,
                outcomes=fixture.outcomes,
                repositories=fixture.repositories,
                maintenance=fixture.maintenance,
                decision_events=fixture.decisions,
                attribution=fixture.attribution,
                output_dir=tmp,
                seed=7,
            )
            samples = load_learner_samples(result.learner_path)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            verified = _verify_run(Path(tmp))

        self.assertEqual({sample.role for sample in samples}, {
            "selection", "action", "writing", "maintenance",
        })
        self.assertTrue(all(sample.policy_identity == identity() for sample in samples))
        self.assertEqual(manifest["trainer_initialization_identity_hash"], identity().identity_hash)
        self.assertTrue(manifest["current_checkpoint_only"])
        self.assertFalse(manifest["replay_enabled"])
        self.assertEqual(manifest["sample_count"], len(samples))
        self.assertEqual(sum(manifest["role_counts"].values()), len(samples))
        self.assertEqual(len(policy.requests), len(samples))
        self.assertTrue(all(request.purpose == "opd_learner" for request in policy.requests))
        self.assertEqual(verified["status"], "ok")

    def test_opd_cli_registers_build_and_verify_commands(self) -> None:
        parser = build_parser()
        build = parser.parse_args([
            "opd",
            "build-round",
            "--run-dir",
            "/tmp/run",
            "--checkpoint",
            "/tmp/checkpoint",
            "--output",
            "/tmp/output",
            "--collection-round",
            "0",
        ])
        verify = parser.parse_args(["opd", "verify-run", "--run-dir", "/tmp/output"])
        self.assertEqual(build.opd_command, "build-round")
        self.assertEqual(verify.opd_command, "verify-run")

    def test_empty_completion_is_rejected_from_formal_dataset(self) -> None:
        fixture = round_fixture()

        class EmptyPolicy(FakeTrainablePolicy):
            def generate_decision(self, request):
                from dataclasses import replace

                response = super().generate_decision(request)
                return replace(
                    response,
                    raw_completion="",
                    completion_token_ids=(),
                    assistant_loss_mask=(),
                )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "trainable completion"):
                build_collection_round(
                    collection_round=0,
                    policy=EmptyPolicy(),
                    tasks=fixture.tasks,
                    outcomes=fixture.outcomes,
                    repositories=fixture.repositories,
                    maintenance=fixture.maintenance,
                    decision_events=fixture.decisions,
                    attribution=fixture.attribution,
                    output_dir=Path(tmp),
                )


if __name__ == "__main__":
    unittest.main()
