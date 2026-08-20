from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from my_agent.cli.commands import opd as opd_command
from my_agent.cli.commands.opd import _verify_run
from my_agent.cli.common import CliContext
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
        replay = parser.parse_args([
            "opd", "build-replay-ablation",
            "--d0", "/tmp/d0",
            "--d1", "/tmp/d1",
            "--output", "/tmp/replay",
        ])
        attribution = parser.parse_args([
            "data",
            "compute-opd-attribution",
            "--run-dir",
            "/tmp/run",
            "--collection-round",
            "0",
        ])
        self.assertEqual(build.opd_command, "build-round")
        self.assertEqual(build.max_new_tokens, 1_024)
        self.assertEqual(verify.opd_command, "verify-run")
        self.assertEqual(replay.opd_command, "build-replay-ablation")
        self.assertEqual(attribution.data_command, "compute-opd-attribution")

    def test_opd_cli_accepts_explicit_learner_token_limit(self) -> None:
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
            "--max-new-tokens",
            "1024",
        ])

        self.assertEqual(build.max_new_tokens, 1_024)

    def test_build_round_uses_cli_environment_config(self) -> None:
        ctx = CliContext(
            run_agent=lambda *args, **kwargs: None,
            agent_repl_cls=object,
            env={"AGENTCLI_POLICY_CHAT_TEMPLATE": "qwen3_5_nothink"},
        )
        args = SimpleNamespace(opd_command="build-round")

        with (
            patch.object(opd_command, "_build_round", return_value={"status": "ok"}) as build,
            redirect_stdout(io.StringIO()),
        ):
            return_code = opd_command.handle(args, ctx)

        self.assertEqual(return_code, 0)
        self.assertEqual(
            build.call_args.kwargs["config"].policy_chat_template,
            "qwen3_5_nothink",
        )

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

    def test_attribution_ablations_change_the_effective_training_evidence(self) -> None:
        fixture = round_fixture()
        for ablation in ("no_attribution", "similarity_only"):
            with self.subTest(ablation=ablation), tempfile.TemporaryDirectory() as tmp:
                result = build_collection_round(
                    collection_round=0,
                    policy=FakeTrainablePolicy(),
                    tasks=fixture.tasks,
                    outcomes=fixture.outcomes,
                    repositories=fixture.repositories,
                    maintenance=fixture.maintenance,
                    decision_events=fixture.decisions,
                    attribution=fixture.attribution,
                    output_dir=tmp,
                    ablation=ablation,
                )

            self.assertEqual(result.manifest.ablation, ablation)
            self.assertNotEqual(
                result.manifest.source_hashes["attribution_input"],
                result.manifest.source_hashes["attribution_effective"],
            )

    def test_role_ablations_remove_the_disabled_training_role(self) -> None:
        fixture = round_fixture()
        cases = (
            ("no_writing_distillation", "writing"),
            ("no_maintenance", "maintenance"),
        )
        for ablation, disabled_role in cases:
            with self.subTest(ablation=ablation), tempfile.TemporaryDirectory() as tmp:
                result = build_collection_round(
                    collection_round=0,
                    policy=FakeTrainablePolicy(),
                    tasks=fixture.tasks,
                    outcomes=fixture.outcomes,
                    repositories=fixture.repositories,
                    maintenance=fixture.maintenance,
                    decision_events=fixture.decisions,
                    attribution=fixture.attribution,
                    output_dir=tmp,
                    ablation=ablation,
                )

            self.assertNotIn(disabled_role, result.manifest.role_counts)
            self.assertEqual(result.manifest.ablation, ablation)

if __name__ == "__main__":
    unittest.main()
