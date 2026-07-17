from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from my_agent.training.collection_round import build_collection_round
from my_agent.training.opd_dataset import OPDLearnerDataset, RoleSampler
from tests.training.opd_round_fixtures import FakeTrainablePolicy, identity, round_fixture


class OPDLearnerDatasetTests(unittest.TestCase):
    def test_dataset_validates_manifest_identity_and_all_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = round_fixture()
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
            )
            dataset = OPDLearnerDataset.from_files(
                result.learner_path,
                result.manifest_path,
            )

        self.assertEqual(dataset.initialization_identity, identity())
        self.assertEqual(dataset.collection_round, 0)
        self.assertEqual(set(dataset.statistics.role_counts), {
            "selection", "action", "writing", "maintenance",
        })
        self.assertEqual(dataset.learner_dataset_hash, result.manifest.learner_dataset_hash)

    def test_identity_mismatch_is_rejected_before_training(self) -> None:
        fixture = round_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            result = build_collection_round(
                collection_round=0,
                policy=FakeTrainablePolicy(),
                tasks=fixture.tasks,
                outcomes=fixture.outcomes,
                repositories=fixture.repositories,
                maintenance=fixture.maintenance,
                decision_events=fixture.decisions,
                attribution=fixture.attribution,
                output_dir=Path(tmp),
            )
            samples = tuple(
                replace(sample, policy_identity=replace(
                    sample.policy_identity,
                    checkpoint_hash="sha256:" + "9" * 64,
                ))
                for sample in OPDLearnerDataset.from_files(
                    result.learner_path,
                    result.manifest_path,
                )
            )

        with self.assertRaisesRegex(ValueError, "identity"):
            OPDLearnerDataset(
                samples,
                initialization_identity=identity(),
                collection_round=0,
            )

    def test_role_sampler_allocates_equal_weights_across_roles(self) -> None:
        fixture = round_fixture()
        with tempfile.TemporaryDirectory() as tmp:
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
            )
            dataset = OPDLearnerDataset.from_files(result.learner_path, result.manifest_path)
            sampler = RoleSampler(
                dataset,
                role_weights={role: 1.0 for role in dataset.statistics.role_counts},
                num_samples=8,
                seed=11,
            )
            first = list(sampler)
            sampler.set_epoch(0)
            second = list(sampler)

        self.assertEqual(first, second)
        self.assertEqual(sampler.sampled_role_counts, {
            "selection": 2,
            "action": 2,
            "writing": 2,
            "maintenance": 2,
        })


if __name__ == "__main__":
    unittest.main()
