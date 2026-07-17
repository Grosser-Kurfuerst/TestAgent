from __future__ import annotations

import tempfile
import unittest

from my_agent.training.collection_round import build_collection_round
from my_agent.training.opd_collator import OPDCollator
from my_agent.training.opd_dataset import OPDLearnerDataset
from tests.training.opd_round_fixtures import FakeTrainablePolicy, round_fixture


class OPDCollatorTests(unittest.TestCase):
    def test_collator_appends_same_completion_and_aligns_causal_indexes(self) -> None:
        import torch

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
            )
            dataset = OPDLearnerDataset.from_files(result.learner_path, result.manifest_path)
            samples = (dataset[0], dataset[1])
            batch = OPDCollator(policy, torch_module=torch)(samples)

        self.assertEqual(batch["student_input_ids"].shape[0], 2)
        self.assertTrue(torch.equal(
            batch["student_input_ids"][:, -1],
            batch["teacher_input_ids"][:, -1],
        ))
        self.assertTrue(torch.equal(
            batch["student_assistant_loss_mask"][:, -1],
            batch["completion_mask"][:, -1],
        ))
        self.assertTrue(torch.equal(
            batch["teacher_assistant_loss_mask"][:, -1],
            batch["completion_mask"][:, -1],
        ))
        self.assertEqual(batch["student_prediction_indices"].tolist(), [[1], [1]])
        self.assertEqual(batch["teacher_prediction_indices"].tolist(), [[1], [1]])


if __name__ == "__main__":
    unittest.main()
