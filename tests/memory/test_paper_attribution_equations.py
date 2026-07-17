from __future__ import annotations

import math
import unittest

from my_agent.memory.evolver.attribution import score_memory
from my_agent.memory.evolver.attribution_schema import CandidateExposure
from my_agent.memory.evolver.paper_attribution import (
    compute_memory_attribution,
    confidence_gamma,
    rho_g,
)
from my_agent.memory.evolver.usage_log import UsageLogEntry
from my_agent.policy.identity import PolicyIdentity, canonical_sha256


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "revision", "sha256:" + "1" * 64, None,
        "tokenizer", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


def _exposure(task_id: str, group: str, selected: bool, reward: float, ordinal: int) -> CandidateExposure:
    return CandidateExposure(
        task_id=task_id,
        task_group=group,
        stream_id="stream-a",
        memory_project_key="project-a",
        memory_id="mem-1",
        tier="skill",
        selected=selected,
        reward=reward,
        collection_round=0,
        task_ordinal=ordinal,
        candidate_snapshot_hash=canonical_sha256({"task": task_id}),
        policy_identity=_identity(),
        repository_revision=f"rev-{ordinal}",
        evaluator_name="pytest",
        evaluator_version="8",
        evaluator_hash=canonical_sha256({"evaluator": "pytest"}),
    )


class PaperAttributionEquationTests(unittest.TestCase):
    def test_single_group_matches_hand_calculated_eq_11_and_12(self) -> None:
        record = compute_memory_attribution(
            memory_id="mem-1",
            tier="skill",
            memory_project_key="project-a",
            exposures=(
                _exposure("task-1", "group-a", True, 1.0, 1),
                _exposure("task-2", "group-a", False, 0.0, 2),
            ),
            collection_round=0,
            as_of_ordinal=2,
        )

        self.assertEqual(record.status, "ready")
        self.assertAlmostEqual(record.groups[0].rho_g, 0.5)
        self.assertAlmostEqual(record.groups[0].delta, 1.0)
        self.assertAlmostEqual(record.attribution, 0.5)
        self.assertAlmostEqual(record.gamma, 1.0 - 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(record.memory_score, record.gamma * 0.5)

    def test_multi_group_contributions_are_summed_without_extra_average(self) -> None:
        record = compute_memory_attribution(
            memory_id="mem-1",
            tier="skill",
            memory_project_key="project-a",
            exposures=(
                _exposure("task-1", "group-a", True, 1.0, 1),
                _exposure("task-2", "group-a", False, 0.0, 2),
                _exposure("task-3", "group-b", True, 1.0, 3),
                _exposure("task-4", "group-b", True, 1.0, 4),
                _exposure("task-5", "group-b", False, 0.0, 5),
            ),
            collection_round=0,
            as_of_ordinal=5,
        )

        self.assertAlmostEqual(record.attribution, 0.5 + 2.0 / 3.0)

    def test_missing_counterfactual_preserves_missing_attribution_not_zero(self) -> None:
        record = compute_memory_attribution(
            memory_id="mem-1",
            tier="skill",
            memory_project_key="project-a",
            exposures=(
                _exposure("task-1", "group-a", True, 1.0, 1),
                _exposure("task-2", "group-a", True, 0.0, 2),
            ),
            collection_round=0,
            as_of_ordinal=2,
        )

        self.assertEqual(record.status, "insufficient_counterfactual_evidence")
        self.assertIsNone(record.attribution)
        self.assertIsNone(record.memory_score)
        self.assertEqual(record.groups[0].status, "insufficient_counterfactual_evidence")

    def test_gamma_exact_values_and_rho_domain(self) -> None:
        for n_plus in (0, 1, 3, 8):
            self.assertAlmostEqual(
                confidence_gamma(n_plus),
                1.0 - 1.0 / math.sqrt(1.0 + n_plus),
            )
        with self.assertRaises(ValueError):
            rho_g(1, 0)

    def test_regression_fixture_differs_from_legacy_pool_mean_formula(self) -> None:
        paper = compute_memory_attribution(
            memory_id="mem-1",
            tier="skill",
            memory_project_key="project-a",
            exposures=(
                _exposure("task-1", "group-a", True, 1.0, 1),
                _exposure("task-2", "group-a", False, 0.0, 2),
            ),
            collection_round=0,
            as_of_ordinal=2,
        )
        legacy = score_memory(
            memory_id="mem-1",
            tier="skill",
            project_key="project-a",
            usage_logs=(
                UsageLogEntry(
                    "task-1", "group-a", stream_id="stream-a", memory_project_key="project-a",
                    retrieved_candidates={"skill": ["mem-1"]},
                    selected_memory_ids={"skill": ["mem-1"]}, env_reward=1.0, success=True,
                ),
                UsageLogEntry(
                    "task-2", "group-a", stream_id="stream-a", memory_project_key="project-a",
                    retrieved_candidates={"skill": ["mem-1"]},
                    selected_memory_ids={"skill": []}, env_reward=0.0, success=False,
                ),
            ),
        )

        self.assertNotAlmostEqual(paper.memory_score, legacy.value)


if __name__ == "__main__":
    unittest.main()
