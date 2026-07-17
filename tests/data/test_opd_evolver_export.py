from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from my_agent.opd_data.export import (
    load_maintenance_evidence,
    load_repository_evidence,
    load_task_evidence,
    load_task_outcomes,
    prepare_round_decisions,
    write_evidence_jsonl,
)
from my_agent.training.role_views import CanonicalMessage, CanonicalToolCall
from tests.training.opd_round_fixtures import identity, round_fixture


class OpdEvolverExportTests(unittest.TestCase):
    def test_evidence_round_trip_and_four_role_join(self) -> None:
        fixture = round_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = write_evidence_jsonl(fixture.tasks, root / "task_evidence.jsonl")
            outcome_path = write_evidence_jsonl(fixture.outcomes, root / "task_outcomes.jsonl")
            repository_path = write_evidence_jsonl(
                fixture.repositories, root / "repository_events.jsonl"
            )
            maintenance_path = write_evidence_jsonl(
                fixture.maintenance, root / "maintenance_evidence.jsonl"
            )
            tasks = load_task_evidence(task_path)
            outcomes = load_task_outcomes(outcome_path)
            repositories = load_repository_evidence(repository_path)
            maintenance = load_maintenance_evidence(maintenance_path)

        prepared = prepare_round_decisions(
            collection_round=0,
            trainer_identity=identity(),
            tasks=tasks,
            outcomes=outcomes,
            repositories=repositories,
            maintenance=maintenance,
            decision_events=fixture.decisions,
            attribution=fixture.attribution,
        )

        self.assertEqual(
            tuple(sorted(tasks, key=lambda item: item.task_ordinal)),
            fixture.tasks,
        )
        self.assertEqual(
            tuple(sorted(outcomes, key=lambda item: item.task_ordinal)),
            fixture.outcomes,
        )
        self.assertEqual(repositories, fixture.repositories)
        self.assertEqual(maintenance, fixture.maintenance)
        self.assertEqual(
            {decision.role for decision in prepared.decisions},
            {"selection", "action", "writing", "maintenance"},
        )
        action = next(item for item in prepared.decisions if item.role == "action")
        self.assertEqual(action.public_view.to_dict()["view_type"], "action_public")
        self.assertEqual(
            action.hindsight_view.to_dict()["positive_memories"][0]["memory_id"],
            "mem-a",
        )
        self.assertEqual(sum(item.selected for item in prepared.writing_score_decisions), 1)

    def test_join_rejects_identity_and_split_drift(self) -> None:
        fixture = round_fixture()
        other_identity = replace(identity(), checkpoint_hash="sha256:" + "9" * 64)
        with self.assertRaisesRegex(ValueError, "policy identity"):
            prepare_round_decisions(
                collection_round=0,
                trainer_identity=other_identity,
                tasks=fixture.tasks,
                outcomes=fixture.outcomes,
                repositories=fixture.repositories,
                maintenance=fixture.maintenance,
                decision_events=fixture.decisions,
                attribution=fixture.attribution,
            )

        drifted = replace(fixture.tasks[1], split="validation")
        with self.assertRaisesRegex(ValueError, "dataset splits"):
            prepare_round_decisions(
                collection_round=0,
                trainer_identity=identity(),
                tasks=(fixture.tasks[0], drifted, *fixture.tasks[2:]),
                outcomes=fixture.outcomes,
                repositories=fixture.repositories,
                maintenance=fixture.maintenance,
                decision_events=fixture.decisions,
                attribution=fixture.attribution,
            )

    def test_join_rejects_invalid_attribution_outcomes_and_trajectory_drift(self) -> None:
        fixture = round_fixture()
        invalid_outcome = replace(fixture.outcomes[0], task_valid=False)
        with self.assertRaisesRegex(ValueError, "invalid or unfinalized"):
            prepare_round_decisions(
                collection_round=0,
                trainer_identity=identity(),
                tasks=fixture.tasks,
                outcomes=(invalid_outcome, *fixture.outcomes[1:]),
                repositories=fixture.repositories,
                maintenance=fixture.maintenance,
                decision_events=fixture.decisions,
                attribution=fixture.attribution,
            )

        drifted_task = replace(
            fixture.tasks[0],
            trajectory=replace(fixture.tasks[0].trajectory, reward=0.0),
        )
        with self.assertRaisesRegex(ValueError, "trajectory reward"):
            prepare_round_decisions(
                collection_round=0,
                trainer_identity=identity(),
                tasks=(drifted_task, *fixture.tasks[1:]),
                outcomes=fixture.outcomes,
                repositories=fixture.repositories,
                maintenance=fixture.maintenance,
                decision_events=fixture.decisions,
                attribution=fixture.attribution,
            )

    def test_action_decisions_preserve_distinct_memory_free_prefixes(self) -> None:
        fixture = round_fixture()
        fast_call = CanonicalToolCall(
            "fast-call-1",
            "shell",
            '{"command":"inspect"}',
        )
        observation = CanonicalMessage(
            "tool",
            "public tool observation",
            tool_call_id=fast_call.call_id,
        )
        first_action = replace(
            fixture.tasks[0].action_decisions[0],
            expected_tool_calls=(fast_call,),
            observation_messages=(observation,),
        )
        second_prefix = (
            CanonicalMessage("user", "Fix public task one"),
            CanonicalMessage("assistant", "", tool_calls=(fast_call,)),
            observation,
        )
        second_action = replace(
            first_action,
            decision_id="dec-action-1b",
            turn_index=1,
            step_index=1,
            prefix_messages=second_prefix,
            expected_tool_calls=(),
            observation_messages=(),
        )
        task = replace(
            fixture.tasks[0],
            action_decisions=(first_action, second_action),
        )
        first_event = next(
            item for item in fixture.decisions if item.decision_id == "dec-action-1"
        )
        first_event = replace(
            first_event,
            parsed_output={"tool_calls": [fast_call.to_dict()]},
        )
        second_event = replace(
            first_event,
            decision_id="dec-action-1b",
            turn_index=1,
            step_index=1,
            canonical_messages=second_prefix,
            parsed_output={"tool_calls": []},
        )

        prepared = prepare_round_decisions(
            collection_round=0,
            trainer_identity=identity(),
            tasks=(task, *fixture.tasks[1:]),
            outcomes=fixture.outcomes,
            repositories=fixture.repositories,
            maintenance=fixture.maintenance,
            decision_events=tuple(
                first_event if item.decision_id == first_event.decision_id else item
                for item in fixture.decisions
            ) + (second_event,),
            attribution=fixture.attribution,
        )

        actions = [
            item
            for item in prepared.decisions
            if item.role == "action" and item.public_view.task == "Fix public task one"
        ]
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].public_view.prefix_messages, first_action.prefix_messages)
        self.assertEqual(actions[1].public_view.prefix_messages, first_action.prefix_messages)
        self.assertEqual(actions[0].action_expected_tool_calls, (fast_call,))
        self.assertEqual(actions[0].action_observation_messages, (observation,))

if __name__ == "__main__":
    unittest.main()
