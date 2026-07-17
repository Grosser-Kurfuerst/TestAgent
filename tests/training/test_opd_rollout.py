from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from my_agent.opd_data.export import prepare_round_decisions
from my_agent.training.opd_rollout import (
    generate_action_rollout_samples,
    generate_learner_sample,
    generate_maintenance_rollout,
    generate_maintenance_rollout_samples,
)
from my_agent.training.role_views import CanonicalMessage, CanonicalToolCall
from tests.training.opd_round_fixtures import FakeTrainablePolicy, identity, round_fixture


class OpdRolloutTests(unittest.TestCase):
    def test_action_regeneration_is_memory_free_and_teacher_adds_hindsight(self) -> None:
        fixture = round_fixture()
        prepared = prepare_round_decisions(
            collection_round=0,
            trainer_identity=identity(),
            tasks=fixture.tasks,
            outcomes=fixture.outcomes,
            repositories=fixture.repositories,
            maintenance=fixture.maintenance,
            decision_events=fixture.decisions,
            attribution=fixture.attribution,
        )
        action = next(item for item in prepared.decisions if item.role == "action")
        policy = FakeTrainablePolicy()

        sample = generate_learner_sample(action, policy=policy)

        public_text = str(sample.student_public_view) + str([
            message.to_dict() for message in sample.canonical_student_messages
        ])
        teacher_text = str([
            message.to_dict() for message in sample.canonical_teacher_messages
        ])
        self.assertNotIn("mem-a", public_text)
        self.assertNotIn("Use the focused public test.", public_text)
        self.assertIn("mem-a", teacher_text)
        self.assertEqual(
            sample.canonical_teacher_messages[:-1],
            sample.canonical_student_messages,
        )
        self.assertEqual(sample.policy_identity, identity())
        self.assertEqual(sample.student_prompt_token_ids, (10, 2))
        self.assertEqual(
            sample.canonical_student_messages,
            action.public_view.prefix_messages,
        )

    def test_action_leakage_and_prompt_token_mismatch_fail_closed(self) -> None:
        fixture = round_fixture()
        prepared = prepare_round_decisions(
            collection_round=0,
            trainer_identity=identity(),
            tasks=fixture.tasks,
            outcomes=fixture.outcomes,
            repositories=fixture.repositories,
            maintenance=fixture.maintenance,
            decision_events=fixture.decisions,
            attribution=fixture.attribution,
        )
        action = next(item for item in prepared.decisions if item.role == "action")
        leaked_public = replace(action.public_view, task="Use mem-a to solve this")
        with self.assertRaisesRegex(ValueError, "leaks selected memory ID"):
            generate_learner_sample(
                replace(action, public_view=leaked_public),
                policy=FakeTrainablePolicy(),
            )

        class BadTokenPolicy(FakeTrainablePolicy):
            def generate_decision(self, request):
                response = super().generate_decision(request)
                return replace(response, prompt_token_ids=(99,))

        with self.assertRaisesRegex(ValueError, "prompt token IDs"):
            generate_learner_sample(action, policy=BadTokenPolicy())

        class BadRoundTripPolicy(FakeTrainablePolicy):
            def verify_completion_round_trip(self, response):
                return False

        with self.assertRaisesRegex(ValueError, "round-trip"):
            generate_learner_sample(action, policy=BadRoundTripPolicy())

    def test_action_rollout_uses_student_calls_and_real_observations(self) -> None:
        fixture = round_fixture()
        prepared = prepare_round_decisions(
            collection_round=0,
            trainer_identity=identity(),
            tasks=fixture.tasks,
            outcomes=fixture.outcomes,
            repositories=fixture.repositories,
            maintenance=fixture.maintenance,
            decision_events=fixture.decisions,
            attribution=fixture.attribution,
        )
        base = next(item for item in prepared.decisions if item.role == "action")
        fast_call = CanonicalToolCall("fast-call", "shell", '{"command":"inspect"}')
        observation = CanonicalMessage(
            "tool",
            "real public observation",
            tool_call_id=fast_call.call_id,
        )
        first = replace(
            base,
            action_rollout_id="rollout-a",
            action_turn_index=0,
            action_expected_tool_calls=(fast_call,),
            action_observation_messages=(observation,),
        )
        second = replace(
            base,
            evidence_refs=("dec-action-next", *base.evidence_refs[1:]),
            action_rollout_id="rollout-a",
            action_turn_index=1,
            action_expected_tool_calls=(),
            action_observation_messages=(),
        )

        class RolloutPolicy(FakeTrainablePolicy):
            def generate_decision(self, request):
                response = super().generate_decision(request)
                if len(self.requests) == 1:
                    return replace(
                        response,
                        parsed_tool_calls=(CanonicalToolCall(
                            "student-call",
                            "shell",
                            '{"command":"inspect"}',
                        ),),
                    )
                return response

            def chat_response_from_decision(self, response):
                return SimpleNamespace(content="student action")

        policy = RolloutPolicy()
        samples = generate_action_rollout_samples((first, second), policy=policy)

        self.assertEqual(len(samples), 2)
        self.assertEqual(len(policy.requests), 2)
        second_prefix = samples[1].canonical_student_messages
        self.assertEqual(second_prefix[-2].role, "assistant")
        self.assertEqual(second_prefix[-2].content, "student action")
        self.assertEqual(second_prefix[-2].tool_calls[0].call_id, "student-call")
        self.assertEqual(second_prefix[-1].role, "tool")
        self.assertEqual(second_prefix[-1].tool_call_id, "student-call")
        self.assertEqual(second_prefix[-1].content, "real public observation")

        class DivergentPolicy(RolloutPolicy):
            def generate_decision(self, request):
                response = super().generate_decision(request)
                if len(self.requests) == 1:
                    return replace(
                        response,
                        parsed_tool_calls=(CanonicalToolCall(
                            "student-call",
                            "shell",
                            '{"command":"different"}',
                        ),),
                    )
                return response

        self.assertEqual(
            len(generate_action_rollout_samples((first, second), policy=DivergentPolicy())),
            1,
        )

    def test_maintenance_rollout_regenerates_each_assistant_turn(self) -> None:
        fixture = round_fixture()
        prepared = prepare_round_decisions(
            collection_round=0,
            trainer_identity=identity(),
            tasks=fixture.tasks,
            outcomes=fixture.outcomes,
            repositories=fixture.repositories,
            maintenance=fixture.maintenance,
            decision_events=fixture.decisions,
            attribution=fixture.attribution,
        )
        base = next(item for item in prepared.decisions if item.role == "maintenance")
        lookup = CanonicalToolCall("fast-lookup", "lookup", '{"query":"duplicate"}')
        finish = CanonicalToolCall("fast-finish", "finish", '{"summary":"done"}')
        observation = CanonicalMessage(
            "tool",
            '{"hits":[]}',
            tool_call_id=lookup.call_id,
        )
        first = replace(
            base,
            maintenance_turn_index=0,
            maintenance_expected_tool_calls=(lookup,),
            maintenance_observation_messages=(observation,),
        )
        second = replace(
            base,
            evidence_refs=("dec-maintenance-2", *base.evidence_refs[1:]),
            maintenance_turn_index=1,
            maintenance_expected_tool_calls=(finish,),
            maintenance_observation_messages=(),
        )

        class MaintenancePolicy(FakeTrainablePolicy):
            def generate_decision(self, request):
                response = super().generate_decision(request)
                call = (lookup, finish)[len(self.requests) - 1]
                return replace(response, parsed_tool_calls=(replace(call, call_id=f"student-{len(self.requests)}"),))

            def chat_response_from_decision(self, response):
                return SimpleNamespace(content="")

        policy = MaintenancePolicy()
        samples = generate_maintenance_rollout_samples((first, second), policy=policy)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[1].canonical_student_messages[-1].role, "tool")
        self.assertEqual(samples[1].canonical_student_messages[-1].tool_call_id, "student-1")
        self.assertEqual(samples[1].evidence_refs[0], "dec-maintenance-2")

        class DivergentMaintenancePolicy(MaintenancePolicy):
            def generate_decision(self, request):
                response = super().generate_decision(request)
                if len(self.requests) == 1:
                    return replace(
                        response,
                        parsed_tool_calls=(CanonicalToolCall(
                            "student-1",
                            "lookup",
                            '{"query":"different"}',
                        ),),
                    )
                return response

        divergent = generate_maintenance_rollout(
            (first, second),
            policy=DivergentMaintenancePolicy(),
        )
        self.assertTrue(divergent.diverged)
        self.assertEqual(len(divergent.samples), 1)


if __name__ == "__main__":
    unittest.main()
