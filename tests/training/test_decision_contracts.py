from __future__ import annotations

import unittest

from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.training.contracts import (
    AuthoritativeTaskOutcome,
    DecisionEvent,
    EvaluatorIdentity,
)
from my_agent.training.formal_contract import require_formal_maintenance_actions
from my_agent.training.role_views import CanonicalMessage


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model",
        "model-rev",
        "sha256:" + "1" * 64,
        None,
        "tokenizer-rev",
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
    )


class DecisionContractTests(unittest.TestCase):
    def test_decision_event_round_trip_verifies_identity_hash(self) -> None:
        event = DecisionEvent(
            role="action",
            purpose="fast_loop_evidence",
            decision_id="dec-1",
            trajectory_id="traj-1",
            turn_index=0,
            step_index=0,
            task_id="task-1",
            task_group="group-a",
            stream_id="stream-a",
            memory_project_key="project-a",
            run_id="run-m0",
            policy_identity=_identity(),
            repository_revision="rev-1",
            candidate_snapshot_hash=canonical_sha256([]),
            canonical_messages=(CanonicalMessage("user", "task"),),
            canonical_tools=(),
            rendered_prompt_hash=canonical_sha256({"prompt": "task"}),
            prompt_token_ids=(1,),
            raw_completion="done",
            completion_token_ids=(2, 3),
            assistant_loss_mask=(1, 1),
            parsed_output={},
            retry_of=None,
            status="success",
        )
        payload = event.to_dict()
        self.assertEqual(DecisionEvent.from_dict(payload), event)

        payload["policy_identity_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            DecisionEvent.from_dict(payload)

    def test_authoritative_outcome_requires_finalization_for_formal_use(self) -> None:
        outcome = AuthoritativeTaskOutcome(
            task_id="task-1",
            task_group="group-a",
            task_valid=True,
            resolved=False,
            reward=0.0,
            evaluator=EvaluatorIdentity("pytest", "8", canonical_sha256({"command": "pytest"})),
            outcome_finalized=False,
        )
        with self.assertRaisesRegex(ValueError, "outcome_finalized"):
            outcome.to_ref()

    def test_direct_evaluator_identity_rejects_null_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluator name"):
            EvaluatorIdentity(None, "8", canonical_sha256({"command": "pytest"}))

    def test_authoritative_outcome_rejects_missing_evaluator_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "EvaluatorIdentity"):
            AuthoritativeTaskOutcome(
                task_id="task-1",
                task_group="group-a",
                task_valid=True,
                resolved=True,
                reward=1.0,
                evaluator=None,
            )

    def test_decision_event_rejects_missing_policy_identity_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "PolicyIdentity"):
            DecisionEvent(
                role="action",
                purpose="fast_loop_evidence",
                decision_id="dec-null-policy",
                trajectory_id="traj-null-policy",
                turn_index=0,
                step_index=0,
                task_id="task-1",
                task_group="group-a",
                stream_id="stream-a",
                memory_project_key="project-a",
                run_id="run-m0",
                policy_identity=None,
                repository_revision="rev-1",
                candidate_snapshot_hash=canonical_sha256([]),
                canonical_messages=(CanonicalMessage("user", "task"),),
                canonical_tools=(),
                rendered_prompt_hash=canonical_sha256({"prompt": "task"}),
                prompt_token_ids=(1,),
                raw_completion="done",
                completion_token_ids=(2,),
                assistant_loss_mask=(1,),
                parsed_output={},
                retry_of=None,
                status="success",
            )

    def test_decision_event_rejects_null_task_group(self) -> None:
        event = DecisionEvent(
            role="action",
            purpose="fast_loop_evidence",
            decision_id="dec-2",
            trajectory_id="traj-2",
            turn_index=0,
            step_index=0,
            task_id="task-2",
            task_group="group-a",
            stream_id="stream-a",
            memory_project_key="project-a",
            run_id="run-m0",
            policy_identity=_identity(),
            repository_revision="rev-1",
            candidate_snapshot_hash=canonical_sha256([]),
            canonical_messages=(CanonicalMessage("user", "task"),),
            canonical_tools=(),
            rendered_prompt_hash=canonical_sha256({"prompt": "task"}),
            prompt_token_ids=(1,),
            raw_completion="done",
            completion_token_ids=(2,),
            assistant_loss_mask=(1,),
            parsed_output={},
            retry_of=None,
            status="success",
        )
        payload = event.to_dict()
        payload["task_group"] = None

        with self.assertRaisesRegex(ValueError, "task_group"):
            DecisionEvent.from_dict(payload)

    def test_formal_maintenance_rejects_promote(self) -> None:
        with self.assertRaisesRegex(ValueError, "promote"):
            require_formal_maintenance_actions(("lookup", "promote", "finish"))


if __name__ == "__main__":
    unittest.main()
