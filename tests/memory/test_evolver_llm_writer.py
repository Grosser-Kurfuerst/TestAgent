from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from my_agent.llm.types import ChatResponse
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact, TaskEvolverSession
from my_agent.memory.experience_store import ExperienceStore
from my_agent.memory.store_errors import MemoryStorePostCommitError
from my_agent.policy.contracts import DecisionResponse
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.training.contracts import AuthoritativeTaskOutcome, EvaluatorIdentity


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "revision", "sha256:" + "1" * 64, None,
        "tokenizer", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


class _Policy:
    def __init__(self, output: str, *, on_generate=None) -> None:
        self.output = output
        self.on_generate = on_generate

    def identity(self):
        return _identity()

    def render_prompt_hash(self, request):
        return canonical_sha256([item.to_dict() for item in request.messages])

    def generate_decision(self, request):
        if self.on_generate is not None:
            self.on_generate()
        return DecisionResponse(
            raw_completion=self.output,
            prompt_token_ids=(10,),
            completion_token_ids=(20,),
            assistant_loss_mask=(1,),
            parsed_tool_calls=(),
            identity=self.identity(),
        )

    def chat_response_from_decision(self, response):
        return ChatResponse(content=self.output)

    def chat(self, *args, **kwargs):
        raise AssertionError("not used")


def _episode(store: ExperienceStore) -> AgentEpisodeArtifact:
    session = TaskEvolverSession(
        task_id="task-1",
        task_group="group-a",
        trajectory_id="traj-1",
        stream_id="stream-a",
        memory_project_key="project-a",
        policy_identity=_identity(),
        repository_revision=store.revision(),
        candidate_snapshot_hash=canonical_sha256([]),
        selected_memory_ids=(),
        rendered_memory_context="",
    )
    return AgentEpisodeArtifact(session, Path(store.path), "assistant_final", "done", (), "fix task")


def _outcome() -> AuthoritativeTaskOutcome:
    return AuthoritativeTaskOutcome(
        "task-1", "group-a", True, True, 1.0,
        EvaluatorIdentity("pytest", "8", canonical_sha256({"command": "pytest"})),
    )


class FormalLLMWriterTests(unittest.TestCase):
    @staticmethod
    def _two_record_output() -> str:
        return json.dumps([
            {
                "tier": "tip",
                "content": "Inspect the focused failure before editing.",
                "payload": {"category": "debugging", "severity": "warning", "trigger": "test failure"},
                "confidence": 0.9,
                "reason": "reusable debugging guidance",
            },
            {
                "tier": "skill",
                "content": "Use a focused verification loop.",
                "payload": {
                    "category": "testing",
                    "technique": "focused verification",
                    "preconditions": [],
                    "steps": ["run the focused test"],
                },
                "confidence": 0.9,
                "reason": "reusable verification method",
            },
        ])

    def test_writer_runs_after_outcome_and_commits_without_fallback(self) -> None:
        output = json.dumps([{
            "tier": "tip",
            "content": "Inspect the focused failure before editing.",
            "payload": {"category": "debugging", "severity": "warning", "trigger": "test failure"},
            "confidence": 0.9,
            "reason": "reusable debugging guidance",
        }])
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_Policy(output),
                trace_sink=lambda event, payload: events.append((event, payload)),
            )

            result = coordinator.finalize_task(_episode(store), _outcome())
            memories = store.all(project_key="project-a")

        self.assertEqual(result.writer_status, "committed")
        self.assertEqual(len(memories), 1)
        names = [event for event, _payload in events]
        self.assertLess(names.index("memory.task_outcome_finalized"), names.index("opd.decision"))
        decision = next(payload for event, payload in events if event == "opd.decision")
        self.assertEqual(decision["role"], "writing")
        self.assertFalse(decision["parsed_output"]["fallback_used"])
        normalized = {**decision, "decision_id": "<decision-id>"}
        self.assertEqual(
            canonical_sha256(normalized),
            "sha256:a120610f499be63e0fb4b212366157fcacdea75dc63c98a0d47ee294f21f4e6c",
        )

    def test_invalid_writer_output_is_audited_no_write(self) -> None:
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_Policy("not-json"),
                trace_sink=lambda event, payload: events.append((event, payload)),
            )

            result = coordinator.finalize_task(_episode(store), _outcome())

        self.assertEqual(result.writer_status, "no_write")
        self.assertEqual(result.written_memory_ids, ())
        decision = next(payload for event, payload in events if event == "opd.decision")
        self.assertEqual(decision["status"], "invalid_output")
        self.assertEqual(decision["completion_token_ids"], [20])

    def test_schema_invalid_field_types_cannot_mutate_repository(self) -> None:
        output = json.dumps([{
            "tier": "tip",
            "content": 123,
            "payload": {"category": "debugging", "severity": "warning", "trigger": "failure"},
            "confidence": "0.9",
            "reason": 456,
        }])
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_Policy(output),
                trace_sink=lambda event, payload: events.append((event, payload)),
            )

            result = coordinator.finalize_task(_episode(store), _outcome())
            memories = store.all(project_key="project-a")

        self.assertEqual(result.writer_status, "no_write")
        self.assertEqual(memories, [])
        decision = next(payload for event, payload in events if event == "opd.decision")
        self.assertEqual(decision["status"], "invalid_output")
        self.assertIn("content must be a string", decision["parsed_output"]["error"])

    def test_revision_change_during_generation_aborts_writer_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)

            def concurrent_write():
                from tests.memory.experience_fixtures import typed_experience
                from my_agent.memory.evolver.types import ExperienceTier

                store.add(typed_experience(
                    "concurrent-tip",
                    "concurrent memory",
                    ExperienceTier.TIP,
                    project_key="project-a",
                ))

            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_Policy(self._two_record_output(), on_generate=concurrent_write),
            )

            result = coordinator.finalize_task(_episode(store), _outcome())
            memories = store.all(project_key="project-a")

        self.assertEqual(result.writer_status, "failed_no_write")
        self.assertEqual([item.id for item in memories], ["concurrent-tip"])

    def test_multi_record_store_failure_cannot_leave_partial_writer_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_Policy(self._two_record_output()),
            )

            with patch.object(
                store,
                "replace_all_atomically",
                side_effect=RuntimeError("injected atomic failure"),
            ) as replace_all:
                result = coordinator.finalize_task(_episode(store), _outcome())
            memories = store.all(project_key="project-a")

        self.assertEqual(result.writer_status, "failed_no_write")
        self.assertEqual(memories, [])
        replace_all.assert_called_once()

    def test_post_commit_verification_failure_recovers_committed_writer_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_Policy(self._two_record_output()),
            )
            original_replace = store.replace_all_atomically

            def commit_then_fail(memories, *, expected_revision):
                written_revision = original_replace(
                    memories,
                    expected_revision=expected_revision,
                )
                raise MemoryStorePostCommitError(
                    "injected post-commit verification failure",
                    expected_revision=written_revision,
                )

            with patch.object(store, "replace_all_atomically", side_effect=commit_then_fail):
                result = coordinator.finalize_task(_episode(store), _outcome())
            memories = store.all(project_key="project-a")

        self.assertEqual(result.writer_status, "committed")
        self.assertEqual(set(result.written_memory_ids), {item.id for item in memories})
        self.assertEqual(len(memories), 2)


if __name__ == "__main__":
    unittest.main()
