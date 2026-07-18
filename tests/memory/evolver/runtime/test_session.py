from __future__ import annotations

import tempfile
import unittest

from my_agent.memory.embedding_retrieval import EmbeddingRetriever
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact
from my_agent.memory.evolver.writing.contracts import ExperienceWriteResult
from my_agent.memory.experience.models import ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.training.contracts import AuthoritativeTaskOutcome, EvaluatorIdentity
from tests.memory.experience.fixtures import typed_experience


class _Encoder:
    model_revision = "embed-rev"
    tokenizer_revision = "tokenizer-rev"

    def encode_queries(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return ((1.0, 0.0),) * len(texts)

    def encode_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


class _Selector:
    def __init__(self) -> None:
        self.calls = 0

    def select(self, *, task, candidates, token_budget, max_items, context):
        del task, token_budget, max_items, context
        self.calls += 1
        return (candidates[0].memory_id,) if candidates else ()


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "rev", "sha256:" + "1" * 64, None,
        "tok", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


class EvolverTaskSessionTests(unittest.TestCase):
    def test_begin_task_retrieves_and_selects_once_then_freezes_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience("tip-a", "alpha tip", ExperienceTier.TIP))
            selector = _Selector()
            trace_events: list[tuple[str, dict[str, object]]] = []
            coordinator = EvolverCoordinator(
                store=store,
                project_key="/repo",
                policy_identity=_identity(),
                retriever=EmbeddingRetriever(_Encoder()),
                selector=selector,
                trace_sink=lambda event, payload: trace_events.append((event, payload)),
            )

            session = coordinator.begin_task(
                task="alpha task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            frozen = coordinator.context_for_session(session)
            store.add(typed_experience("skill-a", "new alpha skill", ExperienceTier.SKILL))
            still_frozen = coordinator.context_for_session(session)

        self.assertEqual(selector.calls, 1)
        self.assertEqual(session.selected_memory_ids, ("tip-a",))
        self.assertEqual(frozen.injected_text, still_frozen.injected_text)
        self.assertNotIn("skill-a", still_frozen.injected_text)
        started = next(payload for event, payload in trace_events if event == "memory.evolver_session_started")
        self.assertEqual(started["candidates"], [item.to_dict() for item in session.candidate_snapshot])
        self.assertEqual(
            started["candidate_snapshot_hash"],
            canonical_sha256(started["candidates"]),
        )

    def test_finalize_runs_writer_only_after_authoritative_outcome(self) -> None:
        calls: list[bool] = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            memory = typed_experience("tip-a", "alpha tip", ExperienceTier.TIP)
            store.add(memory)
            coordinator = EvolverCoordinator(
                store=store,
                project_key="/repo",
                policy_identity=_identity(),
                retriever=EmbeddingRetriever(_Encoder()),
                selector=_Selector(),
                writer=lambda episode, outcome: (
                    calls.append(outcome.outcome_finalized) or ExperienceWriteResult(saved=(memory,))
                ),
            )
            session = coordinator.begin_task(
                task="alpha task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            episode = AgentEpisodeArtifact(session, store.path, "finish", "done", ())
            outcome = AuthoritativeTaskOutcome(
                "task-1", "group-a", True, True, 1.0,
                EvaluatorIdentity("pytest", "8", canonical_sha256({"command": "pytest"})),
            )

            result = coordinator.finalize_task(episode, outcome)

        self.assertEqual(calls, [True])
        self.assertEqual(result.writer_status, "committed")
        self.assertEqual(result.written_memory_ids, ("tip-a",))

    def test_finalized_outcome_event_is_persisted_before_writer_event(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            coordinator = EvolverCoordinator(
                store=store,
                project_key="/repo",
                policy_identity=_identity(),
                retriever=EmbeddingRetriever(_Encoder()),
                writer=lambda episode, outcome: (
                    events.append((
                        "memory.writer_decision",
                        {
                            "trajectory_id": episode.session.trajectory_id,
                            "outcome_finalized": outcome.outcome_finalized,
                        },
                    ))
                    or ExperienceWriteResult()
                ),
                trace_sink=lambda event, payload: events.append((event, payload)),
            )
            session = coordinator.begin_task(
                task="task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            episode = AgentEpisodeArtifact(session, store.path, "finish", "done", ())
            outcome = AuthoritativeTaskOutcome(
                "task-1", "group-a", True, True, 1.0,
                EvaluatorIdentity("pytest", "8", canonical_sha256({"command": "pytest"})),
            )

            coordinator.finalize_task(episode, outcome)

        event_names = [event for event, _payload in events]
        outcome_index = event_names.index("memory.task_outcome_finalized")
        writer_index = event_names.index("memory.writer_decision")
        finalized_index = event_names.index("memory.evolver_task_finalized")
        self.assertLess(outcome_index, writer_index)
        self.assertLess(writer_index, finalized_index)
        outcome_payload = events[outcome_index][1]
        finalized_payload = events[finalized_index][1]
        self.assertEqual(finalized_payload["outcome_event_id"], outcome_payload["outcome_event_id"])

    def test_unfinalized_outcome_is_rejected_before_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            coordinator = EvolverCoordinator(
                store=store,
                project_key="/repo",
                policy_identity=_identity(),
                retriever=EmbeddingRetriever(_Encoder()),
            )
            session = coordinator.begin_task(
                task="task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            episode = AgentEpisodeArtifact(session, store.path, "finish", "done", ())
            outcome = AuthoritativeTaskOutcome(
                "task-1", "group-a", True, False, 0.0,
                EvaluatorIdentity("pytest", "8", canonical_sha256({"command": "pytest"})),
                outcome_finalized=False,
            )
            with self.assertRaisesRegex(ValueError, "outcome_finalized"):
                coordinator.finalize_task(episode, outcome)

    def test_stale_repository_revision_aborts_without_calling_writer(self) -> None:
        writer_calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience("tip-a", "alpha tip", ExperienceTier.TIP))
            coordinator = EvolverCoordinator(
                store=store,
                project_key="/repo",
                policy_identity=_identity(),
                retriever=EmbeddingRetriever(_Encoder()),
                selector=_Selector(),
                writer=lambda episode, outcome: (
                    writer_calls.append(episode.session.trajectory_id) or ExperienceWriteResult()
                ),
            )
            session = coordinator.begin_task(
                task="task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            store.add(typed_experience("skill-a", "concurrent skill", ExperienceTier.SKILL))
            episode = AgentEpisodeArtifact(session, store.path, "finish", "done", ())
            outcome = AuthoritativeTaskOutcome(
                "task-1", "group-a", True, True, 1.0,
                EvaluatorIdentity("pytest", "8", canonical_sha256({"command": "pytest"})),
            )

            result = coordinator.finalize_task(episode, outcome)

        self.assertEqual(writer_calls, [])
        self.assertEqual(result.writer_status, "failed_no_write")


if __name__ == "__main__":
    unittest.main()
