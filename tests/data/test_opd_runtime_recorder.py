from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from my_agent.llm.types import ChatResponse
from my_agent.memory.embedding_retrieval import EmbeddingRetriever
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact
from my_agent.memory.evolver.writer import ExperienceWriteStep
from my_agent.memory.experience_store import ExperienceStore
from my_agent.opd_data.export import (
    load_maintenance_evidence,
    load_repository_evidence,
    load_task_evidence,
    load_task_outcomes,
    load_runtime_exclusions,
    prepare_round_decisions,
)
from my_agent.policy.contracts import DecisionRequest, DecisionResponse
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.training.contracts import AuthoritativeTaskOutcome, EvaluatorIdentity
from my_agent.training.decision_log import DecisionEventContext, load_decision_events
from my_agent.training.role_views import (
    SELECTED_MEMORY_CONTEXT_HEADER,
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCall,
)


class _Encoder:
    model_revision = "embed-rev"
    tokenizer_revision = "tokenizer-rev"

    def encode_queries(self, texts):
        return ((1.0, 0.0),) * len(texts)

    def encode_documents(self, texts):
        return ((1.0, 0.0),) * len(texts)


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "revision", "sha256:" + "1" * 64, None,
        "tokenizer", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


class _RolePolicy:
    def __init__(self, *, invalid_roles: frozenset[str] = frozenset()) -> None:
        self.invalid_roles = invalid_roles

    def identity(self):
        return _identity()

    def render_prompt_hash(self, request):
        return canonical_sha256({
            "messages": [item.to_dict() for item in request.messages],
            "tools": [item.to_dict() for item in request.tools],
        })

    def generate_decision(self, request):
        raw_completion = {
            "selection": canonical_json_bytes({
                "selected_skills": [],
                "selected_tips": [],
                "selected_tools": [],
                "selected_trajectories": [],
                "reasoning": "no candidates",
            }).decode("utf-8"),
            "action": "done",
            "writing": "[]",
            "maintenance": "finish",
        }[request.role]
        if request.role in self.invalid_roles:
            raw_completion = "invalid"
        tool_calls = (
            (CanonicalToolCall(
                "call-finish",
                "finish",
                canonical_json_bytes({"summary": "no changes"}).decode("utf-8"),
            ),)
            if request.role == "maintenance" and request.role not in self.invalid_roles
            else ()
        )
        return DecisionResponse(
            raw_completion=raw_completion,
            prompt_token_ids=(10,),
            completion_token_ids=(20,),
            assistant_loss_mask=(1,),
            parsed_tool_calls=tool_calls,
            identity=self.identity(),
        )

    def chat_response_from_decision(self, response):
        return ChatResponse(content=response.raw_completion)

    def chat(self, *args, **kwargs):
        raise AssertionError("not used")


def _action_tool() -> CanonicalTool:
    parameters = {"type": "object", "properties": {}}
    return CanonicalTool(
        "finish",
        "finish task",
        canonical_json_bytes(parameters).decode("utf-8"),
        canonical_sha256(parameters),
    )


def _record_action(
    coordinator: EvolverCoordinator,
    session,
    *,
    messages: tuple[CanonicalMessage, ...] | None = None,
) -> None:
    action_request = DecisionRequest(
        role="action",
        purpose="fast_loop_evidence",
        messages=messages or (CanonicalMessage("user", "Fix the public task"),),
        tools=(_action_tool(),),
        max_new_tokens=32,
        temperature=1.0,
        top_p=0.95,
    )
    assert coordinator.decision_recorder is not None
    coordinator.decision_recorder.generate(
        action_request,
        context=DecisionEventContext(
            trajectory_id=session.trajectory_id,
            turn_index=0,
            step_index=0,
            task_id=session.task_id,
            task_group=session.task_group,
            stream_id=session.stream_id,
            memory_project_key=session.memory_project_key,
            run_id=session.trajectory_id,
            repository_revision=session.repository_revision,
            candidate_snapshot_hash=session.candidate_snapshot_hash,
        ),
    )


def _episode(session, store: ExperienceStore) -> AgentEpisodeArtifact:
    return AgentEpisodeArtifact(
        session=session,
        trace_path=store.path,
        stop_reason="assistant_final",
        final_answer="done",
        tool_history=(ExperienceWriteStep(0, "finish", {}, True, "done"),),
        task="Fix the public task",
    )


def _outcome() -> AuthoritativeTaskOutcome:
    return AuthoritativeTaskOutcome(
        "task-1",
        "group-a",
        True,
        True,
        1.0,
        EvaluatorIdentity(
            "pytest",
            "8",
            canonical_sha256({"command": "pytest"}),
        ),
    )


class OpdRuntimeRecorderTests(unittest.TestCase):
    def test_formal_coordinator_writes_all_exporter_evidence_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            store = ExperienceStore.from_dir(root / "memory")
            policy = _RolePolicy()
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=policy,
                retriever=EmbeddingRetriever(_Encoder()),
                dataset_dir=dataset,
                maintenance_interval_tasks=1,
            )
            session = coordinator.begin_task(
                task="Fix the public task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            _record_action(
                coordinator,
                session,
                messages=(
                    CanonicalMessage("system", "public repository context and project rules"),
                    CanonicalMessage(
                        "system",
                        SELECTED_MEMORY_CONTEXT_HEADER + "\n[mem-a | tip]\nprivate memory",
                    ),
                    CanonicalMessage("user", "Fix the public task"),
                ),
            )
            coordinator.finalize_task(_episode(session, store), _outcome())
            second = coordinator.begin_task(
                task="Fix the public task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-2",
                stream_id="stream-b",
            )
            _record_action(coordinator, second)
            coordinator.finalize_task(_episode(second, store), _outcome())

            decisions = load_decision_events(dataset / "decision_events.jsonl")
            tasks = load_task_evidence(dataset / "task_evidence.jsonl")
            outcomes = load_task_outcomes(dataset / "task_outcomes.jsonl")
            repositories = load_repository_evidence(dataset / "repository_events.jsonl")
            maintenance = load_maintenance_evidence(dataset / "maintenance_evidence.jsonl")
            prepared = prepare_round_decisions(
                collection_round=0,
                trainer_identity=_identity(),
                tasks=tasks,
                outcomes=outcomes,
                repositories=repositories,
                maintenance=maintenance,
                decision_events=decisions,
                attribution=(),
            )

        self.assertEqual({item.role for item in decisions}, {
            "selection", "action", "writing", "maintenance",
        })
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(len(repositories), 2)
        task_repository = next(item for item in repositories if item.stream_id == "stream-a")
        other_stream_repository = next(
            item for item in repositories if item.stream_id == "stream-b"
        )
        self.assertNotEqual(
            task_repository.snapshot.snapshot_hash,
            other_stream_repository.snapshot.snapshot_hash,
        )
        self.assertEqual(len(maintenance), 2)
        self.assertEqual(
            len([item for item in prepared.decisions if item.role == "selection"]),
            2,
        )
        self.assertEqual(tasks[0].trajectory.reward, outcomes[0].outcome.reward)
        stream_a_task = next(item for item in tasks if item.stream_id == "stream-a")
        self.assertEqual(stream_a_task.action_decisions[0].prefix_messages, (
            CanonicalMessage("system", "public repository context and project rules"),
            CanonicalMessage("user", "Fix the public task"),
        ))

    def test_invalid_roles_are_audited_without_breaking_finalize(self) -> None:
        cases = (
            (frozenset({"selection"}), True, 30, "selection"),
            (frozenset(), False, 30, "action"),
            (frozenset({"maintenance"}), True, 1, "maintenance"),
        )
        for invalid_roles, record_action, interval, expected_role in cases:
            with self.subTest(role=expected_role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset = root / "dataset"
                store = ExperienceStore.from_dir(root / "memory")
                policy = _RolePolicy(invalid_roles=invalid_roles)
                coordinator = EvolverCoordinator(
                    store=store,
                    project_key="project-a",
                    policy_identity=_identity(),
                    policy=policy,
                    retriever=EmbeddingRetriever(_Encoder()),
                    dataset_dir=dataset,
                    maintenance_interval_tasks=interval,
                )
                session = coordinator.begin_task(
                    task="Fix the public task",
                    task_id="task-1",
                    task_group="group-a",
                    trajectory_id="traj-1",
                    stream_id="stream-a",
                )
                if record_action:
                    _record_action(coordinator, session)

                result = coordinator.finalize_task(_episode(session, store), _outcome())
                exclusions = load_runtime_exclusions(dataset / "runtime_exclusions.jsonl")
                outcomes = load_task_outcomes(dataset / "task_outcomes.jsonl")
                task_records = load_task_evidence(dataset / "task_evidence.jsonl")
                task_count = coordinator.cadence_ledger.task_count(
                    stream_id="stream-a",
                    memory_project_key="project-a",
                )

            self.assertEqual(len(outcomes), 1)
            self.assertEqual(task_count, 1)
            self.assertIn(expected_role, {item.role for item in exclusions})
            if expected_role in {"selection", "action"}:
                self.assertEqual(task_records, ())
            if expected_role == "maintenance":
                self.assertEqual(result.maintenance_status, "aborted")


if __name__ == "__main__":
    unittest.main()
