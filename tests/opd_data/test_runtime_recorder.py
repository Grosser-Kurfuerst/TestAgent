from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from my_agent.llm.types import ChatResponse
from my_agent.memory.embedding_retrieval import EmbeddingRetriever
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact
from my_agent.memory.evolver.writing.contracts import ExperienceWriteStep
from my_agent.memory.experience.models import ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from my_agent.opd_data.export import (
    load_action_execution_evidence,
    load_maintenance_evidence,
    load_maintenance_attempts,
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
from tests.memory.experience.fixtures import typed_experience


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
        self.requests: list[DecisionRequest] = []

    def identity(self):
        return _identity()

    def render_prompt_hash(self, request):
        return canonical_sha256({
            "messages": [item.to_dict() for item in request.messages],
            "tools": [item.to_dict() for item in request.tools],
        })

    def generate_decision(self, request):
        self.requests.append(request)
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


class _ActionToolPolicy(_RolePolicy):
    def generate_decision(self, request):
        response = super().generate_decision(request)
        if request.role != "action":
            return response
        call = CanonicalToolCall(
            "call-action-finish",
            "finish",
            canonical_json_bytes({}).decode("utf-8"),
        )
        return replace(
            response,
            raw_completion="finish",
            parsed_tool_calls=(call,),
        )


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
) -> object:
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
    return coordinator.decision_recorder.generate(
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


def _outcome(task_id: str = "task-1") -> AuthoritativeTaskOutcome:
    return AuthoritativeTaskOutcome(
        task_id,
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
    def test_empty_repository_keeps_task_evidence_without_selection_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            store = ExperienceStore.from_dir(root / "memory")
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_RolePolicy(),
                retriever=EmbeddingRetriever(_Encoder()),
                dataset_dir=dataset,
                maintenance_interval_tasks=30,
            )
            session = coordinator.begin_task(
                task="Fix the public task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            action = _record_action(coordinator, session)
            coordinator.finalize_task(_episode(session, store), _outcome())

            decisions = load_decision_events(dataset / "decision_events.jsonl")
            tasks = load_task_evidence(dataset / "task_evidence.jsonl")
            exclusions = load_runtime_exclusions(dataset / "runtime_exclusions.jsonl")
            prepared = prepare_round_decisions(
                collection_round=0,
                trainer_identity=_identity(),
                tasks=tasks,
                outcomes=load_task_outcomes(dataset / "task_outcomes.jsonl"),
                repositories=load_repository_evidence(dataset / "repository_events.jsonl"),
                maintenance=(),
                decision_events=decisions,
                attribution=(),
            )

        self.assertEqual(session.candidate_snapshot, ())
        self.assertEqual({event.role for event in decisions}, {"action", "writing"})
        self.assertEqual(len(tasks), 1)
        self.assertIsNone(tasks[0].selection_decision_id)
        self.assertIn(action.decision_id, tasks[0].source_decision_ids)
        self.assertFalse(any(item.role == "selection" for item in exclusions))
        self.assertFalse(any(item.role == "selection" for item in prepared.decisions))

    def test_action_execution_is_joined_to_decision_and_materialized_on_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            store = ExperienceStore.from_dir(root / "memory")
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_ActionToolPolicy(),
                retriever=EmbeddingRetriever(_Encoder()),
                dataset_dir=dataset,
                maintenance_interval_tasks=30,
            )
            session = coordinator.begin_task(
                task="Fix the public task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            logged = _record_action(coordinator, session)
            assert coordinator.runtime_evidence_recorder is not None
            coordinator.runtime_evidence_recorder.record_action_execution(
                session=session,
                decision_id=logged.decision_id,
                turn_index=0,
                step_index=0,
                call_index=0,
                call_id="call-action-finish",
                tool_name="finish",
                arguments={},
                ok=True,
                blocked=False,
                error_code="",
                output="done",
            )
            coordinator.finalize_task(_episode(session, store), _outcome())
            records = load_action_execution_evidence(
                dataset / "tool_execution_evidence.jsonl"
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.decision_id, logged.decision_id)
        self.assertEqual(record.task_ordinal, 1)
        self.assertTrue(record.ok)
        self.assertEqual(record.execution_evidence_id, canonical_sha256(record._payload()))

    def test_action_execution_conflicting_idempotency_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = EvolverCoordinator(
                store=ExperienceStore.from_dir(root / "memory"),
                project_key="project-a",
                policy_identity=_identity(),
                policy=_ActionToolPolicy(),
                retriever=EmbeddingRetriever(_Encoder()),
                dataset_dir=root / "dataset",
                maintenance_interval_tasks=30,
            )
            session = coordinator.begin_task(
                task="Fix the public task",
                task_id="task-1",
                task_group="group-a",
                trajectory_id="traj-1",
                stream_id="stream-a",
            )
            logged = _record_action(coordinator, session)
            recorder = coordinator.runtime_evidence_recorder
            assert recorder is not None
            common = {
                "session": session,
                "decision_id": logged.decision_id,
                "turn_index": 0,
                "step_index": 0,
                "call_index": 0,
                "call_id": "call-action-finish",
                "tool_name": "finish",
                "arguments": {},
                "blocked": False,
                "error_code": "",
                "output": "done",
            }
            recorder.record_action_execution(ok=True, **common)
            with self.assertRaisesRegex(ValueError, "idempotency key conflicts"):
                recorder.record_action_execution(ok=False, **common)

    def test_interrupted_partial_maintenance_attempt_is_excluded_before_retry(self) -> None:
        class CrashAfterLookupPolicy(_RolePolicy):
            def __init__(self) -> None:
                super().__init__()
                self.maintenance_calls = 0

            def generate_decision(self, request):
                if request.role != "maintenance":
                    return super().generate_decision(request)
                self.maintenance_calls += 1
                if self.maintenance_calls > 1:
                    raise SystemExit("simulated crash after lookup")
                response = super().generate_decision(request)
                lookup = CanonicalToolCall(
                    "call-lookup",
                    "lookup",
                    canonical_json_bytes({"query": "public"}).decode("utf-8"),
                )
                return replace(
                    response,
                    raw_completion="lookup",
                    parsed_tool_calls=(lookup,),
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            store = ExperienceStore.from_dir(root / "memory")
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=CrashAfterLookupPolicy(),
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
            _record_action(coordinator, session)
            with self.assertRaisesRegex(SystemExit, "after lookup"):
                coordinator.finalize_task(_episode(session, store), _outcome())

            restarted = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_RolePolicy(),
                dataset_dir=dataset,
                maintenance_interval_tasks=1,
            )
            attempts = load_maintenance_attempts(dataset / "maintenance_attempts.jsonl")
            exclusions = load_runtime_exclusions(dataset / "runtime_exclusions.jsonl")
            maintenance = load_maintenance_evidence(dataset / "maintenance_evidence.jsonl")
            decisions = load_decision_events(dataset / "decision_events.jsonl")
            prepared = prepare_round_decisions(
                collection_round=0,
                trainer_identity=_identity(),
                tasks=load_task_evidence(dataset / "task_evidence.jsonl"),
                outcomes=load_task_outcomes(dataset / "task_outcomes.jsonl"),
                repositories=load_repository_evidence(dataset / "repository_events.jsonl"),
                maintenance=maintenance,
                decision_events=decisions,
                attribution=(),
            )
            records = restarted.cadence_ledger.cadence_records(
                stream_id="stream-a",
                memory_project_key="project-a",
            )

        self.assertEqual([item.status for item in attempts], [
            "started", "abandoned", "started", "noop",
        ])
        self.assertIn(
            "maintenance_attempt_abandoned_after_interruption",
            {item.reason for item in exclusions},
        )
        self.assertEqual(len(maintenance), 1)
        self.assertEqual(
            len([item for item in prepared.decisions if item.role == "maintenance"]),
            1,
        )
        self.assertEqual([record.status for record in records], ["committed"])

    def test_completion_recovery_rebuilds_missing_maintenance_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            store = ExperienceStore.from_dir(root / "memory")
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_RolePolicy(),
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
            _record_action(coordinator, session)
            assert coordinator.runtime_evidence_recorder is not None
            with mock.patch.object(
                coordinator.runtime_evidence_recorder,
                "finish_maintenance",
                side_effect=SystemExit("simulated crash before evidence finalize"),
            ):
                with self.assertRaisesRegex(SystemExit, "before evidence finalize"):
                    coordinator.finalize_task(_episode(session, store), _outcome())

            restart_policy = _RolePolicy()
            restarted = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=restart_policy,
                dataset_dir=dataset,
                maintenance_interval_tasks=1,
            )
            attempts = load_maintenance_attempts(dataset / "maintenance_attempts.jsonl")
            maintenance = load_maintenance_evidence(dataset / "maintenance_evidence.jsonl")
            decisions = load_decision_events(dataset / "decision_events.jsonl")
            records = restarted.cadence_ledger.cadence_records(
                stream_id="stream-a",
                memory_project_key="project-a",
            )

        self.assertEqual([item.status for item in attempts], ["started", "noop"])
        self.assertEqual(len(maintenance), 1)
        self.assertEqual(len([item for item in decisions if item.role == "maintenance"]), 1)
        self.assertFalse(any(request.role == "maintenance" for request in restart_policy.requests))
        self.assertEqual([record.status for record in records], ["committed"])

    def test_maintenance_uses_the_complete_cadence_outcome_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            store = ExperienceStore.from_dir(root / "memory")
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=_RolePolicy(),
                retriever=EmbeddingRetriever(_Encoder()),
                dataset_dir=dataset,
                maintenance_interval_tasks=2,
            )
            for ordinal in range(1, 3):
                session = coordinator.begin_task(
                    task=f"Fix public task {ordinal}",
                    task_id=f"task-{ordinal}",
                    task_group="group-a",
                    trajectory_id=f"traj-{ordinal}",
                    stream_id="stream-a",
                )
                _record_action(coordinator, session)
                coordinator.finalize_task(
                    _episode(session, store),
                    _outcome(f"task-{ordinal}"),
                )
            outcomes = load_task_outcomes(dataset / "task_outcomes.jsonl")
            maintenance = load_maintenance_evidence(dataset / "maintenance_evidence.jsonl")
            outcome_ids = {outcome.outcome.task_id: outcome.outcome_id for outcome in outcomes}

        self.assertEqual(len(maintenance), 1)
        self.assertEqual(maintenance[0].outcome_ids, (
            outcome_ids["task-1"],
            outcome_ids["task-2"],
        ))

    def test_maintenance_evidence_persists_pairwise_redundancy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            store = ExperienceStore.from_dir(root / "memory")
            store.add(typed_experience(
                "tip-a",
                "inspect duplicate public failures",
                ExperienceTier.TIP,
                project_key="project-a",
            ))
            store.add(typed_experience(
                "tip-b",
                "inspect duplicate public errors",
                ExperienceTier.TIP,
                project_key="project-a",
            ))
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
            _record_action(coordinator, session)
            coordinator.finalize_task(_episode(session, store), _outcome())
            maintenance = load_maintenance_evidence(dataset / "maintenance_evidence.jsonl")

        self.assertEqual(len(maintenance), 1)
        self.assertEqual(len(maintenance[0].redundancy_diagnostics), 1)
        diagnostic = maintenance[0].redundancy_diagnostics[0]
        self.assertEqual(
            (diagnostic.left_memory_id, diagnostic.right_memory_id),
            ("tip-a", "tip-b"),
        )
        self.assertGreater(diagnostic.score, 0.0)

    def test_formal_coordinator_writes_all_exporter_evidence_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            store = ExperienceStore.from_dir(root / "memory")
            store.add(typed_experience(
                "tip-a",
                "Run focused tests before the full suite.",
                ExperienceTier.TIP,
                project_key="project-a",
            ))
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
            0,
        )
        self.assertEqual(
            len([
                item
                for item in prepared.exclusions
                if item["role"] == "selection"
                and item["reason"] == "missing_candidate_attribution"
            ]),
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
                if expected_role == "selection":
                    store.add(typed_experience(
                        "tip-a",
                        "Run focused tests before the full suite.",
                        ExperienceTier.TIP,
                        project_key="project-a",
                    ))
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
