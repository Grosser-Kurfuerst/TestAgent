from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from my_agent.llm.types import ChatResponse
from my_agent.memory.evolver.maintenance.cadence.ledger import (
    MAINTENANCE_HISTORY_FILENAME,
    CadenceLedger,
    load_formal_maintenance_history,
    stable_cadence_id,
)
from my_agent.memory.evolver.maintenance.cadence.scheduler import (
    MaintenanceCadenceScheduler,
)
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.maintenance.formal.agent import (
    FormalMaintenanceAgent,
    FormalMaintenanceResult,
)
from my_agent.memory.evolver.maintenance.formal.tools import (
    MaintenanceToolCommand,
    build_delete_operation,
)
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact, TaskEvolverSession
from my_agent.memory.evolver.maintenance.formal.transaction import (
    apply_formal_maintenance_operations,
)
from my_agent.memory.evolver.maintenance.legacy import transaction as maintenance_transaction
from my_agent.memory.evolver.writing.contracts import ExperienceWriteResult
from my_agent.memory.experience.models import ExperienceCreatedBy, ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from my_agent.policy.contracts import DecisionResponse
from my_agent.policy.identity import PolicyIdentity, canonical_json_bytes, canonical_sha256
from my_agent.training.contracts import AuthoritativeTaskOutcome, EvaluatorIdentity
from my_agent.training.decision_log import DecisionEventRecorder
from my_agent.training.role_views import CanonicalToolCall, TaskOutcomeRef
from tests.memory.experience.fixtures import typed_experience


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "revision", "sha256:" + "1" * 64, None,
        "tokenizer", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


def _completion(
    ledger: CadenceLedger,
    *,
    stream_id: str,
    project_key: str,
    task_id: str,
):
    return ledger.record_task_completion(
        stream_id=stream_id,
        memory_project_key=project_key,
        task_id=task_id,
        task_valid=True,
        outcome_finalized=True,
        writer_terminal_status="no_write",
        repository_revision_after_writer="rev-1",
        outcome=TaskOutcomeRef(
            task_id,
            "group-a",
            1.0,
            True,
            "pytest",
            "8",
            canonical_sha256({"evaluator": "pytest"}),
        ),
    )


class _Policy:
    def __init__(self, calls: list[CanonicalToolCall]) -> None:
        self.calls = list(calls)
        self.generated = 0

    def identity(self):
        return _identity()

    def render_prompt_hash(self, request):
        return canonical_sha256([item.to_dict() for item in request.messages])

    def generate_decision(self, request):
        del request
        self.generated += 1
        call = self.calls.pop(0)
        return DecisionResponse(
            raw_completion=f"<tool_call>{call.arguments_json}</tool_call>",
            prompt_token_ids=(1,),
            completion_token_ids=(2,),
            assistant_loss_mask=(1,),
            parsed_tool_calls=(call,),
            identity=self.identity(),
        )

    def chat_response_from_decision(self, response):
        del response
        return ChatResponse(content="", tool_calls=[])

    def chat(self, *args, **kwargs):
        raise AssertionError("not used")


def _call(name: str, arguments: dict) -> CanonicalToolCall:
    return CanonicalToolCall(
        f"call-{name}",
        name,
        canonical_json_bytes(arguments).decode("utf-8"),
    )


class CadenceLedgerTests(unittest.TestCase):
    def test_maintenance_trace_retains_abort_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            ledger = CadenceLedger(Path(tmp) / "evolver_state.sqlite3", interval_tasks=1)
            advance = _completion(
                ledger,
                stream_id="stream-a",
                project_key="project-a",
                task_id="task-1",
            )
            assert advance.cadence is not None
            traces = []
            revision = store.revision()
            scheduler = MaintenanceCadenceScheduler(
                store=store,
                ledger=ledger,
                history_path=Path(tmp) / MAINTENANCE_HISTORY_FILENAME,
                project_key="project-a",
                maintenance_enabled=True,
                run_maintenance=lambda **_kwargs: FormalMaintenanceResult(
                    status="aborted",
                    maintenance_id=advance.cadence.cadence_id,
                    plan_id="",
                    transaction_id="",
                    turns=3,
                    operation_ids=("op-1",),
                    before_revision=revision,
                    after_revision=revision,
                    error=(
                        "ValueError: formal maintenance requires exactly one tool call "
                        "per assistant turn"
                    ),
                ),
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            status = scheduler.run_or_reconcile(advance.cadence)

        cadence_trace = next(
            payload
            for event, payload in traces
            if event == "memory.evolver_maintenance_cadence"
        )
        self.assertEqual(status, "aborted")
        self.assertEqual(cadence_trace["turns"], 3)
        self.assertEqual(cadence_trace["operation_ids"], ["op-1"])
        self.assertIn("exactly one tool call", cadence_trace["error"])

    def test_cadence_context_contains_the_complete_boundary_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = CadenceLedger(Path(tmp) / "evolver_state.sqlite3", interval_tasks=3)
            advance = None
            for ordinal in range(1, 4):
                advance = _completion(
                    ledger,
                    stream_id="stream-a",
                    project_key="project-a",
                    task_id=f"task-{ordinal}",
                )
            assert advance is not None and advance.cadence is not None

            task_group, history = ledger.cadence_context(advance.cadence)

        self.assertEqual(task_group, "group-a")
        self.assertEqual(tuple(item.task_id for item in history), (
            "task-1", "task-2", "task-3",
        ))

    def test_duplicate_task_conflicting_terminal_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = CadenceLedger(Path(tmp) / "evolver_state.sqlite3", interval_tasks=30)
            _completion(
                ledger,
                stream_id="stream-a",
                project_key="project-a",
                task_id="task-1",
            )

            with self.assertRaisesRegex(ValueError, "conflicts"):
                ledger.record_task_completion(
                    stream_id="stream-a",
                    memory_project_key="project-a",
                    task_id="task-1",
                    task_valid=True,
                    outcome_finalized=True,
                    writer_terminal_status="committed",
                    repository_revision_after_writer="rev-2",
                    outcome=TaskOutcomeRef(
                        "task-1",
                        "group-a",
                        1.0,
                        True,
                        "pytest",
                        "8",
                        canonical_sha256({"evaluator": "pytest"}),
                    ),
                )

    def test_started_without_intent_retries_during_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            seed = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                writer=lambda episode, outcome: ExperienceWriteResult(),
                maintenance_interval_tasks=1,
            )
            advance = _completion(
                seed.cadence_ledger,
                stream_id="stream-a",
                project_key="project-a",
                task_id="task-1",
            )
            assert advance.cadence is not None
            seed.cadence_ledger.mark_started(advance.cadence.cadence_id)
            policy = _Policy([_call("finish", {"summary": "recovered"})])

            restarted = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                policy=policy,
                writer=lambda episode, outcome: ExperienceWriteResult(),
                maintenance_interval_tasks=1,
            )
            records = restarted.cadence_ledger.cadence_records(
                stream_id="stream-a",
                memory_project_key="project-a",
            )

        self.assertEqual(policy.generated, 1)
        self.assertEqual([record.status for record in records], ["committed"])

    def test_boundaries_duplicates_and_streams_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = CadenceLedger(Path(tmp) / "evolver_state.sqlite3", interval_tasks=30)
            boundaries = []
            for ordinal in range(1, 61):
                result = _completion(
                    ledger,
                    stream_id="stream-a",
                    project_key="project-a",
                    task_id=f"task-{ordinal}",
                )
                if result.cadence is not None:
                    boundaries.append((result.task_ordinal, result.cadence.cadence_index))
            duplicate = _completion(
                ledger,
                stream_id="stream-a",
                project_key="project-a",
                task_id="task-60",
            )
            for ordinal in range(1, 31):
                _completion(
                    ledger,
                    stream_id="stream-b",
                    project_key="project-a",
                    task_id=f"task-{ordinal}",
                )

            invalid = ledger.record_task_completion(
                stream_id="stream-a",
                memory_project_key="project-a",
                task_id="invalid-task",
                task_valid=False,
                outcome_finalized=True,
                writer_terminal_status="failed_no_write",
                repository_revision_after_writer="rev-1",
                outcome=TaskOutcomeRef(
                    "invalid-task",
                    "group-a",
                    0.0,
                    False,
                    "pytest",
                    "8",
                    canonical_sha256({"evaluator": "pytest"}),
                ),
            )
            stream_a = ledger.cadence_records(
                stream_id="stream-a", memory_project_key="project-a"
            )
            stream_b = ledger.cadence_records(
                stream_id="stream-b", memory_project_key="project-a"
            )
            stream_a_count = ledger.task_count(
                stream_id="stream-a", memory_project_key="project-a"
            )

        self.assertEqual(boundaries, [(30, 1), (60, 2)])
        self.assertFalse(duplicate.counted)
        self.assertEqual(duplicate.task_ordinal, 60)
        self.assertFalse(invalid.counted)
        self.assertEqual(stream_a_count, 60)
        self.assertEqual([item.boundary_ordinal for item in stream_a], [30, 60])
        self.assertEqual([item.boundary_ordinal for item in stream_b], [30])
        self.assertNotEqual(stream_a[0].cadence_id, stream_b[0].cadence_id)
        self.assertEqual(
            stream_a[1].cadence_id,
            stable_cadence_id(
                stream_id="stream-a",
                memory_project_key="project-a",
                interval_tasks=30,
                cadence_index=2,
            ),
        )

    def test_intent_before_commit_reuses_the_same_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            memory = typed_experience(
                "tip-a",
                "obsolete tip",
                ExperienceTier.TIP,
                project_key="project-a",
                created_by=ExperienceCreatedBy.WRITER,
            )
            store.add(memory)
            before = store.revision()
            operation = build_delete_operation(
                MaintenanceToolCommand(
                    "call-delete",
                    "delete",
                    {"source_ids": ["tip-a"], "reason": "obsolete"},
                ),
                repository_entries=(memory,),
            )
            cadence_id = stable_cadence_id(
                stream_id="stream-a",
                memory_project_key="project-a",
                interval_tasks=30,
                cadence_index=1,
            )
            history_path = Path(tmp) / MAINTENANCE_HISTORY_FILENAME
            with patch.object(store, "replace_all_atomically", side_effect=OSError("before commit")):
                with self.assertRaisesRegex(OSError, "before commit"):
                    apply_formal_maintenance_operations(
                        store=store,
                        cadence_id=cadence_id,
                        stream_id="stream-a",
                        expected_revision=before,
                        project_key="project-a",
                        operations=(operation,),
                        history_path=history_path,
                    )
            pending = load_formal_maintenance_history(history_path, cadence_id=cadence_id)
            self.assertIsNotNone(pending.intent)
            self.assertIsNone(pending.completion)
            assert pending.intent is not None
            self.assertEqual(
                pending.intent["transaction_id"],
                maintenance_transaction.formal_maintenance_transaction_id(
                    cadence_id=cadence_id,
                    plan_id=str(pending.intent["plan_id"]),
                ),
            )
            legacy_state = maintenance_transaction._load_maintenance_history_state(
                history_path,
                SimpleNamespace(plan_id="maint-" + "0" * 24),
            )
            self.assertIsNone(legacy_state.intent)
            self.assertIsNone(legacy_state.completion)
            self.assertIsNotNone(store.get("tip-a"))

            recovered = apply_formal_maintenance_operations(
                store=store,
                cadence_id=cadence_id,
                stream_id="stream-a",
                expected_revision=before,
                project_key="project-a",
                operations=(operation,),
                history_path=history_path,
            )
            repeated = apply_formal_maintenance_operations(
                store=store,
                cadence_id=cadence_id,
                stream_id="stream-a",
                expected_revision=before,
                project_key="project-a",
                operations=(operation,),
                history_path=history_path,
            )
            memory_after = store.get("tip-a")
            complete = load_formal_maintenance_history(history_path, cadence_id=cadence_id)

        self.assertEqual(recovered.status, "committed")
        self.assertEqual(repeated.plan_id, recovered.plan_id)
        self.assertEqual(repeated.transaction_id, recovered.transaction_id)
        self.assertIsNone(memory_after)
        self.assertIsNotNone(complete.completion)
        assert complete.intent is not None and complete.completion is not None
        self.assertEqual(
            complete.intent["transaction_id"],
            complete.completion["transaction_id"],
        )

    def test_repository_commit_is_reconciled_after_ledger_update_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)
            store.add(typed_experience(
                "tip-a",
                "obsolete tip",
                ExperienceTier.TIP,
                project_key="project-a",
                created_by=ExperienceCreatedBy.WRITER,
            ))
            policy = _Policy([
                _call("delete", {"source_ids": ["tip-a"], "reason": "obsolete"}),
                _call("finish", {"summary": "deleted obsolete tip"}),
            ])
            coordinator = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                writer=lambda episode, outcome: ExperienceWriteResult(),
                maintenance_interval_tasks=1,
            )
            coordinator.maintainer = FormalMaintenanceAgent(
                policy=policy,
                recorder=DecisionEventRecorder(policy=policy),
                store=store,
                project_key="project-a",
            )
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
            episode = AgentEpisodeArtifact(session, store.path, "finish", "done", ())
            outcome = AuthoritativeTaskOutcome(
                "task-1",
                "group-a",
                True,
                True,
                1.0,
                EvaluatorIdentity("pytest", "8", canonical_sha256({"command": "pytest"})),
            )
            with patch.object(
                coordinator.cadence_ledger,
                "mark_committed",
                side_effect=OSError("ledger update failed"),
            ):
                with self.assertRaisesRegex(OSError, "ledger update failed"):
                    coordinator.finalize_task(episode, outcome)

            due = coordinator.cadence_ledger.oldest_open_cadence(
                stream_id="stream-a", memory_project_key="project-a"
            )
            assert due is not None
            restarted = EvolverCoordinator(
                store=store,
                project_key="project-a",
                policy_identity=_identity(),
                writer=lambda episode, outcome: ExperienceWriteResult(),
                maintenance_interval_tasks=1,
            )
            records = restarted.cadence_ledger.cadence_records(
                stream_id="stream-a", memory_project_key="project-a"
            )
            memory_after = store.get("tip-a")

        self.assertEqual(policy.generated, 2)
        self.assertIsNone(memory_after)
        self.assertEqual([item.status for item in records], ["committed"])

    def test_concurrent_same_cadence_writes_one_intent_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_store = ExperienceStore.from_dir(tmp)
            memory = typed_experience(
                "tip-a",
                "obsolete tip",
                ExperienceTier.TIP,
                project_key="project-a",
                created_by=ExperienceCreatedBy.WRITER,
            )
            first_store.add(memory)
            before = first_store.revision()
            operation = build_delete_operation(
                MaintenanceToolCommand(
                    "call-delete",
                    "delete",
                    {"source_ids": ["tip-a"], "reason": "obsolete"},
                ),
                repository_entries=(memory,),
            )
            cadence_id = stable_cadence_id(
                stream_id="stream-a",
                memory_project_key="project-a",
                interval_tasks=30,
                cadence_index=1,
            )
            history_path = Path(tmp) / MAINTENANCE_HISTORY_FILENAME

            def apply_once():
                store = ExperienceStore.from_dir(tmp)
                return apply_formal_maintenance_operations(
                    store=store,
                    cadence_id=cadence_id,
                    stream_id="stream-a",
                    expected_revision=before,
                    project_key="project-a",
                    operations=(operation,),
                    history_path=history_path,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(executor.map(lambda _index: apply_once(), range(2)))
            records = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
            ]
            remaining = ExperienceStore.from_dir(tmp).get("tip-a")

        self.assertEqual([result.status for result in results], ["committed", "committed"])
        self.assertEqual(results[0].transaction_id, results[1].transaction_id)
        self.assertEqual([record["record_type"] for record in records], ["intent", "completion"])
        self.assertIsNone(remaining)

    def test_concurrent_finalizers_run_one_maintainer_for_the_due_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_store = ExperienceStore.from_dir(tmp)
            seed_store.add(typed_experience(
                "tip-a",
                "obsolete tip",
                ExperienceTier.TIP,
                project_key="project-a",
                created_by=ExperienceCreatedBy.WRITER,
            ))
            policies = [
                _Policy([
                    _call("delete", {"source_ids": ["tip-a"], "reason": "obsolete"}),
                    _call("finish", {"summary": "deleted obsolete tip"}),
                ])
                for _ in range(2)
            ]
            coordinators = []
            for policy in policies:
                store = ExperienceStore.from_dir(tmp)
                coordinator = EvolverCoordinator(
                    store=store,
                    project_key="project-a",
                    policy_identity=_identity(),
                    writer=lambda episode, outcome: ExperienceWriteResult(),
                    maintenance_interval_tasks=1,
                )
                coordinator.maintainer = FormalMaintenanceAgent(
                    policy=policy,
                    recorder=DecisionEventRecorder(policy=policy),
                    store=store,
                    project_key="project-a",
                )
                coordinators.append(coordinator)
            advance = _completion(
                coordinators[0].cadence_ledger,
                stream_id="stream-a",
                project_key="project-a",
                task_id="task-1",
            )
            assert advance.cadence is not None

            def maintain(index: int):
                return coordinators[index]._run_or_reconcile_cadence(advance.cadence)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(executor.map(maintain, (0, 1)))
            history = [
                json.loads(line)
                for line in (Path(tmp) / MAINTENANCE_HISTORY_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            cadence_rows = coordinators[0].cadence_ledger.cadence_records(
                stream_id="stream-a", memory_project_key="project-a"
            )

        self.assertEqual(sum(policy.generated for policy in policies), 2)
        self.assertEqual([row["record_type"] for row in history], ["intent", "completion"])
        self.assertEqual(results, ("committed", "committed"))
        self.assertEqual([row.status for row in cadence_rows], ["committed"])


if __name__ == "__main__":
    unittest.main()
