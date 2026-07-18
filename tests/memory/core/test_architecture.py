from __future__ import annotations

# ruff: noqa: E402 - tests add the src layout before importing project modules

import ast
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from tests._path import add_src_to_path

add_src_to_path()

import my_agent.memory as memory_api
import my_agent.memory.compression as legacy_compression_api
import my_agent.memory.disabled as disabled_memory_api
import my_agent.memory.embedding_cache as legacy_embedding_cache_api
import my_agent.memory.embedding_index as legacy_embedding_index_api
import my_agent.memory.embedding_retrieval as legacy_embedding_retrieval_api
import my_agent.memory.experience as experience_api
import my_agent.memory.experience_attribution as legacy_attribution_api
import my_agent.memory.experience_retrieval as legacy_retrieval_api
import my_agent.memory.experience_store as legacy_repository_api
import my_agent.memory.noop as legacy_noop_api
import my_agent.memory.short_term as short_term_api
import my_agent.memory.evolver as evolver_api
import my_agent.memory.evolver.artifacts as legacy_maintenance_artifacts_api
import my_agent.memory.evolver.attribution as legacy_offline_attribution_api
import my_agent.memory.evolver.attribution_export as legacy_attribution_io_api
import my_agent.memory.evolver.attribution_schema as legacy_attribution_schema_api
import my_agent.memory.evolver.cadence_ledger as legacy_cadence_ledger_api
import my_agent.memory.evolver.cadence_schema as legacy_cadence_schema_api
import my_agent.memory.evolver.contracts as legacy_maintenance_contracts_api
import my_agent.memory.evolver.dataset_scoring as legacy_dataset_scoring_api
import my_agent.memory.evolver.maintenance_agent as legacy_maintenance_agent_api
import my_agent.memory.evolver.maintenance_prompt as legacy_maintenance_prompt_api
import my_agent.memory.evolver.maintenance_tools as legacy_maintenance_tools_api
import my_agent.memory.evolver.planner as legacy_maintenance_planner_api
import my_agent.memory.evolver.paper_attribution as legacy_attribution_equations_api
import my_agent.memory.evolver.repository_rules as legacy_repository_rules_api
import my_agent.memory.evolver.repository_reducer as legacy_repository_reducer_api
import my_agent.memory.evolver.serialization as legacy_serialization_api
import my_agent.memory.evolver.service as legacy_maintenance_service_api
import my_agent.memory.evolver.formal_writer as legacy_formal_writer_api
import my_agent.memory.evolver.selector as legacy_selector_api
import my_agent.memory.evolver.selector_prompt as legacy_selector_prompt_api
import my_agent.memory.evolver.transaction as legacy_maintenance_transaction_api
import my_agent.memory.evolver.trace_join as legacy_trace_join_api
import my_agent.memory.evolver.types as legacy_models_api
import my_agent.memory.evolver.usage_log as legacy_usage_log_api
import my_agent.memory.evolver.validation as legacy_maintenance_validation_api
import my_agent.memory.evolver.writer as legacy_writer_api
from my_agent.config import SUPPORTED_MEMORY_EVOLVER_MODES
from my_agent.memory.experience import attribution as experience_attribution_api
from my_agent.memory.experience import repository as experience_repository_api
from my_agent.memory.experience import repository_rules as experience_repository_rules_api
from my_agent.memory.experience import serialization as experience_serialization_api
from my_agent.memory.experience.retrieval import embedding as embedding_api
from my_agent.memory.experience.retrieval import embedding_cache as embedding_cache_api
from my_agent.memory.experience.retrieval import embedding_index as embedding_index_api
from my_agent.memory.experience.retrieval import lexical as lexical_api
from my_agent.memory.short_term import compression as short_term_compression_api
from my_agent.memory.evolver import maintenance as maintenance_api
from my_agent.memory.evolver.maintenance import contracts as maintenance_contracts_api
from my_agent.memory.evolver.maintenance import repository_reducer as maintenance_reducer_api
from my_agent.memory.evolver.maintenance.cadence import ledger as cadence_ledger_api
from my_agent.memory.evolver.maintenance.cadence import schema as cadence_schema_api
from my_agent.memory.evolver.maintenance.formal import agent as maintenance_agent_api
from my_agent.memory.evolver.maintenance.formal import history as formal_history_api
from my_agent.memory.evolver.maintenance.formal import prompt as maintenance_prompt_api
from my_agent.memory.evolver.maintenance.formal import tools as maintenance_tools_api
from my_agent.memory.evolver.maintenance.legacy import artifacts as maintenance_artifacts_api
from my_agent.memory.evolver.maintenance.legacy import history_io as legacy_history_api
from my_agent.memory.evolver.maintenance.legacy import history_state as legacy_history_state_api
from my_agent.memory.evolver.maintenance.legacy import planner as maintenance_planner_api
from my_agent.memory.evolver.maintenance.legacy import service as maintenance_service_api
from my_agent.memory.evolver.maintenance.legacy import transaction as maintenance_transaction_api
from my_agent.memory.evolver.maintenance.legacy import validation as maintenance_validation_api
from my_agent.memory.evolver.selection import formal as formal_selection_api
from my_agent.memory.evolver.selection import legacy as weighted_selection_api
from my_agent.memory.evolver.selection import contracts as selection_contracts_api
from my_agent.memory.evolver.writing import contracts as writing_contracts_api
from my_agent.memory.evolver.writing import formal as formal_writing_api
from my_agent.memory.evolver.writing import legacy as legacy_writing_api
from my_agent.memory.evolver import ExperienceWriteResult
from my_agent.memory.evolver.attribution_schema import PAPER_ATTRIBUTION_SCHEMA_VERSION
from my_agent.memory.evolver.cadence_ledger import CADENCE_SCHEMA_VERSION
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.contracts import MaintenanceOperation
from my_agent.memory.evolver.repository_rules import experience_memories_revision
from my_agent.memory.evolver.serialization import experience_canonical_json
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact
from my_agent.memory.evolver.types import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceTier,
    TipPayload,
)
from my_agent.memory.types import MemoryScope, content_fingerprint
from my_agent.memory.experience_retrieval import ExperienceRetrievalMetrics
from my_agent.memory.experience_store import ExperienceStore
import my_agent.opd_data.attribution as attribution_api
from my_agent.opd_data.attribution import equations as attribution_equations_api
from my_agent.opd_data.attribution import io as attribution_io_api
from my_agent.opd_data.attribution import round as attribution_round_api
from my_agent.opd_data.attribution import schema as attribution_schema_api
from my_agent.opd_data.legacy import attribution as offline_attribution_api
from my_agent.opd_data.legacy import dataset_scoring as dataset_scoring_api
from my_agent.opd_data.legacy import trace_join as trace_join_api
from my_agent.opd_data.legacy import usage_log as usage_log_api
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.training.contracts import (
    DECISION_EVENT_SCHEMA_VERSION,
    AuthoritativeTaskOutcome,
    EvaluatorIdentity,
)


SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "my_agent"
TESTS_ROOT = Path(__file__).resolve().parents[3] / "tests"


def _contract_memory() -> ExperienceMemory:
    content = "Always rerun the focused parser test."
    return ExperienceMemory(
        id="exp-contract-tip",
        content=content,
        tier=ExperienceTier.TIP,
        payload=TipPayload(
            category="testing",
            severity="warning",
            trigger="parser behavior changes",
        ),
        scope=MemoryScope.PROJECT,
        project_key="repo:contract",
        created_at=datetime(2026, 7, 18, 9, 30, tzinfo=timezone.utc),
        token_count=9,
        fingerprint=content_fingerprint(content),
        source_task="task-contract",
        run_id="run-contract",
        stream_id="stream-contract",
        created_by=ExperienceCreatedBy.WRITER,
        writer_confidence=0.875,
    )


def _imported_modules(path: Path, *, source_root: Path = SRC_ROOT) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    relative = path.relative_to(source_root).with_suffix("")
    module_parts = ("my_agent", *relative.parts)
    package_parts = module_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                ascend = node.level - 1
                if ascend > len(package_parts):
                    continue
                base_parts = package_parts[: len(package_parts) - ascend]
            else:
                base_parts = ()
            imported_parts = (*base_parts, *((node.module or "").split(".") if node.module else ()))
            imported_module = ".".join(imported_parts)
            if imported_module:
                imported.add(imported_module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported.add(".".join((*imported_parts, alias.name)))
    return imported


def _policy_identity() -> PolicyIdentity:
    return PolicyIdentity(
        base_model="model",
        base_revision="model-revision",
        checkpoint_hash="sha256:" + "1" * 64,
        adapter_hash=None,
        tokenizer_revision="tokenizer-revision",
        tokenizer_hash="sha256:" + "2" * 64,
        chat_template_hash="sha256:" + "3" * 64,
    )


class _OrderRetriever:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.last_metrics = ExperienceRetrievalMetrics()

    def retrieve_candidates(self, query, *, store, project_key, top_k_per_tier):
        del query, project_key, top_k_per_tier
        self.events.append("retrieve")
        self.last_metrics = ExperienceRetrievalMetrics(
            repository_revision=store.revision(),
            retrieval_backend="embedding_cosine",
        )
        return ()


class _OrderSelector:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def select(self, *, task, candidates, token_budget, max_items, context):
        del task, candidates, token_budget, max_items, context
        self.events.append("select")
        return ()


class MemoryPublicContractTests(unittest.TestCase):
    def test_memory_and_evolver_barrels_preserve_type_identity(self) -> None:
        self.assertIs(experience_api.ExperienceMemory, ExperienceMemory)
        self.assertIs(experience_api.ExperienceTier, ExperienceTier)
        self.assertIs(memory_api.ExperienceMemory, ExperienceMemory)
        self.assertIs(memory_api.ExperienceTier, ExperienceTier)
        self.assertIs(evolver_api.ExperienceMemory, ExperienceMemory)
        self.assertIs(evolver_api.ExperienceTier, ExperienceTier)
        self.assertIs(evolver_api.TipPayload, TipPayload)

    def test_legacy_experience_modules_are_thin_identity_preserving_facades(self) -> None:
        self.assertIs(legacy_models_api.ExperienceMemory, experience_api.ExperienceMemory)
        self.assertIs(
            legacy_serialization_api.experience_to_dict,
            experience_serialization_api.experience_to_dict,
        )
        self.assertIs(
            legacy_repository_rules_api.experience_memories_revision,
            experience_repository_rules_api.experience_memories_revision,
        )
        self.assertIs(
            legacy_attribution_api.replace_experience_attribution,
            experience_attribution_api.replace_experience_attribution,
        )
        self.assertIs(legacy_repository_api, experience_repository_api)

    def test_retrieval_and_short_term_facades_preserve_identity(self) -> None:
        self.assertIs(legacy_retrieval_api, lexical_api)
        self.assertIs(legacy_embedding_retrieval_api, embedding_api)
        self.assertIs(legacy_embedding_cache_api, embedding_cache_api)
        self.assertIs(legacy_embedding_index_api, embedding_index_api)
        self.assertIs(legacy_compression_api, short_term_compression_api)
        self.assertNotIn("ExperienceRetriever", memory_api.__all__)
        self.assertIs(
            short_term_api.MemoryCompressor,
            short_term_compression_api.MemoryCompressor,
        )
        self.assertIs(legacy_noop_api, disabled_memory_api)
        self.assertIs(
            legacy_noop_api.NoopMemoryManager,
            disabled_memory_api.DisabledMemoryManager,
        )

    def test_selection_and_writing_facades_preserve_identity(self) -> None:
        self.assertIs(
            legacy_selector_api.ExperienceSelector,
            weighted_selection_api.ExperienceSelector,
        )
        self.assertIs(
            legacy_selector_api.SelectionResult,
            selection_contracts_api.SelectionResult,
        )
        self.assertIs(
            legacy_selector_prompt_api.LLMTaskSelectionPolicy,
            formal_selection_api.LLMTaskSelectionPolicy,
        )
        self.assertIs(
            legacy_writer_api.ExperienceWriter,
            legacy_writing_api.ExperienceWriter,
        )
        self.assertIs(
            legacy_writer_api.ExperienceWriteResult,
            writing_contracts_api.ExperienceWriteResult,
        )
        self.assertIs(
            legacy_formal_writer_api.FormalExperienceWriter,
            formal_writing_api.FormalExperienceWriter,
        )

    def test_formal_selection_and_writing_do_not_import_legacy_policies(self) -> None:
        formal_selection_imports = _imported_modules(
            SRC_ROOT / "memory" / "evolver" / "selection" / "formal.py"
        )
        formal_writing_imports = _imported_modules(
            SRC_ROOT / "memory" / "evolver" / "writing" / "formal.py"
        )

        self.assertNotIn(
            "my_agent.memory.evolver.selection.legacy",
            formal_selection_imports,
        )
        self.assertNotIn(
            "my_agent.memory.evolver.writing.legacy",
            formal_writing_imports,
        )
        self.assertIn(
            "my_agent.memory.evolver.writing.persistence.ExperienceRepositoryWriter",
            formal_writing_imports,
        )

    def test_all_selection_strategies_share_the_policy_protocol(self) -> None:
        policy = selection_contracts_api.TaskSelectionPolicy

        self.assertTrue(
            issubclass(weighted_selection_api.LegacyWeightedSelectionPolicy, policy)
        )
        self.assertTrue(
            issubclass(formal_selection_api.LLMTaskSelectionPolicy, policy)
        )
        self.assertTrue(
            issubclass(formal_selection_api.SimilarityTaskSelectionPolicy, policy)
        )
        self.assertTrue(
            issubclass(formal_selection_api.EmptyTaskSelectionPolicy, policy)
        )

    def test_maintenance_facade_preserves_contract_identity(self) -> None:
        self.assertIs(maintenance_api.MaintenanceOperation, MaintenanceOperation)
        self.assertIs(
            maintenance_api.build_maintenance_plan,
            maintenance_service_api.build_maintenance_plan,
        )
        self.assertIs(
            maintenance_api.apply_maintenance_plan,
            maintenance_transaction_api.apply_maintenance_plan,
        )

    def test_maintenance_compatibility_modules_preserve_module_identity(self) -> None:
        aliases = (
            (legacy_maintenance_artifacts_api, maintenance_artifacts_api),
            (legacy_cadence_ledger_api, cadence_ledger_api),
            (legacy_cadence_schema_api, cadence_schema_api),
            (legacy_maintenance_contracts_api, maintenance_contracts_api),
            (legacy_maintenance_agent_api, maintenance_agent_api),
            (legacy_maintenance_prompt_api, maintenance_prompt_api),
            (legacy_maintenance_tools_api, maintenance_tools_api),
            (legacy_maintenance_planner_api, maintenance_planner_api),
            (legacy_repository_reducer_api, maintenance_reducer_api),
            (legacy_maintenance_service_api, maintenance_service_api),
            (legacy_maintenance_transaction_api, maintenance_transaction_api),
            (legacy_maintenance_validation_api, maintenance_validation_api),
        )
        for legacy_module, new_module in aliases:
            with self.subTest(legacy=legacy_module.__name__, new=new_module.__name__):
                self.assertIs(legacy_module, new_module)

    def test_attribution_compatibility_modules_preserve_module_identity(self) -> None:
        aliases = (
            (legacy_attribution_schema_api, attribution_schema_api),
            (legacy_attribution_equations_api, attribution_equations_api),
            (legacy_attribution_io_api, attribution_io_api),
            (legacy_offline_attribution_api, offline_attribution_api),
            (legacy_usage_log_api, usage_log_api),
            (legacy_trace_join_api, trace_join_api),
            (legacy_dataset_scoring_api, dataset_scoring_api),
        )
        for legacy_module, new_module in aliases:
            with self.subTest(legacy=legacy_module.__name__, new=new_module.__name__):
                self.assertIs(legacy_module, new_module)

        self.assertIs(attribution_api.CandidateExposure, attribution_schema_api.CandidateExposure)
        self.assertIs(
            attribution_api.PaperAttributionRecord,
            attribution_schema_api.PaperAttributionRecord,
        )
        self.assertIs(
            attribution_api.compute_round_attribution,
            attribution_equations_api.compute_round_attribution,
        )
        self.assertIs(
            attribution_api.build_round_attribution,
            attribution_round_api.build_round_attribution,
        )
        self.assertNotIn("MemoryAttributionRecord", evolver_api.__all__)

    def test_memory_and_evolver_barrels_only_export_stable_contracts(self) -> None:
        self.assertLessEqual(len(memory_api.__all__), 25)
        self.assertLessEqual(len(evolver_api.__all__), 30)
        self.assertTrue({
            "MemoryManager",
            "MemoryService",
            "MemoryEntry",
            "ExperienceMemory",
        }.issubset(memory_api.__all__))
        self.assertTrue({
            "EvolverRuntime",
            "TaskEvolverSession",
            "ExperienceCandidate",
            "ExperienceWriteResult",
        }.issubset(evolver_api.__all__))
        self.assertTrue({
            "ExperienceStore",
            "ExperienceRetriever",
            "MemoryCompressor",
        }.isdisjoint(memory_api.__all__))
        self.assertTrue({
            "MemoryAttributionRecord",
            "UsageLogger",
            "apply_maintenance_plan",
            "build_maintenance_plan",
            "ExperienceWriter",
        }.isdisjoint(evolver_api.__all__))

    def test_formal_maintenance_package_does_not_import_legacy_policy(self) -> None:
        formal_root = SRC_ROOT / "memory" / "evolver" / "maintenance" / "formal"
        forbidden_prefixes = (
            "my_agent.memory.evolver.maintenance.legacy",
            "my_agent.memory.evolver.planner",
            "my_agent.memory.evolver.service",
        )
        violations: list[str] = []
        for path in formal_root.rglob("*.py"):
            for imported in _imported_modules(path):
                if any(
                    imported == prefix or imported.startswith(prefix + ".")
                    for prefix in forbidden_prefixes
                ):
                    violations.append(f"{path.relative_to(SRC_ROOT)} -> {imported}")
        self.assertEqual(violations, [])

    def test_importing_formal_maintenance_does_not_load_legacy_modules(self) -> None:
        script = (
            "import json, sys; "
            f"sys.path.insert(0, {str(SRC_ROOT.parent)!r}); "
            "import my_agent.memory.evolver.maintenance.formal.agent; "
            "legacy=[name for name in sys.modules if "
            "name.startswith('my_agent.memory.evolver.maintenance.legacy') or "
            "name in {'my_agent.memory.evolver.planner', "
            "'my_agent.memory.evolver.service', "
            "'my_agent.memory.evolver.transaction', "
            "'my_agent.memory.evolver.validation'}]; "
            "print(json.dumps(sorted(legacy)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(json.loads(completed.stdout), [])

    def test_formal_and_legacy_history_schemas_are_isolated(self) -> None:
        repository_revision = experience_memories_revision(())
        legacy_plan = maintenance_api.build_maintenance_plan(
            entries=(),
            attribution={},
            repository_revision=repository_revision,
            project_key="repo:contract",
            as_of=datetime(2026, 7, 18, 9, 30, tzinfo=timezone.utc),
        )
        legacy_record = legacy_history_state_api._pre_commit_history_record(
            legacy_plan,
            before_revision=repository_revision,
            backup_path="",
            stage="validation",
            error="contract test",
            should_retry=False,
        )
        cadence_id = canonical_sha256({"cadence": "contract"})
        plan_id = formal_history_api.formal_maintenance_plan_id(
            cadence_id=cadence_id,
            before_revision=repository_revision,
            expected_after_revision=repository_revision,
            operations=(),
        )
        formal_record = formal_history_api.formal_intent_record(
            cadence_id=cadence_id,
            plan_id=plan_id,
            transaction_id=formal_history_api.formal_maintenance_transaction_id(
                cadence_id=cadence_id,
                plan_id=plan_id,
            ),
            stream_id="stream-contract",
            memory_project_key="repo:contract",
            before_revision=repository_revision,
            expected_after_revision=repository_revision,
            operations=(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "maintenance_history.jsonl"
            history_path.write_text(
                "\n".join((json.dumps(legacy_record), json.dumps(formal_record))) + "\n",
                encoding="utf-8",
            )
            formal_state = formal_history_api.load_formal_maintenance_history(
                history_path,
                cadence_id=cadence_id,
            )
            legacy_state = legacy_history_api._load_maintenance_history_state(
                history_path,
                legacy_plan,
            )

        self.assertEqual(formal_state.intent, formal_record)
        self.assertIsNone(formal_state.completion)
        self.assertIsNone(legacy_state.intent)
        self.assertEqual(legacy_state.completion, legacy_record)

    def test_coordinator_cadence_helpers_only_delegate_to_scheduler(self) -> None:
        coordinator = object.__new__(EvolverCoordinator)
        coordinator.maintainer = None
        coordinator.cadence_scheduler = SimpleNamespace(
            run_maintenance=object(),
            advance=Mock(return_value=("cadence-id", "committed")),
            run_or_reconcile=Mock(return_value="noop"),
            reconcile_persisted=Mock(),
        )
        episode = object()
        outcome = object()
        cadence = object()

        self.assertEqual(
            coordinator._advance_cadence(
                episode=episode,
                outcome=outcome,
                writer_status="committed",
                repository_revision_after="revision",
                written_memory_ids=("memory-id",),
            ),
            ("cadence-id", "committed"),
        )
        self.assertEqual(coordinator._run_or_reconcile_cadence(cadence), "noop")
        coordinator._reconcile_persisted_cadences()

        self.assertIsNone(coordinator.cadence_scheduler.run_maintenance)
        coordinator.cadence_scheduler.advance.assert_called_once_with(
            episode=episode,
            outcome=outcome,
            writer_status="committed",
            repository_revision_after="revision",
            written_memory_ids=("memory-id",),
        )
        coordinator.cadence_scheduler.run_or_reconcile.assert_called_once_with(cadence)
        coordinator.cadence_scheduler.reconcile_persisted.assert_called_once_with()

    def test_current_cli_data_training_and_evaluation_consumers_import(self) -> None:
        modules = (
            "my_agent.cli.memory_maintenance",
            "my_agent.cli.commands.opd",
            "my_agent.opd_data.attribution",
            "my_agent.opd_data.export",
            "my_agent.training.collection_round",
            "my_agent.evaluation.manifest_benchmark",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertEqual(importlib.import_module(module_name).__name__, module_name)

    def test_runtime_modes_and_versioned_schemas_are_frozen(self) -> None:
        self.assertEqual(
            SUPPORTED_MEMORY_EVOLVER_MODES,
            {"off", "formal", "retrieve_select", "full"},
        )
        self.assertEqual(DECISION_EVENT_SCHEMA_VERSION, "opd-decision-v2")
        self.assertEqual(PAPER_ATTRIBUTION_SCHEMA_VERSION, "opd-paper-attribution-v1")
        self.assertEqual(CADENCE_SCHEMA_VERSION, "opd-maintenance-cadence-v1")

    def test_formal_runtime_order_includes_writer_before_cadence(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore.from_dir(tmp)

            def trace_sink(event: str, payload: dict[str, object]) -> None:
                del payload
                event_names = {
                    "memory.task_outcome_finalized": "outcome",
                    "memory.evolver_cadence_advanced": "cadence",
                }
                if event in event_names:
                    events.append(event_names[event])

            coordinator = EvolverCoordinator(
                store=store,
                project_key="repo:contract",
                policy_identity=_policy_identity(),
                retriever=_OrderRetriever(events),
                selector=_OrderSelector(events),
                writer=lambda episode, outcome: (
                    events.append("write") or ExperienceWriteResult()
                ),
                trace_sink=trace_sink,
                maintenance_interval_tasks=1,
            )
            session = coordinator.begin_task(
                task="contract task",
                task_id="task-contract",
                task_group="group-contract",
                trajectory_id="trajectory-contract",
                stream_id="stream-contract",
            )
            events.append("act")
            episode = AgentEpisodeArtifact(session, store.path, "finish", "done", ())
            coordinator.finalize_task(
                episode,
                AuthoritativeTaskOutcome(
                    task_id="task-contract",
                    task_group="group-contract",
                    task_valid=True,
                    resolved=True,
                    reward=1.0,
                    evaluator=EvaluatorIdentity(
                        "pytest",
                        "8",
                        canonical_sha256({"evaluator": "pytest"}),
                    ),
                ),
            )

        self.assertEqual(events, ["retrieve", "select", "act", "outcome", "write", "cadence"])

    def test_experience_canonical_json_and_repository_revision_are_frozen(self) -> None:
        memory = _contract_memory()
        expected_json = (
            '{"attribution_confidence":0.0,"attribution_updated_at":null,'
            '"attribution_value":0.0,"candidate_count":0,'
            '"content":"Always rerun the focused parser test.",'
            '"created_at":"2026-07-18T09:30:00+00:00","created_by":"writer",'
            '"fingerprint":"579df3ef036e7f5a8fadaba1768a186c37d96d020dc96f09aac6653406f7ef60",'
            '"id":"exp-contract-tip","invalidated":false,"last_used":null,'
            '"maintenance_operation_id":"","not_selected_count":0,"parent_id":"",'
            '"parent_tier":null,"payload":{"category":"testing","severity":"warning",'
            '"trigger":"parser behavior changes"},"project_key":"repo:contract",'
            '"promoted_to":"","protected":false,'
            '"reward_when_candidate_not_selected":null,"reward_when_selected":null,'
            '"run_id":"run-contract","schema_version":2,"scope":"project",'
            '"selected_count":0,"source_task":"task-contract",'
            '"stream_id":"stream-contract","success_when_candidate_not_selected":null,'
            '"success_when_selected":null,"tier":"tip","token_count":9,'
            '"writer_confidence":0.875}'
        )
        self.assertEqual(experience_canonical_json(memory), expected_json)
        self.assertEqual(
            experience_memories_revision((memory,)),
            "sha256:d2d05320e4dbbc76b03c715b50672d884b7933e06abe9d684beaa2ce87de4abf",
        )


class PlannedMemoryArchitectureTests(unittest.TestCase):
    def test_internal_source_does_not_import_memory_barrels(self) -> None:
        excluded = {
            SRC_ROOT / "memory" / "__init__.py",
            SRC_ROOT / "memory" / "evolver" / "__init__.py",
        }
        violations: list[str] = []
        for path in SRC_ROOT.rglob("*.py"):
            if path in excluded:
                continue
            imported = _imported_modules(path)
            for barrel in ("my_agent.memory", "my_agent.memory.evolver"):
                if barrel in imported:
                    violations.append(f"{path.relative_to(SRC_ROOT)} -> {barrel}")
        self.assertEqual(violations, [])

    def test_tests_follow_source_domains_and_large_suites_are_split(self) -> None:
        required_directories = (
            TESTS_ROOT / "memory" / "core",
            TESTS_ROOT / "memory" / "short_term",
            TESTS_ROOT / "memory" / "experience",
            TESTS_ROOT / "memory" / "manager",
            TESTS_ROOT / "memory" / "evolver" / "runtime",
            TESTS_ROOT / "memory" / "evolver" / "selection",
            TESTS_ROOT / "memory" / "evolver" / "writing",
            TESTS_ROOT / "memory" / "evolver" / "maintenance",
            TESTS_ROOT / "opd_data" / "attribution",
            TESTS_ROOT / "opd_data" / "legacy",
        )
        self.assertTrue(all(path.is_dir() for path in required_directories))
        self.assertEqual(list((TESTS_ROOT / "memory").glob("test_*.py")), [])

        oversized = [
            str(path.relative_to(TESTS_ROOT))
            for root in (TESTS_ROOT / "memory", TESTS_ROOT / "opd_data")
            for path in root.rglob("*.py")
            if len(path.read_text(encoding="utf-8").splitlines()) > 900
        ]
        self.assertEqual(oversized, [])

    def test_experience_import_does_not_initialize_higher_level_packages(self) -> None:
        script = (
            "import json, sys; "
            f"sys.path.insert(0, {str(SRC_ROOT.parent)!r}); "
            "import my_agent.memory.experience.models; "
            "loaded=tuple(sys.modules); "
            "print(json.dumps({"
            "'evolver': any(name == 'my_agent.memory.evolver' or "
            "name.startswith('my_agent.memory.evolver.') for name in loaded),"
            "'training': any(name == 'my_agent.training' or "
            "name.startswith('my_agent.training.') for name in loaded),"
            "'opd_data': any(name == 'my_agent.opd_data' or "
            "name.startswith('my_agent.opd_data.') for name in loaded)"
            "}, sort_keys=True))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(completed.stdout),
            {"evolver": False, "opd_data": False, "training": False},
        )

    def test_import_parser_resolves_relative_and_alias_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "my_agent"
            module = source_root / "memory" / "experience" / "module.py"
            package_init = module.parent / "__init__.py"
            module.parent.mkdir(parents=True)
            module.write_text(
                "from ..evolver.types import ExperienceMemory\n"
                "from my_agent.memory import evolver\n"
                "from my_agent import training\n",
                encoding="utf-8",
            )
            package_init.write_text(
                "from ..evolver.types import ExperienceMemory\n",
                encoding="utf-8",
            )

            imported = _imported_modules(module, source_root=source_root)
            init_imported = _imported_modules(package_init, source_root=source_root)

        self.assertTrue({
            "my_agent.memory.evolver.types",
            "my_agent.memory.evolver",
            "my_agent.training",
        }.issubset(imported))
        self.assertIn("my_agent.memory.evolver.types", init_imported)

    def test_new_domain_packages_cannot_reverse_memory_dependencies(self) -> None:
        boundaries = {
            SRC_ROOT / "memory" / "experience": (
                "my_agent.memory.evolver",
                "my_agent.training",
                "my_agent.opd_data",
            ),
            SRC_ROOT / "memory" / "core": (
                "my_agent.memory.experience",
                "my_agent.memory.evolver",
            ),
        }
        violations: list[str] = []
        for package_root, forbidden_prefixes in boundaries.items():
            for path in package_root.rglob("*.py") if package_root.exists() else ():
                for imported in _imported_modules(path):
                    if any(
                        imported == prefix or imported.startswith(prefix + ".")
                        for prefix in forbidden_prefixes
                    ):
                        violations.append(f"{path.relative_to(SRC_ROOT)} -> {imported}")
        self.assertEqual(violations, [])

    def test_memory_only_uses_collection_attribution_through_compatibility_facades(
        self,
    ) -> None:
        allowed_facades = {
            Path("memory/evolver/attribution_export.py"),
            Path("memory/evolver/attribution_schema.py"),
            Path("memory/evolver/paper_attribution.py"),
        }
        violations: list[str] = []
        memory_root = SRC_ROOT / "memory"
        for path in memory_root.rglob("*.py"):
            relative = path.relative_to(SRC_ROOT)
            if relative in allowed_facades:
                continue
            for imported in _imported_modules(path):
                if imported == "my_agent.opd_data.attribution" or imported.startswith(
                    "my_agent.opd_data.attribution."
                ):
                    violations.append(f"{relative} -> {imported}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
