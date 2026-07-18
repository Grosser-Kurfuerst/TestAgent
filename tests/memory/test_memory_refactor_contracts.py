from __future__ import annotations

# ruff: noqa: E402 - tests add the src layout before importing project modules

import ast
import importlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

import my_agent.memory as memory_api
import my_agent.memory.evolver as evolver_api
from my_agent.config import SUPPORTED_MEMORY_EVOLVER_MODES
from my_agent.memory.evolver import maintenance as maintenance_api
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
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.training.contracts import (
    DECISION_EVENT_SCHEMA_VERSION,
    AuthoritativeTaskOutcome,
    EvaluatorIdentity,
)


SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "my_agent"


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
        self.assertIs(memory_api.ExperienceMemory, ExperienceMemory)
        self.assertIs(memory_api.ExperienceTier, ExperienceTier)
        self.assertIs(evolver_api.ExperienceMemory, ExperienceMemory)
        self.assertIs(evolver_api.ExperienceTier, ExperienceTier)
        self.assertIs(evolver_api.TipPayload, TipPayload)

    def test_maintenance_facade_preserves_contract_identity(self) -> None:
        self.assertIs(maintenance_api.MaintenanceOperation, MaintenanceOperation)
        self.assertIs(evolver_api.MaintenanceOperation, MaintenanceOperation)
        self.assertIs(evolver_api.build_maintenance_plan, maintenance_api.build_maintenance_plan)
        self.assertIs(evolver_api.apply_maintenance_plan, maintenance_api.apply_maintenance_plan)

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


if __name__ == "__main__":
    unittest.main()
