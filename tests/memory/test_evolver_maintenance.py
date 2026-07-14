from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tests._path import add_src_to_path

add_src_to_path()

import my_agent.memory.evolver.maintenance as maintenance_module
import my_agent.memory.evolver.contracts as maintenance_contracts
import my_agent.memory.evolver.planner as maintenance_planner
import my_agent.memory.evolver.service as maintenance_service
import my_agent.memory.evolver.transaction as maintenance_transaction
import my_agent.memory.evolver.validation as maintenance_validation
from my_agent.memory.evolver import (
    ExperienceCreatedBy,
    MaintenanceAction,
    MaintenanceAttributionError,
    MaintenanceConfig,
    MaintenanceOperation,
    MaintenancePlan,
    MemoryAttributionRecord,
    build_experience_entry,
    build_maintenance_plan,
    load_maintenance_plan,
    load_project_attribution,
    maintenance_evidence_for_entry,
    maintenance_plan_json,
    lookup_experiences,
    redundancy_score,
    write_maintenance_plan,
)
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryEntry, MemoryScope, MemoryType


PROJECT_KEY = "manifest:demo:memory:shared_stream:stream:python"
NOW = datetime(2026, 7, 10, tzinfo=timezone.utc)


def _entry(
    memory_id: str = "mem-1",
    *,
    tier: str = "skill",
    project_key: str = PROJECT_KEY,
    scope: MemoryScope = MemoryScope.PROJECT,
    metadata: dict | None = None,
    content: str | None = None,
    created_by: ExperienceCreatedBy = ExperienceCreatedBy.MANUAL,
    created_at: datetime = NOW,
    run_id: str = "",
):
    return build_experience_entry(
        id=memory_id,
        content=content or f"experience {memory_id}",
        tier=tier,
        project_key=project_key,
        scope=scope,
        created_at=created_at,
        created_by=created_by,
        run_id=run_id,
        source_task="task-1",
        extra_metadata=metadata,
    )


def _record(
    memory_id: str = "mem-1",
    *,
    tier: str = "skill",
    project_key: str = PROJECT_KEY,
    value: float = 0.25,
    confidence: float = 0.75,
    candidate_count: int = 8,
    selected_count: int = 4,
    not_selected_count: int | None = None,
    last_used: str = "2026-07-09T00:00:00+00:00",
) -> MemoryAttributionRecord:
    return MemoryAttributionRecord(
        memory_id=memory_id,
        tier=tier,
        memory_project_key=project_key,
        candidate_count=candidate_count,
        selected_count=selected_count,
        not_selected_count=(
            candidate_count - selected_count
            if not_selected_count is None
            else not_selected_count
        ),
        value=value,
        confidence=confidence,
        last_used=last_used,
    )


def _refresh_plan_id(payload: dict) -> None:
    operations = tuple(
        maintenance_module.MaintenanceOperation.from_dict(item)
        for item in payload["operations"]
    )
    config = MaintenanceConfig.from_dict(payload["config"]).to_dict()
    payload["plan_id"] = maintenance_module._plan_id(
        repository_revision=payload["repository_revision"],
        project_key=payload["memory_project_key"],
        as_of=payload["as_of"],
        config=config,
        input_summary=payload["input_summary"],
        operations=operations,
        summary=payload["summary"],
    )


class MaintenanceConfigTests(unittest.TestCase):
    def test_defaults_round_trip(self) -> None:
        config = MaintenanceConfig()

        self.assertEqual(MaintenanceConfig.from_dict(config.to_dict()), config)

    def test_invalid_thresholds_and_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MaintenanceConfig(merge_threshold_skill=1.1)
        with self.assertRaises(ValueError):
            MaintenanceConfig(delete_min_candidate_count=-1)
        with self.assertRaises(ValueError):
            MaintenanceConfig(merge_max_cluster_size=1)
        with self.assertRaises(ValueError):
            MaintenanceConfig.from_dict({"unknown": 1})
        with self.assertRaises(ValueError):
            MaintenanceConfig.from_dict({"max_merged_content_chars": 1_600})
        with self.assertRaises(ValueError):
            MaintenanceConfig(protect_manual=False)
        with self.assertRaises(ValueError):
            MaintenanceConfig(stale_min_candidate_count=0)


class MaintenanceModuleBoundaryTests(unittest.TestCase):
    def test_facade_reexports_layer_owners_without_duplicating_implementations(self) -> None:
        self.assertIs(maintenance_module.MaintenancePlan, maintenance_contracts.MaintenancePlan)
        self.assertIs(
            maintenance_module.build_maintenance_plan,
            maintenance_service.build_maintenance_plan,
        )
        self.assertIs(
            maintenance_module.apply_maintenance_plan,
            maintenance_transaction.apply_maintenance_plan,
        )
        self.assertFalse(hasattr(maintenance_contracts, "LongTermMemoryStore"))
        self.assertFalse(hasattr(maintenance_planner, "LongTermMemoryStore"))
        self.assertFalse(hasattr(maintenance_contracts, "validate_plan_semantics"))
        self.assertFalse(hasattr(maintenance_planner, "validate_plan_semantics"))

    def test_planner_all_only_names_existing_exports(self) -> None:
        for name in maintenance_planner.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(maintenance_planner, name))


class ProjectAttributionLoaderTests(unittest.TestCase):
    def _write(self, path: Path, records: list[dict | str]) -> None:
        lines = [item if isinstance(item, str) else json.dumps(item) for item in records]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_loads_composite_keys_for_one_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attribution.jsonl"
            self._write(path, [_record().to_dict(), _record("mem-2", tier="tip").to_dict()])

            loaded = load_project_attribution(path, memory_project_key=PROJECT_KEY)

            self.assertEqual(
                set(loaded),
                {
                    ("mem-1", "skill", PROJECT_KEY),
                    ("mem-2", "tip", PROJECT_KEY),
                },
            )

    def test_mixed_project_bad_json_and_duplicate_key_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attribution.jsonl"

            self._write(path, [_record(project_key="other-project").to_dict()])
            with self.assertRaises(MaintenanceAttributionError):
                load_project_attribution(path, memory_project_key=PROJECT_KEY)

            self._write(path, ["{bad}"])
            with self.assertRaises(MaintenanceAttributionError):
                load_project_attribution(path, memory_project_key=PROJECT_KEY)

            self._write(path, [_record().to_dict(), _record().to_dict()])
            with self.assertRaises(MaintenanceAttributionError):
                load_project_attribution(path, memory_project_key=PROJECT_KEY)

    def test_invalid_numeric_and_timestamp_semantics_fail_closed(self) -> None:
        invalid_cases = {
            "non_finite_value": {"value": float("nan")},
            "confidence_out_of_range": {"confidence": 2.0},
            "success_rate_out_of_range": {"success_when_selected": -0.1},
            "negative_count": {"selected_count": -1},
            "inconsistent_counts": {
                "candidate_count": 8,
                "selected_count": 3,
                "not_selected_count": 4,
            },
            "invalid_last_used": {"last_used": "not-a-timestamp"},
        }
        for case, updates in invalid_cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "attribution.jsonl"
                payload = _record().to_dict()
                payload.update(updates)
                self._write(path, [payload])

                with self.assertRaises(MaintenanceAttributionError):
                    load_project_attribution(path, memory_project_key=PROJECT_KEY)


class MaintenanceEvidenceTests(unittest.TestCase):
    def test_explicit_attribution_wins_and_writer_confidence_stays_separate(self) -> None:
        entry = _entry(metadata={
            "confidence": 0.33,
            "evolver_value": -0.4,
            "evolver_confidence": 0.2,
            "evolver_candidate_count": 1,
            "evolver_last_used": "2026-01-01T00:00:00+00:00",
            "evolver_attribution_memory_project_key": PROJECT_KEY,
        })
        record = _record()

        evidence = maintenance_evidence_for_entry(
            entry,
            attribution={(record.memory_id, record.tier, PROJECT_KEY): record},
            project_key=PROJECT_KEY,
        )

        self.assertEqual(evidence.value, 0.25)
        self.assertEqual(evidence.confidence, 0.75)
        self.assertEqual(evidence.candidate_count, 8)
        self.assertEqual(evidence.last_used, "2026-07-09T00:00:00+00:00")
        self.assertEqual(evidence.writer_confidence, 0.33)
        self.assertTrue(evidence.has_attribution)

    def test_metadata_fallback_and_mismatched_record_is_not_consumed(self) -> None:
        entry = _entry(metadata={
            "confidence": 0.9,
            "evolver_value": -0.1,
            "evolver_confidence": 0.6,
            "evolver_candidate_count": 5,
            "evolver_selected_count": 1,
            "evolver_not_selected_count": 4,
            "evolver_last_used": "2026-06-01T00:00:00+00:00",
            "evolver_attribution_memory_project_key": PROJECT_KEY,
        })
        wrong = _record(tier="tip")

        evidence = maintenance_evidence_for_entry(
            entry,
            attribution={(wrong.memory_id, wrong.tier, PROJECT_KEY): wrong},
            project_key=PROJECT_KEY,
        )

        self.assertEqual(evidence.value, -0.1)
        self.assertEqual(evidence.confidence, 0.6)
        self.assertEqual(evidence.candidate_count, 5)
        self.assertEqual(evidence.writer_confidence, 0.9)
        self.assertTrue(evidence.has_attribution)

    def test_cross_project_or_unprovenanced_metadata_is_not_consumed(self) -> None:
        for metadata_project_key in ("other-project", ""):
            with self.subTest(metadata_project_key=metadata_project_key):
                entry = _entry(metadata={
                    "confidence": 0.8,
                    "evolver_value": -0.4,
                    "evolver_confidence": 0.9,
                    "evolver_candidate_count": 20,
                    "evolver_attribution_memory_project_key": metadata_project_key,
                })

                evidence = maintenance_evidence_for_entry(
                    entry,
                    attribution={},
                    project_key=PROJECT_KEY,
                )

                self.assertFalse(evidence.has_attribution)
                self.assertEqual(evidence.value, 0.0)
                self.assertEqual(evidence.confidence, 0.0)
                self.assertEqual(evidence.candidate_count, 0)
                self.assertEqual(evidence.writer_confidence, 0.8)

    def test_explicit_record_payload_must_match_composite_mapping_key(self) -> None:
        entry = _entry()
        wrong_record = _record(project_key="other-project")

        evidence = maintenance_evidence_for_entry(
            entry,
            attribution={(entry.id, "skill", PROJECT_KEY): wrong_record},
            project_key=PROJECT_KEY,
        )

        self.assertFalse(evidence.has_attribution)
        self.assertEqual(evidence.value, 0.0)

    def test_invalid_metadata_attribution_fails_closed(self) -> None:
        invalid_cases = {
            "value_out_of_range": {"evolver_value": -2.0},
            "confidence_out_of_range": {"evolver_confidence": 2.0},
            "negative_count": {"evolver_selected_count": -1},
            "inconsistent_counts": {
                "evolver_candidate_count": 8,
                "evolver_selected_count": 2,
                "evolver_not_selected_count": 5,
            },
            "invalid_timestamp": {"evolver_last_used": "not-a-timestamp"},
        }
        base = {
            "evolver_value": -0.2,
            "evolver_confidence": 0.8,
            "evolver_candidate_count": 8,
            "evolver_selected_count": 2,
            "evolver_not_selected_count": 6,
            "evolver_last_used": "2026-07-09T00:00:00+00:00",
            "evolver_attribution_memory_project_key": PROJECT_KEY,
        }
        for case, updates in invalid_cases.items():
            with self.subTest(case=case):
                entry = _entry(
                    f"invalid-{case}",
                    tier="tip",
                    created_by=ExperienceCreatedBy.WRITER,
                    metadata={**base, **updates},
                )
                with self.assertRaises(ValueError):
                    build_maintenance_plan(
                        entries=[entry],
                        attribution={},
                        repository_revision=f"sha256:{case}",
                        project_key=PROJECT_KEY,
                        as_of=NOW,
                    )

    def test_invalid_direct_attribution_mapping_fails_closed(self) -> None:
        entry = _entry(
            "invalid-direct",
            tier="tip",
            created_by=ExperienceCreatedBy.WRITER,
        )
        record = _record(
            entry.id,
            tier="tip",
            value=-2.0,
            confidence=2.0,
            candidate_count=8,
            selected_count=2,
            not_selected_count=6,
        )

        with self.assertRaises(ValueError):
            build_maintenance_plan(
                entries=[entry],
                attribution={(entry.id, "tip", PROJECT_KEY): record},
                repository_revision="sha256:invalid-direct",
                project_key=PROJECT_KEY,
                as_of=NOW,
            )


class LookupAndRedundancyTests(unittest.TestCase):
    def test_lookup_filters_project_tier_and_non_experience_entries(self) -> None:
        project_skill = _entry(
            "project-skill",
            content="Parse the project manifest before editing task metadata",
            tier="skill",
        )
        project_tip = _entry(
            "project-tip",
            content="Manifest parser errors should be reproduced first",
            tier="tip",
        )
        other_project = _entry(
            "other-project",
            content="Parse the project manifest",
            project_key="other-project",
        )
        global_skill = _entry(
            "global-skill",
            content="Use a parser for manifest input",
            project_key="",
            scope=MemoryScope.GLOBAL,
        )
        plain = MemoryEntry.build(
            id="plain",
            content="Parse the project manifest",
            type=MemoryType.FACT,
            scope=MemoryScope.PROJECT,
            source="manual",
            token_count=estimate_tokens("Parse the project manifest"),
            project_key=PROJECT_KEY,
            created_at=NOW,
        )

        hits = lookup_experiences(
            [other_project, plain, project_tip, global_skill, project_skill],
            "project manifest",
            project_key=PROJECT_KEY,
            tiers=("skill",),
        )

        self.assertEqual([hit.entry.id for hit in hits], ["project-skill", "global-skill"])
        self.assertTrue(all(hit.tier == "skill" for hit in hits))
        self.assertGreaterEqual(hits[0].score, hits[1].score)

    def test_lookup_rejects_invalid_scope_inputs(self) -> None:
        with self.assertRaises(ValueError):
            lookup_experiences([], "query", project_key="")
        with self.assertRaises(ValueError):
            lookup_experiences([], "query", project_key=PROJECT_KEY, tiers=("future",))
        with self.assertRaises(ValueError):
            lookup_experiences([], "query", project_key=PROJECT_KEY, limit=-1)

    def test_redundancy_is_exact_for_normalized_match_and_supports_cjk(self) -> None:
        left = _entry("left", tier="tip", content="Run  tests before editing")
        exact = _entry("exact", tier="tip", content="run tests before editing")
        cjk_left = _entry("cjk-left", tier="skill", content="先运行测试再修改解析器")
        cjk_right = _entry("cjk-right", tier="skill", content="先运行测试再修改配置器")

        self.assertEqual(redundancy_score(left, exact), 1.0)
        self.assertGreater(redundancy_score(cjk_left, cjk_right), 0.0)

    def test_redundancy_refuses_cross_tier_scope_or_project_pairs(self) -> None:
        tip = _entry("tip", tier="tip", content="same content")
        skill = _entry("skill", tier="skill", content="same content")
        other_project = _entry(
            "other",
            tier="tip",
            content="same content",
            project_key="other-project",
        )
        global_tip = _entry(
            "global",
            tier="tip",
            content="same content",
            project_key="",
            scope=MemoryScope.GLOBAL,
        )

        self.assertEqual(redundancy_score(tip, skill), 0.0)
        self.assertEqual(redundancy_score(tip, other_project), 0.0)
        self.assertEqual(redundancy_score(tip, global_tip), 0.0)


class RetentionPlannerTests(unittest.TestCase):
    def _keyed(self, *records: MemoryAttributionRecord):
        return {
            (record.memory_id, record.tier, record.memory_project_key): record
            for record in records
        }

    def test_planner_applies_boundaries_and_delete_rules(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        manual = _entry("manual", created_by=ExperienceCreatedBy.MANUAL)
        global_entry = _entry(
            "global",
            project_key="",
            scope=MemoryScope.GLOBAL,
            created_by=writer,
        )
        protected = _entry(
            "protected",
            created_by=writer,
            metadata={"maintenance_protected": True, "maintenance_invalidated": True},
        )
        negative = _entry("negative", tier="tip", created_by=writer)
        stale = _entry(
            "stale",
            created_by=writer,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        low_confidence = _entry("low-confidence", created_by=writer)
        other_project = _entry("other-project", project_key="other-project", created_by=writer)
        plain = MemoryEntry.build(
            id="plain",
            content="ordinary fact",
            type=MemoryType.FACT,
            scope=MemoryScope.PROJECT,
            source="manual",
            token_count=2,
            project_key=PROJECT_KEY,
            created_at=NOW,
        )
        attribution = self._keyed(
            _record(
                "negative",
                tier="tip",
                value=-0.2,
                confidence=0.8,
                candidate_count=8,
                selected_count=2,
                not_selected_count=6,
            ),
            _record(
                "stale",
                value=0.0,
                confidence=0.0,
                candidate_count=7,
                selected_count=0,
                not_selected_count=7,
                last_used="2025-01-01T00:00:00+00:00",
            ),
            _record(
                "low-confidence",
                value=-0.4,
                confidence=0.1,
                candidate_count=10,
                selected_count=2,
                not_selected_count=8,
            ),
        )

        plan = build_maintenance_plan(
            entries=[
                other_project,
                plain,
                low_confidence,
                stale,
                negative,
                protected,
                global_entry,
                manual,
            ],
            attribution=attribution,
            repository_revision="sha256:iteration-2",
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        by_source = {operation.source_ids[0]: operation for operation in plan.operations}
        self.assertEqual(set(by_source), {
            "manual", "global", "protected", "negative", "stale", "low-confidence"
        })
        self.assertEqual(by_source["negative"].action, MaintenanceAction.DELETE)
        self.assertEqual(by_source["negative"].reason_codes, ("negative_attribution_with_control",))
        self.assertEqual(by_source["negative"].remove_ids, ("negative",))
        self.assertEqual(by_source["stale"].action, MaintenanceAction.DELETE)
        self.assertEqual(by_source["stale"].reason_codes, ("stale_retrieved_never_selected",))
        self.assertEqual(by_source["manual"].reason_codes, ("protected_manual",))
        self.assertEqual(by_source["global"].reason_codes, ("protected_global",))
        self.assertEqual(by_source["protected"].reason_codes, ("protected_metadata",))
        self.assertEqual(
            by_source["low-confidence"].reason_codes,
            ("insufficient_attribution_evidence",),
        )
        self.assertEqual(plan.summary["delete"], 2)
        self.assertEqual(plan.summary["keep"], 4)
        self.assertEqual(plan.summary["source_entries_removed"], 2)
        self.assertEqual(plan.input_summary["experiences_considered"], 6)
        self.assertEqual(len({item.source_ids[0] for item in plan.operations}), len(plan.operations))

    def test_explicit_invalidation_deletes_unprotected_writer_entry(self) -> None:
        invalid = _entry(
            "invalid",
            created_by=ExperienceCreatedBy.WRITER,
            metadata={"maintenance_invalidated": True},
        )

        plan = build_maintenance_plan(
            entries=[invalid],
            attribution={},
            repository_revision="sha256:invalid",
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        self.assertEqual(plan.operations[0].action, MaintenanceAction.DELETE)
        self.assertEqual(plan.operations[0].reason_codes, ("explicitly_invalidated",))

    def test_never_retrieved_old_memory_is_not_stale_deleted(self) -> None:
        entry = _entry(
            "never-retrieved",
            created_by=ExperienceCreatedBy.WRITER,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        plan = build_maintenance_plan(
            entries=[entry],
            attribution={},
            repository_revision="sha256:never-retrieved",
            project_key=PROJECT_KEY,
            as_of=NOW,
            config=MaintenanceConfig(stale_min_candidate_count=1),
        )

        self.assertEqual(plan.operations[0].action, MaintenanceAction.KEEP)
        self.assertNotEqual(
            plan.operations[0].reason_codes,
            ("stale_retrieved_never_selected",),
        )

    def test_missing_or_unknown_provenance_is_never_a_destructive_candidate(self) -> None:
        for created_by in (None, "legacy-import"):
            with self.subTest(created_by=created_by):
                payload = _entry(
                    "legacy-tip",
                    tier="tip",
                    created_by=ExperienceCreatedBy.WRITER,
                ).to_dict()
                if created_by is None:
                    payload["metadata"].pop("created_by", None)
                else:
                    payload["metadata"]["created_by"] = created_by
                entry = MemoryEntry.from_dict(payload)
                record = _record(
                    "legacy-tip",
                    tier="tip",
                    value=-0.4,
                    confidence=0.9,
                    candidate_count=10,
                    selected_count=2,
                    not_selected_count=8,
                )

                plan = build_maintenance_plan(
                    entries=[entry],
                    attribution={(entry.id, "tip", PROJECT_KEY): record},
                    repository_revision=f"sha256:legacy:{created_by}",
                    project_key=PROJECT_KEY,
                    as_of=NOW,
                )

                self.assertEqual(plan.operations[0].action, MaintenanceAction.KEEP)
                self.assertEqual(
                    plan.operations[0].reason_codes,
                    ("protected_unknown_provenance",),
                )

    def test_plan_is_stable_for_reordered_inputs(self) -> None:
        entries = [
            _entry("keep", created_by=ExperienceCreatedBy.WRITER),
            _entry("delete", created_by=ExperienceCreatedBy.WRITER),
        ]
        record = _record(
            "delete",
            value=-0.2,
            confidence=0.8,
            candidate_count=8,
            selected_count=2,
            not_selected_count=6,
        )
        attribution = self._keyed(record)

        first = build_maintenance_plan(
            entries=entries,
            attribution=attribution,
            repository_revision="sha256:stable",
            project_key=PROJECT_KEY,
            as_of=NOW,
        )
        second = build_maintenance_plan(
            entries=list(reversed(entries)),
            attribution=dict(reversed(list(attribution.items()))),
            repository_revision="sha256:stable",
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

        self.assertEqual(first, second)
        self.assertEqual(maintenance_plan_json(first), maintenance_plan_json(second))

    def test_planner_requires_single_project_and_aware_as_of(self) -> None:
        with self.assertRaises(ValueError):
            build_maintenance_plan(
                entries=[],
                attribution={},
                repository_revision="sha256:x",
                project_key="",
                as_of=NOW,
            )
        with self.assertRaises(ValueError):
            build_maintenance_plan(
                entries=[],
                attribution={},
                repository_revision="sha256:x",
                project_key=PROJECT_KEY,
                as_of=datetime(2026, 7, 10),
            )


class MergePlannerTests(unittest.TestCase):
    def _keyed(self, *records: MemoryAttributionRecord):
        return {
            (record.memory_id, record.tier, record.memory_project_key): record
            for record in records
        }

    def test_merge_guards_reject_cross_boundary_trajectory_and_tool_payloads(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        entries = [
            _entry("tip", tier="tip", content="Inspect parser failures first", created_by=writer),
            _entry("skill", tier="skill", content="Inspect parser failures first", created_by=writer),
            _entry(
                "global-tip",
                tier="tip",
                content="Inspect parser failure first",
                project_key="",
                scope=MemoryScope.GLOBAL,
                created_by=writer,
            ),
            _entry(
                "other-project",
                tier="tip",
                content="Inspect parser failure first",
                project_key="other-project",
                created_by=writer,
            ),
            _entry("trajectory-a", tier="trajectory", content="Parser repair trace A", created_by=writer),
            _entry("trajectory-b", tier="trajectory", content="Parser repair trace B", created_by=writer),
            _entry(
                "tool-a",
                tier="tool",
                content="Run the focused parser test command",
                metadata={"language": "bash", "command": "pytest tests/test_parser.py -q"},
                created_by=writer,
            ),
            _entry(
                "tool-b",
                tier="tool",
                content="Run focused parser tests with this command",
                metadata={"language": "bash", "command": "pytest tests/test_other.py -q"},
                created_by=writer,
            ),
        ]

        plan = build_maintenance_plan(
            entries=entries,
            attribution={},
            repository_revision="sha256:merge-guards",
            project_key=PROJECT_KEY,
            as_of=NOW,
            config=MaintenanceConfig(
                merge_threshold_tip=0.1,
                merge_threshold_skill=0.1,
                merge_threshold_tool=0.1,
                max_promotions=0,
            ),
        )

        self.assertFalse(any(item.action == MaintenanceAction.MERGE for item in plan.operations))
        by_source = {item.source_ids[0]: item for item in plan.operations}
        self.assertEqual(by_source["global-tip"].reason_codes, ("protected_global",))
        self.assertNotIn("other-project", by_source)
        self.assertIn("trajectory-a", by_source)
        self.assertIn("trajectory-b", by_source)

    def test_complete_link_does_not_transitively_merge_three_entries(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        entries = [
            _entry("a", content="alpha parser workflow", created_by=writer),
            _entry("b", content="beta parser workflow", created_by=writer),
            _entry("c", content="gamma parser workflow", created_by=writer),
        ]
        scores = {
            frozenset(("a", "b")): 0.91,
            frozenset(("b", "c")): 0.92,
            frozenset(("a", "c")): 0.70,
        }

        with patch(
            "my_agent.memory.evolver.planner.redundancy_score",
            side_effect=lambda left, right: scores[frozenset((left.id, right.id))],
        ):
            plan = build_maintenance_plan(
                entries=entries,
                attribution={},
                repository_revision="sha256:complete-link",
                project_key=PROJECT_KEY,
                as_of=NOW,
                config=MaintenanceConfig(merge_threshold_skill=0.8, max_promotions=0),
            )

        merges = [item for item in plan.operations if item.action == MaintenanceAction.MERGE]
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0].source_ids, ("a", "b"))
        self.assertEqual(merges[0].redundancy_score, 0.91)
        self.assertTrue(any(item.source_ids == ("c",) for item in plan.operations))

    def test_tool_payload_guard_preserves_command_case_and_code_indentation(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        cases = [
            (
                "command-case",
                {"language": "bash", "command": "python Build.py"},
                {"language": "BASH", "command": "python build.py"},
            ),
            (
                "code-indentation",
                {"language": "python", "code": "if ready:\n    run()"},
                {"language": "PYTHON", "code": "if ready:\nrun()"},
            ),
        ]
        for name, left_metadata, right_metadata in cases:
            with self.subTest(name=name):
                entries = [
                    _entry(
                        f"{name}-left",
                        tier="tool",
                        content=f"Reusable tool description left {name}",
                        metadata=left_metadata,
                        created_by=writer,
                    ),
                    _entry(
                        f"{name}-right",
                        tier="tool",
                        content=f"Reusable tool description right {name}",
                        metadata=right_metadata,
                        created_by=writer,
                    ),
                ]
                with patch(
                    "my_agent.memory.evolver.planner.redundancy_score",
                    return_value=1.0,
                ):
                    plan = build_maintenance_plan(
                        entries=entries,
                        attribution={},
                        repository_revision=f"sha256:{name}",
                        project_key=PROJECT_KEY,
                        as_of=NOW,
                        config=MaintenanceConfig(max_promotions=0),
                    )
                self.assertFalse(
                    any(item.action == MaintenanceAction.MERGE for item in plan.operations)
                )

    def test_anchor_priority_uses_value_confidence_selected_created_and_id(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        earlier = datetime(2025, 1, 1, tzinfo=timezone.utc)
        later = datetime(2026, 1, 1, tzinfo=timezone.utc)
        cases = [
            (
                "value",
                _record("left", value=0.3, confidence=0.1, selected_count=1),
                _record(
                    "right",
                    value=0.2,
                    confidence=0.9,
                    candidate_count=10,
                    selected_count=9,
                ),
                later,
                earlier,
                "left",
            ),
            (
                "confidence",
                _record("left", value=0.3, confidence=0.9, selected_count=1),
                _record(
                    "right",
                    value=0.3,
                    confidence=0.8,
                    candidate_count=10,
                    selected_count=9,
                ),
                later,
                earlier,
                "left",
            ),
            ("selected", _record("left", value=0.3, confidence=0.9, selected_count=5),
             _record("right", value=0.3, confidence=0.9, selected_count=4), later, earlier, "left"),
            ("created", _record("left", value=0.3, confidence=0.9, selected_count=5),
             _record("right", value=0.3, confidence=0.9, selected_count=5), earlier, later, "left"),
            ("id", _record("a", value=0.3, confidence=0.9, selected_count=5),
             _record("b", value=0.3, confidence=0.9, selected_count=5), earlier, earlier, "a"),
        ]
        for name, left_record, right_record, left_created, right_created, expected in cases:
            with self.subTest(name=name):
                left_id = left_record.memory_id
                right_id = right_record.memory_id
                entries = [
                    _entry(
                        left_id,
                        content=f"left content {name}",
                        created_by=writer,
                        created_at=left_created,
                    ),
                    _entry(
                        right_id,
                        content=f"right content {name}",
                        created_by=writer,
                        created_at=right_created,
                    ),
                ]
                with patch(
                    "my_agent.memory.evolver.planner.redundancy_score",
                    return_value=1.0,
                ):
                    plan = build_maintenance_plan(
                        entries=entries,
                        attribution=self._keyed(left_record, right_record),
                        repository_revision=f"sha256:anchor-{name}",
                        project_key=PROJECT_KEY,
                        as_of=NOW,
                        config=MaintenanceConfig(max_promotions=0),
                    )
                merge = next(item for item in plan.operations if item.action == MaintenanceAction.MERGE)
                self.assertEqual(merge.source_ids[0], expected)
                self.assertEqual(merge.target_ids, (expected,))

    def test_merge_preserves_anchor_identity_and_records_complete_lineage(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        anchor = _entry(
            "anchor",
            content="Run parser tests before editing",
            created_by=writer,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            run_id="run-anchor",
            metadata={
                "steps": ["run focused tests"],
                "tags": ["parser"],
                "evolver_value": 0.4,
                "evolver_candidate_count": 8,
            },
        )
        source = _entry(
            "source",
            content="Run parser test before edits",
            created_by=writer,
            metadata={
                "steps": ["run focused tests", "run full suite"],
                "tags": ["pytest", "parser"],
                "evolver_value": 0.2,
                "evolver_candidate_count": 20,
            },
        )
        records = self._keyed(
            _record("anchor", value=0.4, confidence=0.9, selected_count=5),
            _record("source", value=0.2, confidence=0.8, selected_count=4),
        )

        with patch.object(maintenance_planner, "redundancy_score", return_value=0.95):
            plan = build_maintenance_plan(
                entries=[source, anchor],
                attribution=records,
                repository_revision="sha256:lineage",
                project_key=PROJECT_KEY,
                as_of=NOW,
                config=MaintenanceConfig(max_promotions=0),
            )

        operation = next(item for item in plan.operations if item.action == MaintenanceAction.MERGE)
        replacement = MemoryEntry.from_dict(operation.replacements[0])
        self.assertEqual(replacement.id, anchor.id)
        self.assertEqual(replacement.content, anchor.content)
        self.assertEqual(replacement.created_at, anchor.created_at)
        self.assertEqual(replacement.project_key, anchor.project_key)
        self.assertEqual(replacement.run_id, "run-anchor")
        self.assertEqual(replacement.metadata["steps"], ["run focused tests", "run full suite"])
        self.assertEqual(replacement.metadata["tags"], ["parser", "pytest"])
        self.assertEqual(replacement.metadata["evolver_value"], 0.4)
        self.assertEqual(replacement.metadata["evolver_candidate_count"], 8)
        self.assertEqual(replacement.metadata["maintenance_source_ids"], ["anchor", "source"])
        self.assertEqual(
            set(replacement.metadata["maintenance_source_fingerprints"]),
            {"anchor", "source"},
        )
        self.assertEqual(
            set(replacement.metadata["maintenance_source_evidence"]),
            {"anchor", "source"},
        )
        self.assertEqual(operation.remove_ids, ("source",))


class PromotionPlannerTests(unittest.TestCase):
    def _keyed(self, *records: MemoryAttributionRecord):
        return {
            (record.memory_id, record.tier, record.memory_project_key): record
            for record in records
        }

    def test_tip_and_trajectory_promotions_are_stable_neutral_and_timestamped(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        tip = _entry(
            "tip-source",
            tier="tip",
            content="Inspect the focused parser failure before broad edits.",
            created_by=writer,
            run_id="run-tip",
            metadata={"category": "debugging", "trigger": "parser test fails", "confidence": 0.82},
        )
        trajectory = _entry(
            "trajectory-source",
            tier="trajectory",
            content="Successful parser repair trajectory",
            created_by=writer,
            metadata={
                "task_description": "Repair parser handling for comments",
                "outcome": "success",
                "key_learnings": ["Strip comments first", "Preserve strings", "Run focused tests", "Ignore fourth"],
                "tags": ["parser", "testing"],
                "steps": [
                    {"action": "inspect", "result": "located parser", "status": "passed"},
                    {"action": "bad attempt", "result": "failed", "status": "failed"},
                    {"action": "run_tests", "result": "focused tests passed", "success": True},
                ],
                "confidence": 0.74,
            },
        )
        attribution = self._keyed(
            _record("tip-source", tier="tip", value=0.3, confidence=0.9, selected_count=5),
            _record(
                "trajectory-source",
                tier="trajectory",
                value=0.2,
                confidence=0.8,
                selected_count=4,
            ),
        )
        kwargs = {
            "attribution": attribution,
            "repository_revision": "sha256:promotion",
            "project_key": PROJECT_KEY,
            "as_of": NOW,
        }

        first = build_maintenance_plan(entries=[tip, trajectory], **kwargs)
        second = build_maintenance_plan(entries=[trajectory, tip], **kwargs)

        self.assertEqual(first, second)
        promotions = {
            item.source_ids[0]: item
            for item in first.operations
            if item.action == MaintenanceAction.PROMOTE
        }
        self.assertEqual(set(promotions), {"tip-source", "trajectory-source"})

        tip_operation = promotions["tip-source"]
        promoted_tip = MemoryEntry.from_dict(tip_operation.additions[0])
        updated_tip = MemoryEntry.from_dict(tip_operation.replacements[0])
        self.assertTrue(promoted_tip.id.startswith("exp_maint_"))
        self.assertEqual(promoted_tip.created_at, NOW)
        self.assertEqual(promoted_tip.content, tip.content)
        self.assertEqual(promoted_tip.fingerprint, tip.fingerprint)
        self.assertEqual(promoted_tip.run_id, "run-tip")
        self.assertEqual(promoted_tip.metadata["created_by"], "maintenance")
        self.assertEqual(promoted_tip.metadata["confidence"], 0.82)
        for key in (
            "evolver_value",
            "evolver_confidence",
            "evolver_candidate_count",
            "evolver_selected_count",
            "evolver_not_selected_count",
        ):
            self.assertNotIn(key, promoted_tip.metadata)
        self.assertEqual(updated_tip.metadata["maintenance_promoted_to"], promoted_tip.id)
        self.assertEqual(updated_tip.metadata["maintenance_promoted_at"], NOW.isoformat())

        trajectory_operation = promotions["trajectory-source"]
        promoted_trajectory = MemoryEntry.from_dict(trajectory_operation.additions[0])
        self.assertEqual(
            promoted_trajectory.content,
            "Strip comments first\nPreserve strings\nRun focused tests",
        )
        self.assertEqual(
            promoted_trajectory.metadata["steps"],
            ["inspect: located parser", "run_tests: focused tests passed"],
        )
        self.assertEqual(promoted_trajectory.created_at, NOW)

    def test_existing_skill_link_uses_skill_tier_and_repeat_is_not_promoted(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        tip = _entry(
            "tip-source",
            tier="tip",
            content="Reuse the focused parser verification loop.",
            created_by=writer,
        )
        existing_skill = _entry(
            "existing-skill",
            tier="skill",
            content=tip.content,
            created_by=writer,
        )
        already_promoted = _entry(
            "already-promoted",
            tier="tip",
            content="Inspect import boundaries before edits.",
            created_by=writer,
            metadata={"maintenance_promoted_to": "skill-old"},
        )
        attribution = self._keyed(
            _record("tip-source", tier="tip", value=0.3, confidence=0.9, selected_count=5),
            _record("already-promoted", tier="tip", value=0.4, confidence=0.9, selected_count=6),
        )

        plan = build_maintenance_plan(
            entries=[tip, existing_skill, already_promoted],
            attribution=attribution,
            repository_revision="sha256:existing-skill",
            project_key=PROJECT_KEY,
            as_of=NOW,
            config=MaintenanceConfig(merge_threshold_tip=1.0),
        )

        promotions = [item for item in plan.operations if item.action == MaintenanceAction.PROMOTE]
        self.assertEqual(len(promotions), 1)
        self.assertEqual(promotions[0].source_ids, ("tip-source",))
        self.assertEqual(promotions[0].target_ids, ("existing-skill",))
        self.assertEqual(promotions[0].reason_codes, ("promotion_linked_existing_skill",))
        self.assertEqual(promotions[0].additions, ())
        self.assertTrue(any(item.source_ids == ("already-promoted",) for item in plan.operations))

    def test_promotion_cap_uses_value_confidence_selected_and_id_order(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        entries = [
            _entry("tip-low", tier="tip", content="Low value parser guidance", created_by=writer),
            _entry("tip-high", tier="tip", content="High value test guidance", created_by=writer),
            _entry("tip-mid", tier="tip", content="Medium value import guidance", created_by=writer),
        ]
        attribution = self._keyed(
            _record("tip-low", tier="tip", value=0.2, confidence=0.9, selected_count=8),
            _record("tip-high", tier="tip", value=0.5, confidence=0.8, selected_count=3),
            _record("tip-mid", tier="tip", value=0.3, confidence=0.9, selected_count=5),
        )

        plan = build_maintenance_plan(
            entries=entries,
            attribution=attribution,
            repository_revision="sha256:promotion-cap",
            project_key=PROJECT_KEY,
            as_of=NOW,
            config=MaintenanceConfig(merge_threshold_tip=1.0, max_promotions=2),
        )

        promoted = {
            item.source_ids[0]
            for item in plan.operations
            if item.action == MaintenanceAction.PROMOTE
        }
        self.assertEqual(promoted, {"tip-high", "tip-mid"})
        self.assertEqual(plan.summary["promote"], 2)


class MaintenancePlanContractTests(unittest.TestCase):
    def _plan(self) -> MaintenancePlan:
        return build_maintenance_plan(
            entries=[_entry("mem-1", created_by=ExperienceCreatedBy.WRITER)],
            attribution={},
            repository_revision="sha256:repository",
            project_key=PROJECT_KEY,
            as_of=NOW,
        )

    def _promotion_fixture(self):
        source = _entry(
            "promotion-source",
            tier="tip",
            content="Preserve parser command case during verification.",
            created_by=ExperienceCreatedBy.WRITER,
        )
        record = _record(
            source.id,
            tier="tip",
            value=0.3,
            confidence=0.9,
            selected_count=5,
        )
        plan = build_maintenance_plan(
            entries=[source],
            attribution={(source.id, "tip", PROJECT_KEY): record},
            repository_revision="sha256:promotion-contract",
            project_key=PROJECT_KEY,
            as_of=NOW,
        )
        operation = next(
            item for item in plan.operations if item.action == MaintenanceAction.PROMOTE
        )
        return source, plan, operation

    def test_plan_round_trip_and_file_output_are_byte_stable(self) -> None:
        plan = self._plan()

        restored = MaintenancePlan.from_dict(plan.to_dict())
        self.assertEqual(restored, plan)
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            write_maintenance_plan(plan, first)
            write_maintenance_plan(restored, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(load_maintenance_plan(first), plan)
            self.assertEqual(first.read_text(encoding="utf-8"), maintenance_plan_json(plan))

    def test_partial_config_is_normalized_before_round_trip(self) -> None:
        plan = self._plan()
        normalized_config = MaintenanceConfig(delete_value_threshold=-0.2).to_dict()
        plan_id = maintenance_module._plan_id(
            repository_revision=plan.repository_revision,
            project_key=plan.memory_project_key,
            as_of=plan.as_of,
            config=normalized_config,
            input_summary=plan.input_summary,
            operations=plan.operations,
            summary=plan.summary,
        )
        partial = MaintenancePlan(
            schema_version=plan.schema_version,
            policy=plan.policy,
            plan_id=plan_id,
            repository_revision=plan.repository_revision,
            scope_mode=plan.scope_mode,
            memory_project_key=plan.memory_project_key,
            as_of=plan.as_of,
            config={"delete_value_threshold": -0.2},
            input_summary=plan.input_summary,
            operations=plan.operations,
            summary=plan.summary,
        )

        restored = MaintenancePlan.from_dict(partial.to_dict())

        self.assertEqual(restored, partial)
        self.assertEqual(partial.config["delete_value_threshold"], -0.2)
        self.assertEqual(partial.config["merge_threshold_skill"], 0.86)

    def test_single_project_scope_and_source_preconditions_are_required(self) -> None:
        payload = self._plan().to_dict()
        payload["memory_project_key"] = ""
        with self.assertRaises(ValueError):
            MaintenancePlan.from_dict(payload)

        payload = self._plan().to_dict()
        payload["operations"][0]["source_preconditions"] = {}
        with self.assertRaises(ValueError):
            MaintenancePlan.from_dict(payload)

        payload = self._plan().to_dict()
        payload["operations"][0]["source_preconditions"]["mem-1"] = {}
        with self.assertRaises(ValueError):
            MaintenancePlan.from_dict(payload)

        payload = self._plan().to_dict()
        payload["operations"][0]["source_preconditions"]["mem-1"]["tier"] = "tip"
        with self.assertRaises(ValueError):
            MaintenancePlan.from_dict(payload)

    def test_plan_rejects_multiple_primary_operations_for_one_source(self) -> None:
        plan = self._plan()

        with self.assertRaises(ValueError):
            MaintenancePlan(
                schema_version=plan.schema_version,
                policy=plan.policy,
                plan_id="maint-conflict",
                repository_revision=plan.repository_revision,
                scope_mode=plan.scope_mode,
                memory_project_key=plan.memory_project_key,
                as_of=plan.as_of,
                config=plan.config,
                input_summary=plan.input_summary,
                operations=(plan.operations[0], plan.operations[0]),
                summary=plan.summary,
            )

    def test_operation_and_mutation_metadata_ids_are_tamper_evident(self) -> None:
        _, plan, _ = self._promotion_fixture()
        operation_id_tampered = plan.to_dict()
        operation_id_tampered["operations"][0]["operation_id"] = "op-tampered"
        with self.assertRaises(ValueError):
            MaintenancePlan.from_dict(operation_id_tampered)

        metadata_id_tampered = plan.to_dict()
        metadata_id_tampered["operations"][0]["replacements"][0]["metadata"][
            "maintenance_operation_id"
        ] = "op-tampered"
        with self.assertRaises(ValueError):
            MaintenancePlan.from_dict(metadata_id_tampered)

    def test_plan_rejects_noncanonical_as_of_and_rehashed_summary(self) -> None:
        payload = self._plan().to_dict()
        payload["as_of"] = "2026-07-10T08:00:00+08:00"
        with self.assertRaisesRegex(ValueError, "canonical timezone-aware UTC"):
            MaintenancePlan.from_dict(payload)

        payload = self._plan().to_dict()
        payload["summary"]["keep"] += 1
        _refresh_plan_id(payload)
        with self.assertRaisesRegex(ValueError, "summary does not match"):
            MaintenancePlan.from_dict(payload)

    def test_reviewed_plan_rejects_invalid_evidence_domain_values(self) -> None:
        payload = self._plan().to_dict()
        payload["operations"][0]["evidence"][0]["confidence"] = 2.0
        _refresh_plan_id(payload)

        with self.assertRaisesRegex(ValueError, "evidence confidence"):
            maintenance_validation.parse_maintenance_plan(payload)

    def test_rehashed_promotion_target_must_match_semantic_contract(self) -> None:
        _, plan, _ = self._promotion_fixture()
        cases = (
            ("created_at", "1999-01-01T00:00:00+00:00"),
            ("source", "manual"),
            ("metadata.created_by", "writer"),
            ("metadata.source_task", "forged-task"),
            ("metadata.maintenance_parent_id", "forged-parent"),
        )
        for path, value in cases:
            with self.subTest(path=path):
                payload = plan.to_dict()
                target = payload["operations"][0]["additions"][0]
                if path.startswith("metadata."):
                    target["metadata"][path.removeprefix("metadata.")] = value
                else:
                    target[path] = value
                _refresh_plan_id(payload)
                with self.assertRaisesRegex(ValueError, "deterministic semantics"):
                    maintenance_validation.parse_maintenance_plan(payload)

    def test_rehashed_promotion_target_id_must_be_deterministic(self) -> None:
        _, plan, _ = self._promotion_fixture()
        payload = plan.to_dict()
        operation = payload["operations"][0]
        forged_id = "exp_maint_0000000000000000"
        operation["target_ids"] = [forged_id]
        operation["additions"][0]["id"] = forged_id
        operation["replacements"][0]["metadata"]["maintenance_promoted_to"] = forged_id
        operation_id = maintenance_module._operation_id(
            action=MaintenanceAction.PROMOTE,
            source_ids=operation["source_ids"],
            target_ids=operation["target_ids"],
            replacements=operation["replacements"],
            additions=operation["additions"],
        )
        operation["operation_id"] = operation_id
        operation["replacements"][0]["metadata"]["maintenance_operation_id"] = operation_id
        operation["additions"][0]["metadata"]["maintenance_operation_id"] = operation_id
        _refresh_plan_id(payload)

        with self.assertRaisesRegex(ValueError, "target id is not deterministic"):
            maintenance_validation.parse_maintenance_plan(payload)

    def test_rehashed_merge_lineage_must_cover_exact_sources(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        anchor = _entry(
            "merge-a",
            content="Run parser tests before editing.",
            created_by=writer,
        )
        source = _entry(
            "merge-b",
            content="Run parser test before edits.",
            created_by=writer,
        )
        with patch.object(maintenance_planner, "redundancy_score", return_value=0.95):
            plan = build_maintenance_plan(
                entries=[anchor, source],
                attribution={},
                repository_revision="sha256:merge-lineage",
                project_key=PROJECT_KEY,
                as_of=NOW,
                config=MaintenanceConfig(max_promotions=0),
            )
        payload = plan.to_dict()
        operation = next(item for item in payload["operations"] if item["action"] == "merge")
        operation["replacements"][0]["metadata"]["maintenance_source_ids"] = ["merge-a"]
        _refresh_plan_id(payload)

        with self.assertRaisesRegex(ValueError, "merge replacement lineage mismatch"):
            maintenance_validation.parse_maintenance_plan(payload)

    def test_mutation_payload_rejects_forged_fingerprint(self) -> None:
        _, plan, _ = self._promotion_fixture()
        payload = plan.to_dict()
        payload["operations"][0]["additions"][0]["fingerprint"] = "f" * 64

        with self.assertRaises(ValueError):
            MaintenancePlan.from_dict(payload)

    def test_existing_promotion_target_must_exist_and_match_skill_identity(self) -> None:
        source, _, valid_operation = self._promotion_fixture()
        wrong_tier = _entry(
            "wrong-tier",
            tier="tool",
            content=source.content,
            created_by=ExperienceCreatedBy.WRITER,
        )
        wrong_fingerprint = _entry(
            "wrong-fingerprint",
            tier="skill",
            content="Different skill content",
            created_by=ExperienceCreatedBy.WRITER,
        )
        cases = [
            ("ghost-skill", [source]),
            (wrong_tier.id, [source, wrong_tier]),
            (wrong_fingerprint.id, [source, wrong_fingerprint]),
        ]
        for target_id, repository in cases:
            with self.subTest(target_id=target_id):
                payload = valid_operation.to_dict()
                payload["target_ids"] = [target_id]
                payload["additions"] = []
                payload["replacements"][0]["metadata"]["maintenance_promoted_to"] = target_id
                operation_id = maintenance_module._operation_id(
                    action=MaintenanceAction.PROMOTE,
                    source_ids=payload["source_ids"],
                    target_ids=payload["target_ids"],
                    replacements=payload["replacements"],
                    additions=payload["additions"],
                )
                payload["operation_id"] = operation_id
                payload["replacements"][0]["metadata"][
                    "maintenance_operation_id"
                ] = operation_id
                malformed = MaintenanceOperation.from_dict(payload)

                with self.assertRaises(ValueError):
                    maintenance_module._validate_operation_conflicts(
                        (malformed,),
                        repository_entries=repository,
                    )

    def test_merge_contract_rejects_cross_tier_truth_and_tier_spoofing(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        tip = _entry(
            "merge-tip",
            tier="tip",
            content="Inspect parser failure before editing.",
            created_by=writer,
        )
        skill = _entry(
            "merge-skill",
            tier="skill",
            content="Run parser verification before editing.",
            created_by=writer,
        )

        def operation_payload(*, declared_tiers: tuple[str, str], anchor: MemoryEntry):
            source_ids = (tip.id, skill.id)
            preconditions = {
                tip.id: {
                    "fingerprint": tip.fingerprint,
                    "tier": declared_tiers[0],
                    "scope": tip.scope.value,
                    "project_key": tip.project_key,
                },
                skill.id: {
                    "fingerprint": skill.fingerprint,
                    "tier": declared_tiers[1],
                    "scope": skill.scope.value,
                    "project_key": skill.project_key,
                },
            }
            replacement = anchor.to_dict()
            replacement["metadata"]["maintenance_action"] = "merge"
            operation_id = maintenance_module._operation_id(
                action=MaintenanceAction.MERGE,
                source_ids=source_ids,
                target_ids=(anchor.id,),
                replacements=(replacement,),
                additions=(),
            )
            replacement["metadata"]["maintenance_operation_id"] = operation_id
            return {
                "operation_id": operation_id,
                "action": "merge",
                "source_ids": list(source_ids),
                "source_tiers": list(declared_tiers),
                "source_preconditions": preconditions,
                "target_ids": [anchor.id],
                "reason_codes": ["near_duplicate_complete_link"],
                "redundancy_score": 0.95,
                "evidence": [],
                "remove_ids": [item for item in source_ids if item != anchor.id],
                "replacements": [replacement],
                "additions": [],
            }

        truthful = operation_payload(declared_tiers=("tip", "skill"), anchor=tip)
        with self.assertRaises(ValueError):
            MaintenanceOperation.from_dict(truthful)

        spoofed = MaintenanceOperation.from_dict(
            operation_payload(declared_tiers=("skill", "skill"), anchor=skill)
        )
        with self.assertRaises(ValueError):
            maintenance_module._validate_operation_conflicts(
                (spoofed,),
                repository_entries=[tip, skill],
            )

    def test_merge_replacement_must_preserve_anchor_repository_identity(self) -> None:
        writer = ExperienceCreatedBy.WRITER
        anchor = _entry(
            "anchor",
            content="Run parser tests before editing.",
            created_by=writer,
            run_id="run-anchor",
        )
        source = _entry(
            "source",
            content="Run parser test before edits.",
            created_by=writer,
        )
        with patch.object(maintenance_planner, "redundancy_score", return_value=0.95):
            plan = build_maintenance_plan(
                entries=[anchor, source],
                attribution={},
                repository_revision="sha256:anchor-contract",
                project_key=PROJECT_KEY,
                as_of=NOW,
                config=MaintenanceConfig(max_promotions=0),
            )
        merge = next(item for item in plan.operations if item.action == MaintenanceAction.MERGE)
        payload = merge.to_dict()
        payload["replacements"][0]["project_key"] = "other-project"
        malformed = MaintenanceOperation.from_dict(payload)

        with self.assertRaises(ValueError):
            maintenance_module._validate_operation_conflicts(
                (malformed,),
                repository_entries=[anchor, source],
            )


if __name__ == "__main__":
    unittest.main()
