from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.memory.evolver import (
    MAINTENANCE_POLICY,
    MAINTENANCE_SCHEMA_VERSION,
    MAINTENANCE_SCOPE_MODE,
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
):
    return build_experience_entry(
        id=memory_id,
        content=content or f"experience {memory_id}",
        tier=tier,
        project_key=project_key,
        scope=scope,
        created_at=created_at,
        created_by=created_by,
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
    not_selected_count: int = 4,
    last_used: str = "2026-07-09T00:00:00+00:00",
) -> MemoryAttributionRecord:
    return MemoryAttributionRecord(
        memory_id=memory_id,
        tier=tier,
        memory_project_key=project_key,
        candidate_count=candidate_count,
        selected_count=selected_count,
        not_selected_count=not_selected_count,
        value=value,
        confidence=confidence,
        last_used=last_used,
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


class MaintenancePlanContractTests(unittest.TestCase):
    def _plan(self) -> MaintenancePlan:
        operation = MaintenanceOperation(
            operation_id="op-keep-1",
            action=MaintenanceAction.KEEP,
            source_ids=("mem-1",),
            source_tiers=("skill",),
            source_preconditions={
                "mem-1": {
                    "fingerprint": "fp-1",
                    "tier": "skill",
                    "scope": "project",
                    "project_key": PROJECT_KEY,
                }
            },
            reason_codes=("no_maintenance_rule",),
        )
        return MaintenancePlan(
            schema_version=MAINTENANCE_SCHEMA_VERSION,
            policy=MAINTENANCE_POLICY,
            plan_id="maint-contract-1",
            repository_revision="sha256:repository",
            scope_mode=MAINTENANCE_SCOPE_MODE,
            memory_project_key=PROJECT_KEY,
            as_of=NOW.isoformat(),
            config=MaintenanceConfig().to_dict(),
            input_summary={"entries_total": 1},
            operations=(operation,),
            summary={"keep": 1, "delete": 0, "merge": 0, "promote": 0},
        )

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
        partial = MaintenancePlan(
            schema_version=plan.schema_version,
            policy=plan.policy,
            plan_id=plan.plan_id,
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


if __name__ == "__main__":
    unittest.main()
