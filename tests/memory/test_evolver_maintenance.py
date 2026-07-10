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
    MaintenanceAction,
    MaintenanceAttributionError,
    MaintenanceConfig,
    MaintenanceOperation,
    MaintenancePlan,
    MemoryAttributionRecord,
    build_experience_entry,
    load_maintenance_plan,
    load_project_attribution,
    maintenance_evidence_for_entry,
    maintenance_plan_json,
    write_maintenance_plan,
)
from my_agent.memory.types import MemoryScope


PROJECT_KEY = "manifest:demo:memory:shared_stream:stream:python"
NOW = datetime(2026, 7, 10, tzinfo=timezone.utc)


def _entry(
    memory_id: str = "mem-1",
    *,
    tier: str = "skill",
    project_key: str = PROJECT_KEY,
    scope: MemoryScope = MemoryScope.PROJECT,
    metadata: dict | None = None,
):
    return build_experience_entry(
        id=memory_id,
        content=f"experience {memory_id}",
        tier=tier,
        project_key=project_key,
        scope=scope,
        created_at=NOW,
        source_task="task-1",
        extra_metadata=metadata,
    )


def _record(
    memory_id: str = "mem-1",
    *,
    tier: str = "skill",
    project_key: str = PROJECT_KEY,
) -> MemoryAttributionRecord:
    return MemoryAttributionRecord(
        memory_id=memory_id,
        tier=tier,
        memory_project_key=project_key,
        candidate_count=8,
        selected_count=4,
        not_selected_count=4,
        value=0.25,
        confidence=0.75,
        last_used="2026-07-09T00:00:00+00:00",
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
