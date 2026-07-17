from __future__ import annotations

import unittest

from my_agent.policy.identity import canonical_sha256
from my_agent.training.role_views import (
    ActionHindsight,
    ActionPublic,
    CandidateSnapshotEntry,
    CanonicalMessage,
    CanonicalTool,
    CanonicalTrajectoryStep,
    MaintenanceHindsight,
    MaintenancePublic,
    MemoryDiagnostic,
    MemoryValueEvidence,
    RedundancyDiagnostic,
    RepositorySnapshotRef,
    SelectionHindsight,
    SelectionPublic,
    TaskOutcomeRef,
    TrajectoryEvidence,
    WritingHindsight,
    WritingPublic,
    role_view_hash,
)


def _tool() -> CanonicalTool:
    parameters = '{"properties":{},"type":"object"}'
    return CanonicalTool("lookup", "lookup memory", parameters, canonical_sha256({"properties": {}, "type": "object"}))


def _value(memory_id: str = "mem-1") -> MemoryValueEvidence:
    return MemoryValueEvidence(memory_id, "skill", 0.5, 0.25, 0.125, "ready")


def _trajectory() -> TrajectoryEvidence:
    return TrajectoryEvidence(
        "traj-1",
        "group-a",
        "resolved",
        1.0,
        (CanonicalTrajectoryStep(0, "obs", "lookup", "{}", "ok", 1.0),),
    )


class RoleViewContractTests(unittest.TestCase):
    def test_all_role_views_round_trip_with_stable_hash(self) -> None:
        candidate = CandidateSnapshotEntry("SKILL_01", "mem-1", "skill", "use pytest", 0.9, 1, 3)
        trajectory = _trajectory()
        tool = _tool()
        snapshot = RepositorySnapshotRef("rev-1", "project-a", ("mem-1",), canonical_sha256(["mem-1"]))
        outcome = TaskOutcomeRef("task-1", "group-a", 1.0, True, "pytest", "8", canonical_sha256({"command": "pytest"}))
        views = (
            SelectionPublic("fix tests", (candidate,), 1800),
            SelectionHindsight((_value(),)),
            ActionPublic("fix tests", (tool,), (CanonicalMessage("user", "fix tests"),)),
            ActionHindsight((_value(),), trajectory),
            WritingPublic("fix tests", trajectory, 1.0, ("mem-1",)),
            WritingHindsight((_value(),)),
            MaintenancePublic(snapshot, (outcome,), (tool,)),
            MaintenanceHindsight(
                (MemoryDiagnostic("mem-1", "skill", 0.125, 0.25, 2, 1, "2026-07-17T00:00:00Z"),),
                (RedundancyDiagnostic("mem-1", "mem-2", 0.8),),
            ),
        )

        for view in views:
            with self.subTest(view=type(view).__name__):
                restored = type(view).from_dict(view.to_dict())
                self.assertEqual(restored, view)
                self.assertEqual(role_view_hash(restored), role_view_hash(view))

    def test_canonical_json_fields_reject_noncanonical_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical"):
            CanonicalTrajectoryStep(0, "obs", "act", '{"b": 1, "a": 2}', "ok", None)

    def test_task_outcome_rejects_null_evaluator_identity(self) -> None:
        payload = TaskOutcomeRef(
            "task-1",
            "group-a",
            1.0,
            True,
            "pytest",
            "8",
            canonical_sha256({"command": "pytest"}),
        ).to_dict()
        payload["evaluator_name"] = None

        with self.assertRaisesRegex(ValueError, "evaluator_name"):
            TaskOutcomeRef.from_dict(payload)

    def test_trajectory_rejects_null_task_group(self) -> None:
        payload = _trajectory().to_dict()
        payload["task_group"] = None

        with self.assertRaisesRegex(ValueError, "task_group"):
            TrajectoryEvidence.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
