from __future__ import annotations

import unittest

from my_agent.memory.evolver.contracts import MaintenancePlanError
from my_agent.memory.evolver.maintenance_tools import (
    MaintenanceToolCommand,
    build_delete_operation,
    build_merge_operation,
)
from my_agent.memory.evolver.repository_reducer import reduce_repository
from my_agent.memory.evolver.types import ExperienceCreatedBy, ExperienceTier
from tests.memory.experience_fixtures import typed_experience


class RepositoryReducerTests(unittest.TestCase):
    def test_merge_accepts_llm_replacement_without_similarity_threshold(self) -> None:
        entries = [
            typed_experience(
                "skill-a", "alpha", ExperienceTier.SKILL,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
            ),
            typed_experience(
                "skill-b", "beta", ExperienceTier.SKILL,
                project_key="project-a", created_by=ExperienceCreatedBy.WRITER,
            ),
        ]
        command = MaintenanceToolCommand("call-1", "merge", {
            "source_ids": ["skill-a", "skill-b"],
            "replacement": {
                "content": "combined technique",
                "payload": {
                    "category": "testing",
                    "technique": "combined",
                    "preconditions": [],
                    "steps": ["apply combined technique"],
                },
            },
            "reason": "model chose to merge",
        })
        operation = build_merge_operation(command, repository_entries=entries)

        reduced = reduce_repository(entries, (operation,), project_key="project-a")

        self.assertEqual([item.id for item in reduced], ["skill-a"])
        self.assertEqual(reduced[0].content, "combined technique")

    def test_manual_source_is_protected_from_delete(self) -> None:
        entry = typed_experience(
            "tip-manual", "manual", ExperienceTier.TIP,
            project_key="project-a", created_by=ExperienceCreatedBy.MANUAL,
        )
        command = MaintenanceToolCommand("call-1", "delete", {
            "source_ids": ["tip-manual"],
            "reason": "model requested delete",
        })
        operation = build_delete_operation(command, repository_entries=(entry,))

        with self.assertRaisesRegex(MaintenancePlanError, "manual"):
            reduce_repository((entry,), (operation,), project_key="project-a")


if __name__ == "__main__":
    unittest.main()
