from __future__ import annotations

import unittest

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.observability.maintenance_events import MaintenanceEventCounters


class MaintenanceEventCountersTests(unittest.TestCase):
    def test_reduces_the_shared_maintenance_event_contract(self) -> None:
        counters = MaintenanceEventCounters()

        self.assertTrue(counters.observe("memory.maintenance_started", {}))
        self.assertTrue(counters.observe(
            "memory.maintenance_proposed",
            {
                "keep": 2,
                "delete": 1,
                "merge": 3,
                "promote": 4,
                "source_entries_removed": 5,
                "entries_added": 6,
            },
        ))
        self.assertTrue(counters.observe(
            "memory.maintenance_completed",
            {"status": "committed_with_audit_error"},
        ))
        self.assertTrue(counters.observe("memory.maintenance_failed", {}))

        self.assertEqual(counters.runs, 1)
        self.assertEqual(counters.applied_runs, 1)
        self.assertEqual(counters.keep, 2)
        self.assertEqual(counters.delete, 1)
        self.assertEqual(counters.merge, 3)
        self.assertEqual(counters.promote, 4)
        self.assertEqual(counters.removed_entries, 5)
        self.assertEqual(counters.added_entries, 6)
        self.assertEqual(counters.failures, 1)
        self.assertEqual(counters.committed_with_audit_error, 1)

    def test_ignores_unknown_events_and_invalid_counts(self) -> None:
        counters = MaintenanceEventCounters()

        self.assertFalse(counters.observe("tool.completed", {}))
        counters.observe(
            "memory.maintenance_proposed",
            {"keep": True, "delete": -1, "merge": "2", "promote": None},
        )

        self.assertEqual(counters.keep, 0)
        self.assertEqual(counters.delete, 0)
        self.assertEqual(counters.merge, 0)
        self.assertEqual(counters.promote, 0)


if __name__ == "__main__":
    unittest.main()
