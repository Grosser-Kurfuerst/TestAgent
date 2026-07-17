from __future__ import annotations

import unittest

from my_agent.memory.evolver.maintenance_tools import (
    formal_maintenance_tools,
    parse_maintenance_tool_call,
)
from my_agent.policy.identity import canonical_json_bytes
from my_agent.training.role_views import CanonicalToolCall


def _call(name: str, arguments: dict) -> CanonicalToolCall:
    return CanonicalToolCall("call-1", name, canonical_json_bytes(arguments).decode("utf-8"))


class FormalMaintenanceToolTests(unittest.TestCase):
    def test_tool_schema_is_fixed_and_has_no_promote(self) -> None:
        self.assertEqual(
            tuple(tool.name for tool in formal_maintenance_tools()),
            ("lookup", "merge", "delete", "finish"),
        )

    def test_parser_rejects_promote_and_extra_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_maintenance_tool_call((_call("promote", {"source_ids": ["tip-a"]}),))
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_maintenance_tool_call((_call("finish", {"summary": "done", "promote": True}),))

    def test_one_assistant_turn_must_issue_exactly_one_tool_call(self) -> None:
        finish = _call("finish", {"summary": "done"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_maintenance_tool_call(())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_maintenance_tool_call((finish, finish))


if __name__ == "__main__":
    unittest.main()
