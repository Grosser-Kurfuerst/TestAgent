from __future__ import annotations

import unittest

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.mcp.schema import sanitize_schema


class McpSchemaSanitizerTests(unittest.TestCase):
    def test_removes_json_schema_metadata_and_refs(self) -> None:
        sanitized = sanitize_schema(
            {
                "$schema": "https://json-schema.org",
                "$id": "tool",
                "type": "object",
                "properties": {"path": {"$ref": "#/$defs/path", "description": "Path"}},
            }
        )

        self.assertNotIn("$schema", sanitized)
        self.assertNotIn("$id", sanitized)
        self.assertNotIn("$ref", sanitized["properties"]["path"])
        self.assertEqual(sanitized["type"], "object")
        self.assertTrue(sanitized["additionalProperties"])

    def test_anyof_and_oneof_are_rendered_into_description(self) -> None:
        sanitized = sanitize_schema(
            {
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [{"type": "string"}, {"type": "integer"}],
                    },
                    "mode": {
                        "description": "Mode",
                        "oneOf": [{"type": "string", "description": "name"}, {"type": "boolean"}],
                    },
                },
            }
        )

        value_schema = sanitized["properties"]["value"]
        mode_schema = sanitized["properties"]["mode"]
        self.assertNotIn("anyOf", value_schema)
        self.assertIn("anyOf options", value_schema["description"])
        self.assertNotIn("oneOf", mode_schema)
        self.assertIn("Mode", mode_schema["description"])
        self.assertIn("oneOf options", mode_schema["description"])

    def test_non_object_schema_is_wrapped_as_object(self) -> None:
        sanitized = sanitize_schema({"type": "string", "description": "raw"})

        self.assertEqual(sanitized["type"], "object")
        self.assertEqual(sanitized["properties"], {})
        self.assertIn("Original MCP schema", sanitized["description"])

    def test_missing_type_and_properties_are_filled(self) -> None:
        sanitized = sanitize_schema({"description": "No type."})

        self.assertEqual(sanitized["type"], "object")
        self.assertEqual(sanitized["properties"], {})

    def test_long_descriptions_are_truncated(self) -> None:
        sanitized = sanitize_schema({"type": "object", "description": "x" * 1200})

        self.assertEqual(len(sanitized["description"]), 1003)
        self.assertTrue(sanitized["description"].endswith("..."))


if __name__ == "__main__":
    unittest.main()
