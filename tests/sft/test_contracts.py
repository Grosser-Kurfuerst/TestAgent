from __future__ import annotations

from pathlib import Path
import json
import unittest

from my_agent.policy.identity import canonical_sha256, hash_artifact_path
from my_agent.sft.contracts import (
    CANONICAL_SFT_SCHEMA_VERSION,
    ENVIRONMENT_EXCLUSION_CODES,
    RENDERED_SFT_SCHEMA_VERSION,
    deterministic_tool_call_id,
    validate_expected_output_contract,
)
from my_agent.training.role_views import CanonicalMessage, CanonicalTool


ROOT = Path(__file__).resolve().parents[2]


class SFTContractTests(unittest.TestCase):
    def test_semantic_and_rendered_fixtures_round_trip_and_hash(self) -> None:
        semantic = _load_json(ROOT / "tests/fixtures/sft/canonical_tool_call.json")
        self.assertEqual(semantic["schema_version"], CANONICAL_SFT_SCHEMA_VERSION)
        messages = tuple(CanonicalMessage.from_dict(item) for item in semantic["messages"])
        tools = tuple(CanonicalTool.from_dict(item) for item in semantic["tools"])
        target = CanonicalMessage.from_dict(semantic["target"])
        self.assertEqual([item.to_dict() for item in messages], semantic["messages"])
        self.assertEqual([item.to_dict() for item in tools], semantic["tools"])
        self.assertEqual(target.to_dict(), semantic["target"])
        validate_expected_output_contract(
            semantic["expected_output_kind"],
            semantic["expected_tool_call_count"],
        )
        semantic_payload = {key: value for key, value in semantic.items() if key != "sample_id"}
        self.assertEqual(semantic["sample_id"], canonical_sha256(semantic_payload))

        rendered = _load_json(ROOT / "tests/fixtures/sft/rendered_tool_call.json")
        self.assertEqual(rendered["schema_version"], RENDERED_SFT_SCHEMA_VERSION)
        self.assertEqual(rendered["sample_id"], semantic["sample_id"])
        self.assertEqual(rendered["semantic_sample_hash"], semantic["sample_id"])
        self.assertEqual(
            rendered["input_ids"],
            rendered["prompt_token_ids"] + rendered["completion_token_ids"],
        )
        expected_labels = [
            token if mask else -100
            for token, mask in zip(rendered["input_ids"], rendered["full_sequence_loss_mask"])
        ]
        self.assertEqual(rendered["labels"], expected_labels)
        rendered_payload = {
            key: value for key, value in rendered.items() if key != "rendered_sample_hash"
        }
        self.assertEqual(rendered["rendered_sample_hash"], canonical_sha256(rendered_payload))

    def test_deterministic_call_ids_and_output_contract(self) -> None:
        call_id = deterministic_tool_call_id(
            call_index=0,
            name="read_file",
            arguments='{"path":"src/foo.py"}',
        )
        self.assertEqual(call_id, "call_770ef254227f")
        validate_expected_output_contract("tool_call", 1)
        validate_expected_output_contract("assistant_text", None)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_expected_output_contract("tool_call", 0)
        with self.assertRaisesRegex(ValueError, "canonical JSON"):
            deterministic_tool_call_id(
                call_index=0,
                name="read_file",
                arguments='{"path": "src/foo.py"}',
            )

    def test_lock_and_openai_fixtures_are_immutable_and_object_typed(self) -> None:
        integration = ROOT / "integrations/llamafactory"
        lock = _load_json(integration / "lock.json")
        self.assertEqual(lock["revision"], "95ac3f2373b82662c1bd855c284d3379e6a763d3")
        self.assertNotIn("REPLACE_", json.dumps(lock))
        self.assertEqual(lock["template_mode"], "pinned_patch")
        self.assertEqual(
            lock["template_artifact_hash"],
            hash_artifact_path(ROOT / lock["patch_path"]),
        )
        fixture_manifest = _load_json(integration / "fixtures/manifest.json")
        self.assertEqual(lock["tool_fixture_manifest_hash"], canonical_sha256(fixture_manifest))
        dataset_info = _load_json(integration / "fixtures/dataset_info.json")
        self.assertEqual(lock["dataset_info_fixture_hash"], canonical_sha256(dataset_info))
        self.assertEqual(
            dataset_info["agentcli_sft_train"]["formatting"],
            "agentcli_openai_tools",
        )
        expected_tags = {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function",
            "system_tag": "system",
        }
        for dataset in dataset_info.values():
            self.assertEqual(dataset["tags"], expected_tags)

        for name, expected_hash in fixture_manifest["fixtures"].items():
            fixture = _load_json(integration / "fixtures" / name)
            self.assertEqual(expected_hash, canonical_sha256(fixture))
            tags = dataset_info["agentcli_sft_train"]["tags"]
            for message in fixture["messages"]:
                self.assertIn(message[tags["role_tag"]], expected_tags.values())
                self.assertIn(tags["content_tag"], message)
            call_ids: list[str] = []
            observation_ids: list[str] = []
            for message in fixture["messages"]:
                for tool_call in message.get("tool_calls", []):
                    self.assertIsInstance(tool_call["function"]["arguments"], dict)
                    call_ids.append(tool_call["id"])
                if message["role"] == "tool":
                    observation_ids.append(message["tool_call_id"])
            self.assertEqual(observation_ids, call_ids[: len(observation_ids)])

        self.assertEqual(ENVIRONMENT_EXCLUSION_CODES, {
            "sandbox_unavailable",
            "fixture_setup_failed",
            "required_dependency_unavailable",
        })


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"fixture must be an object: {path}")
    return payload


if __name__ == "__main__":
    unittest.main()
