from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
import json

from my_agent.policy.contracts import (
    DecisionResponse,
    completion_mask_to_full_sequence,
)
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_sha256,
    hash_artifact_path,
    load_policy_identity_manifest,
    policy_identity_manifest_payload,
    require_matching_policy_identity,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


class PolicyIdentityContractTests(unittest.TestCase):
    def test_identity_round_trip_and_hash_include_null_adapter(self) -> None:
        identity = PolicyIdentity(
            base_model="model",
            base_revision="revision-1",
            checkpoint_hash=HASH_A,
            adapter_hash=None,
            tokenizer_revision="tokenizer-1",
            tokenizer_hash=HASH_B,
            chat_template_hash=HASH_C,
        )

        payload = identity.to_dict()

        self.assertIn("adapter_hash", payload)
        self.assertIsNone(payload["adapter_hash"])
        self.assertEqual(PolicyIdentity.from_dict(payload), identity)
        self.assertEqual(identity.identity_hash, canonical_sha256(payload))

    def test_identity_missing_field_fails_closed(self) -> None:
        payload = {
            "base_model": "model",
            "base_revision": "revision-1",
            "checkpoint_hash": HASH_A,
            "adapter_hash": None,
            "tokenizer_revision": "tokenizer-1",
            "tokenizer_hash": HASH_B,
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            PolicyIdentity.from_dict(payload)

    def test_direct_identity_rejects_null_required_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "base_model"):
            PolicyIdentity(None, "revision-1", HASH_A, None, "tokenizer-1", HASH_B, HASH_C)

    def test_completion_only_mask_expands_to_full_sequence(self) -> None:
        self.assertEqual(
            completion_mask_to_full_sequence(prompt_token_count=3, completion_mask=(1, 0, 1)),
            (0, 0, 0, 1, 0, 1),
        )

    def test_response_rejects_misaligned_completion_mask(self) -> None:
        identity = PolicyIdentity(
            "model", "rev", HASH_A, None, "tok", HASH_B, HASH_C
        )
        with self.assertRaisesRegex(ValueError, "align"):
            DecisionResponse(
                raw_completion="ok",
                prompt_token_ids=(1,),
                completion_token_ids=(2, 3),
                assistant_loss_mask=(1,),
                parsed_tool_calls=(),
                identity=identity,
            )

    def test_response_rejects_missing_policy_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "PolicyIdentity"):
            DecisionResponse(
                raw_completion="ok",
                prompt_token_ids=(1,),
                completion_token_ids=(2,),
                assistant_loss_mask=(1,),
                parsed_tool_calls=(),
                identity=None,
            )

    def test_artifact_hash_is_stable_and_detects_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.bin").write_bytes(b"b")
            (root / "a.bin").write_bytes(b"a")
            first = hash_artifact_path(root)
            second = hash_artifact_path(root)
            (root / "a.bin").write_bytes(b"changed")
            changed = hash_artifact_path(root)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_identity_mismatch_fails_closed(self) -> None:
        expected = PolicyIdentity("model", "rev", HASH_A, None, "tok", HASH_B, HASH_C)
        actual = PolicyIdentity("model", "rev-2", HASH_A, None, "tok", HASH_B, HASH_C)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            require_matching_policy_identity(expected, actual)

    def test_identity_manifest_round_trip_and_hash_validation(self) -> None:
        identity = PolicyIdentity("model", "rev", HASH_A, None, "tok", HASH_B, HASH_C)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity.json"
            payload = policy_identity_manifest_payload(identity)
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_policy_identity_manifest(path), identity)

            payload["policy_identity_hash"] = "sha256:" + "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                load_policy_identity_manifest(path)


if __name__ == "__main__":
    unittest.main()
