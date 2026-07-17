from __future__ import annotations

import unittest

from my_agent.policy.contracts import (
    DecisionResponse,
    completion_mask_to_full_sequence,
)
from my_agent.policy.identity import PolicyIdentity, canonical_sha256


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


if __name__ == "__main__":
    unittest.main()
