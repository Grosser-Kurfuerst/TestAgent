from __future__ import annotations

import unittest

from my_agent.config import AgentConfig


def _formal_env() -> dict[str, str]:
    return {
        "MY_AGENT_LLM_PROVIDER": "fake",
        "AGENTCLI_MEMORY_EVOLVER_MODE": "formal",
        "AGENTCLI_POLICY_BACKEND": "transformers",
        "AGENTCLI_POLICY_BASE_REVISION": "model-revision-1",
        "AGENTCLI_POLICY_TOKENIZER_REVISION": "tokenizer-revision-1",
        "AGENTCLI_POLICY_IDENTITY_MANIFEST": "/tmp/policy-identity.json",
        "AGENTCLI_EMBEDDING_REVISION": "embedding-revision-1",
    }


class FormalConfigTests(unittest.TestCase):
    def test_formal_config_loads_new_contract_fields(self) -> None:
        config = AgentConfig.from_env(_formal_env(), require_env_file=False)

        self.assertEqual(config.memory_evolver_mode, "formal")
        self.assertEqual(config.memory_evolver_candidate_top_k_per_tier, 50)
        self.assertEqual(config.memory_evolver_maintenance_interval_tasks, 30)
        self.assertEqual(config.memory_evolver_writing_top_fraction, 0.30)

    def test_formal_config_rejects_legacy_rule_fields(self) -> None:
        env = _formal_env()
        env["AGENTCLI_MEMORY_EVOLVER_WRITER_MODE"] = "llm"
        with self.assertRaisesRegex(ValueError, "legacy rule fields"):
            AgentConfig.from_env(env, require_env_file=False)

    def test_formal_config_rejects_floating_or_missing_revisions(self) -> None:
        env = _formal_env()
        env["AGENTCLI_POLICY_BASE_REVISION"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            AgentConfig.from_env(env, require_env_file=False)

    def test_formal_config_requires_identity_manifest(self) -> None:
        env = _formal_env()
        del env["AGENTCLI_POLICY_IDENTITY_MANIFEST"]
        with self.assertRaisesRegex(ValueError, "policy_identity_manifest"):
            AgentConfig.from_env(env, require_env_file=False)


if __name__ == "__main__":
    unittest.main()
