from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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
        env = _formal_env()
        env["AGENTCLI_MEMORY_EVOLVER_COLLECTION_ROUND"] = "2"
        env["AGENTCLI_MEMORY_EVOLVER_DATASET_SPLIT"] = "validation"
        env["AGENTCLI_MEMORY_EVOLVER_GENERATION_TEMPERATURE"] = "0"
        env["AGENTCLI_MEMORY_EVOLVER_GENERATION_TOP_P"] = "1"
        config = AgentConfig.from_env(env, require_env_file=False)

        self.assertEqual(config.memory_evolver_mode, "formal")
        self.assertEqual(config.memory_evolver_candidate_top_k_per_tier, 50)
        self.assertEqual(config.memory_evolver_maintenance_interval_tasks, 30)
        self.assertEqual(config.memory_evolver_generation_temperature, 0.0)
        self.assertEqual(config.memory_evolver_generation_top_p, 1.0)
        self.assertEqual(config.memory_evolver_writing_top_fraction, 0.30)
        self.assertEqual(config.memory_evolver_collection_round, 2)
        self.assertEqual(config.memory_evolver_dataset_split, "validation")

    def test_formal_config_rejects_zero_generation_top_p(self) -> None:
        env = _formal_env()
        env["AGENTCLI_MEMORY_EVOLVER_GENERATION_TOP_P"] = "0"

        with self.assertRaisesRegex(ValueError, "generation top_p"):
            AgentConfig.from_env(env, require_env_file=False)

    def test_formal_config_rejects_unknown_dataset_split(self) -> None:
        env = _formal_env()
        env["AGENTCLI_MEMORY_EVOLVER_DATASET_SPLIT"] = "holdout"
        with self.assertRaisesRegex(ValueError, "dataset split"):
            AgentConfig.from_env(env, require_env_file=False)

    def test_formal_config_rejects_legacy_rule_fields(self) -> None:
        env = _formal_env()
        env["AGENTCLI_MEMORY_EVOLVER_WRITER_MODE"] = "llm"
        with self.assertRaisesRegex(ValueError, "legacy rule fields"):
            AgentConfig.from_env(env, require_env_file=False)

    def test_direct_formal_config_rejects_non_default_legacy_rule_fields(self) -> None:
        config = AgentConfig.from_env(_formal_env(), require_env_file=False)
        cases = {
            "memory_evolver_top_k_per_tier": 7,
            "memory_evolver_min_score": 0.8,
            "memory_evolver_min_experience_entries": 2,
            "memory_evolver_tier_caps": {"trajectory": 2, "tip": 2, "skill": 2, "tool": 2},
            "memory_evolver_tier_weights": {"trajectory": 1.0, "tip": 1.0, "skill": 1.0, "tool": 1.0},
            "memory_evolver_writer_enabled": True,
            "memory_evolver_writer_mode": "llm",
            "memory_evolver_writer_min_confidence": 0.9,
            "memory_evolver_writer_max_records": 2,
            "memory_evolver_writer_max_input_chars": 1_000,
            "memory_evolver_writer_max_content_chars": 100,
            "memory_evolver_writer_dataset_path": Path("/tmp/formal-writer.jsonl"),
        }
        for field_name, value in cases.items():
            with self.subTest(field_name=field_name):
                invalid = replace(config, **{field_name: value})
                with self.assertRaisesRegex(ValueError, field_name):
                    invalid.require_valid_formal_evolver()

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

    def test_formal_ablation_requires_its_runtime_backend(self) -> None:
        env = _formal_env()
        env["AGENTCLI_OPD_ABLATION"] = "lexical_retrieval"
        with self.assertRaisesRegex(ValueError, "runtime contract mismatch"):
            AgentConfig.from_env(env, require_env_file=False)

        env["AGENTCLI_MEMORY_EVOLVER_RETRIEVAL_BACKEND"] = "lexical_ablation"
        config = AgentConfig.from_env(env, require_env_file=False)

        self.assertEqual(config.opd_ablation, "lexical_retrieval")
        self.assertEqual(config.memory_evolver_retrieval_backend, "lexical_ablation")

    def test_no_maintenance_ablation_disables_maintainer_execution(self) -> None:
        env = _formal_env()
        env["AGENTCLI_OPD_ABLATION"] = "no_maintenance"
        env["AGENTCLI_MEMORY_EVOLVER_MAINTENANCE_ENABLED"] = "0"

        config = AgentConfig.from_env(env, require_env_file=False)

        self.assertFalse(config.memory_evolver_maintenance_enabled)


if __name__ == "__main__":
    unittest.main()
