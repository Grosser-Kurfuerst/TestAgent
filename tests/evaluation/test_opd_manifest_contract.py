from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import (
    ManifestSettings,
    _config_env_values,
    _config_for_eval_env,
    _prepare_manifest_tasks,
)


class OpdManifestContractTests(unittest.TestCase):
    def test_formal_manifest_requires_explicit_task_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_group"):
            _prepare_manifest_tasks(
                ({"id": "task-1", "source": "unit"},),
                settings=ManifestSettings(),
                formal=True,
            )

    def test_smoke_manifest_may_explicitly_fallback_to_source(self) -> None:
        tasks = _prepare_manifest_tasks(
            ({"id": "task-1", "source": "unit"},),
            settings=ManifestSettings(task_group_fallback="source"),
            formal=True,
        )
        self.assertEqual(tasks[0]["task_group"], "unit")

    def test_formal_eval_config_round_trip_uses_only_formal_fields(self) -> None:
        config = AgentConfig.from_env(
            {
                "MY_AGENT_LLM_PROVIDER": "fake",
                "AGENTCLI_MEMORY_EVOLVER_MODE": "formal",
                "AGENTCLI_POLICY_BASE_REVISION": "model-revision-1",
                "AGENTCLI_POLICY_TOKENIZER_REVISION": "tokenizer-revision-1",
                "AGENTCLI_EMBEDDING_REVISION": "embedding-revision-1",
            },
            require_env_file=False,
        )
        values = _config_env_values(config)

        self.assertNotIn("AGENTCLI_MEMORY_EVOLVER_MIN_SCORE", values)
        self.assertNotIn("AGENTCLI_MEMORY_EVOLVER_WRITER_MODE", values)
        self.assertEqual(values["AGENTCLI_POLICY_BASE_REVISION"], "model-revision-1")
        self.assertEqual(values["AGENTCLI_EMBEDDING_REVISION"], "embedding-revision-1")

        with tempfile.TemporaryDirectory() as tmp:
            resolved = _config_for_eval_env(
                config,
                {},
                trace_dir=Path(tmp) / "traces",
                memory_dir=Path(tmp) / "memory",
                command_timeout=10,
            )
        self.assertEqual(resolved.memory_evolver_mode, "formal")
        self.assertEqual(resolved.policy_base_revision, "model-revision-1")


if __name__ == "__main__":
    unittest.main()
