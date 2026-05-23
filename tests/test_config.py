from __future__ import annotations

import unittest

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig


class AgentConfigTests(unittest.TestCase):
    def test_config_defaults_without_environment(self) -> None:
        config = AgentConfig.from_env(env={})

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.api_key, "")
        self.assertIsNone(config.base_url)
        self.assertEqual(config.model, "gpt-4o-mini")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_steps, 8)
        self.assertEqual(config.command_timeout, 60)
        self.assertEqual(str(config.trace_dir), "traces")
        self.assertFalse(config.use_fake_llm)

    def test_openai_provider_requires_api_key(self) -> None:
        config = AgentConfig.from_env(env={"MY_AGENT_LLM_PROVIDER": "openai"})

        with self.assertRaisesRegex(RuntimeError, "No API key configured"):
            config.require_api_key()

    def test_fake_provider_does_not_require_api_key(self) -> None:
        config = AgentConfig.from_env(env={"MY_AGENT_LLM_PROVIDER": "fake"})

        config.require_api_key()
        self.assertTrue(config.use_fake_llm)

    def test_unsupported_provider_is_rejected(self) -> None:
        config = AgentConfig.from_env(env={"MY_AGENT_LLM_PROVIDER": "local"})

        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            config.require_valid_provider()


if __name__ == "__main__":
    unittest.main()
