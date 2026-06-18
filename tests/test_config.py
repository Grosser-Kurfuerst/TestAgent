from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig


class AgentConfigTests(unittest.TestCase):
    def test_missing_dotenv_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / ".env"

            with self.assertRaisesRegex(FileNotFoundError, "Configuration file not found"):
                AgentConfig.from_env(env_file=missing)

    def test_config_defaults_from_empty_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(env_file=env_file)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.api_key, "")
        self.assertIsNone(config.base_url)
        self.assertEqual(config.model, "gpt-4o-mini")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_steps, 8)
        self.assertEqual(config.command_timeout, 60)
        self.assertEqual(str(config.trace_dir), "traces")
        self.assertFalse(config.use_fake_llm)
        self.assertEqual(config.plan_task_max_steps, 6)
        self.assertEqual(config.plan_max_tasks, 12)
        self.assertEqual(config.plan_max_replans, 1)
        self.assertEqual(config.agent_mode, "auto")

    def test_openai_provider_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MY_AGENT_LLM_PROVIDER=openai\n", encoding="utf-8")

            config = AgentConfig.from_env(env_file=env_file)

        with self.assertRaisesRegex(RuntimeError, "No API key configured"):
            config.require_api_key()

    def test_fake_provider_does_not_require_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MY_AGENT_LLM_PROVIDER=fake\n", encoding="utf-8")

            config = AgentConfig.from_env(env_file=env_file)

        config.require_api_key()
        self.assertTrue(config.use_fake_llm)

    def test_config_loads_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "# local model config",
                        "MY_AGENT_LLM_PROVIDER=openai",
                        "MY_AGENT_API_KEY='file-key'",
                        'MY_AGENT_BASE_URL="https://example.test/v1"',
                        "MY_AGENT_MODEL=test-model",
                        "MY_AGENT_TEMPERATURE=0.2",
                        "MY_AGENT_MAX_STEPS=5",
                        "MY_AGENT_COMMAND_TIMEOUT=12",
                        "MY_AGENT_TRACE_DIR=tmp-traces",
                    ]
                ),
                encoding="utf-8",
            )

            config = AgentConfig.from_env(env_file=env_file)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.api_key, "file-key")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.max_steps, 5)
        self.assertEqual(config.command_timeout, 12)
        self.assertEqual(str(config.trace_dir), "tmp-traces")

    def test_tool_flags_default_off_and_can_be_enabled_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            default_config = AgentConfig.from_env(env_file=env_file)
            enabled_config = AgentConfig.from_env(
                env={
                    "AGENTCLI_ENABLE_PROJECT_TOOLS": "1",
                    "AGENTCLI_ENABLE_PROJECT_PLUGINS": "true",
                },
                env_file=env_file,
            )

        self.assertFalse(default_config.enable_project_tools)
        self.assertFalse(default_config.enable_project_plugins)
        self.assertTrue(enabled_config.enable_project_tools)
        self.assertTrue(enabled_config.enable_project_plugins)

    def test_system_environment_does_not_override_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MY_AGENT_LLM_PROVIDER=fake\nMY_AGENT_API_KEY=file-key\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"MY_AGENT_LLM_PROVIDER": "openai", "MY_AGENT_API_KEY": "env-key"}, clear=True):
                config = AgentConfig.from_env(env_file=env_file)

        self.assertEqual(config.provider, "fake")
        self.assertEqual(config.api_key, "file-key")
        self.assertTrue(config.use_fake_llm)

    def test_dotenv_export_syntax_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("export MY_AGENT_LLM_PROVIDER=fake\n", encoding="utf-8")

            config = AgentConfig.from_env(env_file=env_file)

        self.assertEqual(config.provider, "fake")
        self.assertTrue(config.use_fake_llm)

    def test_unsupported_provider_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MY_AGENT_LLM_PROVIDER=local\n", encoding="utf-8")

            config = AgentConfig.from_env(env_file=env_file)

        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            config.require_valid_provider()

    def test_plan_config_values_are_loaded_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_PLAN_TASK_MAX_STEPS": "3",
                    "AGENTCLI_PLAN_MAX_TASKS": "5",
                    "AGENTCLI_PLAN_MAX_REPLANS": "0",
                    "AGENTCLI_AGENT_MODE": "plan",
                },
                env_file=env_file,
            )

        self.assertEqual(config.plan_task_max_steps, 3)
        self.assertEqual(config.plan_max_tasks, 5)
        self.assertEqual(config.plan_max_replans, 0)
        self.assertEqual(config.agent_mode, "plan")

    def test_invalid_agent_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("AGENTCLI_AGENT_MODE=invalid\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported AGENTCLI_AGENT_MODE"):
                AgentConfig.from_env(env_file=env_file)


if __name__ == "__main__":
    unittest.main()
