from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._path import add_src_to_path

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
        self.assertFalse(config.context_window_explicit)
        self.assertFalse(config.response_reserve_tokens_explicit)
        self.assertFalse(config.compression_buffer_tokens_explicit)
        self.assertFalse(config.repo_context_budget_tokens_explicit)
        self.assertFalse(config.tool_schema_budget_tokens_explicit)
        self.assertEqual(config.plan_task_max_steps, 6)
        self.assertEqual(config.plan_max_tasks, 12)
        self.assertEqual(config.plan_max_replans, 1)
        self.assertEqual(config.agent_mode, "auto")
        self.assertEqual(config.team_worker_count, 2)
        self.assertEqual(config.team_max_steps, 12)
        self.assertEqual(config.team_max_retries, 2)
        self.assertEqual(config.team_step_max_steps, 6)
        self.assertEqual(config.team_dependency_context_chars, 4_000)
        self.assertTrue(config.team_parallel_enabled)
        self.assertFalse(config.team_allow_unapproved_results)

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
                        "MY_AGENT_CONTEXT_WINDOW=32000",
                        "MY_AGENT_RESPONSE_RESERVE_TOKENS=5000",
                        "MY_AGENT_COMPRESSION_BUFFER_TOKENS=2500",
                        "AGENTCLI_REPO_CONTEXT_BUDGET_TOKENS=20000",
                        "AGENTCLI_TOOL_SCHEMA_BUDGET_TOKENS=12000",
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
        self.assertEqual(config.context_window, 32_000)
        self.assertTrue(config.context_window_explicit)
        self.assertEqual(config.response_reserve_tokens, 5_000)
        self.assertTrue(config.response_reserve_tokens_explicit)
        self.assertEqual(config.compression_buffer_tokens, 2_500)
        self.assertTrue(config.compression_buffer_tokens_explicit)
        self.assertEqual(config.repo_context_budget_tokens, 20_000)
        self.assertTrue(config.repo_context_budget_tokens_explicit)
        self.assertEqual(config.tool_schema_budget_tokens, 12_000)
        self.assertTrue(config.tool_schema_budget_tokens_explicit)

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
                    "AGENTCLI_AGENT_MODE": "team",
                },
                env_file=env_file,
            )

        self.assertEqual(config.plan_task_max_steps, 3)
        self.assertEqual(config.plan_max_tasks, 5)
        self.assertEqual(config.plan_max_replans, 0)
        self.assertEqual(config.agent_mode, "team")

    def test_team_config_values_are_loaded_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_TEAM_WORKERS": "3",
                    "AGENTCLI_TEAM_MAX_STEPS": "9",
                    "AGENTCLI_TEAM_MAX_RETRIES": "1",
                    "AGENTCLI_TEAM_STEP_MAX_STEPS": "4",
                    "AGENTCLI_TEAM_DEPENDENCY_CONTEXT_CHARS": "1500",
                    "AGENTCLI_TEAM_PARALLEL": "0",
                    "AGENTCLI_TEAM_ALLOW_UNAPPROVED_RESULTS": "yes",
                },
                env_file=env_file,
            )

        self.assertEqual(config.team_worker_count, 3)
        self.assertEqual(config.team_max_steps, 9)
        self.assertEqual(config.team_max_retries, 1)
        self.assertEqual(config.team_step_max_steps, 4)
        self.assertEqual(config.team_dependency_context_chars, 1_500)
        self.assertFalse(config.team_parallel_enabled)
        self.assertTrue(config.team_allow_unapproved_results)

    def test_invalid_agent_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("AGENTCLI_AGENT_MODE=invalid\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported AGENTCLI_AGENT_MODE"):
                AgentConfig.from_env(env_file=env_file)

    def test_memory_config_defaults_from_empty_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(env_file=env_file)

        self.assertEqual(config.memory_dir, Path("~/.agentcli/memory").expanduser())
        self.assertEqual(config.memory_project_key, "")
        self.assertEqual(config.memory_short_term_tokens, 24_000)
        self.assertEqual(config.memory_short_term_entries, 500)
        self.assertEqual(config.memory_context_tokens, 2_000)
        self.assertFalse(config.memory_context_tokens_explicit)
        self.assertEqual(config.memory_retrieval_limit, 8)
        self.assertEqual(config.memory_compression_trigger_ratio, 0.8)
        self.assertEqual(config.memory_retain_recent_turns, 3)
        self.assertEqual(config.memory_map_chunk_size, 5)
        self.assertEqual(config.memory_tool_result_chars, 500)
        self.assertFalse(config.memory_short_term_tokens_explicit)
        self.assertFalse(config.memory_tool_result_chars_explicit)
        self.assertEqual(config.memory_evolver_mode, "off")
        self.assertEqual(config.memory_evolver_top_k_per_tier, 50)
        self.assertEqual(config.memory_evolver_selected_max_items, 20)
        self.assertEqual(config.memory_evolver_min_score, 0.0)
        self.assertEqual(config.memory_evolver_min_experience_entries, 0)
        self.assertEqual(config.memory_evolver_tier_caps["trajectory"], 1)
        self.assertEqual(config.memory_evolver_tier_weights["skill"], 1.20)
        self.assertFalse(config.memory_evolver_writer_enabled)
        self.assertEqual(config.memory_evolver_writer_mode, "fallback")
        self.assertEqual(config.memory_evolver_writer_min_confidence, 0.70)
        self.assertEqual(config.memory_evolver_writer_max_records, 6)
        self.assertEqual(config.memory_evolver_writer_max_input_chars, 12_000)
        self.assertEqual(config.memory_evolver_writer_max_content_chars, 1_200)
        self.assertIsNone(config.memory_evolver_writer_dataset_path)
        # memory_auto_extract defaults to True per plan §13 (line 544).
        self.assertTrue(config.memory_auto_extract)
        self.assertFalse(config.hitl_enabled)
        self.assertEqual(config.hitl_audit_dir, Path("~/.agentcli/audit").expanduser())
        self.assertEqual(config.hitl_non_interactive, "reject")
        self.assertEqual(config.hitl_medium_risk_mode, "ask")
        self.assertFalse(config.hitl_llm_judge_enabled)
        self.assertEqual(config.max_parallel_tools, 4)
        self.assertEqual(config.tool_batch_timeout_seconds, 60)
        self.assertEqual(config.tool_shutdown_grace_seconds, 2)
        self.assertEqual(config.max_process_output_chars, 8_000)
        self.assertTrue(config.plan_parallel_enabled)
        self.assertEqual(config.plan_max_parallel_tasks, 4)
        self.assertEqual(config.plan_task_batch_timeout_seconds, 1_800)
        self.assertEqual(config.team_step_batch_timeout_seconds, 1_800)
        self.assertTrue(config.mcp_enabled)
        self.assertEqual(config.mcp_startup_wait_seconds, 8)
        self.assertEqual(config.mcp_initialize_timeout_seconds, 60)
        self.assertEqual(config.mcp_call_timeout_seconds, 60)
        self.assertEqual(config.mcp_max_startup_workers, 8)
        self.assertTrue(config.mcp_require_approval)
        self.assertTrue(config.mcp_enable_project_servers)

    def test_parallel_config_loaded_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MAX_PARALLEL_TOOLS": "3",
                    "AGENTCLI_TOOL_BATCH_TIMEOUT_SECONDS": "9",
                    "AGENTCLI_TOOL_SHUTDOWN_GRACE_SECONDS": "1",
                    "AGENTCLI_MAX_PROCESS_OUTPUT_CHARS": "2000",
                    "AGENTCLI_PLAN_PARALLEL": "0",
                    "AGENTCLI_PLAN_MAX_PARALLEL_TASKS": "2",
                    "AGENTCLI_PLAN_TASK_BATCH_TIMEOUT_SECONDS": "30",
                    "AGENTCLI_TEAM_STEP_BATCH_TIMEOUT_SECONDS": "40",
                },
                env_file=env_file,
            )

        self.assertEqual(config.max_parallel_tools, 3)
        self.assertEqual(config.tool_batch_timeout_seconds, 9)
        self.assertEqual(config.tool_shutdown_grace_seconds, 1)
        self.assertEqual(config.max_process_output_chars, 2_000)
        self.assertFalse(config.plan_parallel_enabled)
        self.assertEqual(config.plan_max_parallel_tasks, 2)
        self.assertEqual(config.plan_task_batch_timeout_seconds, 30)
        self.assertEqual(config.team_step_batch_timeout_seconds, 40)

    def test_mcp_config_loaded_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MCP": "0",
                    "AGENTCLI_MCP_STARTUP_WAIT_SECONDS": "0",
                    "AGENTCLI_MCP_INITIALIZE_TIMEOUT_SECONDS": "7",
                    "AGENTCLI_MCP_CALL_TIMEOUT_SECONDS": "8",
                    "AGENTCLI_MCP_MAX_STARTUP_WORKERS": "2",
                    "AGENTCLI_MCP_REQUIRE_APPROVAL": "false",
                    "AGENTCLI_MCP_ENABLE_PROJECT_SERVERS": "0",
                },
                env_file=env_file,
            )

        self.assertFalse(config.mcp_enabled)
        self.assertEqual(config.mcp_startup_wait_seconds, 0)
        self.assertEqual(config.mcp_initialize_timeout_seconds, 7)
        self.assertEqual(config.mcp_call_timeout_seconds, 8)
        self.assertEqual(config.mcp_max_startup_workers, 2)
        self.assertFalse(config.mcp_require_approval)
        self.assertFalse(config.mcp_enable_project_servers)

    def test_memory_config_loaded_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY": "1",
                    "AGENTCLI_MEMORY_DIR": str(Path(tmp) / "mem"),
                    "AGENTCLI_MEMORY_PROJECT_KEY": "stream:alpha",
                    "AGENTCLI_MEMORY_SHORT_TERM_TOKENS": "1000",
                    "AGENTCLI_MEMORY_SHORT_TERM_ENTRIES": "50",
                    "AGENTCLI_MEMORY_CONTEXT_TOKENS": "512",
                    "AGENTCLI_MEMORY_RETRIEVAL_LIMIT": "4",
                    "AGENTCLI_MEMORY_COMPRESSION_TRIGGER_RATIO": "0.6",
                    "AGENTCLI_MEMORY_RETAIN_RECENT_TURNS": "2",
                    "AGENTCLI_MEMORY_MAP_CHUNK_SIZE": "3",
                    "AGENTCLI_MEMORY_TOOL_RESULT_CHARS": "250",
                    "AGENTCLI_MEMORY_AUTO_EXTRACT": "true",
                },
                env_file=env_file,
            )

        self.assertEqual(config.memory_dir, Path(tmp) / "mem")
        self.assertEqual(config.memory_project_key, "stream:alpha")
        self.assertTrue(config.memory_enabled)
        self.assertEqual(config.memory_short_term_tokens, 1_000)
        self.assertTrue(config.memory_short_term_tokens_explicit)
        self.assertEqual(config.memory_short_term_entries, 50)
        self.assertEqual(config.memory_context_tokens, 512)
        self.assertTrue(config.memory_context_tokens_explicit)
        self.assertEqual(config.memory_retrieval_limit, 4)
        self.assertEqual(config.memory_compression_trigger_ratio, 0.6)
        self.assertEqual(config.memory_retain_recent_turns, 2)
        self.assertEqual(config.memory_map_chunk_size, 3)
        self.assertEqual(config.memory_tool_result_chars, 250)
        self.assertTrue(config.memory_tool_result_chars_explicit)
        self.assertTrue(config.memory_auto_extract)

    def test_hitl_config_loaded_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_HITL": "1",
                    "AGENTCLI_HITL_AUDIT_DIR": str(Path(tmp) / "audit"),
                    "AGENTCLI_HITL_NON_INTERACTIVE": "reject",
                    "AGENTCLI_HITL_MEDIUM_RISK_MODE": "allow",
                    "AGENTCLI_HITL_LLM_JUDGE": "true",
                },
                env_file=env_file,
            )

        self.assertTrue(config.hitl_enabled)
        self.assertEqual(config.hitl_audit_dir, Path(tmp) / "audit")
        self.assertEqual(config.hitl_non_interactive, "reject")
        self.assertEqual(config.hitl_medium_risk_mode, "allow")
        self.assertTrue(config.hitl_llm_judge_enabled)

    def test_invalid_hitl_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("AGENTCLI_HITL_MEDIUM_RISK_MODE=judge\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "AGENTCLI_HITL_MEDIUM_RISK_MODE"):
                AgentConfig.from_env(env_file=env_file)

    def test_memory_auto_extract_can_be_disabled_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={"AGENTCLI_MEMORY_AUTO_EXTRACT": "0"},
                env_file=env_file,
            )

        self.assertFalse(config.memory_auto_extract)

    def test_memory_can_be_disabled_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={"AGENTCLI_MEMORY": "0"},
                env_file=env_file,
            )

        self.assertFalse(config.memory_enabled)

    def test_my_agent_prefixed_memory_vars_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "MY_AGENT_MEMORY_CONTEXT_TOKENS": "1024",
                    "MY_AGENT_MEMORY_PROJECT_KEY": "stream:legacy",
                    "MY_AGENT_MEMORY_AUTO_EXTRACT": "on",
                },
                env_file=env_file,
            )

        self.assertEqual(config.memory_context_tokens, 1_024)
        self.assertEqual(config.memory_project_key, "stream:legacy")
        self.assertTrue(config.memory_auto_extract)

    def test_memory_evolver_bool_alias_enables_retrieve_select(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={"AGENTCLI_MEMORY_EVOLVER": "1"},
                env_file=env_file,
            )

        self.assertEqual(config.memory_evolver_mode, "retrieve_select")

    def test_memory_evolver_mode_env_takes_precedence_over_bool_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY_EVOLVER": "1",
                    "AGENTCLI_MEMORY_EVOLVER_MODE": "off",
                },
                env_file=env_file,
            )

        self.assertEqual(config.memory_evolver_mode, "off")

    def test_memory_evolver_config_values_are_loaded_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY_EVOLVER_MODE": "retrieve_select",
                    "AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER": "7",
                    "AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS": "6",
                    "AGENTCLI_MEMORY_EVOLVER_MIN_SCORE": "0.25",
                    "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES": "3",
                },
                env_file=env_file,
            )

        self.assertEqual(config.memory_evolver_mode, "retrieve_select")
        self.assertEqual(config.memory_evolver_top_k_per_tier, 7)
        self.assertEqual(config.memory_evolver_selected_max_items, 6)
        self.assertEqual(config.memory_evolver_min_score, 0.25)
        self.assertEqual(config.memory_evolver_min_experience_entries, 3)

    def test_my_agent_prefixed_memory_evolver_vars_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "MY_AGENT_MEMORY_EVOLVER": "true",
                    "MY_AGENT_MEMORY_EVOLVER_TOP_K_PER_TIER": "4",
                    "MY_AGENT_MEMORY_EVOLVER_SELECTED_MAX_ITEMS": "5",
                    "MY_AGENT_MEMORY_EVOLVER_MIN_SCORE": "0.5",
                    "MY_AGENT_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES": "2",
                },
                env_file=env_file,
            )

        self.assertEqual(config.memory_evolver_mode, "retrieve_select")
        self.assertEqual(config.memory_evolver_top_k_per_tier, 4)
        self.assertEqual(config.memory_evolver_selected_max_items, 5)
        self.assertEqual(config.memory_evolver_min_score, 0.5)
        self.assertEqual(config.memory_evolver_min_experience_entries, 2)

    def test_invalid_memory_evolver_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "AGENTCLI_MEMORY_EVOLVER_MODE"):
                AgentConfig.from_env(
                    env={"AGENTCLI_MEMORY_EVOLVER_MODE": "legacy"},
                    env_file=env_file,
                )

    def test_invalid_memory_evolver_numeric_values_fall_back_or_clamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER": "0",
                    "AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS": "-10",
                    "AGENTCLI_MEMORY_EVOLVER_MIN_SCORE": "-0.1",
                    "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES": "-1",
                },
                env_file=env_file,
            )
            fallback = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY_EVOLVER_TOP_K_PER_TIER": "bad",
                    "AGENTCLI_MEMORY_EVOLVER_SELECTED_MAX_ITEMS": "bad",
                    "AGENTCLI_MEMORY_EVOLVER_MIN_SCORE": "bad",
                    "AGENTCLI_MEMORY_EVOLVER_MIN_EXPERIENCE_ENTRIES": "bad",
                },
                env_file=env_file,
            )

        self.assertEqual(config.memory_evolver_top_k_per_tier, 1)
        self.assertEqual(config.memory_evolver_selected_max_items, 1)
        self.assertEqual(config.memory_evolver_min_score, 0.0)
        self.assertEqual(config.memory_evolver_min_experience_entries, 0)
        self.assertEqual(fallback.memory_evolver_top_k_per_tier, 50)
        self.assertEqual(fallback.memory_evolver_selected_max_items, 20)
        self.assertEqual(fallback.memory_evolver_min_score, 0.0)
        self.assertEqual(fallback.memory_evolver_min_experience_entries, 0)

    def test_memory_evolver_writer_config_values_are_loaded_from_env_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")
            dataset_path = Path(tmp) / "writer.jsonl"

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY_EVOLVER_WRITER": "1",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MODE": "llm",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE": "0.8",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_RECORDS": "3",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS": "4000",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS": "500",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_DATASET_PATH": str(dataset_path),
                },
                env_file=env_file,
            )

        self.assertTrue(config.memory_evolver_writer_enabled)
        self.assertEqual(config.memory_evolver_writer_mode, "llm")
        self.assertEqual(config.memory_evolver_writer_min_confidence, 0.8)
        self.assertEqual(config.memory_evolver_writer_max_records, 3)
        self.assertEqual(config.memory_evolver_writer_max_input_chars, 4_000)
        self.assertEqual(config.memory_evolver_writer_max_content_chars, 500)
        self.assertEqual(config.memory_evolver_writer_dataset_path, dataset_path)

    def test_my_agent_prefixed_memory_evolver_writer_vars_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "MY_AGENT_MEMORY_EVOLVER_WRITER": "true",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MODE": "fallback",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE": "0.6",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_RECORDS": "2",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS": "3000",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS": "400",
                    "MY_AGENT_MEMORY_EVOLVER_WRITER_DATASET_PATH": "~/writer.jsonl",
                },
                env_file=env_file,
            )

        self.assertTrue(config.memory_evolver_writer_enabled)
        self.assertEqual(config.memory_evolver_writer_mode, "fallback")
        self.assertEqual(config.memory_evolver_writer_min_confidence, 0.6)
        self.assertEqual(config.memory_evolver_writer_max_records, 2)
        self.assertEqual(config.memory_evolver_writer_max_input_chars, 3_000)
        self.assertEqual(config.memory_evolver_writer_max_content_chars, 400)
        self.assertEqual(config.memory_evolver_writer_dataset_path, Path("~/writer.jsonl").expanduser())

    def test_invalid_memory_evolver_writer_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "AGENTCLI_MEMORY_EVOLVER_WRITER_MODE"):
                AgentConfig.from_env(
                    env={"AGENTCLI_MEMORY_EVOLVER_WRITER_MODE": "agent"},
                    env_file=env_file,
                )

    def test_memory_evolver_full_does_not_implicitly_enable_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            full_config = AgentConfig.from_env(
                env={"AGENTCLI_MEMORY_EVOLVER_MODE": "full"},
                env_file=env_file,
            )
            writer_config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY_EVOLVER_MODE": "full",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER": "1",
                },
                env_file=env_file,
            )

        self.assertEqual(full_config.memory_evolver_mode, "full")
        self.assertFalse(full_config.memory_evolver_writer_enabled)
        self.assertEqual(writer_config.memory_evolver_mode, "full")
        self.assertTrue(writer_config.memory_evolver_writer_enabled)

    def test_invalid_memory_evolver_writer_numeric_values_fall_back_or_clamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            config = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE": "1.5",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_RECORDS": "-1",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS": "-20",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS": "-30",
                },
                env_file=env_file,
            )
            fallback = AgentConfig.from_env(
                env={
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MIN_CONFIDENCE": "bad",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_RECORDS": "bad",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_INPUT_CHARS": "bad",
                    "AGENTCLI_MEMORY_EVOLVER_WRITER_MAX_CONTENT_CHARS": "bad",
                },
                env_file=env_file,
            )

        self.assertEqual(config.memory_evolver_writer_min_confidence, 1.0)
        self.assertEqual(config.memory_evolver_writer_max_records, 0)
        self.assertEqual(config.memory_evolver_writer_max_input_chars, 0)
        self.assertEqual(config.memory_evolver_writer_max_content_chars, 0)
        self.assertEqual(fallback.memory_evolver_writer_min_confidence, 0.70)
        self.assertEqual(fallback.memory_evolver_writer_max_records, 6)
        self.assertEqual(fallback.memory_evolver_writer_max_input_chars, 12_000)
        self.assertEqual(fallback.memory_evolver_writer_max_content_chars, 1_200)

    def test_invalid_compression_ratio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "memory_compression_trigger_ratio"):
                AgentConfig.from_env(
                    env={"AGENTCLI_MEMORY_COMPRESSION_TRIGGER_RATIO": "1.5"},
                    env_file=env_file,
                )

    def test_zero_compression_ratio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "memory_compression_trigger_ratio"):
                AgentConfig.from_env(
                    env={"AGENTCLI_MEMORY_COMPRESSION_TRIGGER_RATIO": "0"},
                    env_file=env_file,
                )

    def test_non_positive_memory_int_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "plan configuration values must be >= 1."):
                AgentConfig.from_env(
                    env={"AGENTCLI_MEMORY_CONTEXT_TOKENS": "0"},
                    env_file=env_file,
                )


if __name__ == "__main__":
    unittest.main()
