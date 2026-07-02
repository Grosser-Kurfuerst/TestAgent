from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.context import (
    AgentContextManager,
    ContextOverBudgetError,
    ContextProfile,
    budget_tool_definitions,
    estimate_tokens,
)
from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message, messages_to_openai
from my_agent.memory import MemoryManager, MemoryScope
from my_agent.tools import ToolExecutionResult


def _config(memory_dir: Path, **overrides: object) -> AgentConfig:
    values: dict[str, object] = {
        "provider": "fake",
        "api_key": "",
        "base_url": None,
        "model": "fake",
        "temperature": 0.0,
        "max_steps": 8,
        "command_timeout": 60,
        "trace_dir": Path("traces"),
        "use_fake_llm": True,
        "memory_dir": memory_dir,
    }
    values.update(overrides)
    return AgentConfig(**values)


class ContextProfileTests(unittest.TestCase):
    def test_explicit_context_window_overrides_model_inference(self) -> None:
        config = _config(
            Path("memory"),
            model="gpt-4.1-mini",
            context_window=32_000,
            context_window_explicit=True,
        )

        profile = ContextProfile.resolve(config, config.model)

        self.assertEqual(profile.max_context_tokens, 32_000)
        self.assertEqual(profile.response_reserve_tokens, 4_000)
        self.assertEqual(profile.compression_buffer_tokens, 2_000)
        self.assertEqual(profile.compression_trigger_tokens, 26_000)
        self.assertEqual(profile.dynamic_profile_source, "config")

    def test_model_window_inference_and_dynamic_budgets(self) -> None:
        config = _config(Path("memory"), model="gpt-4.1-mini")

        profile = ContextProfile.resolve(config, config.model)

        self.assertEqual(profile.max_context_tokens, 1_000_000)
        self.assertEqual(profile.dynamic_profile_source, "model")
        self.assertEqual(profile.short_term_storage_token_limit, 850_000)
        self.assertEqual(profile.memory_context_tokens, 10_000)
        self.assertEqual(profile.tool_result_char_limit, 60_000)
        self.assertEqual(profile.response_reserve_tokens, 100_000)
        self.assertEqual(profile.compression_buffer_tokens, 50_000)
        self.assertEqual(profile.compression_trigger_tokens, 850_000)
        self.assertEqual(profile.repo_context_budget_tokens, 120_000)
        self.assertEqual(profile.tool_schema_budget_tokens, 80_000)

    def test_explicit_memory_context_tokens_override_dynamic_budget(self) -> None:
        config = _config(
            Path("memory"),
            model="gpt-4.1-mini",
            memory_context_tokens=2_048,
            memory_context_tokens_explicit=True,
        )

        profile = ContextProfile.resolve(config, config.model)

        self.assertEqual(profile.max_context_tokens, 1_000_000)
        self.assertEqual(profile.memory_context_tokens, 2_048)

    def test_compression_trigger_uses_configured_reserve_and_buffer(self) -> None:
        config = _config(
            Path("memory"),
            context_window=32_000,
            context_window_explicit=True,
            response_reserve_tokens=4_000,
            compression_buffer_tokens=3_000,
            memory_short_term_tokens=50_000,
            memory_short_term_tokens_explicit=True,
        )

        profile = ContextProfile.resolve(config, config.model)

        self.assertEqual(profile.response_reserve_tokens, 4_000)
        self.assertEqual(profile.compression_buffer_tokens, 3_000)
        self.assertEqual(profile.compression_trigger_tokens, 25_000)
        self.assertEqual(profile.short_term_storage_token_limit, 50_000)

    def test_unknown_model_falls_back_to_default_window(self) -> None:
        config = _config(Path("memory"), model="unknown-local-model")

        profile = ContextProfile.resolve(config, config.model)

        self.assertEqual(profile.max_context_tokens, 128_000)
        self.assertEqual(profile.memory_context_tokens, 1_280)
        self.assertEqual(profile.response_reserve_tokens, 12_800)
        self.assertEqual(profile.compression_buffer_tokens, 6_400)
        self.assertEqual(profile.compression_trigger_tokens, 108_800)
        self.assertEqual(profile.short_term_storage_token_limit, 108_800)
        self.assertEqual(profile.repo_context_budget_tokens, 15_360)
        self.assertEqual(profile.tool_schema_budget_tokens, 10_240)
        self.assertEqual(profile.dynamic_profile_source, "default")

    def test_explicit_repo_and_tool_schema_budgets_override_dynamic_budget(self) -> None:
        config = _config(
            Path("memory"),
            model="gpt-4.1-mini",
            repo_context_budget_tokens=32_000,
            repo_context_budget_tokens_explicit=True,
            tool_schema_budget_tokens=16_000,
            tool_schema_budget_tokens_explicit=True,
        )

        profile = ContextProfile.resolve(config, config.model)

        self.assertEqual(profile.max_context_tokens, 1_000_000)
        self.assertEqual(profile.repo_context_budget_tokens, 32_000)
        self.assertEqual(profile.tool_schema_budget_tokens, 16_000)

    def test_tool_budget_omits_non_core_tools_without_truncating_schema(self) -> None:
        config = _config(
            Path("memory"),
            tool_schema_budget_tokens=500,
            tool_schema_budget_tokens_explicit=True,
        )
        profile = ContextProfile.resolve(config, config.model)
        core = _tool_definition("read_file", "core tool" * 1000)
        extra = _tool_definition("mcp_big_tool", "extra tool" * 2000)

        result = budget_tool_definitions([core, extra], profile)

        self.assertEqual(result.included_names, ("read_file",))
        self.assertEqual(result.omitted_names, ("mcp_big_tool",))
        self.assertEqual(result.definitions[0]["function"]["description"], core["function"]["description"])
        self.assertTrue(result.over_budget)

    def test_tool_budget_uses_full_list_estimate_before_including_non_core_tool(self) -> None:
        first = _tool_definition("project_a", "small")
        second = _tool_definition("project_b", "small")
        budget = estimate_tokens([first, second]) - 1
        profile = ContextProfile(tool_schema_budget_tokens=budget)

        result = budget_tool_definitions([first, second], profile)

        self.assertEqual(result.included_names, ("project_a",))
        self.assertEqual(result.omitted_names, ("project_b",))
        self.assertLessEqual(estimate_tokens(result.definitions), budget)

    def test_small_remaining_memory_budget_is_available_to_long_term_memory(self) -> None:
        config = _config(
            Path("memory"),
            context_window=8_000,
            context_window_explicit=True,
            response_reserve_tokens=7_000,
            response_reserve_tokens_explicit=True,
            compression_buffer_tokens=900,
            compression_buffer_tokens_explicit=True,
        )
        manager = AgentContextManager(ContextProfile.resolve(config, config.model))

        budget = manager.budget_for_messages(base_messages=[Message(role="system", content="base")], tools=[])

        self.assertGreater(budget.memory_budget_tokens, 0)
        self.assertLess(budget.memory_budget_tokens, 500)
        self.assertEqual(budget.long_term_limit, budget.memory_budget_tokens)


class AgentContextManagerTests(unittest.TestCase):
    def test_prepared_trace_includes_context_profile_fields(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: events.append((event, payload)),
            )
            manager.save_fact("用户偏好：回答中文", scope=MemoryScope.PROJECT)

            messages, _, _ = AgentContextManager(manager.context_profile).prepare_messages(
                base_messages=[Message(role="system", content="base")],
                query="回答中文",
                tools=[],
                memory=manager,
            )

        self.assertTrue(messages)
        prepared = [payload for event, payload in events if event == "memory.prepared"][-1]
        self.assertEqual(prepared["context_window"], 128_000)
        self.assertEqual(prepared["short_term_storage_token_limit"], 108_800)
        self.assertEqual(prepared["tool_result_char_limit"], 32_000)
        self.assertEqual(prepared["dynamic_profile_source"], "model")
        self.assertIn("estimated_prompt_tokens", prepared)
        self.assertIn("fixed_tokens", prepared)
        self.assertIn("memory_budget_tokens", prepared)
        self.assertIn("long_term_limit", prepared)
        self.assertIn("short_term_allowed", prepared)
        self.assertEqual(prepared["repo_context_budget_tokens"], 15_360)
        self.assertEqual(prepared["tool_schema_budget_tokens"], 10_240)

    def test_prepare_messages_reduces_long_term_limit_after_fixed_content(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    context_window=32_000,
                    context_window_explicit=True,
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: events.append((event, payload)),
            )
            manager.save_fact("Project fact about subtract.", scope=MemoryScope.PROJECT)
            base = [Message(role="system", content="x" * 20_000)]

            AgentContextManager(manager.context_profile).prepare_messages(
                base_messages=base,
                query="subtract",
                tools=[],
                memory=manager,
            )

        prepared = [payload for event, payload in events if event == "memory.prepared"][-1]
        self.assertLess(prepared["long_term_limit"], manager.context_profile.memory_context_tokens)
        self.assertGreaterEqual(prepared["short_term_allowed"], 0)

    def test_prepare_messages_raises_when_prompt_remains_over_budget(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    context_window=8_000,
                    context_window_explicit=True,
                    response_reserve_tokens=7_000,
                    response_reserve_tokens_explicit=True,
                    compression_buffer_tokens=900,
                    compression_buffer_tokens_explicit=True,
                    memory_auto_extract=False,
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: events.append((event, payload)),
            )

            with self.assertRaises(ContextOverBudgetError) as raised:
                AgentContextManager(manager.context_profile).prepare_messages(
                    base_messages=[Message(role="system", content="x" * 2_000)],
                    query="oversized prompt",
                    tools=[],
                    memory=manager,
                )

        self.assertIn("LLM request was not sent", str(raised.exception))
        self.assertTrue([payload for event, payload in events if event == "context.over_budget"])
        prepared = [payload for event, payload in events if event == "memory.prepared"][-1]
        self.assertTrue(prepared["over_budget"])

    def test_task_goal_and_fuller_tool_result_rebuild_as_session_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_task_goal("fix calculator subtract")
            manager.append_assistant_response(
                ChatResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call_1",
                            name="read_file",
                            arguments={"path": "calculator.py"},
                            arguments_json='{"path": "calculator.py"}',
                        )
                    ],
                )
            )
            manager.append_tool_result(ToolExecutionResult(id="call_1", name="read_file", ok=True, content="x" * 1_000))

            messages, _, _ = AgentContextManager(manager.context_profile).prepare_messages(
                base_messages=[Message(role="system", content="base")],
                query="subtract",
                tools=[],
                memory=manager,
            )

        self.assertTrue(any("[Task goal]" in (message.content or "") for message in messages if isinstance(message, Message)))
        assistant_messages = [message for message in messages if isinstance(message, Message) and message.role == "assistant"]
        tool_messages = [message for message in messages if isinstance(message, Message) and message.role == "tool"]
        self.assertEqual(assistant_messages[-1].tool_calls[0].name, "read_file")
        self.assertEqual(tool_messages[-1].tool_call_id, "call_1")
        payload = json.loads(tool_messages[-1].content or "{}")
        self.assertEqual(payload["content"], "x" * 1_000)
        self.assertNotIn("truncated", payload)

    def test_short_term_rendering_uses_rendered_tool_call_payload_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_task_goal("fix file")
            manager.append_assistant_response(
                ChatResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call_big",
                            name="write_file",
                            arguments={"path": "big.txt", "content": "x" * 20_000},
                            arguments_json=json.dumps({"path": "big.txt", "content": "x" * 20_000}),
                        )
                    ],
                )
            )
            manager.append_tool_result(ToolExecutionResult(id="call_big", name="write_file", ok=True, content="ok"))

            messages = manager.render_short_term_messages(max_tokens=50)

        self.assertLessEqual(estimate_tokens(messages_to_openai(messages)), 50)
        self.assertFalse(any(isinstance(message, Message) and message.role == "assistant" for message in messages))


def _tool_definition(name: str, description: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


if __name__ == "__main__":
    unittest.main()
