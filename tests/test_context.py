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
from my_agent.context import AgentContextManager, ContextProfile
from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message
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
        self.assertEqual(profile.short_term_token_limit, 450_000)
        self.assertEqual(profile.memory_context_tokens, 5_000)
        self.assertEqual(profile.tool_result_char_limit, 60_000)
        self.assertEqual(profile.response_reserve_tokens, 100_000)
        self.assertEqual(profile.compression_buffer_tokens, 50_000)
        self.assertEqual(profile.compression_trigger_tokens, 850_000)

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

    def test_unknown_model_falls_back_to_default_window(self) -> None:
        config = _config(Path("memory"), model="unknown-local-model")

        profile = ContextProfile.resolve(config, config.model)

        self.assertEqual(profile.max_context_tokens, 128_000)
        self.assertEqual(profile.response_reserve_tokens, 12_800)
        self.assertEqual(profile.compression_buffer_tokens, 6_400)
        self.assertEqual(profile.compression_trigger_tokens, 108_800)
        self.assertEqual(profile.dynamic_profile_source, "default")


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
        self.assertEqual(prepared["short_term_token_limit"], 57_600)
        self.assertEqual(prepared["tool_result_char_limit"], 32_000)
        self.assertEqual(prepared["dynamic_profile_source"], "model")
        self.assertIn("estimated_prompt_tokens", prepared)

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


if __name__ == "__main__":
    unittest.main()
