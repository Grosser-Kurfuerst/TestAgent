from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

try:
    from ._path import add_src_to_path
except ImportError:  # unittest discover -s tests imports modules as top-level files
    from _path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message, MessageLike
from my_agent.memory import MemoryManager, MemoryScope
from my_agent.memory.long_term import LongTermMemoryStore
from my_agent.tools import ToolExecutionResult


NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


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
        "memory_context_tokens": 200,
        "memory_retrieval_limit": 8,
    }
    values.update(overrides)
    return AgentConfig(**values)


class RecordingMemoryLLM:
    supports_tools = True

    def __init__(self, *, fail_map: bool = False, fact_response: str | None = None) -> None:
        self.fail_map = fail_map
        self.fact_response = fact_response or "[]"
        self.map_prompts: list[str] = []
        self.reduce_prompts: list[str] = []
        self.fact_prompts: list[str] = []

    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        prompt = _messages_text(messages)
        if "请压缩以下 Agent 对话片段" in prompt:
            if self.fail_map:
                raise RuntimeError("map failed")
            self.map_prompts.append(prompt)
            return ChatResponse(content=f"map summary {len(self.map_prompts)}")
        if "请合并以下多个对话摘要" in prompt:
            self.reduce_prompts.append(prompt)
            return ChatResponse(content="reduced summary")
        if "稳定事实" in prompt and "JSON 数组" in prompt:
            self.fact_prompts.append(prompt)
            return ChatResponse(content=self.fact_response)
        return ChatResponse(content="noop")


class FailingFactLLM:
    supports_tools = True

    def chat(self, messages: list[MessageLike], tools: list[dict[str, object]] | None = None) -> ChatResponse:
        raise RuntimeError("fact llm failed")


def _messages_text(messages: list[MessageLike]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, Message):
            lines.append(message.content or "")
        elif isinstance(message, dict):
            value = message.get("content", "")
            if isinstance(value, str):
                lines.append(value)
    return "\n".join(lines)


def _tool_call(name: str, call_id: str, arguments: dict[str, object] | None = None) -> LLMToolCall:
    args = arguments or {}
    return LLMToolCall(
        id=call_id,
        name=name,
        arguments=dict(args),
        arguments_json=json.dumps(args, ensure_ascii=False),
    )


def _append_turn(manager: MemoryManager, index: int, *, payload: str = "") -> None:
    suffix = f" {payload}" if payload else ""
    manager.append_user_message(f"user turn {index}{suffix}")
    manager.append_assistant_response(
        ChatResponse(
            content=f"assistant turn {index}{suffix}",
            tool_calls=[_tool_call("read_file", f"call_{index}", {"path": f"file_{index}.py"})],
        )
    )
    manager.append_tool_result(
        ToolExecutionResult(
            id=f"call_{index}",
            name="read_file",
            ok=True,
            content=f"result {index}{suffix}",
        )
    )


class MemoryManagerBuildContextTests(unittest.TestCase):
    def test_build_context_for_query_returns_token_bounded_injection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_fact("用户偏好：回答中文，先给结论", scope=MemoryScope.PROJECT)

            ctx = manager.build_context_for_query("用户偏好 回答中文")

            self.assertTrue(ctx.injected_text.startswith("Relevant long-term memory:"))
            self.assertGreater(ctx.estimated_tokens, 0)
            self.assertLessEqual(ctx.estimated_tokens, manager.config.memory_context_tokens)

    def test_build_context_for_query_injects_nothing_when_no_relevant_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_fact("项目使用 FastAPI", scope=MemoryScope.PROJECT)

            ctx = manager.build_context_for_query("completely unrelated query xyz")

            self.assertEqual(ctx.injected_text, "")
            self.assertEqual(ctx.estimated_tokens, 0)

    def test_build_context_for_query_never_exceeds_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_context_tokens=1),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_fact("用户偏好：回答中文", scope=MemoryScope.PROJECT)

            ctx = manager.build_context_for_query("用户偏���")

            self.assertEqual(ctx.injected_text, "")
            self.assertEqual(ctx.estimated_tokens, 0)


class MemoryManagerSaveFactTests(unittest.TestCase):
    def test_save_fact_persists_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            entry, created = manager.save_fact("用户偏好：回答中文", scope=MemoryScope.PROJECT)
            self.assertTrue(created)

            same, created2 = manager.save_fact("用户偏好：回答中文", scope=MemoryScope.PROJECT)
            self.assertFalse(created2)
            self.assertEqual(same.id, entry.id)

    def test_save_fact_survives_new_manager_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            config = _config(Path(tmp) / "memory")
            manager = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            manager.save_fact("durable fact about config", scope=MemoryScope.PROJECT)

            reopened = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            ctx = reopened.build_context_for_query("config")
            self.assertIn("durable fact about config", ctx.injected_text)

    def test_save_fact_global_scope_visible_to_other_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "repo_a"
            repo_b = Path(tmp) / "repo_b"
            repo_a.mkdir()
            repo_b.mkdir()
            config = _config(Path(tmp) / "memory")
            manager_a = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo_a)
            manager_a.save_fact("global rule about config", scope=MemoryScope.GLOBAL)

            manager_b = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo_b)
            ctx = manager_b.build_context_for_query("config")
            self.assertIn("global rule about config", ctx.injected_text)


class MemoryManagerAppendTests(unittest.TestCase):
    def test_append_messages_record_short_term_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_user_message("fix the subtract bug")
            manager.append_assistant_response(ChatResponse(content="I will inspect calculator.py."))
            manager.append_tool_result(
                ToolExecutionResult(id="call_1", name="read_file", ok=True, content="def subtract(a,b): return a + b")
            )

            self.assertEqual(len(manager.short_term), 3)
            self.assertGreater(manager.short_term.token_count(), 0)

    def test_append_tool_result_truncates_to_configured_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_tool_result_chars=10),
                llm=FakeLLM(),
                repo_path=repo,
            )
            long_output = "x" * 500
            manager.append_tool_result(
                ToolExecutionResult(
                    id="c1",
                    name="run_tests",
                    ok=False,
                    content=long_output,
                    elapsed_ms=123,
                    error_code="tool_failed",
                    retryable=True,
                    blocked=True,
                    timed_out=True,
                )
            )
            entry = manager.short_term.all()[0]
            payload = json.loads(entry.content)
            self.assertEqual(payload["tool"], "run_tests")
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["blocked"])
            self.assertTrue(payload["timed_out"])
            self.assertTrue(payload["retryable"])
            self.assertEqual(payload["error_code"], "tool_failed")
            self.assertEqual(payload["elapsed_ms"], 123)
            self.assertEqual(payload["content"], "x" * 10 + "...(truncated)")
            self.assertTrue(payload["truncated"])
            self.assertEqual(payload["original_content_chars"], 500)


class MemoryManagerStatusTests(unittest.TestCase):
    def test_status_reports_short_and_long_term_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_user_message("hello")
            manager.save_fact("用户偏好：回答中文", scope=MemoryScope.PROJECT)

            status = manager.status()

            self.assertEqual(status.short_term_entries, 1)
            self.assertEqual(status.long_term_entries, 1)
            self.assertEqual(status.project_key, str(repo.resolve()))
            self.assertTrue(status.storage_path.endswith("long_term_memory.jsonl"))
            self.assertEqual(status.compression_trigger_ratio, 0.8)
            self.assertEqual(status.retain_recent_turns, 3)
            self.assertEqual(status.map_chunk_size, 5)
            self.assertEqual(len(status.long_term_entries_detail), 1)

    def test_status_without_entries_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_fact("a fact", scope=MemoryScope.PROJECT)
            status = manager.status(include_entries=False)
            self.assertEqual(status.long_term_entries_detail, ())


class MemoryManagerCompressionTests(unittest.TestCase):
    def test_prepare_messages_forced_compaction_keeps_recent_three_turns_and_runs_map_reduce(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            llm = RecordingMemoryLLM()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_map_chunk_size=5),
                llm=llm,
                repo_path=repo,
            )
            for index in range(12):
                _append_turn(manager, index)

            _, _, compaction = manager.prepare_messages(
                base_messages=[Message(role="system", content="base"), Message(role="user", content="runtime")],
                query="user turn",
                tools=[],
                force_compact=True,
            )

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)
            self.assertEqual(compaction.map_count, 2)
            self.assertTrue(compaction.reduce_used)
            contents = "\n".join(entry.content for entry in manager.short_term.all())
            self.assertIn("[Compressed memory summary]", contents)
            self.assertNotIn("user turn 8", contents)
            self.assertIn("user turn 9", contents)
            self.assertIn("user turn 10", contents)
            self.assertIn("user turn 11", contents)
            self.assertIn("assistant turn 0", llm.map_prompts[0])
            self.assertIn('"tool": "read_file"', llm.map_prompts[0])
            self.assertIn('"content": "result 0"', llm.map_prompts[0])

    def test_prepare_messages_auto_compacts_above_configured_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_short_term_tokens=20_000,
                    memory_compression_trigger_ratio=0.8,
                    memory_map_chunk_size=5,
                ),
                llm=RecordingMemoryLLM(),
                repo_path=repo,
            )
            payload = "x" * 2500
            for index in range(12):
                _append_turn(manager, index, payload=payload)

            _, _, compaction = manager.prepare_messages(
                base_messages=[Message(role="system", content="base")],
                query="user turn",
                tools=[],
            )

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)

    def test_prepare_messages_compacts_when_full_prompt_crosses_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            llm = RecordingMemoryLLM()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_short_term_tokens=1_000,
                    memory_compression_trigger_ratio=0.8,
                ),
                llm=llm,
                repo_path=repo,
            )
            for index in range(4):
                _append_turn(manager, index)

            _, _, compaction = manager.prepare_messages(
                base_messages=[
                    Message(role="system", content="base"),
                    Message(role="user", content="runtime " + ("x" * 4000)),
                ],
                query="user turn",
                tools=[{"type": "function", "function": {"name": "large_tool", "description": "x" * 1000}}],
            )

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)
            self.assertEqual(len(llm.map_prompts), 1)

    def test_compaction_trace_uses_full_prompt_after_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_short_term_tokens=1_000),
                llm=RecordingMemoryLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            for index in range(4):
                _append_turn(manager, index)

            _, _, compaction = manager.prepare_messages(
                base_messages=[
                    Message(role="system", content="base"),
                    Message(role="user", content="runtime " + ("x" * 4000)),
                ],
                query="user turn",
                tools=[],
            )

            self.assertIsNotNone(compaction)
            compacted_events = [payload for event, payload in traces if event == "memory.compacted"]
            self.assertEqual(len(compacted_events), 1)
            self.assertEqual(compacted_events[0]["after_tokens"], compaction.after_tokens)

    def test_compaction_fallback_keeps_main_flow_alive_when_llm_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(fail_map=True),
                repo_path=repo,
            )
            for index in range(5):
                _append_turn(manager, index)

            _, _, compaction = manager.prepare_messages(
                base_messages=[Message(role="system", content="base")],
                query="user turn",
                tools=[],
                force_compact=True,
            )

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)
            self.assertTrue(compaction.fallback)
            self.assertIn("[Fallback memory summary]", manager.short_term.all()[0].content)

    def test_compaction_extracts_stable_facts_and_filters_temporary_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            fact_response = json.dumps(
                [
                    {"content": "用户偏好：回答中文，先给结论", "scope": "project", "confidence": 0.95},
                    {"content": "本次任务：修复 calculator.py", "scope": "project", "confidence": 0.99},
                ],
                ensure_ascii=False,
            )
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(fact_response=fact_response),
                repo_path=repo,
            )
            for index in range(5):
                _append_turn(manager, index, payload="用户偏好：回答中文，先给结论")

            _, _, compaction = manager.prepare_messages(
                base_messages=[Message(role="system", content="base")],
                query="用户偏好",
                tools=[],
                force_compact=True,
            )

            self.assertIsNotNone(compaction)
            self.assertEqual(compaction.extracted_facts, 1)
            facts = "\n".join(entry.content for entry in manager.long_term.all(project_key=str(repo.resolve())))
            self.assertIn("用户偏好：回答中文，先给结论", facts)
            self.assertNotIn("本次任务", facts)

    def test_compaction_extracts_english_project_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            fact_response = json.dumps(
                [
                    {
                        "content": "Project uses FastAPI for REST APIs",
                        "scope": "project",
                        "confidence": 0.92,
                    }
                ],
                ensure_ascii=False,
            )
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(fact_response=fact_response),
                repo_path=repo,
            )
            for index in range(5):
                _append_turn(manager, index, payload="Project uses FastAPI for REST APIs")

            _, _, compaction = manager.prepare_messages(
                base_messages=[Message(role="system", content="base")],
                query="FastAPI",
                tools=[],
                force_compact=True,
            )

            self.assertIsNotNone(compaction)
            self.assertEqual(compaction.extracted_facts, 1)
            facts = [entry.content for entry in manager.long_term.all(project_key=str(repo.resolve()))]
            self.assertIn("Project uses FastAPI for REST APIs", facts)


class MemoryManagerFacadeTests(unittest.TestCase):
    def test_prepare_messages_injects_memory_and_rebuilds_tool_call_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_fact("用户偏好：回答中文，先给结论", scope=MemoryScope.PROJECT)
            manager.append_user_message("please inspect calculator")
            manager.append_assistant_response(
                ChatResponse(
                    content="",
                    tool_calls=[_tool_call("read_file", "call_read", {"path": "calculator.py"})],
                )
            )
            manager.append_tool_result(
                ToolExecutionResult(
                    id="call_read",
                    name="read_file",
                    ok=True,
                    content="def subtract(a, b): return a - b",
                )
            )

            messages, context, compaction = manager.prepare_messages(
                base_messages=[Message(role="system", content="base"), Message(role="user", content="runtime")],
                query="用户偏好 回答中文",
                tools=[],
            )

            self.assertIsNone(compaction)
            self.assertIn("用户偏好：回答中文，先给结论", context.injected_text)
            self.assertEqual(messages[1].role, "system")
            self.assertIn("Relevant long-term memory:", messages[1].content or "")
            assistant_messages = [message for message in messages if isinstance(message, Message) and message.role == "assistant"]
            tool_messages = [message for message in messages if isinstance(message, Message) and message.role == "tool"]
            self.assertEqual(assistant_messages[-1].tool_calls[0].name, "read_file")
            self.assertEqual(tool_messages[-1].tool_call_id, "call_read")
            tool_payload = json.loads(tool_messages[-1].content or "{}")
            self.assertTrue(tool_payload["ok"])
            self.assertEqual(tool_payload["error_code"], "")
            self.assertEqual(tool_payload["content"], "def subtract(a, b): return a - b")

    def test_manual_save_upgrades_existing_auto_extracted_fact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            fact_response = json.dumps(
                [{"content": "用户偏好：回答中文", "scope": "project", "confidence": 0.9}],
                ensure_ascii=False,
            )
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(fact_response=fact_response),
                repo_path=repo,
            )
            manager.append_user_message("用户偏好：回答中文")
            extracted = manager.extract_facts(reason="test")
            self.assertEqual(len(extracted), 1)
            auto_entry = extracted[0]
            self.assertEqual(auto_entry.source, "fact_extractor")

            manual_entry, created = manager.save_fact("用户偏好：回答中文", scope=MemoryScope.PROJECT)

            self.assertFalse(created)
            self.assertEqual(manual_entry.id, auto_entry.id)
            self.assertEqual(manual_entry.created_at, auto_entry.created_at)
            self.assertEqual(manual_entry.source, "manual")
            self.assertEqual(manual_entry.metadata.get("source"), "manual")
            entries = manager.long_term.all(project_key=str(repo.resolve()))
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].source, "manual")

    def test_prepare_messages_downgrades_orphan_tool_result_to_user_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_tool_result(
                ToolExecutionResult(
                    id="orphan_call",
                    name="read_file",
                    ok=True,
                    content="orphan tool output",
                )
            )

            messages, _, _ = manager.prepare_messages(
                base_messages=[Message(role="system", content="base")],
                query="orphan",
                tools=[],
            )

            self.assertFalse(any(isinstance(message, Message) and message.role == "tool" for message in messages))
            self.assertTrue(
                any(
                    isinstance(message, Message)
                    and message.role == "user"
                    and "[Tool result memory]" in (message.content or "")
                    for message in messages
                )
            )

    def test_prepare_messages_downgrades_incomplete_multi_tool_call_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_user_message("run two tools")
            manager.append_assistant_response(
                ChatResponse(
                    content="",
                    tool_calls=[
                        _tool_call("read_file", "call_one", {"path": "a.py"}),
                        _tool_call("read_file", "call_two", {"path": "b.py"}),
                    ],
                )
            )
            manager.append_tool_result(
                ToolExecutionResult(id="call_one", name="read_file", ok=True, content="a.py")
            )
            manager.append_user_message("next user turn")

            messages, _, _ = manager.prepare_messages(
                base_messages=[Message(role="system", content="base")],
                query="tools",
                tools=[],
            )

            self.assertFalse(any(isinstance(message, Message) and message.role == "tool" for message in messages))
            self.assertFalse(
                any(
                    isinstance(message, Message)
                    and message.role == "assistant"
                    and message.tool_calls
                    for message in messages
                )
            )
            self.assertTrue(
                any(
                    isinstance(message, Message)
                    and message.role == "user"
                    and "[Incomplete tool-call memory]" in (message.content or "")
                    for message in messages
                )
            )

    def test_clear_short_term_without_extract_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_user_message("hello")
            count, extracted = manager.clear_short_term(extract_first=False)
            self.assertEqual(count, 1)
            self.assertEqual(extracted, [])
            self.assertEqual(len(manager.short_term), 0)

    def test_clear_short_term_with_extract_saves_facts_before_clearing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            fact_response = json.dumps(
                [{"content": "用户偏好：回答中文", "scope": "project", "confidence": 0.9}],
                ensure_ascii=False,
            )
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(fact_response=fact_response),
                repo_path=repo,
            )
            manager.append_user_message("用户偏好：回答中文")
            count, extracted = manager.clear_short_term(extract_first=True)
            self.assertEqual(count, 1)
            self.assertEqual(len(extracted), 1)
            self.assertEqual(len(manager.short_term), 0)
            self.assertIn("用户偏好：回答中文", manager.long_term.all(project_key=str(repo.resolve()))[0].content)

    def test_extract_facts_does_not_raise_when_auto_save_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            fact_response = json.dumps(
                [{"content": "用户偏好：回答中文", "scope": "project", "confidence": 0.9}],
                ensure_ascii=False,
            )
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(fact_response=fact_response),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.append_user_message("用户偏好：回答中文", run_id="run_1")

            def fail_add(entry: object) -> tuple[object, bool]:
                raise OSError("disk full")

            manager.long_term.add = fail_add  # type: ignore[method-assign]

            extracted = manager.extract_facts(reason="run_completed", run_id="run_1")

            self.assertEqual(extracted, [])
            self.assertTrue(manager.last_fact_save_errors)
            self.assertIn("memory.save_failed", [event for event, _ in traces])

    def test_extract_facts_rolls_back_memory_when_persist_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            fact_response = json.dumps(
                [{"content": "用户偏好：回答中文", "scope": "project", "confidence": 0.9}],
                ensure_ascii=False,
            )
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(fact_response=fact_response),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.append_user_message("用户偏好：回答中文", run_id="run_1")
            original_persist = manager.long_term._persist

            def fail_persist() -> None:
                raise OSError("disk full")

            manager.long_term._persist = fail_persist  # type: ignore[method-assign]

            extracted = manager.extract_facts(reason="run_completed", run_id="run_1")

            self.assertEqual(extracted, [])
            self.assertFalse(any("用户偏好：回答中文" in entry.content for entry in manager.long_term.all()))
            self.assertIn("memory.save_failed", [event for event, _ in traces])
            self.assertNotIn("memory.fact_extracted", [event for event, _ in traces])

            manager.long_term._persist = original_persist  # type: ignore[method-assign]
            entry, created = manager.save_fact("用户偏好：回答中文")
            self.assertTrue(created)
            self.assertIn("用户偏好：回答中文", entry.content)

    def test_extract_facts_records_llm_failure_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FailingFactLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.append_user_message("用户偏好：回答中文")

            count, extracted = manager.clear_short_term(extract_first=True)

            self.assertEqual(count, 1)
            self.assertEqual(extracted, [])
            self.assertIn("RuntimeError: fact llm failed", manager.last_fact_extraction_error)
            self.assertIn("memory.fact_extraction_failed", [event for event, _ in traces])
            self.assertEqual(len(manager.short_term), 0)

    def test_run_completed_fact_extraction_only_uses_matching_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            fact_response = json.dumps(
                [
                    {"content": "用户偏好：回答中文", "scope": "project", "confidence": 0.95},
                    {"content": "项目 AgentCli 使用 Python src 布局", "scope": "project", "confidence": 0.9},
                ],
                ensure_ascii=False,
            )
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(fact_response=fact_response),
                repo_path=repo,
            )
            manager.append_user_message("用户偏好：回答中文", run_id="old")
            manager.append_user_message("项目 AgentCli 使用 Python src 布局", run_id="new")

            extracted = manager.extract_facts(reason="run_completed", run_id="new")

            contents = [entry.content for entry in extracted]
            self.assertIn("项目 AgentCli 使用 Python src 布局", contents)
            prompts = manager.compressor.llm.fact_prompts  # type: ignore[union-attr]
            self.assertEqual(len(prompts), 1)
            self.assertNotIn("用户偏好：回答中文", prompts[0])


if __name__ == "__main__":
    unittest.main()
