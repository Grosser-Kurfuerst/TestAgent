from __future__ import annotations

# ruff: noqa: F403, F405 - shared test support exports the frozen fixtures

from tests.memory.manager.support import *

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
            save_typed_experience(manager, "用户偏好：回答中文", tier="tip", scope=MemoryScope.PROJECT)

            status = manager.status()

            self.assertEqual(status.short_term_entries, 1)
            self.assertEqual(status.long_term_entries, 1)
            self.assertEqual(status.project_key, str(repo.resolve()))
            self.assertTrue(status.storage_path.endswith("experience_memory.jsonl"))
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
            save_typed_experience(manager, "a typed tip", tier="tip", scope=MemoryScope.PROJECT)
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

            _, _, compaction = _prepare_messages(
                manager,
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
                    context_window=8_000,
                    context_window_explicit=True,
                    response_reserve_tokens=6_000,
                    compression_buffer_tokens=1_000,
                    memory_short_term_tokens=100_000,
                    memory_map_chunk_size=5,
                ),
                llm=RecordingMemoryLLM(),
                repo_path=repo,
            )
            payload = "x" * 2500
            for index in range(12):
                _append_turn(manager, index, payload=payload)

            _, _, compaction = _prepare_messages(
                manager,
                base_messages=[Message(role="system", content="base")],
                query="user turn",
                tools=[],
            )

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)

    def test_prepare_messages_compacts_single_task_goal_tool_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    context_window=8_000,
                    context_window_explicit=True,
                    response_reserve_tokens=6_000,
                    compression_buffer_tokens=1_000,
                    memory_short_term_tokens=100_000,
                    memory_map_chunk_size=3,
                ),
                llm=RecordingMemoryLLM(),
                repo_path=repo,
            )
            manager.append_task_goal("fix subtract")
            payload = "x" * 3000
            for index in range(6):
                manager.append_assistant_response(ChatResponse(content=f"assistant cycle {index} {payload}"))
                manager.append_tool_result(
                    ToolExecutionResult(
                        id=f"call_{index}",
                        name="read_file",
                        ok=True,
                        content=f"tool cycle {index} {payload}",
                    )
                )

            _, _, compaction = _prepare_messages(
                manager,
                base_messages=[Message(role="system", content="base")],
                query="subtract",
                tools=[],
            )

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)
            entries = manager.short_term.all()
            self.assertTrue(any(entry.source == "task_goal" and entry.content == "fix subtract" for entry in entries))
            self.assertLess(len([entry for entry in entries if entry.source == "assistant"]), 6)

    def test_prepare_messages_compacts_when_full_prompt_crosses_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            llm = RecordingMemoryLLM()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    context_window=8_000,
                    context_window_explicit=True,
                    response_reserve_tokens=5_500,
                    compression_buffer_tokens=1_000,
                    memory_short_term_tokens=100_000,
                ),
                llm=llm,
                repo_path=repo,
            )
            for index in range(4):
                _append_turn(manager, index)

            _, _, compaction = _prepare_messages(
                manager,
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
                config=_config(
                    Path(tmp) / "memory",
                    context_window=8_000,
                    context_window_explicit=True,
                    response_reserve_tokens=5_500,
                    compression_buffer_tokens=1_000,
                    memory_short_term_tokens=100_000,
                ),
                llm=RecordingMemoryLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            for index in range(4):
                _append_turn(manager, index)

            _, _, compaction = _prepare_messages(
                manager,
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
            self.assertEqual(compacted_events[0]["estimated_prompt_tokens"], compaction.after_tokens)

    def test_direct_compact_short_term_trace_includes_estimated_prompt_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            for index in range(5):
                _append_turn(manager, index)

            compaction = manager.compact_short_term(tools=[], force=True)

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)
            compacted = [payload for event, payload in traces if event == "memory.compacted"][-1]
            self.assertEqual(compacted["estimated_prompt_tokens"], compaction.after_tokens)
            self.assertEqual(compacted["context_window"], manager.context_profile.max_context_tokens)
            self.assertEqual(compacted["compression_trigger_tokens"], manager.context_profile.compression_trigger_tokens)
            self.assertEqual(
                compacted["short_term_storage_token_limit"],
                manager.context_profile.short_term_storage_token_limit,
            )
            self.assertEqual(compacted["tool_result_char_limit"], manager.context_profile.tool_result_char_limit)
            self.assertEqual(compacted["dynamic_profile_source"], manager.context_profile.dynamic_profile_source)

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

            _, _, compaction = _prepare_messages(
                manager,
                base_messages=[Message(role="system", content="base")],
                query="user turn",
                tools=[],
                force_compact=True,
            )

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)
            self.assertTrue(compaction.fallback)
            self.assertIn("[Fallback memory summary]", manager.short_term.all()[0].content)

    def test_compaction_keeps_short_term_summary_without_extracting_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(),
                repo_path=repo,
            )
            for index in range(5):
                _append_turn(manager, index, payload="Project uses FastAPI for REST APIs")

            _, _, compaction = _prepare_messages(
                manager,
                base_messages=[Message(role="system", content="base")],
                query="FastAPI",
                tools=[],
                force_compact=True,
            )

            self.assertIsNotNone(compaction)
            self.assertTrue(compaction.compacted)
            self.assertEqual(compaction.extracted_facts, 0)
            self.assertEqual(manager.experience_store.all(project_key=manager.project_key), [])
            self.assertTrue(any(entry.type.value == "summary" for entry in manager.short_term.all()))
class AgentContextManagerPromptTests(unittest.TestCase):
    def test_prepare_messages_injects_memory_and_rebuilds_tool_call_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            save_typed_experience(manager, "用户偏好：回答中文，先给结论", tier="tip")
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

            messages, context, compaction = _prepare_messages(
                manager,
                base_messages=[Message(role="system", content="base"), Message(role="user", content="runtime")],
                query="用户偏好 回答中文",
                tools=[],
            )

            self.assertIsNone(compaction)
            self.assertIn("用户偏好：回答中文，先给结论", context.injected_text)
            self.assertEqual(messages[1].role, "system")
            self.assertIn("Relevant selected experience:", messages[1].content or "")
            assistant_messages = [message for message in messages if isinstance(message, Message) and message.role == "assistant"]
            tool_messages = [message for message in messages if isinstance(message, Message) and message.role == "tool"]
            self.assertEqual(assistant_messages[-1].tool_calls[0].name, "read_file")
            self.assertEqual(tool_messages[-1].tool_call_id, "call_read")
            tool_payload = json.loads(tool_messages[-1].content or "{}")
            self.assertTrue(tool_payload["ok"])
            self.assertEqual(tool_payload["error_code"], "")
            self.assertEqual(tool_payload["content"], "def subtract(a, b): return a - b")

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

            messages, _, _ = _prepare_messages(
                manager,
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

            messages, _, _ = _prepare_messages(
                manager,
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

    def test_clear_short_term_with_extract_flag_does_not_write_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=RecordingMemoryLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.append_user_message("用户偏好：回答中文")

            count, extracted = manager.clear_short_term(extract_first=True)

            self.assertEqual(count, 1)
            self.assertEqual(extracted, [])
            self.assertEqual(manager.experience_store.all(project_key=manager.project_key), [])
            clear_payload = [payload for event, payload in traces if event == "memory.clear"][-1]
            self.assertEqual(clear_payload["extracted_facts"], 0)
