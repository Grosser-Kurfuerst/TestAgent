from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._path import add_src_to_path

add_src_to_path()

from my_agent.config import AgentConfig
from my_agent.context import AgentContextManager
from my_agent.llm import FakeLLM
from my_agent.llm.types import ChatResponse, LLMToolCall, Message, MessageLike
from my_agent.memory import (
    ExperienceCreatedBy,
    ExperienceTier,
    MemoryManager,
    MemoryScope,
    NoopMemoryManager,
    experience_record_from_entry,
    is_experience_entry,
)
from my_agent.memory.evolver import ExperienceWriteProposal, ExperienceWriteResult
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


def _tool_record(
    tool: str,
    *,
    ok: bool,
    output: str = "",
    reason: str = "",
    arguments: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "call": {"tool": tool, "arguments": arguments or {"command": "pytest tests/test_example.py -q"}},
        "result": {"ok": ok, "output": output or ("passed" if ok else "failed"), "reason": reason},
    }


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


def _prepare_messages(manager: MemoryManager, **kwargs):
    return AgentContextManager(manager.context_profile).prepare_messages(memory=manager, **kwargs)


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

    def test_memory_project_key_override_controls_project_scope_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "repo_a"
            repo_b = Path(tmp) / "repo_b"
            repo_c = Path(tmp) / "repo_c"
            repo_a.mkdir()
            repo_b.mkdir()
            repo_c.mkdir()
            memory_dir = Path(tmp) / "memory"
            config_x = _config(memory_dir, memory_project_key="stream:x")
            config_y = _config(memory_dir, memory_project_key="stream:y")

            manager_a = MemoryManager.from_config(config=config_x, llm=FakeLLM(), repo_path=repo_a)
            manager_a.save_fact("stream marker VALUE equals 1", scope=MemoryScope.PROJECT)

            manager_b = MemoryManager.from_config(config=config_x, llm=FakeLLM(), repo_path=repo_b)
            manager_c = MemoryManager.from_config(config=config_y, llm=FakeLLM(), repo_path=repo_c)

            ctx_b = manager_b.build_context_for_query("stream marker VALUE")
            ctx_c = manager_c.build_context_for_query("stream marker VALUE")

        self.assertEqual(manager_a.project_key, "stream:x")
        self.assertEqual(manager_b.project_key, "stream:x")
        self.assertEqual(manager_c.project_key, "stream:y")
        self.assertIn("stream marker VALUE equals 1", ctx_b.injected_text)
        self.assertNotIn("stream marker VALUE equals 1", ctx_c.injected_text)


class MemoryManagerSaveExperienceTests(unittest.TestCase):
    def test_save_experience_persists_metadata_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            entry, created = manager.save_experience(
                "Always run compileall after package migration.",
                tier=ExperienceTier.SKILL,
                source_task="task-1",
                created_by=ExperienceCreatedBy.WRITER,
                run_id="run-1",
                metadata={
                    "category": "debugging",
                    "technique": "import smoke",
                    "steps": ["compileall", "import smoke"],
                    "usage_count": 3,
                    "success_count": 2,
                    "last_used": "2026-06-18T12:00:00+00:00",
                },
            )

            self.assertTrue(created)
            self.assertTrue(is_experience_entry(entry))
            self.assertEqual(entry.source, "evolver:skill")
            self.assertEqual(entry.scope, MemoryScope.PROJECT)
            self.assertEqual(entry.project_key, str(repo.resolve()))
            self.assertEqual(entry.run_id, "run-1")
            self.assertEqual(entry.metadata["evolver_tier"], "skill")
            self.assertEqual(entry.metadata["source_task"], "task-1")
            self.assertEqual(entry.metadata["created_by"], "writer")
            self.assertEqual(entry.metadata["technique"], "import smoke")
            self.assertEqual(entry.metadata["usage_count"], 3)

            events = [(event, payload) for event, payload in traces if event == "memory.evolver_saved"]
            self.assertEqual(len(events), 1)
            payload = events[0][1]
            self.assertEqual(payload["id"], entry.id)
            self.assertEqual(payload["created"], True)
            self.assertEqual(payload["tier"], "skill")
            self.assertEqual(payload["scope"], "project")
            self.assertEqual(payload["tokens"], entry.token_count)
            self.assertEqual(payload["source_task"], "task-1")
            self.assertEqual(payload["created_by"], "writer")

    def test_save_experience_dedup_and_project_visibility_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "repo_a"
            repo_b = Path(tmp) / "repo_b"
            repo_c = Path(tmp) / "repo_c"
            repo_a.mkdir()
            repo_b.mkdir()
            repo_c.mkdir()
            memory_dir = Path(tmp) / "memory"
            config_x = _config(memory_dir, memory_project_key="stream:x")
            config_y = _config(memory_dir, memory_project_key="stream:y")
            manager_a = MemoryManager.from_config(config=config_x, llm=FakeLLM(), repo_path=repo_a)

            entry, created = manager_a.save_experience("same project tip", tier="tip")
            duplicate, created_duplicate = manager_a.save_experience("same project tip", tier="tip")
            global_entry, created_global = manager_a.save_experience(
                "global evolver skill",
                tier="skill",
                scope=MemoryScope.GLOBAL,
            )
            manager_b = MemoryManager.from_config(config=config_x, llm=FakeLLM(), repo_path=repo_b)
            manager_c = MemoryManager.from_config(config=config_y, llm=FakeLLM(), repo_path=repo_c)

            self.assertTrue(created)
            self.assertFalse(created_duplicate)
            self.assertEqual(duplicate.id, entry.id)
            self.assertTrue(created_global)
            self.assertEqual(global_entry.project_key, "")
            visible_b = {item.content for item in manager_b.long_term.all(project_key=manager_b.project_key)}
            visible_c = {item.content for item in manager_c.long_term.all(project_key=manager_c.project_key)}
            self.assertIn("same project tip", visible_b)
            self.assertIn("global evolver skill", visible_b)
            self.assertNotIn("same project tip", visible_c)
            self.assertIn("global evolver skill", visible_c)

    def test_save_experience_preserves_tool_and_trajectory_metadata_after_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            config = _config(Path(tmp) / "memory")
            manager = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            manager.save_experience(
                "Reusable command template for one pytest file.",
                tier="tool",
                metadata={
                    "name": "pytest_single_file",
                    "language": "bash",
                    "code": "pytest {test_path} -q",
                    "input_description": "test_path: path to one pytest file",
                    "output_description": "pytest summary",
                    "tool_name": "run_tests",
                    "command": "pytest tests/test_parser.py -q",
                    "args_schema": {"test_path": "str"},
                    "repo_context": "run from repo root",
                    "template": "pytest {test_path} -q",
                },
            )
            manager.save_experience(
                "Parser fix trajectory.",
                tier="trajectory",
                source_task="task-trajectory",
                metadata={
                    "task_description": "Fix parser test",
                    "steps": [
                        {
                            "step_num": 1,
                            "observation": "pytest failed",
                            "action": "run_tests",
                            "action_params": {"command": "pytest tests/test_parser.py -q"},
                            "result": "passed after fix",
                            "reward": 1.0,
                        }
                    ],
                    "outcome": "success",
                    "total_reward": 1.0,
                    "key_learnings": ["parser strips comments before tokenization"],
                    "tags": ["parser"],
                    "usage_count": 1,
                    "success_count": 1,
                    "last_used": "2026-06-18T12:00:00+00:00",
                },
            )

            reopened = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            entries = {entry.metadata["evolver_tier"]: entry for entry in reopened.long_term.all(project_key=str(repo.resolve()))}

            tool = entries["tool"]
            self.assertEqual(tool.metadata["name"], "pytest_single_file")
            self.assertEqual(tool.metadata["code"], "pytest {test_path} -q")
            self.assertEqual(tool.metadata["tool_name"], "run_tests")
            self.assertEqual(tool.metadata["args_schema"], {"test_path": "str"})
            trajectory = entries["trajectory"]
            self.assertEqual(trajectory.metadata["source_task"], "task-trajectory")
            self.assertEqual(trajectory.metadata["steps"][0]["action"], "run_tests")
            self.assertEqual(trajectory.metadata["usage_count"], 1)
            record = experience_record_from_entry(trajectory)
            self.assertIsNotNone(record)
            self.assertEqual(record.tier, ExperienceTier.TRAJECTORY)

    def test_experience_entries_are_retrievable_as_regular_long_term_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_fact("ordinary durable fact about config", scope=MemoryScope.PROJECT)
            manager.save_experience(
                "pytest fixture cleanup issue: clear tmp_path state before rerun",
                tier="tip",
            )

            fact_ctx = manager.build_context_for_query("ordinary config")
            experience_ctx = manager.build_context_for_query("pytest fixture cleanup")

            self.assertIn("ordinary durable fact about config", fact_ctx.injected_text)
            self.assertIn("pytest fixture cleanup issue", experience_ctx.injected_text)

    def test_noop_save_experience_returns_entry_without_writing_or_tracing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(Path(tmp) / "memory")
            manager = NoopMemoryManager(
                config=config,
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            entry, created = manager.save_experience(
                "No-op tool template",
                tier="tool",
                source_task="task-noop",
                created_by="manual",
                run_id="run-noop",
                metadata={"tool_name": "run_tests", "command": "pytest -q"},
            )

            self.assertFalse(created)
            self.assertTrue(is_experience_entry(entry))
            self.assertEqual(entry.metadata["evolver_tier"], "tool")
            self.assertEqual(entry.metadata["source_task"], "task-noop")
            self.assertEqual(entry.metadata["tool_name"], "run_tests")
            self.assertEqual(entry.run_id, "run-noop")
            self.assertFalse((Path(config.memory_dir) / "long_term_memory.jsonl").exists())
            self.assertNotIn("memory.evolver_saved", [event for event, _ in traces])

    def test_write_experiences_disabled_does_not_write_or_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(Path(tmp) / "memory")
            manager = MemoryManager.from_config(
                config=config,
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                stop_reason="finish_called",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual(result.saved, ())
            self.assertEqual([event for event, _ in traces if event.startswith("memory.evolver_writer")], [])
            self.assertFalse((Path(config.memory_dir) / "long_term_memory.jsonl").exists())

    def test_write_experiences_success_saves_writer_skill_and_tool_with_trace_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(
                Path(tmp) / "memory",
                memory_evolver_writer_enabled=True,
                memory_project_key="stream:a",
            )
            manager = MemoryManager.from_config(
                config=config,
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                trace_path=Path(tmp) / "trace.jsonl",
                stop_reason="finish_called",
                outcome="success",
                outcome_source="runtime",
                source_task="manifest-task-1",
                stream_id="stream-a",
                task_type="manifest",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual({entry.metadata["evolver_tier"] for entry in result.saved}, {"skill", "tool"})
            entry = result.saved[0]
            self.assertEqual(entry.metadata["created_by"], "writer")
            self.assertEqual(entry.metadata["confidence"], 0.8)
            self.assertEqual(entry.metadata["outcome"], "success")
            self.assertEqual(entry.metadata["outcome_source"], "runtime")
            self.assertEqual(entry.metadata["source_task"], "manifest-task-1")
            self.assertEqual(entry.metadata["stream_id"], "stream-a")
            self.assertEqual(entry.metadata["task_type"], "manifest")
            self.assertEqual(entry.metadata["memory_project_key"], "stream:a")
            self.assertEqual(entry.metadata["writer_policy"], "fallback_runtime_v1")
            self.assertEqual(entry.run_id, "run-1")

            events = [event for event, _ in traces]
            self.assertIn("memory.evolver_writer_started", events)
            self.assertIn("memory.evolver_writer_proposed", events)
            self.assertIn("memory.evolver_writer_saved", events)
            saved_payload = [payload for event, payload in traces if event == "memory.evolver_writer_saved"][-1]
            self.assertEqual(saved_payload["saved_count"], 2)
            self.assertEqual(saved_payload["memory_project_key"], "stream:a")
            self.assertEqual(saved_payload["source_task"], "manifest-task-1")
            self.assertEqual(saved_payload["saved_records"][0]["tier"], "skill")

    def test_write_experiences_appends_self_describing_dataset_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "datasets" / "writer.jsonl"
            config = _config(
                Path(tmp) / "memory",
                memory_evolver_mode="retrieve_select",
                memory_evolver_writer_enabled=True,
                memory_evolver_writer_dataset_path=dataset_path,
                memory_project_key="stream:a",
            )
            manager = MemoryManager.from_config(config=config, llm=FakeLLM(), repo_path=repo)
            manager.save_experience("pytest selected skill", tier="skill", source_task="task-skill")
            manager.save_experience("pytest useful tip", tier="tip", source_task="task-tip")
            manager.build_evolver_context_for_query("pytest")

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                trace_path=Path(tmp) / "trace.jsonl",
                stop_reason="finish_called",
                outcome="success",
                outcome_source="runtime",
                source_task="manifest-task-1",
                stream_id="stream-a",
                task_type="manifest",
                memory_mode="shared_stream",
                tool_history=[
                    _tool_record(
                        "run_tests",
                        ok=True,
                        output="hidden_test_output should not be persisted",
                    )
                ],
            )

            rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            record = rows[0]
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(record["run_id"], "run-1")
            self.assertEqual(record["trace_path"], str(Path(tmp) / "trace.jsonl"))
            self.assertEqual(record["source_task"], "manifest-task-1")
            self.assertEqual(record["task_id"], "manifest-task-1")
            self.assertEqual(record["task_type"], "manifest")
            self.assertEqual(record["stream_id"], "stream-a")
            self.assertEqual(record["memory_project_key"], "stream:a")
            self.assertEqual(record["memory_mode"], "shared_stream")
            self.assertEqual(record["outcome"], "success")
            self.assertEqual(record["selected_memory_ids"], [item.candidate.id for item in manager.last_evolver_selection.selected])
            self.assertIn("skill", record["candidate_memory_ids_by_tier"])
            self.assertIn("skill", record["selected_memory_ids_by_tier"])
            self.assertEqual(record["saved_ids"], [entry.id for entry in result.saved])
            self.assertEqual(record["saved_records"][0]["tier"], "skill")
            self.assertTrue(record["proposals"])
            self.assertEqual(record["steps"][0]["output"], "")
            self.assertTrue(record["steps"][0]["output_redacted"])
            self.assertNotIn("hidden_test_output", json.dumps(record, ensure_ascii=False))

    def test_write_experiences_dataset_append_error_traces_failure_without_losing_saved_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "writer-as-directory"
            dataset_path.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_writer_enabled=True,
                    memory_evolver_writer_dataset_path=dataset_path,
                    memory_project_key="project?token=project-token-value",
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                source_task="task?api_key=plain-secret-value",
                stream_id="stream?cookie=session-cookie-value",
                task_type="manifest?password=plain-password-value",
                memory_mode="mode?secret=plain-mode-secret",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual(len(result.saved), 2)
            failed = [payload for event, payload in traces if event == "memory.evolver_writer_failed"][-1]
            self.assertEqual(failed["phase"], "dataset")
            self.assertIn("IsADirectoryError", failed["error"])
            failed_text = json.dumps(failed, ensure_ascii=False)
            traces_text = json.dumps(traces, ensure_ascii=False)
            self.assertNotIn("plain-secret-value", failed_text)
            self.assertNotIn("plain-password-value", failed_text)
            self.assertNotIn("session-cookie-value", failed_text)
            self.assertNotIn("plain-mode-secret", failed_text)
            self.assertNotIn("project-token-value", failed_text)
            self.assertNotIn("plain-secret-value", traces_text)
            self.assertNotIn("plain-password-value", traces_text)
            self.assertNotIn("session-cookie-value", traces_text)
            self.assertNotIn("plain-mode-secret", traces_text)
            self.assertNotIn("project-token-value", traces_text)
            self.assertTrue(str(failed["source_task"]).startswith("redacted_"))

    def test_write_experiences_dataset_redacts_rejected_and_result_errors(self) -> None:
        class SecretRejectedWriter:
            def propose(self, *_: object, **__: object) -> ExperienceWriteResult:
                return ExperienceWriteResult(
                    rejected=(
                        {
                            "reason": "llm_parse_failed",
                            "error": "token=plain-token-value password=plain-password-value",
                        },
                    ),
                    error="cookie=session-cookie-value",
                    llm_used=True,
                    fallback_used=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "writer.jsonl"
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_writer_enabled=True,
                    memory_evolver_writer_dataset_path=dataset_path,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.evolver_writer = SecretRejectedWriter()  # type: ignore[assignment]

            manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            record_text = dataset_path.read_text(encoding="utf-8")
            record = json.loads(record_text)
            self.assertNotIn("plain-token-value", record_text)
            self.assertNotIn("plain-password-value", record_text)
            self.assertNotIn("session-cookie-value", record_text)
            self.assertEqual(record["rejected"][0]["error"], "")
            self.assertTrue(record["error"].startswith("redacted_"))

    def test_write_experiences_dataset_redacts_secret_arguments_and_join_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "writer.jsonl"
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_writer_enabled=True,
                    memory_evolver_writer_dataset_path=dataset_path,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )

            manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                trace_path=Path(tmp) / "trace-api_key=plain-secret-value.jsonl",
                outcome="success",
                source_task="task?api_key=plain-secret-value",
                stream_id="stream-a",
                task_type="manifest",
                tool_history=[
                    _tool_record(
                        "run_tests",
                        ok=True,
                        arguments={
                            "api_key": "plain-secret-value",
                            "token": "plain-token-value",
                            "password": "plain-password-value",
                            "cookie": "session-cookie-value",
                            "secret": "generic-secret-value",
                            "access_key": "plain-access-key-value",
                            "nested": {"private_key": "nested-secret-value"},
                            "command": "pytest tests/test_example.py -q",
                        },
                    )
                ],
            )

            record_text = dataset_path.read_text(encoding="utf-8")
            record = json.loads(record_text)
            self.assertNotIn("plain-secret-value", record_text)
            self.assertNotIn("nested-secret-value", record_text)
            self.assertNotIn("plain-token-value", record_text)
            self.assertNotIn("plain-password-value", record_text)
            self.assertNotIn("session-cookie-value", record_text)
            self.assertNotIn("generic-secret-value", record_text)
            self.assertNotIn("plain-access-key-value", record_text)
            self.assertTrue(record["source_task"].startswith("redacted_"))
            self.assertTrue(record["task_id"].startswith("redacted_"))
            self.assertTrue(record["trace_path"].startswith("redacted_"))
            self.assertEqual(record["steps"][0]["arguments"]["command"], "pytest tests/test_example.py -q")
            self.assertTrue(any(key.startswith("redacted_") for key in record["steps"][0]["arguments"]))
            self.assertTrue(any(key.startswith("redacted_") for key in record["steps"][0]["arguments"]["nested"]))

    def test_write_experiences_reserved_metadata_fields_override_proposal_metadata(self) -> None:
        long_key = "k" * 2_000

        class ReservedOverrideWriter:
            def propose(self, *_: object, **__: object) -> ExperienceWriteResult:
                return ExperienceWriteResult(
                    proposals=(
                        ExperienceWriteProposal(
                            ExperienceTier.SKILL,
                            "Use focused pytest verification after a small patch.",
                            0.9,
                            metadata={
                                "source_task": "llm-task",
                                "stream_id": "llm-stream",
                                "task_type": "llm-type",
                                "memory_project_key": "llm-project",
                                "writer_policy": "llm-policy",
                                "source_trace": "llm-trace",
                                long_key: "long key value",
                            },
                            reason="llm reason",
                        ),
                    ),
                    llm_used=True,
                )

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_writer_enabled=True,
                    memory_project_key="stream:a",
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.evolver_writer = ReservedOverrideWriter()  # type: ignore[assignment]

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                trace_path=Path(tmp) / "trace.jsonl",
                outcome="success",
                source_task="manifest-task-1",
                stream_id="stream-a",
                task_type="manifest",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            metadata = result.saved[0].metadata
            self.assertEqual(metadata["source_task"], "manifest-task-1")
            self.assertEqual(metadata["stream_id"], "stream-a")
            self.assertEqual(metadata["task_type"], "manifest")
            self.assertEqual(metadata["memory_project_key"], "stream:a")
            self.assertEqual(metadata["writer_policy"], "fallback_runtime_v1")
            self.assertEqual(metadata["source_trace"], str(Path(tmp) / "trace.jsonl"))
            self.assertNotIn(long_key, metadata)
            self.assertTrue(all(len(key) <= 1_000 for key in metadata))

    def test_write_experiences_failure_saves_tip_and_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_writer_enabled=True),
                llm=FakeLLM(),
                repo_path=repo,
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                stop_reason="max_steps_reached",
                outcome="failure",
                tool_history=[
                    _tool_record("read_file", ok=True, output="code"),
                    _tool_record("run_tests", ok=False, output="failed", reason="failed"),
                ],
            )

            self.assertEqual({entry.metadata["evolver_tier"] for entry in result.saved}, {"tip", "trajectory"})

    def test_write_experiences_duplicate_content_is_reported_without_new_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_writer_enabled=True),
                llm=FakeLLM(),
                repo_path=repo,
            )
            kwargs = {
                "task": "Fix focused pytest failure",
                "run_id": "run-1",
                "stop_reason": "finish_called",
                "outcome": "success",
                "tool_history": [_tool_record("run_tests", ok=True)],
            }

            first = manager.write_experiences_from_run(**kwargs)
            second = manager.write_experiences_from_run(**kwargs)

            self.assertEqual(len(first.saved), 2)
            self.assertEqual(second.saved, ())
            self.assertEqual(len(second.duplicate_ids), 2)

    def test_write_experiences_writer_exception_traces_failure_without_raising(self) -> None:
        class BrokenWriter:
            def propose(self, *_: object, **__: object) -> object:
                raise RuntimeError("writer exploded")

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_writer_enabled=True),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.evolver_writer = BrokenWriter()  # type: ignore[assignment]

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertIn("writer exploded", result.error)
            failed = [payload for event, payload in traces if event == "memory.evolver_writer_failed"][-1]
            self.assertIn("RuntimeError", failed["error"])
            self.assertEqual(failed["phase"], "unknown")

    def test_noop_write_experiences_returns_empty_result_without_writing_or_tracing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            config = _config(Path(tmp) / "memory", memory_evolver_writer_enabled=True)
            manager = NoopMemoryManager(
                config=config,
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            result = manager.write_experiences_from_run(
                task="Fix focused pytest failure",
                run_id="run-1",
                outcome="success",
                tool_history=[_tool_record("run_tests", ok=True)],
            )

            self.assertEqual(result.saved, ())
            self.assertEqual(traces, [])
            self.assertFalse((Path(config.memory_dir) / "long_term_memory.jsonl").exists())


class MemoryManagerEvolverContextTests(unittest.TestCase):
    def test_evolver_disabled_keeps_legacy_context_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.save_fact("ordinary durable fact about config", scope=MemoryScope.PROJECT)

            context = manager.build_context_for_query("ordinary config")

            self.assertIn("Relevant long-term memory:", context.injected_text)
            self.assertIn("ordinary durable fact about config", context.injected_text)
            self.assertIn("memory.retrieved", [event for event, _ in traces])
            self.assertNotIn("memory.evolver_selected", [event for event, _ in traces])

    def test_evolver_enabled_injects_only_selected_experience(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_selected_max_items=1,
                    memory_evolver_tier_caps={"trajectory": 1, "tip": 1, "skill": 1, "tool": 1},
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_fact("ordinary pytest fact should stay out", scope=MemoryScope.PROJECT)
            manager.save_experience(
                "pytest weak tip",
                tier="tip",
                source_task="task-weak",
            )
            manager.save_experience(
                "pytest boosted skill",
                tier="skill",
                source_task="task-strong",
                metadata={"evolver_value": 0.5, "confidence": 1.2},
            )

            context = manager.build_context_for_query("pytest")

            self.assertTrue(context.injected_text.startswith("Relevant selected experience:"))
            self.assertIn("pytest boosted skill", context.injected_text)
            self.assertNotIn("pytest weak tip", context.injected_text)
            self.assertNotIn("ordinary pytest fact", context.injected_text)
            self.assertEqual([hit.entry.content for hit in context.hits], ["pytest boosted skill"])
            self.assertIsNotNone(manager.last_evolver_selection)

    def test_evolver_enabled_ignores_legacy_include_short_term_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_selected_max_items=1,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_fact("ordinary pytest fact should not be injected", scope=MemoryScope.PROJECT)
            manager.append_user_message("short-term pytest note should not be injected")
            manager.save_experience(
                "selected pytest tip",
                tier="tip",
                source_task="task-selected",
            )

            context = manager.build_context_for_query("pytest", include_short_term=True)

            self.assertTrue(context.injected_text.startswith("Relevant selected experience:"))
            self.assertIn("selected pytest tip", context.injected_text)
            self.assertNotIn("ordinary pytest fact", context.injected_text)
            self.assertNotIn("short-term pytest note", context.injected_text)
            self.assertEqual([hit.entry.content for hit in context.hits], ["selected pytest tip"])

    def test_evolver_context_traces_candidates_and_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_selected_max_items=1,
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.save_experience("pytest tip", tier="tip", source_task="task-tip")
            manager.save_experience("pytest skill", tier="skill", source_task="task-skill")

            context = manager.build_context_for_query("pytest")

            candidates = [payload for event, payload in traces if event == "memory.evolver_candidates"][-1]
            selected = [payload for event, payload in traces if event == "memory.evolver_selected"][-1]
            retrieved = [payload for event, payload in traces if event == "memory.retrieved"][-1]
            self.assertEqual(candidates["candidate_count"], 2)
            self.assertEqual(candidates["selection_policy"], "rule_tier_weighted_v1")
            self.assertEqual(candidates["memory_project_key"], manager.project_key)
            self.assertEqual(selected["selected_count"], 1)
            self.assertEqual(selected["estimated_tokens"], context.estimated_tokens)
            self.assertEqual(retrieved["hits"], len(context.hits))
            self.assertEqual(retrieved["mode"], "retrieve_select")
            summary = candidates["candidate_summaries"][0]
            self.assertIn("score", summary)
            self.assertIn("tokens", summary)
            self.assertIn("source_task", summary)

    def test_evolver_candidate_trace_uses_selector_visible_candidates_for_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_min_score=2.0,
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.save_experience("pytest tip below selector threshold", tier="tip")

            context = manager.build_context_for_query("pytest")

            candidates = [payload for event, payload in traces if event == "memory.evolver_candidates"][-1]
            self.assertEqual(context.injected_text, "")
            self.assertEqual(candidates["candidate_count"], 0)
            self.assertEqual(candidates["candidate_ids"], [])
            self.assertEqual(candidates["candidate_summaries"], [])
            self.assertEqual(candidates["tiers"], {})
            self.assertEqual(candidates["retrieved_candidate_count"], 1)
            self.assertEqual(candidates["retrieved_tiers"], {"tip": 1})

    def test_evolver_selector_failure_returns_empty_context_without_legacy_fallback(self) -> None:
        class RaisingSelector:
            def select(self, **_: object) -> object:
                raise RuntimeError("selector unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )
            manager.save_fact("ordinary pytest fact must not leak through fallback", scope=MemoryScope.PROJECT)
            manager.save_experience("pytest tip selected before selector fails", tier="tip")
            manager.evolver_selector = RaisingSelector()  # type: ignore[assignment]

            context = manager.build_context_for_query("pytest")

            failed = [payload for event, payload in traces if event == "memory.evolver_selection_failed"][-1]
            selected = [payload for event, payload in traces if event == "memory.evolver_selected"][-1]
            retrieved = [payload for event, payload in traces if event == "memory.retrieved"][-1]
            self.assertEqual(context.injected_text, "")
            self.assertEqual(context.hits, [])
            self.assertEqual(failed["fallback"], "empty_context")
            self.assertTrue(selected["fallback"])
            self.assertEqual(retrieved["hits"], 0)
            self.assertNotIn("ordinary pytest fact", context.injected_text)

    def test_evolver_candidate_retrieval_is_top_k_per_tier_not_global_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_experience("pytest tip one", tier="tip")
            manager.save_experience("pytest tip two", tier="tip")
            manager.save_experience("pytest skill one", tier="skill")
            manager.save_experience("pytest skill two", tier="skill")

            candidates = manager.retrieve_evolver_candidates("pytest", top_k_per_tier=1)

            tiers = [hit.entry.metadata["evolver_tier"] for hit in candidates]
            self.assertEqual(tiers.count("tip"), 1)
            self.assertEqual(tiers.count("skill"), 1)

    def test_evolver_context_respects_project_and_global_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "repo_a"
            repo_b = Path(tmp) / "repo_b"
            repo_a.mkdir()
            repo_b.mkdir()
            memory_dir = Path(tmp) / "memory"
            config_a = _config(memory_dir, memory_evolver_mode="retrieve_select", memory_project_key="stream:a")
            config_b = _config(memory_dir, memory_evolver_mode="retrieve_select", memory_project_key="stream:b")
            manager_a = MemoryManager.from_config(config=config_a, llm=FakeLLM(), repo_path=repo_a)
            manager_a.save_experience("pytest project a tip", tier="tip")
            manager_a.save_experience("pytest global skill", tier="skill", scope=MemoryScope.GLOBAL)
            manager_b = MemoryManager.from_config(config=config_b, llm=FakeLLM(), repo_path=repo_b)

            context_b = manager_b.build_context_for_query("pytest")

            self.assertNotIn("pytest project a tip", context_b.injected_text)
            self.assertIn("pytest global skill", context_b.injected_text)

    def test_evolver_context_can_be_disabled_by_min_experience_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    memory_evolver_mode="retrieve_select",
                    memory_evolver_min_experience_entries=2,
                ),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.save_experience("pytest one tip", tier="tip")

            context = manager.build_context_for_query("pytest")

            self.assertEqual(context.injected_text, "")
            self.assertTrue(manager.last_evolver_selection.metadata["insufficient_experience_entries"])

    def test_noop_evolver_context_returns_empty_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            config = _config(Path(tmp) / "memory", memory_evolver_mode="retrieve_select")
            manager = NoopMemoryManager(config=config, repo_path=repo)

            context = manager.build_evolver_context_for_query("pytest")

            self.assertEqual(context.injected_text, "")
            self.assertEqual(context.hits, [])
            self.assertFalse((Path(config.memory_dir) / "long_term_memory.jsonl").exists())


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

            _, _, compaction = _prepare_messages(
                manager,
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

            _, _, compaction = _prepare_messages(
                manager,
                base_messages=[Message(role="system", content="base")],
                query="FastAPI",
                tools=[],
                force_compact=True,
            )

            self.assertIsNotNone(compaction)
            self.assertEqual(compaction.extracted_facts, 1)
            facts = [entry.content for entry in manager.long_term.all(project_key=str(repo.resolve()))]
            self.assertIn("Project uses FastAPI for REST APIs", facts)


class AgentContextManagerPromptTests(unittest.TestCase):
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

            messages, context, compaction = _prepare_messages(
                manager,
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
