from __future__ import annotations

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
from my_agent.llm.types import ChatResponse
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
                ToolExecutionResult(id="c1", name="run_tests", ok=True, content=long_output)
            )
            entry = manager.short_term.all()[0]
            self.assertLessEqual(len(entry.content), 50)
            self.assertIn("truncated", entry.content)


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


class MemoryManagerDeferredTests(unittest.TestCase):
    def test_prepare_messages_not_yet_implemented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            with self.assertRaises(NotImplementedError):
                manager.prepare_messages(base_messages=[], query="q", tools=[])

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
            count, removed = manager.clear_short_term(extract_first=False)
            self.assertEqual(count, 1)
            self.assertEqual(len(manager.short_term), 0)

    def test_clear_short_term_with_extract_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(Path(tmp) / "memory"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            with self.assertRaises(NotImplementedError):
                manager.clear_short_term(extract_first=True)


if __name__ == "__main__":
    unittest.main()
