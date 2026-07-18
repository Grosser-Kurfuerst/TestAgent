from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from my_agent.config import AgentConfig
from my_agent.llm import FakeLLM
from my_agent.memory import (
    DisabledMemoryManager,
    MemoryManager,
    MemoryService,
    NoopMemoryManager,
)
from my_agent.memory.evolver.runtime import (
    DisabledEvolverRuntime,
    LegacyEvolverRuntime,
)


class MemoryRuntimeStrategyTests(unittest.TestCase):
    def test_factory_selects_disabled_and_legacy_strategies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()

            disabled = MemoryManager.from_config(
                config=_config(root / "off", mode="off"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            retrieve = MemoryManager.from_config(
                config=_config(root / "retrieve", mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            full = MemoryManager.from_config(
                config=_config(root / "full", mode="full", writer_enabled=True),
                llm=FakeLLM(),
                repo_path=repo,
            )

        self.assertIsInstance(disabled.evolver_runtime, DisabledEvolverRuntime)
        self.assertIsInstance(retrieve.evolver_runtime, LegacyEvolverRuntime)
        self.assertIsInstance(full.evolver_runtime, LegacyEvolverRuntime)
        self.assertFalse(retrieve.evolver_runtime.write_enabled)
        self.assertTrue(full.evolver_runtime.write_enabled)

    def test_memory_services_satisfy_protocol_and_disabled_alias_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            config = _config(root / "memory", mode="off")
            manager = MemoryManager.from_config(
                config=config,
                llm=FakeLLM(),
                repo_path=repo,
            )
            disabled = DisabledMemoryManager(config=config, repo_path=repo)

        self.assertIsInstance(manager, MemoryService)
        self.assertIsInstance(disabled, MemoryService)
        self.assertIs(DisabledMemoryManager, NoopMemoryManager)

    def test_manager_contains_no_mode_dispatch_or_concrete_policy_construction(self) -> None:
        source = inspect.getsource(MemoryManager)

        self.assertNotIn("memory_evolver_mode", source)
        self.assertNotIn("ExperienceSelector(", source)
        self.assertNotIn("ExperienceWriter(", source)
        self.assertNotIn("EvolverCoordinator(", source)
        self.assertNotIn("EmbeddingRetriever(", source)
        self.assertNotIn("FormalEvolverRuntime", source)
        self.assertNotIn("LegacyEvolverRuntime", source)

    def test_fork_uses_service_forks_and_keeps_repository_shared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            manager = MemoryManager.from_config(
                config=_config(root / "memory", mode="retrieve_select"),
                llm=FakeLLM(),
                repo_path=repo,
            )
            manager.append_user_message("parent")

            forked = manager.fork_for_task(session_id="child")

        self.assertIs(forked.experience_store, manager.experience_store)
        self.assertIsNot(forked.short_term, manager.short_term)
        self.assertEqual(forked.short_term.all(), [])
        self.assertIsNot(forked.compressor, manager.compressor)
        self.assertIsNot(forked.evolver_runtime, manager.evolver_runtime)
        self.assertIsNot(forked.experience_retriever, manager.experience_retriever)
        self.assertIsNot(forked.evolver_selector, manager.evolver_selector)
        self.assertIsNot(forked.evolver_writer, manager.evolver_writer)


def _config(
    memory_dir: Path,
    *,
    mode: str,
    writer_enabled: bool = False,
) -> AgentConfig:
    return AgentConfig(
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        max_steps=8,
        command_timeout=60,
        trace_dir=memory_dir / "traces",
        use_fake_llm=True,
        memory_dir=memory_dir,
        memory_evolver_mode=mode,
        memory_evolver_writer_enabled=writer_enabled,
    )


if __name__ == "__main__":
    unittest.main()
