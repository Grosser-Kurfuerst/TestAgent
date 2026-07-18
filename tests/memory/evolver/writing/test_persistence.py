from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from my_agent.config import AgentConfig
from my_agent.llm import FakeLLM
from my_agent.memory.manager import MemoryManager


def _config(memory_dir: Path, *, dataset_path: Path | None = None) -> AgentConfig:
    return AgentConfig(
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        max_steps=8,
        command_timeout=60,
        trace_dir=Path("traces"),
        use_fake_llm=True,
        memory_dir=memory_dir,
        memory_context_tokens=200,
        memory_retrieval_limit=8,
        memory_evolver_mode="full",
        memory_evolver_writer_enabled=True,
        memory_evolver_writer_dataset_path=dataset_path,
    )


def _successful_tool_history() -> list[dict[str, object]]:
    return [{
        "call": {
            "tool": "run_tests",
            "arguments": {"command": "pytest tests/test_example.py -q"},
        },
        "result": {"ok": True, "output": "passed", "reason": ""},
    }]


class LegacyWritingPersistenceTests(unittest.TestCase):
    def test_bulk_writer_preserves_per_memory_created_and_duplicate_events(self) -> None:
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
            kwargs = {
                "task": "Fix focused pytest failure",
                "run_id": "run-1",
                "outcome": "success",
                "tool_history": _successful_tool_history(),
            }

            manager.write_experiences_from_run(**kwargs)
            manager.write_experiences_from_run(**kwargs)

        saved_events = [
            payload
            for event, payload in traces
            if event == "memory.evolver_saved"
        ]
        self.assertEqual(len(saved_events), 4)
        self.assertEqual(
            [payload["created"] for payload in saved_events],
            [True, True, False, False],
        )

    def test_atomic_write_failure_emits_failure_without_saved_event_or_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            dataset_path = Path(tmp) / "writer.jsonl"
            traces: list[tuple[str, dict[str, object]]] = []
            manager = MemoryManager.from_config(
                config=_config(
                    Path(tmp) / "memory",
                    dataset_path=dataset_path,
                ),
                llm=FakeLLM(),
                repo_path=repo,
                trace_sink=lambda event, payload: traces.append((event, payload)),
            )

            with patch.object(
                manager.experience_store,
                "append_all_atomically",
                side_effect=OSError("disk full"),
            ):
                result = manager.write_experiences_from_run(
                    task="Fix focused pytest failure",
                    run_id="run-1",
                    outcome="success",
                    tool_history=_successful_tool_history(),
                )

        event_names = [event for event, _payload in traces]
        self.assertIn("OSError: disk full", result.error)
        self.assertNotIn("memory.evolver_writer_saved", event_names)
        self.assertNotIn("memory.evolver_saved", event_names)
        self.assertIn("memory.evolver_writer_failed", event_names)
        self.assertFalse(dataset_path.exists())


if __name__ == "__main__":
    unittest.main()
