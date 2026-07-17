from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import _formal_manifest_mode, run_manifest_benchmark
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact, TaskEvolverSession
from my_agent.memory.experience_store import ExperienceStore
from my_agent.observability.tracing import TraceWriter
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from my_agent.schema import AgentState


def _config(root: Path) -> AgentConfig:
    return AgentConfig(
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        max_steps=1,
        command_timeout=20,
        trace_dir=root / "traces",
        use_fake_llm=True,
        memory_dir=root / "memory",
        memory_evolver_mode="formal",
        policy_base_revision="model-revision",
        policy_tokenizer_revision="tokenizer-revision",
        policy_identity_manifest=root / "identity.json",
        embedding_revision="embedding-revision",
    )


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model", "model-revision", "sha256:" + "1" * 64, None,
        "tokenizer-revision", "sha256:" + "2" * 64, "sha256:" + "3" * 64,
    )


class EvolverCoordinatorEvaluationTests(unittest.TestCase):
    def test_formal_manifest_forces_auto_to_react_and_rejects_other_modes(self) -> None:
        self.assertEqual(_formal_manifest_mode("auto", formal=True), "react")
        self.assertEqual(_formal_manifest_mode("react", formal=True), "react")
        with self.assertRaisesRegex(ValueError, "mode=react"):
            _formal_manifest_mode("plan", formal=True)

    def test_manifest_finalizes_formal_episode_only_after_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "value.py").write_text("VALUE = 0\n", encoding="utf-8")
            test_command = [
                sys.executable,
                "-c",
                "from value import VALUE; raise SystemExit(0 if VALUE == 1 else 1)",
            ]
            manifest = root / "tasks.json"
            manifest.write_text(json.dumps({
                "tasks": [{
                    "id": "task-1",
                    "task_group": "group-a",
                    "source": "unit",
                    "repo": str(repo),
                    "task": "set VALUE to one",
                    "test_command": test_command,
                }]
            }), encoding="utf-8")

            seen_modes: list[str] = []

            def agent_runner(**kwargs):
                seen_modes.append(kwargs["mode"])
                work_repo = Path(kwargs["repo_path"])
                (work_repo / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
                state = AgentState.initial(
                    work_repo,
                    kwargs["task"],
                    metadata=kwargs["metadata"],
                )
                writer = TraceWriter.create(Path(kwargs["trace_dir"]), state.run_id)
                state.trace_path = writer.path
                state.done = True
                state.stop_reason = "assistant_final"
                state.final_answer = "done"
                repository_revision = ExperienceStore.from_dir(
                    kwargs["config"].memory_dir
                ).revision()
                session = TaskEvolverSession(
                    task_id="task-1",
                    task_group="group-a",
                    trajectory_id=state.run_id,
                    stream_id=str(kwargs["metadata"]["stream_id"]),
                    memory_project_key="project-a",
                    policy_identity=_identity(),
                    repository_revision=repository_revision,
                    candidate_snapshot_hash=canonical_sha256([]),
                    selected_memory_ids=(),
                    rendered_memory_context="",
                    candidate_snapshot=(),
                )
                state.evolver_episode = AgentEpisodeArtifact(
                    session,
                    writer.path,
                    state.stop_reason,
                    state.final_answer,
                    (),
                )
                return state

            result = run_manifest_benchmark(
                tasks_path=manifest,
                output_dir=root / "output",
                config=_config(root),
                agent_runner=agent_runner,
            ).results[0]

            events = [
                json.loads(line)
                for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(result.resolved)
        self.assertEqual(seen_modes, ["react"])
        self.assertEqual(result.evolver_writer_status, "no_write")
        finalized = [item for item in events if item["event"] == "memory.evolver_task_finalized"]
        self.assertEqual(len(finalized), 1)
        self.assertTrue(finalized[0]["payload"]["outcome_finalized"])
        self.assertEqual(finalized[0]["payload"]["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
