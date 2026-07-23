from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json

from my_agent.evaluation.manifest_benchmark import CommandResult, ManifestEvalResult
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    ACTION_LOG_SCHEMA_VERSION,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    BenchmarkTask,
    PreparedBenchmarkTask,
    PublicEpisode,
)
from my_agent.evaluation.memory_benchmark.public_episode import (
    build_public_episode,
    public_episode_char_count,
)
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256


HASH = canonical_sha256({"fixture": "public-episode"})


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="fixture",
        subset="os",
        task_id="task-1",
        order_index=1,
        task_group="fixture:os",
        instruction="PUBLIC GOAL: create result.txt",
        split="test",
        source_revision="a" * 40,
        content_hash=HASH,
        environment_spec={"image": "fixture"},
        evaluator_spec={"name": "fixture", "version": "v1", "hash": HASH},
    )


def _prepared(tmp_path: Path, actions: list[dict[str, Any]]) -> PreparedBenchmarkTask:
    task = _task()
    repo = tmp_path / "repo"
    repo.mkdir()
    state = repo / ".agentcli" / "benchmark_state.json"
    state.parent.mkdir()
    state.write_text("{}\n", encoding="utf-8")
    action_log = tmp_path / "actions.jsonl"
    action_log.write_bytes(
        b"".join(canonical_json_bytes(action) + b"\n" for action in actions)
    )
    adapter_state = tmp_path / "adapter_state.json"
    adapter_state.write_text("{}\n", encoding="utf-8")
    return PreparedBenchmarkTask(
        task=task,
        repo_path=repo,
        public_prompt=task.instruction,
        agent_test_command=None,
        initial_environment_command=("python", "ready.py"),
        hidden_evaluator_command=("python", "hidden-evaluator-secret.py"),
        env_overrides={},
        action_log_path=action_log,
        runtime_action_log_path=tmp_path / "runtime-actions.jsonl",
        adapter_state_path=adapter_state,
        public_tool_state_path=state,
        official_result_path=tmp_path / "official_result.json",
    )


def _result(*, final_answer: str = "Created the public file.") -> ManifestEvalResult:
    return ManifestEvalResult(
        task_id="task-1",
        status="passed",
        resolved=True,
        task_valid=True,
        failure_type="",
        initial_visible=CommandResult("", True, 0, skipped=True),
        evaluation_kind="external_state",
        agent_final_answer=final_answer,
        reward=1.0,
        evaluator_name="fixture",
        evaluator_version="v1",
        evaluator_hash=HASH,
        outcome_finalized=True,
    )


def _action(
    sequence: int, *, stdout: str = "public", stderr: str = ""
) -> dict[str, Any]:
    return {
        "schema_version": ACTION_LOG_SCHEMA_VERSION,
        "sequence": sequence,
        "command": f"echo action-{sequence}",
        "returncode": 0,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
        "elapsed_sec": 0.01,
    }


def test_builds_from_public_actions_and_manifest_final_answer_only(
    tmp_path: Path,
) -> None:
    action = _action(1, stdout="s" * 80, stderr="e" * 80)
    action.update(
        {
            "hidden_evaluator_output": "HIDDEN OUTPUT SECRET",
            "evaluator_command": "python hidden-evaluator-secret.py",
            "reference_answer": "REFERENCE ANSWER SECRET",
        }
    )
    episode = build_public_episode(
        _prepared(tmp_path, [action]),
        _result(final_answer="AUTHORITATIVE FINAL ANSWER"),
        observation_max_chars=40,
    )

    rendered = json.dumps(episode.to_dict(), ensure_ascii=False)
    assert episode.final_response == "AUTHORITATIVE FINAL ANSWER"
    assert (
        len(str(episode.actions[0]["stdout"])) + len(str(episode.actions[0]["stderr"]))
        <= 40
    )
    assert "HIDDEN OUTPUT SECRET" not in rendered
    assert "hidden-evaluator-secret.py" not in rendered
    assert "REFERENCE ANSWER SECRET" not in rendered


def test_episode_limit_keeps_goal_recent_action_and_final_result(
    tmp_path: Path,
) -> None:
    actions = [
        _action(index, stdout=f"observation-{index}-" + "x" * 900)
        for index in range(1, 5)
    ]
    prepared = _prepared(tmp_path, actions)
    prepared = replace(
        prepared,
        public_prompt="GOAL-START " + "g" * 1_500,
    )
    episode = build_public_episode(
        prepared,
        _result(final_answer="FINAL-START " + "f" * 1_500),
        observation_max_chars=1_000,
        episode_max_chars=1_024,
    )

    assert public_episode_char_count(episode) <= 1_024
    assert episode.instruction.startswith("GOAL-START")
    assert episode.final_response.startswith("FINAL-START")
    assert episode.actions
    assert episode.actions[-1]["sequence"] == 4
    assert episode.resolved is True
    assert episode.reward == 1.0


def test_saved_episode_round_trips_for_mem0_add(tmp_path: Path) -> None:
    episode = build_public_episode(_prepared(tmp_path, [_action(1)]), _result())
    saved = tmp_path / "public_episode.json"
    saved.write_text(json.dumps(episode.to_dict()), encoding="utf-8")

    restored = PublicEpisode.from_dict(json.loads(saved.read_text(encoding="utf-8")))

    assert restored == episode
