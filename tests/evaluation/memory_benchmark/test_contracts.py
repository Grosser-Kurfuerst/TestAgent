from __future__ import annotations

from dataclasses import replace

import pytest

from my_agent.evaluation.memory_benchmark.contracts import (
    BenchmarkTask,
    MemoryBenchmarkTaskResult,
    MemoryContextSelection,
    OfficialEvaluatorResult,
    ProviderUsage,
    PublicEpisode,
)
from my_agent.policy.identity import canonical_sha256


HASH = canonical_sha256({"fixture": True})
REVISION = "a" * 40


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="lifelong_os",
        subset="os",
        task_id="os-1",
        order_index=1,
        task_group="lifelong_os:os",
        instruction="Create the requested file.",
        split="test",
        source_revision=REVISION,
        content_hash=HASH,
        environment_spec={"image": "fixture"},
        evaluator_spec={"name": "official"},
        tags=("os",),
    )


def test_benchmark_task_round_trips() -> None:
    task = _task()

    assert BenchmarkTask.from_dict(task.to_dict()) == task


@pytest.mark.parametrize(
    "environment_spec",
    [
        {"expected_output": "secret"},
        {"nested": {"reference_solution": "secret"}},
        {"items": [{"ground_truth": "secret"}]},
    ],
)
def test_benchmark_task_rejects_hidden_answer_fields(
    environment_spec: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="hidden answer field"):
        replace(_task(), environment_spec=environment_spec)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("task_group", "wrong", "task_group"),
        ("split", "train", "split"),
        ("source_revision", "main", "source_revision"),
        ("order_index", 0, "order_index"),
    ],
)
def test_benchmark_task_rejects_invalid_identity_fields(
    field_name: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_task(), **{field_name: value})


def test_memory_context_selection_enforces_item_and_content_budgets() -> None:
    with pytest.raises(ValueError, match="20 items"):
        MemoryContextSelection(
            backend="mem0",
            candidate_count=21,
            selected_ids=tuple(f"id-{index}" for index in range(21)),
            selected_texts=tuple("memory" for _ in range(21)),
            selected_content_tokens=21,
            injected_text="memory",
            estimated_tokens=30,
            retrieval_elapsed_sec=0.1,
        )

    with pytest.raises(ValueError, match="1800 tokens"):
        MemoryContextSelection(
            backend="mem0",
            candidate_count=1,
            selected_ids=("id-1",),
            selected_texts=("memory",),
            selected_content_tokens=1801,
            injected_text="memory",
            estimated_tokens=1810,
            retrieval_elapsed_sec=0.1,
        )


def test_empty_no_memory_selection_is_explicitly_zero() -> None:
    selection = MemoryContextSelection(
        backend="no_memory",
        candidate_count=0,
        selected_ids=(),
        selected_texts=(),
        selected_content_tokens=0,
        injected_text="",
        estimated_tokens=0,
        retrieval_elapsed_sec=0.0,
    )

    assert MemoryContextSelection.from_dict(selection.to_dict()) == selection


def test_no_memory_selection_rejects_candidates_or_retrieval_cost() -> None:
    with pytest.raises(ValueError, match="no_memory"):
        MemoryContextSelection(
            backend="no_memory",
            candidate_count=1,
            selected_ids=(),
            selected_texts=(),
            selected_content_tokens=0,
            injected_text="",
            estimated_tokens=0,
            retrieval_elapsed_sec=0.0,
        )
    with pytest.raises(ValueError, match="no_memory"):
        MemoryContextSelection(
            backend="no_memory",
            candidate_count=0,
            selected_ids=(),
            selected_texts=(),
            selected_content_tokens=0,
            injected_text="",
            estimated_tokens=0,
            retrieval_elapsed_sec=0.01,
        )


def test_provider_usage_distinguishes_unknown_from_zero() -> None:
    assert ProviderUsage().available is False
    assert ProviderUsage().resolved_total_tokens is None
    assert ProviderUsage(prompt_tokens=0, completion_tokens=0).available is True
    assert ProviderUsage(prompt_tokens=7, completion_tokens=3).resolved_total_tokens == 10


def test_official_result_requires_valid_schema_hash_and_reward() -> None:
    result = OfficialEvaluatorResult(
        task_id="os-1",
        evaluator_hash=HASH,
        resolved=True,
        reward=1.0,
    )

    assert OfficialEvaluatorResult.from_dict(result.to_dict()) == result
    with pytest.raises(ValueError, match="reward"):
        replace(result, reward=1.1)
    with pytest.raises(ValueError, match="schema"):
        OfficialEvaluatorResult.from_dict({**result.to_dict(), "schema_version": "old"})


def test_public_episode_round_trips_public_actions() -> None:
    episode = PublicEpisode(
        task_id="os-1",
        instruction="Create the requested file.",
        actions=({"command": "touch /tmp/ok", "stdout": ""},),
        final_response="Done",
        resolved=True,
        reward=1.0,
        failure_type="",
    )

    assert PublicEpisode.from_dict(episode.to_dict()) == episode


def _task_result() -> MemoryBenchmarkTaskResult:
    return MemoryBenchmarkTaskResult(
        run_id="run-1",
        seed=42,
        actor_sampling_seed_supported=False,
        actor_sampling_seed_effective=None,
        arm="no_memory",
        benchmark="lifelong_os",
        subset="os",
        task_id="os-1",
        order_index=1,
        task_content_hash=HASH,
        actor_identity_hash=HASH,
        tools_hash=HASH,
        evaluator_hash=HASH,
        resolved=True,
        reward=1.0,
        outcome_finalized=True,
        failure_type="",
        agent_steps=2,
        tool_calls=1,
        actor_prompt_tokens=10,
        actor_completion_tokens=2,
        actor_total_tokens=12,
        memory_prompt_tokens=0,
        memory_completion_tokens=0,
        memory_total_tokens=0,
        memory_tokens_by_role={},
        actor_usage_available=True,
        memory_usage_available=True,
        memory_usage_unavailable_reason="",
        system_total_tokens=12,
        embedding_calls=0,
        embedding_elapsed_sec=0.0,
        elapsed_sec=1.5,
        memory={"selected_count": 0, "written_count": 0},
        trace_path="trace.jsonl",
        action_log_path="actions.jsonl",
        official_result_path="official_result.json",
        public_episode_path="public_episode.json",
        protocol_hash=HASH,
        backend_config_hash=HASH,
    )


def test_task_result_round_trips_and_preserves_known_zero_usage() -> None:
    result = _task_result()

    assert MemoryBenchmarkTaskResult.from_dict(result.to_dict()) == result
    assert result.memory_total_tokens == 0
    assert result.system_total_tokens == 12


def test_task_result_rejects_system_total_when_usage_is_unknown() -> None:
    with pytest.raises(ValueError, match="system_total_tokens"):
        replace(
            _task_result(),
            memory_usage_available=False,
            memory_usage_unavailable_reason="provider omitted usage",
        )
