from __future__ import annotations

from pathlib import Path

import pytest

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import CommandResult, ManifestEvalResult
from my_agent.evaluation.memory_benchmark.backends import (
    AgentCliFourTierBackend,
    NoMemoryBackend,
    memory_stream_project_key,
)
from my_agent.evaluation.memory_benchmark.contracts import BenchmarkTask, PublicEpisode
from my_agent.policy.identity import canonical_sha256
from my_agent.runtime import run_agent


HASH = canonical_sha256({"fixture": True})


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="lifelong_os",
        subset="os",
        task_id="task-1",
        order_index=1,
        task_group="lifelong_os:os",
        instruction="Perform the task.",
        split="test",
        source_revision="a" * 40,
        content_hash=HASH,
        environment_spec={"image": "fixture"},
        evaluator_spec={"name": "fixture", "version": "v1", "hash": HASH},
    )


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        max_steps=4,
        command_timeout=20,
        trace_dir=tmp_path / "traces",
        use_fake_llm=True,
        memory_dir=tmp_path / "base-memory",
    )


def _manifest_result(*, formal: bool) -> ManifestEvalResult:
    return ManifestEvalResult(
        task_id="task-1",
        status="passed",
        resolved=True,
        task_valid=True,
        failure_type="",
        initial_visible=CommandResult("", True, 0, skipped=True),
        evaluation_kind="external_state",
        reward=1.0,
        evaluator_hash=HASH,
        outcome_finalized=True,
        evolver_writer_status="committed" if formal else "",
        written_memory_ids=["memory-1"] if formal else [],
    )


def _episode() -> PublicEpisode:
    return PublicEpisode(
        task_id="task-1",
        instruction="Perform the task.",
        actions=(),
        final_response="Done",
        resolved=True,
        reward=1.0,
        failure_type="",
    )


@pytest.mark.parametrize(
    ("backend_type", "expected_mode"),
    [(NoMemoryBackend, "off"), (AgentCliFourTierBackend, "formal")],
)
def test_backends_freeze_common_tool_and_isolation_config(
    tmp_path: Path,
    backend_type: type[NoMemoryBackend] | type[AgentCliFourTierBackend],
    expected_mode: str,
) -> None:
    memory_dir = tmp_path / "stream" / "memory"
    project_key = "memory-benchmark:run-1:42:lifelong_os:test"
    backend = backend_type(
        stream_memory_dir=memory_dir,
        stream_project_key=project_key,
    )
    context = backend.prepare_context(_task())

    configured = backend.configure_task(
        _config(tmp_path),
        stream_memory_dir=memory_dir,
        stream_project_key=project_key,
        context=context,
    )

    assert configured.agent_mode == "react"
    assert configured.memory_enabled is True
    assert configured.memory_evolver_mode == expected_mode
    assert configured.memory_dir == memory_dir.resolve()
    assert configured.memory_project_key == project_key
    assert configured.enable_project_tools is True
    assert configured.tool_config_paths == ()
    assert configured.enable_project_plugins is False
    assert configured.mcp_enabled is False
    assert configured.mcp_enable_project_servers is False
    assert configured.hitl_enabled is False


def test_no_memory_context_and_finalize_are_known_zero(tmp_path: Path) -> None:
    backend = NoMemoryBackend(
        stream_memory_dir=tmp_path / "memory",
        stream_project_key="stream-key",
    )

    context = backend.prepare_context(_task())
    finalized = backend.finalize_task(_episode(), _manifest_result(formal=False))
    snapshot = backend.snapshot()

    assert context.candidate_count == 0
    assert context.selected_count == 0
    assert context.injected_text == ""
    assert finalized.status == "not_applicable"
    assert finalized.llm_usage.available is True
    assert finalized.llm_usage.resolved_total_tokens == 0
    assert snapshot.entry_count == 0
    assert sum(snapshot.tier_counts.values()) == 0


def test_no_memory_rejects_reported_persistent_writes(tmp_path: Path) -> None:
    backend = NoMemoryBackend(
        stream_memory_dir=tmp_path / "memory",
        stream_project_key="stream-key",
    )

    with pytest.raises(RuntimeError, match="persistent memory writes"):
        backend.finalize_task(_episode(), _manifest_result(formal=True))


def test_four_tier_uses_native_deferred_context_and_original_runner(tmp_path: Path) -> None:
    backend = AgentCliFourTierBackend(
        stream_memory_dir=tmp_path / "memory",
        stream_project_key="stream-key",
    )

    context = backend.prepare_context(_task())

    assert context.candidate_count == 0
    assert context.selected_count == 0
    assert backend.build_agent_runner(context=context) is run_agent


def test_four_tier_finalize_only_reads_manifest_owned_result(tmp_path: Path) -> None:
    backend = AgentCliFourTierBackend(
        stream_memory_dir=tmp_path / "memory",
        stream_project_key="stream-key",
    )

    finalized = backend.finalize_task(_episode(), _manifest_result(formal=True))

    assert finalized.status == "committed"
    assert finalized.written_ids == ("memory-1",)
    assert finalized.llm_usage.available is False
    assert backend.snapshot().entry_count == 0


def test_stream_project_key_isolates_arm_seed_and_benchmark() -> None:
    identities = {
        memory_stream_project_key(
            run_id="run-1",
            seed=seed,
            benchmark=benchmark,
            arm=arm,
        )
        for seed, benchmark, arm in (
            (42, "lifelong_os", "no_memory"),
            (42, "lifelong_os", "agentcli_four_tier"),
            (43, "lifelong_os", "agentcli_four_tier"),
            (42, "intercode_bash", "agentcli_four_tier"),
        )
    }

    assert len(identities) == 4


def test_backend_rejects_mismatched_stream_identity(tmp_path: Path) -> None:
    backend = NoMemoryBackend(
        stream_memory_dir=tmp_path / "memory",
        stream_project_key="stream-key",
    )

    with pytest.raises(ValueError, match="stream_memory_dir"):
        backend.configure_task(
            _config(tmp_path),
            stream_memory_dir=tmp_path / "other",
            stream_project_key="stream-key",
            context=backend.prepare_context(_task()),
        )
