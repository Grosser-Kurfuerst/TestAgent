from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import CommandResult, ManifestEvalResult
from my_agent.evaluation.memory_benchmark.backends import (
    AgentCliFourTierBackend,
    Mem0Backend,
    NoMemoryBackend,
    _four_tier_maintenance_metrics,
    memory_stream_project_key,
)
from my_agent.evaluation.memory_benchmark.api_config import ApiEndpoint
from my_agent.evaluation.memory_benchmark.api_embedding import (
    MemoryBenchmarkApiEmbeddingEncoder,
)
from my_agent.evaluation.memory_benchmark.api_policy import MemoryBenchmarkApiPolicy
from my_agent.evaluation.memory_benchmark.contracts import (
    BenchmarkTask,
    ExternalMemoryItem,
    Mem0SearchResult,
    Mem0WriteResult,
    ProviderUsage,
    PublicEpisode,
)
from my_agent.evaluation.memory_benchmark.external_memory import (
    ExternalContextMemoryManager,
)
from my_agent.memory.manager import MemoryManager
from my_agent.memory.token import estimate_tokens
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
        task_group="lifelong_os:os",
        reward=1.0,
        evaluator_name="fixture",
        evaluator_version="v1",
        evaluator_hash=HASH,
        outcome_finalized=True,
        trace_path="trace.jsonl",
        agent_stop_reason="assistant_final",
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


class _EmbeddingApi:
    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        assert model == "text-embedding-v4"
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[1.0, 0.0])
                for index, _text in enumerate(input)
            ]
        )


class _PolicyCompletions:
    def __init__(self) -> None:
        self.writing_calls = 0
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        system = str(messages[0]["content"])
        tool_calls: list[SimpleNamespace] = []
        if system.startswith("Select zero or more"):
            user = __import__("json").loads(str(messages[1]["content"]))
            allowed = user["allowed_labels_by_tier"]
            selected = {tier: [] for tier in ("skill", "tip", "tool", "trajectory")}
            for tier in selected:
                if allowed[tier]:
                    selected[tier] = [allowed[tier][0]]
                    break
            content = __import__("json").dumps(
                {
                    "selected_skills": selected["skill"],
                    "selected_tips": selected["tip"],
                    "selected_tools": selected["tool"],
                    "selected_trajectories": selected["trajectory"],
                    "reasoning": "fixture",
                }
            )
        elif system.startswith("Extract only reusable"):
            self.writing_calls += 1
            content = (
                '[{"tier":"tip","content":"Reuse the verified benchmark action pattern.",'
                '"payload":{"category":"benchmark","severity":"info",'
                '"trigger":"when executing a similar benchmark task"},'
                '"confidence":0.9,"reason":"supported by the public episode"}]'
                if self.writing_calls == 1
                else "[]"
            )
        else:
            content = ""
            tool_calls = [
                SimpleNamespace(
                    id="finish-1",
                    function=SimpleNamespace(
                        name="finish", arguments='{"summary":"fixture"}'
                    ),
                )
            ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=tool_calls),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        )


def _four_tier_backend(
    tmp_path: Path,
    *,
    stream_project_key: str = "stream-key",
    maintenance_interval_tasks: int = 30,
) -> AgentCliFourTierBackend:
    endpoint = ApiEndpoint(
        api_key="secret",
        base_url="https://example.test/v1",
        model="qwen-plus",
        endpoint_hash=HASH,
    )
    completions = _PolicyCompletions()
    policy = MemoryBenchmarkApiPolicy(
        endpoint,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
    )
    embedding = MemoryBenchmarkApiEmbeddingEncoder(
        replace(endpoint, model="text-embedding-v4"),
        client=SimpleNamespace(embeddings=_EmbeddingApi()),
    )
    return AgentCliFourTierBackend(
        stream_memory_dir=tmp_path / "memory",
        stream_project_key=stream_project_key,
        policy=policy,
        embedding_encoder=embedding,
        maintenance_interval_tasks=maintenance_interval_tasks,
    )


@pytest.mark.parametrize("backend_name", ["no_memory", "agentcli_four_tier"])
def test_backends_freeze_common_tool_and_isolation_config(
    tmp_path: Path,
    backend_name: str,
) -> None:
    memory_dir = tmp_path / "stream" / "memory"
    project_key = "memory-benchmark:run-1:42:lifelong_os:test"
    backend = (
        NoMemoryBackend(
            stream_memory_dir=memory_dir,
            stream_project_key=project_key,
        )
        if backend_name == "no_memory"
        else _four_tier_backend(
            tmp_path / "stream",
            stream_project_key=project_key,
        )
    )
    context = backend.prepare_context(_task())

    configured = backend.configure_task(
        replace(
            _config(tmp_path),
            policy_adapter_path=tmp_path / "adapter",
            policy_identity_manifest=tmp_path / "identity.json",
            embedding_revision="local-revision",
        ),
        stream_memory_dir=memory_dir,
        stream_project_key=project_key,
        context=context,
    )

    assert configured.agent_mode == "react"
    assert configured.memory_enabled is True
    assert configured.memory_evolver_mode == "off"
    assert configured.memory_dir == memory_dir.resolve()
    assert configured.memory_project_key == project_key
    assert configured.enable_project_tools is True
    assert configured.tool_config_paths == ()
    assert configured.enable_project_plugins is False
    assert configured.mcp_enabled is False
    assert configured.mcp_enable_project_servers is False
    assert configured.hitl_enabled is False
    if backend_name == "agentcli_four_tier":
        assert configured.policy_adapter_path is None
        assert configured.policy_identity_manifest is None
        assert configured.embedding_revision == "local-revision"


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


def test_four_tier_prepares_context_once_and_requires_finalize(tmp_path: Path) -> None:
    backend = _four_tier_backend(tmp_path)

    context = backend.prepare_context(_task())

    assert context.candidate_count == 0
    assert context.selected_count == 0
    assert backend.build_agent_runner(context=context) is not run_agent
    with pytest.raises(RuntimeError, match="not been finalized"):
        backend.prepare_context(_task())


def test_four_tier_finalize_writes_only_after_authoritative_result(tmp_path: Path) -> None:
    backend = _four_tier_backend(tmp_path)
    backend.prepare_context(_task())
    unfinalized = _manifest_result(formal=False)
    unfinalized.outcome_finalized = False

    with pytest.raises(RuntimeError, match="unfinalized"):
        backend.finalize_task(_episode(), unfinalized)
    assert backend.snapshot().entry_count == 0

    finalized = backend.finalize_task(_episode(), _manifest_result(formal=False))

    assert finalized.status == "committed"
    assert len(finalized.written_ids) == 1
    assert finalized.llm_usage.available is True
    assert finalized.usage_by_role["writing"].resolved_total_tokens == 12
    assert finalized.embedding_calls == 1
    assert backend.snapshot().entry_count == 1


def test_four_tier_second_task_retrieves_first_task_memory(tmp_path: Path) -> None:
    backend = _four_tier_backend(tmp_path)
    backend.prepare_context(_task())
    first = backend.finalize_task(_episode(), _manifest_result(formal=False))

    task_two = replace(
        _task(),
        task_id="task-2",
        order_index=2,
        instruction="Perform a similar task.",
    )
    context = backend.prepare_context(task_two)

    assert context.candidate_count == 1
    assert context.selected_count == 1
    assert context.selected_ids == first.written_ids
    assert context.selected_texts == (
        "Reuse the verified benchmark action pattern.",
    )


def test_four_tier_writes_failed_authoritative_outcome(tmp_path: Path) -> None:
    backend = _four_tier_backend(tmp_path)
    backend.prepare_context(_task())
    episode = replace(
        _episode(),
        final_response="Could not finish.",
        resolved=False,
        reward=0.0,
        failure_type="official_evaluator_failed",
    )
    result = _manifest_result(formal=False)
    result.resolved = False
    result.reward = 0.0
    result.failure_type = "official_evaluator_failed"

    finalized = backend.finalize_task(episode, result)

    assert finalized.status == "committed"
    assert backend.snapshot().entry_count == 1


def test_four_tier_maintenance_interval_is_owned_by_backend(tmp_path: Path) -> None:
    backend = _four_tier_backend(tmp_path, maintenance_interval_tasks=2)
    backend.prepare_context(_task())
    backend.finalize_task(_episode(), _manifest_result(formal=False))

    task_two = replace(
        _task(),
        task_id="task-2",
        order_index=2,
        instruction="Perform a similar task.",
    )
    episode_two = replace(
        _episode(),
        task_id="task-2",
        instruction=task_two.instruction,
    )
    result_two = _manifest_result(formal=False)
    result_two.task_id = "task-2"
    backend.prepare_context(task_two)

    finalized = backend.finalize_task(episode_two, result_two)

    assert finalized.metrics["maintenance_runs"] == 1
    assert finalized.metrics["maintenance_applied_runs"] == 1
    assert finalized.metrics["maintenance_status"] == "noop"
    backend_events = [
        json.loads(line)
        for line in (tmp_path / "backend_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    cadence = [
        event["payload"]
        for event in backend_events
        if event["event"] == "memory.evolver_maintenance_cadence"
    ]
    assert len(cadence) == 1
    assert cadence[0]["turns"] == 1
    assert cadence[0]["operation_ids"] == []
    assert cadence[0]["error"] == ""


def test_four_tier_maintenance_metrics_expose_abort_error() -> None:
    metrics = _four_tier_maintenance_metrics(
        [
            (
                "memory.evolver_maintenance_cadence",
                {
                    "status": "aborted",
                    "turns": 3,
                    "operation_ids": ["op-1"],
                    "error": (
                        "ValueError: formal maintenance requires exactly one tool call "
                        "per assistant turn"
                    ),
                },
            )
        ],
        status="aborted",
    )

    assert metrics["maintenance_status"] == "aborted"
    assert metrics["maintenance_failures"] == 1
    assert metrics["maintenance_turns"] == 3
    assert metrics["maintenance_operation_ids"] == ["op-1"]
    assert "exactly one tool call" in metrics["maintenance_error"]


def test_four_tier_actor_never_loads_local_transformers_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import my_agent.evaluation.memory_benchmark.backends as backends_module
    from my_agent.policy.transformers_policy import TransformersPolicy

    backend = _four_tier_backend(tmp_path)
    context = backend.prepare_context(_task())
    configured = backend.configure_task(
        replace(
            _config(tmp_path),
            policy_identity_manifest=tmp_path / "identity.json",
        ),
        stream_memory_dir=backend.stream_memory_dir,
        stream_project_key=backend.stream_project_key,
        context=context,
    )
    actor = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        TransformersPolicy,
        "from_config",
        classmethod(
            lambda cls, config: (_ for _ in ()).throw(
                AssertionError("local Transformers policy loaded")
            )
        ),
    )
    monkeypatch.setattr(backends_module, "build_llm", lambda config: actor)
    monkeypatch.setattr(
        backends_module,
        "run_agent",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )

    backend.build_agent_runner(context=context)(
        repo_path=tmp_path,
        task="Perform the task.",
        config=configured,
        mode="react",
    )

    assert captured["llm"] is actor
    assert isinstance(captured["memory_manager"], ExternalContextMemoryManager)
    backend.close()


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


class _FakeMem0Client:
    def __init__(
        self,
        persistence_dir: Path,
        *,
        items: tuple[ExternalMemoryItem, ...] = (),
        search_usage: ProviderUsage = ProviderUsage(),
        add_usage: ProviderUsage = ProviderUsage(),
    ) -> None:
        self.persistence_dir = persistence_dir
        self.items = list(items)
        self.search_usage = search_usage
        self.add_usage = add_usage
        self.search_limits: list[int] = []
        self.added: list[PublicEpisode] = []
        self.closed = False

    def search(self, query: str, *, stream_key: str, limit: int) -> Mem0SearchResult:
        del query, stream_key
        self.search_limits.append(limit)
        return Mem0SearchResult(
            items=tuple(self.items),
            llm_usage=self.search_usage,
            embedding_calls=1,
            embedding_elapsed_sec=0.01,
            elapsed_sec=0.02,
        )

    def add(self, episode: PublicEpisode, *, stream_key: str) -> Mem0WriteResult:
        del stream_key
        self.added.append(episode)
        item = ExternalMemoryItem(
            memory_id=f"written-{len(self.items) + 1}",
            text=f"Stored public episode for {episode.task_id}",
        )
        self.items.append(item)
        return Mem0WriteResult(
            written_ids=(item.memory_id,),
            llm_usage=self.add_usage,
            embedding_calls=2,
            embedding_elapsed_sec=0.02,
            elapsed_sec=0.03,
        )

    def count(self, *, stream_key: str) -> int:
        del stream_key
        return len(self.items)

    def close(self) -> None:
        self.closed = True


def test_mem0_backend_caps_candidates_items_and_selected_content(tmp_path: Path) -> None:
    memory_dir = tmp_path / "stream" / "memory"
    items = tuple(
        ExternalMemoryItem(memory_id=f"memory-{index}", text="x" * 360)
        for index in range(60)
    )
    client = _FakeMem0Client(memory_dir / "mem0", items=items)
    backend = Mem0Backend(
        stream_memory_dir=memory_dir,
        stream_project_key="memory-benchmark:run-1:42:lifelong_os:mem0",
        client=client,
    )

    context = backend.prepare_context(_task())

    assert client.search_limits == [50]
    assert context.candidate_count == 50
    assert context.selected_count == 20
    assert context.selected_content_tokens == sum(
        estimate_tokens(text) for text in context.selected_texts
    )
    assert context.selected_content_tokens <= 1_800
    assert context.injected_text.startswith("Relevant selected external memory:")
    assert "memory-50" not in context.injected_text


def test_mem0_backend_writes_failure_episode_and_preserves_unknown_usage(tmp_path: Path) -> None:
    memory_dir = tmp_path / "stream" / "memory"
    client = _FakeMem0Client(memory_dir / "mem0")
    backend = Mem0Backend(
        stream_memory_dir=memory_dir,
        stream_project_key="memory-benchmark:run-1:42:lifelong_os:mem0",
        client=client,
    )
    backend.prepare_context(_task())
    failed_episode = PublicEpisode(
        task_id="task-1",
        instruction="Perform the task.",
        actions=(),
        final_response="Could not finish.",
        resolved=False,
        reward=0.0,
        failure_type="official_evaluator_failed",
    )
    failed_result = _manifest_result(formal=False)
    failed_result.resolved = False
    failed_result.reward = 0.0
    failed_result.failure_type = "official_evaluator_failed"

    finalized = backend.finalize_task(failed_episode, failed_result)

    assert client.added == [failed_episode]
    assert finalized.status == "committed"
    assert finalized.llm_usage.available is False
    assert finalized.usage_by_role["search"].available is False
    assert finalized.usage_by_role["add"].available is False
    assert finalized.embedding_calls == 3
    assert finalized.embedding_elapsed_sec == pytest.approx(0.03)
    assert backend.snapshot().entry_count == 1


@pytest.mark.parametrize("backend_name", ["no_memory", "mem0"])
def test_non_formal_task_runner_shares_one_llm_with_memory_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
) -> None:
    import my_agent.evaluation.memory_benchmark.backends as backends_module

    memory_dir = tmp_path / "stream" / "memory"
    project_key = f"memory-benchmark:run-1:42:lifelong_os:{backend_name}"
    if backend_name == "mem0":
        client = _FakeMem0Client(memory_dir / "mem0")
        backend: NoMemoryBackend | Mem0Backend = Mem0Backend(
            stream_memory_dir=memory_dir,
            stream_project_key=project_key,
            client=client,
        )
    else:
        backend = NoMemoryBackend(
            stream_memory_dir=memory_dir,
            stream_project_key=project_key,
        )
    context = backend.prepare_context(_task())
    config = backend.configure_task(
        _config(tmp_path),
        stream_memory_dir=memory_dir,
        stream_project_key=project_key,
        context=context,
    )
    llm = object()
    build_calls: list[AgentConfig] = []
    captured: dict[str, object] = {}

    def fake_build_llm(task_config: AgentConfig) -> object:
        build_calls.append(task_config)
        return llm

    def fake_run_agent(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(backends_module, "build_llm", fake_build_llm)
    monkeypatch.setattr(backends_module, "run_agent", fake_run_agent)

    backend.build_agent_runner(context=context)(
        repo_path=tmp_path,
        task="Perform the task.",
        config=config,
        mode="react",
    )

    assert len(build_calls) == 1
    assert captured["llm"] is llm
    manager = captured["memory_manager"]
    if backend_name == "mem0":
        assert isinstance(manager, ExternalContextMemoryManager)
        assert manager.inner.llm is llm
    else:
        assert isinstance(manager, MemoryManager)
        assert manager.llm is llm


def test_no_memory_runner_uses_provider_actor_even_when_legacy_identity_is_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import my_agent.evaluation.memory_benchmark.backends as backends_module
    memory_dir = tmp_path / "stream" / "memory"
    backend = NoMemoryBackend(
        stream_memory_dir=memory_dir,
        stream_project_key="memory-benchmark:run-1:42:lifelong_os:no_memory",
    )
    context = backend.prepare_context(_task())
    config = backend.configure_task(
        replace(
            _config(tmp_path),
            policy_identity_manifest=tmp_path / "policy_identity_manifest.json",
        ),
        stream_memory_dir=memory_dir,
        stream_project_key=backend.stream_project_key,
        context=context,
    )
    actor = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        backends_module,
        "build_llm",
        lambda _config: actor,
    )
    monkeypatch.setattr(
        backends_module,
        "run_agent",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )

    backend.build_agent_runner(context=context)(
        repo_path=tmp_path,
        task="Perform the task.",
        config=config,
        mode="react",
    )

    assert captured["llm"] is actor
    assert captured["memory_manager"].llm is actor
