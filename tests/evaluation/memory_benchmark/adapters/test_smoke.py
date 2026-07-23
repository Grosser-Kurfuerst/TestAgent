from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import subprocess

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import CommandResult, ManifestEvalResult
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    BENCHMARK_LABEL,
    MANAGED_LABEL,
    RUN_LABEL,
    SEED_LABEL,
    TASK_LABEL,
    DockerContainer,
    benchmark_container_name,
)
from my_agent.evaluation.memory_benchmark.adapters.smoke import (
    SMOKE_EVALUATOR_HASH,
    SmokeAdapter,
    smoke_action_main,
    smoke_cli_main,
)
from my_agent.evaluation.memory_benchmark.backends import AgentCliFourTierBackend
from my_agent.evaluation.memory_benchmark.contracts import load_official_evaluator_result
from my_agent.evaluation.memory_benchmark.public_episode import build_public_episode
from my_agent.memory.embedding_retrieval import EmbeddingRetriever
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.experience.models import ExperienceTier
from my_agent.memory.experience_store import ExperienceStore
from my_agent.policy.identity import PolicyIdentity, canonical_sha256
from tests.memory.experience.fixtures import typed_experience


IMAGE_DIGEST = canonical_sha256({"image": "smoke"})


class _FakeDockerRuntime:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.cleaned: list[DockerContainer] = []

    def create_container(self, **kwargs: Any) -> DockerContainer:
        self.created.append(dict(kwargs))
        labels = {
            MANAGED_LABEL: "true",
            RUN_LABEL: str(kwargs["run_id"]),
            SEED_LABEL: str(kwargs["seed"]),
            BENCHMARK_LABEL: str(kwargs["benchmark"]),
            TASK_LABEL: str(kwargs["task_id"]),
        }
        return DockerContainer(
            container_id=f"container-{len(self.created)}",
            name=benchmark_container_name(
                run_id=str(kwargs["run_id"]),
                seed=int(kwargs["seed"]),
                benchmark=str(kwargs["benchmark"]),
                task_id=str(kwargs["task_id"]),
            ),
            image=str(kwargs["image"]),
            image_digest=str(kwargs["expected_digest"]),
            labels=labels,
        )

    def cleanup_container(self, container: DockerContainer) -> None:
        self.cleaned.append(container)


def _adapter(tmp_path: Path, runtime: _FakeDockerRuntime | None = None) -> SmokeAdapter:
    return SmokeAdapter(
        data_root=tmp_path / "data",
        run_id="smoke-test",
        runtime_root=tmp_path / "runtime",
        container_image="ubuntu:locked",
        container_digest=IMAGE_DIGEST,
        docker_runtime=runtime or _FakeDockerRuntime(),
    )


class _Encoder:
    model_revision = "smoke-embedding-revision"
    tokenizer_revision = "smoke-tokenizer-revision"

    def encode_queries(self, texts):
        return tuple((1.0, 0.0) for _ in texts)

    def encode_documents(self, texts):
        return tuple((1.0, 0.0) for _ in texts)


class _AllSelector:
    def select(self, *, task, candidates, token_budget, max_items, context):
        del task, token_budget, context
        return tuple(candidate.memory_id for candidate in candidates[:max_items])


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        "model",
        "revision",
        "sha256:" + "1" * 64,
        None,
        "tokenizer",
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
    )


def test_smoke_adapter_generates_stable_eight_task_suite(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    first = tuple(adapter.load_tasks(limit=8))
    first_bytes = (tmp_path / "data" / "smoke" / "tasks.jsonl").read_bytes()
    second = tuple(adapter.load_tasks(limit=8))

    assert len(first) == 8
    assert first == second
    assert first_bytes == (tmp_path / "data" / "smoke" / "tasks.jsonl").read_bytes()
    assert [task.order_index for task in first] == list(range(1, 9))
    assert [task.tags[-1] for task in first] == ["source"] * 4 + ["variant"] * 4
    assert len({task.content_hash for task in first}) == 8


def test_smoke_public_tool_state_is_arm_neutral(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    task = adapter.load_tasks(limit=1)[0]
    states: list[bytes] = []

    for arm in ("no_memory", "agentcli_four_tier", "mem0"):
        task_dir = tmp_path / arm
        task_dir.mkdir()
        prepared = adapter.prepare_task(task, task_dir=task_dir, seed=42)
        states.append(prepared.public_tool_state_path.read_bytes())
        adapter.cleanup_task(prepared)

    assert len(set(states)) == 1
    rendered = states[0].decode("utf-8")
    assert all(arm not in rendered for arm in ("no_memory", "agentcli_four_tier", "mem0"))


def test_smoke_action_and_official_scorer_preserve_public_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _FakeDockerRuntime()
    adapter = _adapter(tmp_path, runtime)
    task = adapter.load_tasks(limit=1)[0]
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    prepared = adapter.prepare_task(task, task_dir=task_dir, seed=42)
    public_state = json.loads(prepared.public_tool_state_path.read_text(encoding="utf-8"))

    assert "expected_files" not in public_state
    assert "evaluator" not in str(public_state)
    assert public_state["container_name"] == "agentcli-mb-smoke-test-42-smoke-smoke-config-parse-source"
    assert Path(public_state["runtime_action_log_path"]).is_absolute()
    assert runtime.created[0]["bind_mounts"] == {
        (prepared.repo_path / "workspace").resolve(): "/workspace"
    }
    assert runtime.created[0]["working_directory"] == "/workspace"
    assert not prepared.adapter_state_path.resolve().is_relative_to(
        next(iter(runtime.created[0]["bind_mounts"])).resolve()
    )
    hidden_state_before = prepared.adapter_state_path.read_bytes()

    def ready_check(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv == [
            "docker",
            "exec",
            public_state["container_name"],
            "sh",
            "-lc",
            "test -d /workspace",
        ]
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "my_agent.evaluation.memory_benchmark.adapters.smoke.subprocess.run",
        ready_check,
    )
    assert (
        smoke_cli_main(
            ["check-ready", "--state", str(prepared.adapter_state_path)]
        )
        == 0
    )

    def isolated_docker_exec(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[:3] == ["docker", "exec", public_state["container_name"]]
        command = argv[-1]
        if command == "cat ../../adapter_state.json":
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr="cat: ../../adapter_state.json: No such file or directory\n",
            )
        if command.startswith("printf hacked"):
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected container command: {command}")

    monkeypatch.chdir(prepared.repo_path)
    assert (
        smoke_action_main(
            ["--command", "cat ../../adapter_state.json"],
            command_runner=isolated_docker_exec,
        )
        == 1
    )
    assert (
        smoke_action_main(
            ["--command", "printf hacked > ../../adapter_state.json"],
            command_runner=isolated_docker_exec,
        )
        == 0
    )
    assert prepared.adapter_state_path.read_bytes() == hidden_state_before

    (prepared.repo_path / "workspace" / "result.json").write_text(
        '{"host":"api.internal","port":8080}\n',
        encoding="utf-8",
    )
    adapter.finalize_task_artifacts(prepared)
    assert smoke_cli_main(
        [
            "score",
            "--state",
            str(prepared.adapter_state_path),
            "--result",
            str(prepared.official_result_path),
        ]
    ) == 0

    official = load_official_evaluator_result(
        prepared.official_result_path,
        expected_task_id=task.task_id,
        expected_evaluator_hash=SMOKE_EVALUATOR_HASH,
        returncode=0,
    )
    assert official.resolved is True
    actions = [
        json.loads(line)
        for line in prepared.action_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [action["command"] for action in actions] == [
        "cat ../../adapter_state.json",
        "printf hacked > ../../adapter_state.json",
    ]
    assert "expected_files" not in json.dumps(actions)
    result = ManifestEvalResult(
        task_id=task.task_id,
        status="passed",
        resolved=True,
        task_valid=True,
        failure_type="",
        initial_visible=CommandResult("", True, 0, skipped=True),
        evaluation_kind="external_state",
        agent_final_answer="done",
        reward=1.0,
        evaluator_name="agentcli-memory-smoke",
        evaluator_version="memory-benchmark-smoke-v1",
        evaluator_hash=SMOKE_EVALUATOR_HASH,
        outcome_finalized=True,
    )
    episode = build_public_episode(prepared, result)
    assert "expected_files" not in json.dumps(episode.to_dict())
    adapter.cleanup_task(prepared)
    assert runtime.cleaned


def test_four_tier_fixture_enters_candidate_snapshot_and_rendered_context(
    tmp_path: Path,
) -> None:
    store = ExperienceStore.from_dir(tmp_path / "memory")
    for tier in ExperienceTier:
        store.add(
            typed_experience(
                f"{tier.value}-fixture",
                f"public {tier.value} memory",
                tier,
                project_key="smoke-project",
            )
        )
    coordinator = EvolverCoordinator(
        store=store,
        project_key="smoke-project",
        policy_identity=_identity(),
        retriever=EmbeddingRetriever(_Encoder()),
        selector=_AllSelector(),
    )

    session = coordinator.begin_task(
        task="reuse public smoke memory",
        task_id="smoke-fixture",
        task_group="smoke:memory",
        trajectory_id="trajectory-fixture",
        stream_id="smoke-stream",
    )
    context = coordinator.context_for_session(session)

    assert {candidate.tier for candidate in session.candidate_snapshot} == {
        tier.value for tier in ExperienceTier
    }
    assert set(session.selected_memory_ids) == {
        f"{tier.value}-fixture" for tier in ExperienceTier
    }
    assert all(
        f"public {tier.value} memory" in context.injected_text
        for tier in ExperienceTier
    )


def test_smoke_backend_overrides_maintenance_interval(tmp_path: Path) -> None:
    backend = AgentCliFourTierBackend(
        stream_memory_dir=tmp_path / "memory",
        stream_project_key="smoke-project",
        maintenance_interval_tasks=4,
    )
    config = AgentConfig(
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
    task = _adapter(tmp_path).load_tasks(limit=1)[0]
    context = backend.prepare_context(task)

    resolved = backend.configure_task(
        config,
        stream_memory_dir=tmp_path / "memory",
        stream_project_key="smoke-project",
        context=context,
    )

    assert resolved.memory_evolver_maintenance_interval_tasks == 4
