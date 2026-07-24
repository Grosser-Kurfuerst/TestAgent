from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import hashlib
import json
import subprocess

import pytest

import my_agent.cli.commands.memory_benchmark as memory_benchmark_command
from my_agent.cli.common import CliContext
from my_agent.cli.commands.memory_benchmark import (
    preflight_memory_benchmark,
    prepare_memory_benchmark_data,
    run_preflighted_memory_benchmark,
)
from my_agent.config import AgentConfig
from my_agent.evaluation.memory_benchmark.api_config import ApiEndpoint
from my_agent.evaluation.memory_benchmark.api_embedding import (
    MemoryBenchmarkApiEmbeddingEncoder,
)
from my_agent.evaluation.memory_benchmark.api_policy import MemoryBenchmarkApiPolicy
from my_agent.llm.types import ChatResponse, LLMToolCall
from my_agent.policy.identity import canonical_sha256


class _FakeDockerRuntime:
    def __init__(self, digests: dict[str, str]) -> None:
        self.digests = digests

    def preflight(self) -> str:
        return "fixture-docker"

    def inspect_image(self, image: str) -> str:
        return self.digests[image]


class _ProbeActor:
    supports_tools = True

    def chat(
        self,
        _messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        if tools is None:
            return ChatResponse(content="OK", finish_reason="stop")
        return ChatResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                LLMToolCall(
                    id="call_probe",
                    name="memory_benchmark_probe",
                    arguments={},
                    arguments_json="{}",
                )
            ],
        )


class _FailingProbeActor(_ProbeActor):
    def chat(
        self,
        _messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        del tools
        raise RuntimeError("fixture Actor API failure")


class _FakeEmbeddingClient:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> Any:
        del model
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index + 1)] * self.dimension)
                for index, _text in enumerate(input)
            ]
        )


def _actor_factory(_config: AgentConfig) -> _ProbeActor:
    return _ProbeActor()


def _embedding_factory(
    dimension: int = 3,
) -> Callable[[ApiEndpoint], MemoryBenchmarkApiEmbeddingEncoder]:
    return lambda endpoint: MemoryBenchmarkApiEmbeddingEncoder(
        endpoint,
        client=_FakeEmbeddingClient(dimension),
    )


def _policy_factory(endpoint: ApiEndpoint) -> MemoryBenchmarkApiPolicy:
    return MemoryBenchmarkApiPolicy(endpoint, client=SimpleNamespace())


def _mem0_probe(config: dict[str, Any], _path: Path) -> str:
    assert config["llm"]["config"]["model"] == "fixture-chat"
    assert config["embedder"]["config"]["model"] == "text-embedding-v4"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 3
    return "2.0.13"


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _git_init(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _base_config(root: Path) -> AgentConfig:
    return AgentConfig(
        provider="openai",
        api_key="fixture-secret",
        base_url="https://workspace.example.com/compatible-mode/v1",
        model="fixture-chat",
        temperature=0.0,
        max_steps=4,
        command_timeout=20,
        trace_dir=root / "traces",
        use_fake_llm=False,
    )


def _fixture_workspace(
    tmp_path: Path,
    *,
    invalid_evaluator: bool = False,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    lifelong = tmp_path / "lifelong"
    intercode = tmp_path / "intercode"
    lifelong.mkdir()
    intercode.mkdir()
    source_fixture_root = Path(__file__).parents[2] / "data" / "memory_benchmark"
    lifelong_data = lifelong / "tasks.jsonl"
    lifelong_data.write_bytes(
        (source_fixture_root / "lifelong_os_sample.jsonl").read_bytes()
    )
    (lifelong / "fixture_lifelong_eval.py").write_text(
        "def score():\n    return True\n",
        encoding="utf-8",
    )
    intercode_data = intercode / "tasks.jsonl"
    intercode_data.write_bytes(
        (source_fixture_root / "intercode_bash_sample.jsonl").read_bytes()
    )
    (intercode / "fixture_intercode_eval.py").write_text(
        "def score():\n    return True\n",
        encoding="utf-8",
    )
    lifelong_revision = _git_init(lifelong)
    intercode_revision = _git_init(intercode)

    configs = repo / "configs" / "memory_benchmark"
    configs.mkdir(parents=True)
    tracked_config = json.loads(
        (Path(__file__).parents[3] / "configs" / "memory_benchmark" / "v2.json").read_text(
            encoding="utf-8"
        )
    )
    tracked_config["benchmarks"]["lifelong_os"]["limit"] = 2
    tracked_config["benchmarks"]["intercode_bash"]["limit"] = 2
    tracked_config["seeds"] = [42]
    config_path = configs / "v2.json"
    config_path.write_text(json.dumps(tracked_config), encoding="utf-8")
    lifelong_digest = canonical_sha256("lifelong-image")
    intercode_digest = canonical_sha256("intercode-image")
    source_lock = {
        "schema_version": "memory-benchmark-source-lock-v1",
        "sources": {
            "lifelong_agent_bench": {
                "url": "https://example.com/lifelong",
                "revision": lifelong_revision,
                "license": "MIT",
                "task_data_path": "tasks.jsonl",
                "task_data_revision": lifelong_revision,
                "task_data_sha256": _sha256_file(lifelong_data),
                "evaluator_entrypoint": (
                    "missing_lifelong_eval.score"
                    if invalid_evaluator
                    else "fixture_lifelong_eval.score"
                ),
                "container_image": "fixture/lifelong:locked",
                "container_digest": lifelong_digest,
            },
            "intercode": {
                "url": "https://example.com/intercode",
                "revision": intercode_revision,
                "license": "MIT",
                "task_data_path": "tasks.jsonl",
                "task_data_revision": intercode_revision,
                "task_data_sha256": _sha256_file(intercode_data),
                "evaluator_entrypoint": (
                    "missing_intercode_eval.score"
                    if invalid_evaluator
                    else "fixture_intercode_eval.score"
                ),
                "container_image": "fixture/intercode:locked",
                "container_digest": intercode_digest,
            },
        },
    }
    (configs / "source-lock.json").write_text(
        json.dumps(source_lock), encoding="utf-8"
    )
    (repo / "uv.lock").write_text("fixture lock\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "data/\nevaluationResults/\n", encoding="utf-8"
    )
    _git_init(repo)
    env = {
        "AGENTCLI_LIFELONG_AGENT_BENCH_ROOT": str(lifelong),
        "AGENTCLI_INTERCODE_ROOT": str(intercode),
        "AGENTCLI_MEMORY_BENCHMARK_DATA_ROOT": str(repo / "data" / "memory_benchmark"),
    }
    return {
        "repo": repo,
        "config": config_path,
        "env": env,
        "docker": _FakeDockerRuntime(
            {
                "fixture/lifelong:locked": lifelong_digest,
                "fixture/intercode:locked": intercode_digest,
            }
        ),
    }


def _preflight(
    fixture: dict[str, Any],
    run_dir: Path,
    *,
    actor_factory: Callable[[AgentConfig], Any] = _actor_factory,
) -> dict[str, Any]:
    return preflight_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        base_config=_base_config(fixture["repo"]),
        env=fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        docker_runtime=fixture["docker"],
        actor_factory=actor_factory,
        embedding_encoder_factory=_embedding_factory(),
        mem0_probe=_mem0_probe,
    )


def _fake_stream_runner(calls: list[dict[str, Any]]) -> Callable[..., Any]:
    def run(**kwargs: Any) -> Any:
        assert set(kwargs["adapter"]._records) == {
            task.task_id for task in kwargs["tasks"]
        }
        calls.append(dict(kwargs))
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        results_path = output / "results.jsonl"
        results_path.write_text("", encoding="utf-8")
        executions = tuple(
            SimpleNamespace(task_result=SimpleNamespace(resolved=True))
            for _task in kwargs["tasks"]
        )
        return SimpleNamespace(
            arm=kwargs["backend"].name,
            seed=kwargs["seed"],
            benchmark=kwargs["tasks"][0].benchmark,
            results_path=results_path,
            executions=executions,
        )

    return run


def _run(
    fixture: dict[str, Any],
    run_dir: Path,
    *,
    arms: str,
    base_config: AgentConfig | None = None,
    env: dict[str, str] | None = None,
    embedding_dimension: int = 3,
    mem0_version: str = "2.0.13",
    stream_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    return run_preflighted_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        seed=42,
        arms=arms,
        base_config=base_config or _base_config(fixture["repo"]),
        env=env or fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        stream_runner=stream_runner or _fake_stream_runner([]),
        api_policy_factory=_policy_factory,
        embedding_encoder_factory=_embedding_factory(embedding_dimension),
        mem0_version_resolver=lambda: mem0_version,
    )


def test_prepare_is_deterministic_and_preflight_is_immutable(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path)
    first = prepare_memory_benchmark_data(
        config_path=fixture["config"], env=fixture["env"]
    )
    suite_path = Path(first["suite_manifest"])
    first_bytes = suite_path.read_bytes()
    second = prepare_memory_benchmark_data(
        config_path=fixture["config"], env=fixture["env"]
    )

    assert first_bytes == suite_path.read_bytes()
    assert first["benchmarks"] == second["benchmarks"]
    run_dir = fixture["repo"] / "evaluationResults" / "pilot"
    preflight = _preflight(fixture, run_dir)
    protocol_bytes = (run_dir / "protocol.json").read_bytes()
    repeated = _preflight(fixture, run_dir)

    assert preflight == repeated
    assert preflight["schema_version"] == "memory-benchmark-preflight-v2"
    assert preflight["actor_api"]["model"] == "fixture-chat"
    assert preflight["embedding_api"]["dimension"] == 3
    assert protocol_bytes == (run_dir / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["pilot"] is True
    assert protocol["ordered_task_ids_by_benchmark"] == {"intercode_bash": ["0"]}
    four_tier = json.loads(
        (run_dir / "arms" / "agentcli_four_tier" / "backend_config.json").read_text()
    )
    mem0 = json.loads(
        (run_dir / "arms" / "mem0" / "backend_config.json").read_text()
    )
    serialized = json.dumps({"preflight": preflight, "four_tier": four_tier, "mem0": mem0})
    assert "fixture-secret" not in serialized
    assert "api_key" not in serialized
    assert four_tier["policy_model"] == mem0["llm_model"] == "fixture-chat"
    assert four_tier["embedding_model"] == mem0["embedding_model"]
    assert four_tier["embedding_dimension"] == mem0["embedding_dimension"] == 3


def test_smoke_cli_passes_memory_generation_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_workspace(tmp_path)
    captured: dict[str, Any] = {}
    env = {
        **fixture["env"],
        "MY_AGENT_LLM_PROVIDER": "openai",
        "MY_AGENT_API_KEY": "fixture-secret",
        "MY_AGENT_BASE_URL": "https://workspace.example.com/compatible-mode/v1",
        "MY_AGENT_MODEL": "fixture-chat",
    }

    class _EmbeddingProbe:
        def __init__(self, _endpoint: ApiEndpoint) -> None:
            pass

        def encode_queries(self, _texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return ((1.0, 0.0, 0.0),)

    monkeypatch.setattr(memory_benchmark_command, "build_llm", lambda _config: _ProbeActor())
    monkeypatch.setattr(
        memory_benchmark_command,
        "MemoryBenchmarkApiEmbeddingEncoder",
        _EmbeddingProbe,
    )
    monkeypatch.setattr(
        memory_benchmark_command,
        "_mem0_package_version",
        lambda: "2.0.13",
    )
    monkeypatch.setattr(
        memory_benchmark_command,
        "run_smoke_benchmark",
        lambda **kwargs: captured.update(kwargs) or {"status": "passed"},
    )
    ctx = CliContext(run_agent=lambda **_kwargs: None, agent_repl_cls=object, env=env)

    result = memory_benchmark_command._run_smoke(
        SimpleNamespace(config=fixture["config"], output_dir=tmp_path / "smoke-output"),
        ctx,
    )

    assert result["status"] == "passed"
    assert captured["generation_temperature"] == 0.2
    assert captured["generation_top_p"] == 1.0


def test_preflight_probe_failure_does_not_write_protocol(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])
    run_dir = fixture["repo"] / "evaluationResults" / "failed"

    with pytest.raises(RuntimeError, match="fixture Actor API failure"):
        _preflight(fixture, run_dir, actor_factory=lambda _config: _FailingProbeActor())

    assert not (run_dir / "protocol.json").exists()
    assert not (run_dir / "preflight.json").exists()


def test_run_rejects_selection_drift_and_existing_stream(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])
    run_dir = fixture["repo"] / "evaluationResults" / "pilot"
    _preflight(fixture, run_dir)
    calls: list[dict[str, Any]] = []

    result = _run(
        fixture,
        run_dir,
        arms="no_memory,agentcli_four_tier,mem0",
        stream_runner=_fake_stream_runner(calls),
    )

    assert result["status"] == "completed"
    assert len(calls) == 3
    assert len({call["protocol_hash"] for call in calls}) == 1
    assert all(call["max_steps"] == 40 for call in calls)
    assert all(call["command_timeout"] == 120 for call in calls)
    with pytest.raises(FileExistsError, match="already exists"):
        _run(fixture, run_dir, arms="no_memory")
    with pytest.raises(ValueError, match="selection does not match preflight"):
        run_preflighted_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            seed=42,
            arms="no_memory",
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="lifelong_os",
            limit=1,
            embedding_encoder_factory=_embedding_factory(),
        )
    with pytest.raises(FileExistsError, match="scored seed output"):
        _preflight(fixture, run_dir)


def test_run_rejects_stale_protocol_hash(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])
    run_dir = fixture["repo"] / "evaluationResults" / "pilot"
    _preflight(fixture, run_dir)
    (run_dir / "protocol_hash.txt").write_text(
        canonical_sha256("stale") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="protocol hash file"):
        _run(fixture, run_dir, arms="no_memory")


def test_preflight_rejects_unimportable_evaluator_entrypoint(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path, invalid_evaluator=True)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])

    with pytest.raises(ImportError, match="evaluator entrypoint"):
        _preflight(fixture, fixture["repo"] / "evaluationResults" / "pilot")


@pytest.mark.parametrize(
    ("drift", "match"),
    (
        ("actor_model", "Actor API model"),
        ("actor_endpoint", "Actor API endpoint"),
        ("embedding_endpoint", "embedding API endpoint"),
        ("embedding_dimension", "embedding API dimension"),
        ("mem0_version", "current backend config"),
    ),
)
def test_run_rejects_api_and_backend_drift(
    tmp_path: Path,
    drift: str,
    match: str,
) -> None:
    fixture = _fixture_workspace(tmp_path)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])
    run_dir = fixture["repo"] / "evaluationResults" / "pilot"
    _preflight(fixture, run_dir)
    base_config = _base_config(fixture["repo"])
    env = dict(fixture["env"])
    embedding_dimension = 3
    mem0_version = "2.0.13"
    arms = "mem0" if drift == "mem0_version" else "no_memory"

    if drift == "actor_model":
        base_config = replace(base_config, model="drifted-chat")
    elif drift == "actor_endpoint":
        base_config = replace(base_config, base_url="https://other.example.com/v1")
    elif drift == "embedding_endpoint":
        env["MY_AGENT_EMBEDDING_BASE_URL"] = "https://embedding.example.com/v1"
    elif drift == "embedding_dimension":
        embedding_dimension = 4
    else:
        mem0_version = "2.0.14"

    with pytest.raises(ValueError, match=match):
        _run(
            fixture,
            run_dir,
            arms=arms,
            base_config=base_config,
            env=env,
            embedding_dimension=embedding_dimension,
            mem0_version=mem0_version,
        )
