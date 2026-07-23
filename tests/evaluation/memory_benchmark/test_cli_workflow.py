from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import hashlib
import json
import subprocess

import pytest

from my_agent.cli.commands.memory_benchmark import (
    preflight_memory_benchmark,
    prepare_memory_benchmark_data,
    run_preflighted_memory_benchmark,
)
from my_agent.config import AgentConfig
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    canonical_sha256,
    policy_identity_manifest_payload,
)


class _FakeDockerRuntime:
    def __init__(self, digests: dict[str, str]) -> None:
        self.digests = digests

    def preflight(self) -> str:
        return "fixture-docker"

    def inspect_image(self, image: str) -> str:
        return self.digests[image]


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
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        max_steps=4,
        command_timeout=20,
        trace_dir=root / "traces",
        use_fake_llm=True,
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
        (Path(__file__).parents[3] / "configs" / "memory_benchmark" / "v1.json").read_text(
            encoding="utf-8"
        )
    )
    tracked_config["benchmarks"]["lifelong_os"]["limit"] = 2
    tracked_config["benchmarks"]["intercode_bash"]["limit"] = 2
    tracked_config["seeds"] = [42]
    config_path = configs / "v1.json"
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

    mem0_config = tmp_path / "mem0.json"
    mem0_config.write_text(
        json.dumps(
            {
                "llm": {
                    "provider": "openai",
                    "config": {"model": "fixture-llm", "api_key": "secret"},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": "fixture-embed", "api_key": "secret"},
                },
                "vector_store": {"provider": "qdrant", "config": {}},
            }
        ),
        encoding="utf-8",
    )
    identity = PolicyIdentity(
        base_model="fixture/model",
        base_revision="revision-1",
        checkpoint_hash=canonical_sha256("checkpoint"),
        adapter_hash=canonical_sha256("adapter"),
        tokenizer_revision="tokenizer-1",
        tokenizer_hash=canonical_sha256("tokenizer"),
        chat_template_hash=canonical_sha256("template"),
    )
    identity_path = repo / "policy_identity_manifest.json"
    identity_path.write_bytes(
        canonical_json_bytes(policy_identity_manifest_payload(identity)) + b"\n"
    )
    checkpoint = repo / "checkpoint"
    checkpoint.mkdir()
    _git_init(repo)
    env = {
        "AGENTCLI_LIFELONG_AGENT_BENCH_ROOT": str(lifelong),
        "AGENTCLI_INTERCODE_ROOT": str(intercode),
        "AGENTCLI_MEMORY_BENCHMARK_DATA_ROOT": str(repo / "data" / "memory_benchmark"),
        "AGENTCLI_MEM0_CONFIG_PATH": str(mem0_config),
        "AGENTCLI_BENCHMARK_EMBEDDING_REVISION": "embedding-revision-1",
    }
    return {
        "repo": repo,
        "config": config_path,
        "identity": identity_path,
        "policy_identity": identity,
        "checkpoint": checkpoint,
        "env": env,
        "docker": _FakeDockerRuntime(
            {
                "fixture/lifelong:locked": lifelong_digest,
                "fixture/intercode:locked": intercode_digest,
            }
        ),
    }


def _identity_validator(
    _config: AgentConfig,
    expected: PolicyIdentity,
) -> PolicyIdentity:
    return expected


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
    preflight = preflight_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        checkpoint=fixture["checkpoint"],
        identity_manifest=fixture["identity"],
        base_config=_base_config(fixture["repo"]),
        env=fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        docker_runtime=fixture["docker"],
        policy_identity_validator=_identity_validator,
    )
    protocol_bytes = (run_dir / "protocol.json").read_bytes()
    repeated = preflight_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        checkpoint=fixture["checkpoint"],
        identity_manifest=fixture["identity"],
        base_config=_base_config(fixture["repo"]),
        env=fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        docker_runtime=fixture["docker"],
        policy_identity_validator=_identity_validator,
    )

    assert preflight == repeated
    assert preflight["pilot"] is True
    assert protocol_bytes == (run_dir / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["pilot"] is True
    assert protocol["ordered_task_ids_by_benchmark"] == {"intercode_bash": ["0"]}
    backend = json.loads(
        (run_dir / "arms" / "mem0" / "backend_config.json").read_text()
    )
    assert "api_key" not in json.dumps(backend)


def test_run_rejects_selection_drift_and_existing_stream(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])
    run_dir = fixture["repo"] / "evaluationResults" / "pilot"
    preflight_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        checkpoint=fixture["checkpoint"],
        identity_manifest=fixture["identity"],
        base_config=_base_config(fixture["repo"]),
        env=fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        docker_runtime=fixture["docker"],
        policy_identity_validator=_identity_validator,
    )
    calls: list[dict[str, Any]] = []

    def fake_stream_runner(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
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

    result = run_preflighted_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        checkpoint=fixture["checkpoint"],
        identity_manifest=fixture["identity"],
        seed=42,
        arms="no_memory,agentcli_four_tier,mem0",
        base_config=_base_config(fixture["repo"]),
        env=fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        stream_runner=fake_stream_runner,
        policy_identity_validator=_identity_validator,
    )

    assert result["status"] == "completed"
    assert len(calls) == 3
    assert len({call["protocol_hash"] for call in calls}) == 1
    assert all(call["max_steps"] == 40 for call in calls)
    assert all(call["command_timeout"] == 120 for call in calls)
    with pytest.raises(FileExistsError, match="already exists"):
        run_preflighted_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            seed=42,
            arms="no_memory",
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="intercode_bash",
            limit=1,
            stream_runner=fake_stream_runner,
            policy_identity_validator=_identity_validator,
        )
    with pytest.raises(ValueError, match="selection does not match preflight|limit/pilot"):
        run_preflighted_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            seed=42,
            arms="no_memory",
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="lifelong_os",
            limit=1,
            stream_runner=fake_stream_runner,
            policy_identity_validator=_identity_validator,
        )
    with pytest.raises(FileExistsError, match="scored seed output"):
        preflight_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="intercode_bash",
            limit=1,
            docker_runtime=fixture["docker"],
            policy_identity_validator=_identity_validator,
        )


def test_run_rejects_stale_protocol_hash(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])
    run_dir = fixture["repo"] / "evaluationResults" / "pilot"
    preflight_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        checkpoint=fixture["checkpoint"],
        identity_manifest=fixture["identity"],
        base_config=_base_config(fixture["repo"]),
        env=fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        docker_runtime=fixture["docker"],
        policy_identity_validator=_identity_validator,
    )
    (run_dir / "protocol_hash.txt").write_text(
        canonical_sha256("stale") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="protocol hash file"):
        run_preflighted_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            seed=42,
            arms="no_memory",
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="intercode_bash",
            limit=1,
            policy_identity_validator=_identity_validator,
        )


def test_preflight_rejects_unimportable_evaluator_entrypoint(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path, invalid_evaluator=True)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])

    with pytest.raises(ImportError, match="evaluator entrypoint"):
        preflight_memory_benchmark(
            config_path=fixture["config"],
            run_dir=fixture["repo"] / "evaluationResults" / "pilot",
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="intercode_bash",
            limit=1,
            docker_runtime=fixture["docker"],
            policy_identity_validator=_identity_validator,
        )


def test_preflight_and_run_reject_policy_identity_mismatch(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])
    mismatched = PolicyIdentity(
        **{
            **fixture["policy_identity"].to_dict(),
            "adapter_hash": canonical_sha256("drifted-adapter"),
        }
    )

    def mismatch_validator(
        _config: AgentConfig,
        _expected: PolicyIdentity,
    ) -> PolicyIdentity:
        return mismatched

    run_dir = fixture["repo"] / "evaluationResults" / "pilot"
    with pytest.raises(ValueError, match="policy identity no longer matches|mismatch"):
        preflight_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="intercode_bash",
            limit=1,
            docker_runtime=fixture["docker"],
            policy_identity_validator=mismatch_validator,
        )

    preflight_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        checkpoint=fixture["checkpoint"],
        identity_manifest=fixture["identity"],
        base_config=_base_config(fixture["repo"]),
        env=fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        docker_runtime=fixture["docker"],
        policy_identity_validator=_identity_validator,
    )
    with pytest.raises(ValueError, match="policy identity mismatch"):
        run_preflighted_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            seed=42,
            arms="no_memory",
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="intercode_bash",
            limit=1,
            policy_identity_validator=mismatch_validator,
        )


def test_run_rejects_current_backend_config_drift(tmp_path: Path) -> None:
    fixture = _fixture_workspace(tmp_path)
    prepare_memory_benchmark_data(config_path=fixture["config"], env=fixture["env"])
    run_dir = fixture["repo"] / "evaluationResults" / "pilot"
    preflight_memory_benchmark(
        config_path=fixture["config"],
        run_dir=run_dir,
        checkpoint=fixture["checkpoint"],
        identity_manifest=fixture["identity"],
        base_config=_base_config(fixture["repo"]),
        env=fixture["env"],
        benchmarks="intercode_bash",
        limit=1,
        docker_runtime=fixture["docker"],
        policy_identity_validator=_identity_validator,
    )

    embedding_env = dict(fixture["env"])
    embedding_env["AGENTCLI_BENCHMARK_EMBEDDING_REVISION"] = "embedding-revision-2"
    with pytest.raises(ValueError, match="current backend config"):
        run_preflighted_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            seed=42,
            arms="agentcli_four_tier",
            base_config=_base_config(fixture["repo"]),
            env=embedding_env,
            benchmarks="intercode_bash",
            limit=1,
            policy_identity_validator=_identity_validator,
        )

    mem0_path = Path(fixture["env"]["AGENTCLI_MEM0_CONFIG_PATH"])
    mem0_config = json.loads(mem0_path.read_text(encoding="utf-8"))
    mem0_config["llm"]["config"]["model"] = "drifted-llm"
    mem0_path.write_text(json.dumps(mem0_config), encoding="utf-8")
    with pytest.raises(ValueError, match="current backend config"):
        run_preflighted_memory_benchmark(
            config_path=fixture["config"],
            run_dir=run_dir,
            checkpoint=fixture["checkpoint"],
            identity_manifest=fixture["identity"],
            seed=42,
            arms="mem0",
            base_config=_base_config(fixture["repo"]),
            env=fixture["env"],
            benchmarks="intercode_bash",
            limit=1,
            policy_identity_validator=_identity_validator,
        )
