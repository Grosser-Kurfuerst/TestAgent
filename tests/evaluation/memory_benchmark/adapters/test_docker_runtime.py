from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import os
import shutil
import subprocess

import pytest

from my_agent.evaluation.memory_benchmark.adapters.base import execute_official_scorer
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    BENCHMARK_ACTION_SCRIPT,
    BENCHMARK_AGENT_INSTRUCTIONS,
    BENCHMARK_LABEL,
    MANAGED_LABEL,
    RUN_LABEL,
    SEED_LABEL,
    TASK_LABEL,
    BenchmarkActionState,
    DockerContainer,
    DockerRuntime,
    DockerRuntimeError,
    benchmark_action_main,
    benchmark_action_tool_config,
    benchmark_action_tools_hash,
    benchmark_container_name,
    execute_benchmark_action,
    finalize_action_log,
    prepare_runtime_action_log,
    write_benchmark_action_files,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    OfficialEvaluatorResult,
    load_official_evaluator_result,
)
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.tools.config_source import ConfigToolSource
from my_agent.tools.spec import ToolContext


HASH = canonical_sha256({"evaluator": "fixture"})


def _completed(
    returncode: int = 0,
    *,
    stdout: str = "",
    stderr: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _labels(
    *,
    run_id: str = "run-1",
    seed: int = 42,
    benchmark: str = "lifelong_os",
    task_id: str = "task-1",
) -> dict[str, str]:
    return {
        MANAGED_LABEL: "true",
        RUN_LABEL: run_id,
        SEED_LABEL: str(seed),
        BENCHMARK_LABEL: benchmark,
        TASK_LABEL: task_id,
    }


def _container(**overrides: Any) -> DockerContainer:
    values: dict[str, Any] = {
        "container_id": "container-1",
        "name": "agentcli-mb-run-1-42-lifelong-os-task-1",
        "image": "fixture:locked",
        "image_digest": HASH,
        "labels": _labels(),
    }
    values.update(overrides)
    return DockerContainer(**values)


class RecordingRunner:
    def __init__(self, handler: Callable[[list[str]], Any]) -> None:
        self.handler = handler
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append((list(argv), dict(kwargs)))
        return self.handler(list(argv))


def test_docker_preflight_requires_a_server_version() -> None:
    runner = RecordingRunner(lambda _argv: _completed(stdout="27.5.1\n"))

    assert DockerRuntime(command_runner=runner).preflight() == "27.5.1"
    assert runner.calls[0][0] == [
        "docker",
        "version",
        "--format",
        "{{.Server.Version}}",
    ]


def test_docker_runtime_uses_argv_lists_and_validates_locked_image(tmp_path: Path) -> None:
    labels = _labels()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def handler(argv: list[str]) -> Any:
        if argv[1:4] == ["image", "inspect", "--format"]:
            return _completed(stdout=f"{HASH}\n")
        if argv[1] == "create":
            return _completed(stdout="container-1\n")
        if argv[1] == "start":
            return _completed()
        if argv[1] == "inspect":
            return _completed(stdout=json.dumps(labels))
        if argv[1] == "rm":
            return _completed()
        raise AssertionError(f"unexpected Docker argv: {argv}")

    runner = RecordingRunner(handler)
    runtime = DockerRuntime(command_runner=runner)
    container = runtime.create_container(
        image="fixture:locked",
        expected_digest=HASH,
        run_id="run-1",
        seed=42,
        benchmark="lifelong_os",
        task_id="task-1",
        bind_mounts={workspace: "/workspace"},
        working_directory="/workspace",
    )
    runtime.cleanup_container(container)

    assert container.image_digest == HASH
    assert container.labels == labels
    assert all(isinstance(argv, list) for argv, _ in runner.calls)
    assert all(kwargs["shell"] is False for _, kwargs in runner.calls)
    assert all(kwargs["check"] is False for _, kwargs in runner.calls)
    create_argv = next(argv for argv, _ in runner.calls if argv[1] == "create")
    assert create_argv[-3:] == [HASH, "sleep", "infinity"]
    assert "fixture:locked" not in create_argv
    assert ["--workdir", "/workspace"] == create_argv[
        create_argv.index("--workdir") : create_argv.index("--workdir") + 2
    ]
    assert (
        f"type=bind,source={workspace.resolve()},target=/workspace" in create_argv
    )
    assert any(argv[1] == "rm" and "--force" in argv for argv, _ in runner.calls)


def test_docker_runtime_rejects_unsafe_bind_mounts(tmp_path: Path) -> None:
    runtime = DockerRuntime(command_runner=RecordingRunner(lambda _argv: _completed()))

    with pytest.raises(ValueError, match="source must be absolute"):
        runtime.create_container(
            image="fixture:locked",
            expected_digest=HASH,
            run_id="run-1",
            seed=42,
            benchmark="smoke",
            task_id="task-1",
            bind_mounts={Path("relative"): "/workspace"},
        )

    with pytest.raises(ValueError, match="absolute non-root"):
        runtime.create_container(
            image="fixture:locked",
            expected_digest=HASH,
            run_id="run-1",
            seed=42,
            benchmark="smoke",
            task_id="task-1",
            bind_mounts={tmp_path: "workspace"},
        )


def test_docker_runtime_rejects_image_id_mismatch_before_create() -> None:
    other_hash = canonical_sha256({"image": "other"})
    runner = RecordingRunner(lambda _argv: _completed(stdout=f"{other_hash}\n"))
    runtime = DockerRuntime(command_runner=runner)

    with pytest.raises(DockerRuntimeError, match="image digest mismatch"):
        runtime.create_container(
            image="fixture:locked",
            expected_digest=HASH,
            run_id="run-1",
            seed=42,
            benchmark="lifelong_os",
            task_id="task-1",
        )

    assert len(runner.calls) == 1
    assert runner.calls[0][0][1:3] == ["image", "inspect"]


def test_create_start_failure_cleans_up_the_partial_container() -> None:
    labels = _labels()

    def handler(argv: list[str]) -> Any:
        if argv[1] == "image":
            return _completed(stdout=f"{HASH}\n")
        if argv[1] == "create":
            return _completed(stdout="container-1\n")
        if argv[1] == "start":
            return _completed(1, stderr="start failed")
        if argv[1] == "inspect":
            return _completed(stdout=json.dumps(labels))
        if argv[1] == "rm":
            return _completed()
        raise AssertionError(f"unexpected Docker argv: {argv}")

    runner = RecordingRunner(handler)

    with pytest.raises(DockerRuntimeError, match="start container"):
        DockerRuntime(command_runner=runner).create_container(
            image="fixture:locked",
            expected_digest=HASH,
            run_id="run-1",
            seed=42,
            benchmark="lifelong_os",
            task_id="task-1",
        )

    assert any(argv[1] == "rm" for argv, _ in runner.calls)


def test_cleanup_requires_matching_identity_labels() -> None:
    labels = _labels()
    labels[TASK_LABEL] = "another-task"
    runner = RecordingRunner(
        lambda argv: _completed(stdout=json.dumps(labels))
        if argv[1] == "inspect"
        else _completed()
    )

    with pytest.raises(DockerRuntimeError, match="mismatched labels"):
        DockerRuntime(command_runner=runner).cleanup_container(_container())

    assert not any(argv[1] == "rm" for argv, _ in runner.calls)


def test_paired_container_identity_and_public_state_do_not_contain_arm(tmp_path: Path) -> None:
    names = {
        benchmark_container_name(
            run_id="run-1",
            seed=42,
            benchmark="lifelong_os",
            task_id="task-1",
        )
        for _arm in ("no_memory", "agentcli_four_tier", "mem0")
    }
    runtime_log = tmp_path / "opaque" / "actions.jsonl"
    states = {
        canonical_json_bytes(
            BenchmarkActionState(
                container_name=next(iter(names)),
                runtime_action_log_path=runtime_log,
                timeout_seconds=120,
                max_output_chars=4000,
            ).to_dict()
        )
        for _arm in ("no_memory", "agentcli_four_tier", "mem0")
    }

    assert len(names) == 1
    assert len(states) == 1
    public_state = next(iter(states)).decode("utf-8")
    assert "no_memory" not in public_state
    assert "agentcli_four_tier" not in public_state
    assert "mem0" not in public_state
    assert "evaluator" not in public_state
    assert set(_labels()) == {
        MANAGED_LABEL,
        RUN_LABEL,
        SEED_LABEL,
        BENCHMARK_LABEL,
        TASK_LABEL,
    }


def test_action_log_is_append_only_ordered_and_output_bounded(tmp_path: Path) -> None:
    runtime_log = prepare_runtime_action_log(tmp_path / "runtime" / "actions.jsonl")
    state = BenchmarkActionState(
        container_name="fixture-container",
        runtime_action_log_path=runtime_log,
        timeout_seconds=5,
        max_output_chars=12,
    )
    replies = iter(
        (
            _completed(stdout="a" * 100, stderr="first-error"),
            _completed(1, stdout="second", stderr="b" * 100),
        )
    )
    runner = RecordingRunner(lambda _argv: next(replies))

    first = execute_benchmark_action(state, "echo first", command_runner=runner)
    second = execute_benchmark_action(state, "false", command_runner=runner)
    records = [json.loads(line) for line in runtime_log.read_text().splitlines()]

    assert first.returncode == 0
    assert second.returncode == 1
    assert [record["sequence"] for record in records] == [1, 2]
    assert [record["command"] for record in records] == ["echo first", "false"]
    assert all(len(record["stdout"]) <= 12 for record in records)
    assert all(len(record["stderr"]) <= 12 for record in records)
    assert all(call[0][:4] == ["docker", "exec", "fixture-container", "bash"] for call in runner.calls)
    assert all(call[1]["shell"] is False for call in runner.calls)


def test_action_timeout_is_logged_as_returncode_124(tmp_path: Path) -> None:
    state = BenchmarkActionState(
        container_name="fixture-container",
        runtime_action_log_path=tmp_path / "actions.jsonl",
        timeout_seconds=1,
        max_output_chars=200,
    )

    def timeout_runner(_argv: list[str], **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("docker", 1, output="partial")

    result = execute_benchmark_action(state, "sleep 10", command_runner=timeout_runner)
    record = json.loads(state.runtime_action_log_path.read_text().strip())

    assert result.returncode == 124
    assert result.timed_out is True
    assert record["timed_out"] is True


def test_runtime_log_must_be_finalized_before_path_reuse(tmp_path: Path) -> None:
    runtime = tmp_path / "opaque" / "actions.jsonl"
    final = tmp_path / "task" / "actions.jsonl"
    runtime.parent.mkdir(parents=True)
    runtime.write_text('{"sequence":1}\n', encoding="utf-8")

    with pytest.raises(DockerRuntimeError, match="previous arm"):
        prepare_runtime_action_log(runtime)

    assert finalize_action_log(runtime, final) == final.resolve()
    assert not runtime.exists()
    assert final.read_text(encoding="utf-8") == '{"sequence":1}\n'
    assert finalize_action_log(runtime, final) == final.resolve()
    assert prepare_runtime_action_log(runtime) == runtime.resolve()


def test_action_log_stays_outside_repo_copies(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = BenchmarkActionState(
        container_name="fixture-container",
        runtime_action_log_path=tmp_path / "opaque" / "actions.jsonl",
        timeout_seconds=5,
        max_output_chars=100,
    )
    write_benchmark_action_files(repo, state)
    execute_benchmark_action(
        state,
        "echo ok",
        command_runner=lambda *_args, **_kwargs: _completed(stdout="ok"),
    )
    copied = tmp_path / "copied"
    shutil.copytree(repo, copied)

    assert state.runtime_action_log_path.exists()
    assert not (repo / "actions.jsonl").exists()
    assert not (copied / "actions.jsonl").exists()


def test_action_wrapper_rejects_runtime_log_inside_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = BenchmarkActionState(
        container_name="fixture-container",
        runtime_action_log_path=repo / "actions.jsonl",
        timeout_seconds=5,
        max_output_chars=100,
    )

    with pytest.raises(ValueError, match="outside"):
        write_benchmark_action_files(repo, state)

    assert not (repo / ".agentcli").exists()
    assert not (repo / "benchmark_action.py").exists()


def test_project_tool_config_loads_only_benchmark_action(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = BenchmarkActionState(
        container_name="fixture-container",
        runtime_action_log_path=tmp_path / "opaque" / "actions.jsonl",
        timeout_seconds=120,
        max_output_chars=4000,
    )
    write_benchmark_action_files(repo, state)
    source = ConfigToolSource(repo / ".agentcli" / "tools.json", source_name="config:project")
    registrations = source.load(ToolContext(repo_root=repo))

    assert [registration.spec.name for registration in registrations] == ["benchmark_action"]
    assert benchmark_action_tool_config()["tools"][0]["command"]["argv"] == [
        "python",
        "benchmark_action.py",
        "--command",
        "{command}",
    ]
    assert (repo / "benchmark_action.py").read_text(encoding="utf-8") == BENCHMARK_ACTION_SCRIPT
    assert (repo / "AGENT.md").read_text(encoding="utf-8") == BENCHMARK_AGENT_INSTRUCTIONS


def test_tools_hash_covers_tool_config_and_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    original = benchmark_action_tools_hash()
    assert original == canonical_sha256(
        {
            "tool_config": benchmark_action_tool_config(),
            "wrapper_sha256": canonical_sha256(BENCHMARK_ACTION_SCRIPT),
        }
    )

    monkeypatch.setattr(
        "my_agent.evaluation.memory_benchmark.adapters.docker_runtime.BENCHMARK_ACTION_SCRIPT",
        BENCHMARK_ACTION_SCRIPT + "\n# changed\n",
    )
    assert benchmark_action_tools_hash() != original


def test_benchmark_action_main_executes_docker_wrapper(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = BenchmarkActionState(
        container_name="fixture-container",
        runtime_action_log_path=tmp_path / "opaque" / "actions.jsonl",
        timeout_seconds=5,
        max_output_chars=100,
    )
    write_benchmark_action_files(repo, state)
    runner = RecordingRunner(lambda _argv: _completed(stdout="ok\n"))

    returncode = benchmark_action_main(
        ["--command", "echo ok"],
        command_runner=runner,
        cwd=repo,
    )

    assert returncode == 0
    assert capsys.readouterr().out == "ok\n"
    assert runner.calls[0][0] == [
        "docker",
        "exec",
        "fixture-container",
        "bash",
        "-lc",
        "echo ok",
    ]
    assert json.loads(state.runtime_action_log_path.read_text())["command"] == "echo ok"


@pytest.mark.parametrize(("resolved", "expected_returncode"), [(True, 0), (False, 1)])
def test_official_scorer_writes_success_and_legal_failure_atomically(
    tmp_path: Path,
    resolved: bool,
    expected_returncode: int,
) -> None:
    result_path = tmp_path / "official_result.json"
    result_path.write_text("stale", encoding="utf-8")
    result = OfficialEvaluatorResult(
        task_id="task-1",
        evaluator_hash=HASH,
        resolved=resolved,
        reward=1.0 if resolved else 0.0,
    )

    assert (
        execute_official_scorer(lambda: result, official_result_path=result_path)
        == expected_returncode
    )
    assert (
        load_official_evaluator_result(
            result_path,
            expected_task_id="task-1",
            expected_evaluator_hash=HASH,
            returncode=expected_returncode,
        )
        == result
    )
    assert not list(tmp_path.glob(".*.tmp"))


def test_official_scorer_exception_removes_stale_result(tmp_path: Path) -> None:
    result_path = tmp_path / "official_result.json"
    result_path.write_text("stale", encoding="utf-8")

    def fail() -> OfficialEvaluatorResult:
        raise RuntimeError("hidden evaluator detail")

    assert execute_official_scorer(fail, official_result_path=result_path) == 2
    assert not result_path.exists()


def test_official_scorer_write_failure_returns_infrastructure_code(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    result = OfficialEvaluatorResult(
        task_id="task-1",
        evaluator_hash=HASH,
        resolved=True,
        reward=1.0,
    )

    assert (
        execute_official_scorer(
            lambda: result,
            official_result_path=parent_file / "official_result.json",
        )
        == 3
    )


def test_official_result_loader_rejects_returncode_mismatch(tmp_path: Path) -> None:
    result_path = tmp_path / "official_result.json"
    result = OfficialEvaluatorResult(
        task_id="task-1",
        evaluator_hash=HASH,
        resolved=True,
        reward=1.0,
    )
    assert execute_official_scorer(lambda: result, official_result_path=result_path) == 0

    with pytest.raises(ValueError, match="conflicts"):
        load_official_evaluator_result(
            result_path,
            expected_task_id="task-1",
            expected_evaluator_hash=HASH,
            returncode=1,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.environ.get("AGENTCLI_MEMORY_BENCHMARK_TEST_IMAGE")
        and os.environ.get("AGENTCLI_MEMORY_BENCHMARK_TEST_IMAGE_DIGEST")
    ),
    reason="set a locally prepared image and its sha256 image ID to run Docker integration",
)
def test_real_docker_tasks_do_not_share_container_state(tmp_path: Path) -> None:
    image = os.environ["AGENTCLI_MEMORY_BENCHMARK_TEST_IMAGE"]
    digest = os.environ["AGENTCLI_MEMORY_BENCHMARK_TEST_IMAGE_DIGEST"]
    runtime = DockerRuntime()
    first = runtime.create_container(
        image=image,
        expected_digest=digest,
        run_id="integration",
        seed=42,
        benchmark="runtime",
        task_id="first",
    )
    try:
        result = runtime.execute_action(
            first,
            "touch /tmp/agentcli-first",
            action_log_path=tmp_path / "first.jsonl",
            timeout_seconds=10,
            max_output_chars=1000,
        )
        assert result.ok
    finally:
        runtime.cleanup_container(first)

    second = runtime.create_container(
        image=image,
        expected_digest=digest,
        run_id="integration",
        seed=42,
        benchmark="runtime",
        task_id="second",
    )
    try:
        result = runtime.execute_action(
            second,
            "test ! -e /tmp/agentcli-first && echo ok",
            action_log_path=tmp_path / "second.jsonl",
            timeout_seconds=10,
            max_output_chars=1000,
        )
        assert result.ok
        assert result.stdout.strip() == "ok"
    finally:
        runtime.cleanup_container(second)


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.environ.get("AGENTCLI_MEMORY_BENCHMARK_TEST_IMAGE")
        and os.environ.get("AGENTCLI_MEMORY_BENCHMARK_TEST_IMAGE_DIGEST")
    ),
    reason="set a locally prepared image and its sha256 image ID to run Docker integration",
)
def test_real_docker_workspace_mount_hides_host_scorer_state(tmp_path: Path) -> None:
    image = os.environ["AGENTCLI_MEMORY_BENCHMARK_TEST_IMAGE"]
    digest = os.environ["AGENTCLI_MEMORY_BENCHMARK_TEST_IMAGE_DIGEST"]
    workspace = tmp_path / "repo" / "workspace"
    workspace.mkdir(parents=True)
    hidden_state = tmp_path / "adapter_state.json"
    hidden_state.write_text('{"expected_files":{"secret":"value"}}\n', encoding="utf-8")
    runtime = DockerRuntime()
    container = runtime.create_container(
        image=image,
        expected_digest=digest,
        run_id="integration",
        seed=42,
        benchmark="smoke",
        task_id="hidden-boundary",
        bind_mounts={workspace: "/workspace"},
        working_directory="/workspace",
    )
    try:
        attack = runtime.execute_action(
            container,
            "cat ../../adapter_state.json",
            action_log_path=tmp_path / "attack.jsonl",
            timeout_seconds=10,
            max_output_chars=1000,
        )
        write = runtime.execute_action(
            container,
            "printf ok > result.txt",
            action_log_path=tmp_path / "attack.jsonl",
            timeout_seconds=10,
            max_output_chars=1000,
        )
        assert not attack.ok
        assert "expected_files" not in attack.stdout
        assert write.ok
        assert (workspace / "result.txt").read_text(encoding="utf-8") == "ok"
        assert hidden_state.read_text(encoding="utf-8") == (
            '{"expected_files":{"secret":"value"}}\n'
        )
    finally:
        runtime.cleanup_container(container)
