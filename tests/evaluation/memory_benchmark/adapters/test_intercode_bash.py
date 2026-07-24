from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import subprocess

import pytest

from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    BENCHMARK_LABEL,
    MANAGED_LABEL,
    RUN_LABEL,
    SEED_LABEL,
    TASK_LABEL,
    DockerContainer,
    benchmark_action_main,
    benchmark_container_name,
)
from my_agent.evaluation.memory_benchmark.adapters.intercode_bash import (
    INTERCODE_BASH_EVALUATOR_NAME,
    InterCodeBashAdapter,
    intercode_bash_cli_main,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    load_official_evaluator_result,
)
from my_agent.policy.identity import canonical_sha256


FIXTURE = (
    Path(__file__).parents[3]
    / "data"
    / "memory_benchmark"
    / "intercode_bash_sample.jsonl"
)
SOURCE_REVISION = "c3e46d827cfc9d4c704ec078f7abf9f41e3191d8"
DATA_REVISION = SOURCE_REVISION
IMAGE_DIGEST = canonical_sha256({"image": "intercode-bash"})


class _FakeDockerRuntime:
    def __init__(self, *, fail_create_number: int | None = None) -> None:
        self.created: list[dict[str, Any]] = []
        self.cleaned: list[DockerContainer] = []
        self.fail_create_number = fail_create_number

    def create_container(self, **kwargs: Any) -> DockerContainer:
        self.created.append(dict(kwargs))
        if len(self.created) == self.fail_create_number:
            raise RuntimeError("container create failed")
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


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _source(path: Path) -> dict[str, str]:
    return {
        "revision": SOURCE_REVISION,
        "task_data_revision": DATA_REVISION,
        "task_data_sha256": _sha256_file(path),
        "container_image": "intercode-nl2bash:locked",
        "container_digest": IMAGE_DIGEST,
        "evaluator_entrypoint": "intercode.envs.BashEnv.get_reward",
    }


def _adapter(
    tmp_path: Path,
    *,
    data_path: Path = FIXTURE,
    runtime: _FakeDockerRuntime | None = None,
) -> InterCodeBashAdapter:
    return InterCodeBashAdapter(
        task_data_path=data_path,
        source=_source(data_path),
        run_id="run-1",
        runtime_root=tmp_path / "runtime",
        docker_runtime=runtime or _FakeDockerRuntime(),
    )


def _row(index: int) -> dict[str, str]:
    return {
        "query": f"Print checksum group {index}.",
        "gold": f"printf hidden-gold-{index}",
    }


def test_intercode_fixture_conversion_preserves_order_hash_and_hidden_gold(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)

    first = tuple(adapter.load_tasks(limit=2))
    second = tuple(adapter.load_tasks(limit=2))

    assert first == second
    assert [task.task_id for task in first] == ["0", "1"]
    assert [task.order_index for task in first] == [1, 2]
    assert all(task.source_revision == SOURCE_REVISION for task in first)
    assert all(task.evaluator_spec["name"] == INTERCODE_BASH_EVALUATOR_NAME for task in first)
    rendered = json.dumps([task.to_dict() for task in first])
    assert "INTERCODE_GOLD_SECRET_ONE" not in rendered
    assert "INTERCODE_GOLD_SECRET_TWO" not in rendered
    assert len({task.content_hash for task in first}) == 2


def test_intercode_limit_returns_exact_first_forty_tasks(tmp_path: Path) -> None:
    data_path = tmp_path / "sixty.json"
    data_path.write_text(
        json.dumps([_row(index) for index in range(60)]),
        encoding="utf-8",
    )

    tasks = tuple(_adapter(tmp_path, data_path=data_path).load_tasks(limit=40))

    assert len(tasks) == 40
    assert [task.task_id for task in tasks] == [str(index) for index in range(40)]
    assert [task.order_index for task in tasks] == list(range(1, 41))


def test_intercode_rejects_task_data_hash_mismatch(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter.task_data_sha256 = canonical_sha256("different")

    with pytest.raises(ValueError, match="hash mismatch"):
        adapter.load_tasks(limit=1)


def test_intercode_prepare_keeps_gold_hidden_and_creates_two_containers(
    tmp_path: Path,
) -> None:
    runtime = _FakeDockerRuntime()
    adapter = _adapter(tmp_path, runtime=runtime)
    task = adapter.load_tasks(limit=1)[0]
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    prepared = adapter.prepare_task(task, task_dir=task_dir, seed=42)

    public_state = prepared.public_tool_state_path.read_text(encoding="utf-8")
    private_state = prepared.adapter_state_path.read_text(encoding="utf-8")
    assert "INTERCODE_GOLD_SECRET_ONE" not in public_state
    assert "INTERCODE_GOLD_SECRET_ONE" in private_state
    assert "evaluation_container_name" not in public_state
    assert len(runtime.created) == 2
    assert runtime.created[0]["task_id"] == "0"
    assert runtime.created[1]["task_id"] == "0-gold"

    adapter.cleanup_task(prepared)
    assert len(runtime.cleaned) == 2


def test_intercode_prepare_failure_cleans_partial_container(tmp_path: Path) -> None:
    runtime = _FakeDockerRuntime(fail_create_number=2)
    adapter = _adapter(tmp_path, runtime=runtime)
    task = adapter.load_tasks(limit=1)[0]
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    with pytest.raises(RuntimeError, match="container create failed"):
        adapter.prepare_task(task, task_dir=task_dir, seed=42)

    assert [container.container_id for container in runtime.cleaned] == ["container-1"]


def test_intercode_uses_last_action_output_and_official_reward_once(
    tmp_path: Path,
) -> None:
    runtime = _FakeDockerRuntime()
    adapter = _adapter(tmp_path, runtime=runtime)
    task = adapter.load_tasks(limit=1)[0]
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    prepared = adapter.prepare_task(task, task_dir=task_dir, seed=42)
    state = json.loads(prepared.adapter_state_path.read_text(encoding="utf-8"))
    agent = state["agent_container_name"]
    gold = state["gold_command"]
    calls: list[tuple[str, str]] = []

    def command_runner(
        argv: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        container, command = argv[2], argv[-1]
        calls.append((container, command))
        if container == agent and command == "first":
            return subprocess.CompletedProcess(argv, 0, stdout="WRONG\n", stderr="")
        if container == agent and command == "second":
            return subprocess.CompletedProcess(argv, 0, stdout="FINAL\n", stderr="")
        if command == gold:
            return subprocess.CompletedProcess(argv, 0, stdout="FINAL\n", stderr="")
        if command == "git status --short;":
            return subprocess.CompletedProcess(
                argv, 0, stdout="?? result.txt\n", stderr=""
            )
        if command == "md5sum result.txt":
            return subprocess.CompletedProcess(argv, 0, stdout="same-md5\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    assert benchmark_action_main(
        ["--command", "first"], cwd=prepared.repo_path, command_runner=command_runner
    ) == 0
    assert benchmark_action_main(
        ["--command", "second"], cwd=prepared.repo_path, command_runner=command_runner
    ) == 0
    assert not any(command == gold for _container, command in calls)
    assert intercode_bash_cli_main(
        ["initialize", "--state", str(prepared.adapter_state_path)],
        command_runner=command_runner,
    ) == 0
    assert not any(command == gold for _container, command in calls)

    assert intercode_bash_cli_main(
        [
            "score",
            "--state",
            str(prepared.adapter_state_path),
            "--result",
            str(prepared.official_result_path),
        ],
        command_runner=command_runner,
    ) == 0
    official = load_official_evaluator_result(
        prepared.official_result_path,
        expected_task_id=task.task_id,
        expected_evaluator_hash=str(task.evaluator_spec["hash"]),
        returncode=0,
    )

    assert official.resolved is True
    assert math.isclose(official.reward, 1.0)
    assert sum(command == gold for _container, command in calls) == 1
    adapter.cleanup_task(prepared)
    action_log = prepared.action_log_path.read_text(encoding="utf-8")
    assert "INTERCODE_GOLD_SECRET_ONE" not in action_log
    assert len(runtime.cleaned) == 2


def test_intercode_legal_failure_and_scorer_error_are_distinct(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    task = adapter.load_tasks(limit=1)[0]
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    prepared = adapter.prepare_task(task, task_dir=task_dir, seed=42)
    state = json.loads(prepared.adapter_state_path.read_text(encoding="utf-8"))
    agent = state["agent_container_name"]
    evaluation = state["evaluation_container_name"]
    gold = state["gold_command"]

    def task_failure(
        argv: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        container, command = argv[2], argv[-1]
        if container == agent and command == "wrong":
            return subprocess.CompletedProcess(argv, 0, stdout="wrongonly", stderr="")
        if container == evaluation and command == gold:
            return subprocess.CompletedProcess(argv, 0, stdout="correctonly", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    assert benchmark_action_main(
        ["--command", "wrong"], cwd=prepared.repo_path, command_runner=task_failure
    ) == 0
    assert intercode_bash_cli_main(
        [
            "score",
            "--state",
            str(prepared.adapter_state_path),
            "--result",
            str(prepared.official_result_path),
        ],
        command_runner=task_failure,
    ) == 1
    official = load_official_evaluator_result(
        prepared.official_result_path,
        expected_task_id=task.task_id,
        expected_evaluator_hash=str(task.evaluator_spec["hash"]),
        returncode=1,
    )
    assert official.resolved is False
    assert math.isclose(official.reward, 0.67)

    def scorer_error(
        argv: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        container, command = argv[2], argv[-1]
        if container == evaluation and command == "git reset --hard; git clean -fd;":
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="reset failed")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    assert intercode_bash_cli_main(
        [
            "score",
            "--state",
            str(prepared.adapter_state_path),
            "--result",
            str(prepared.official_result_path),
        ],
        command_runner=scorer_error,
    ) == 2
    assert not prepared.official_result_path.exists()
    adapter.cleanup_task(prepared)


def test_intercode_tasks_use_fresh_agent_and_evaluation_containers(
    tmp_path: Path,
) -> None:
    runtime = _FakeDockerRuntime()
    adapter = _adapter(tmp_path, runtime=runtime)
    first, second = adapter.load_tasks(limit=2)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first_prepared = adapter.prepare_task(first, task_dir=first_dir, seed=42)
    second_prepared = adapter.prepare_task(second, task_dir=second_dir, seed=42)

    names = {
        json.loads(first_prepared.adapter_state_path.read_text())[key]
        for key in ("agent_container_name", "evaluation_container_name")
    } | {
        json.loads(second_prepared.adapter_state_path.read_text())[key]
        for key in ("agent_container_name", "evaluation_container_name")
    }
    assert len(names) == 4
    adapter.cleanup_task(first_prepared)
    adapter.cleanup_task(second_prepared)
    assert len(runtime.cleaned) == 4


def test_intercode_unknown_evaluator_state_fails_closed(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    result = tmp_path / "official_result.json"
    state.write_text('{"schema_version":"unknown"}\n', encoding="utf-8")
    result.write_text("stale\n", encoding="utf-8")

    assert intercode_bash_cli_main(
        ["score", "--state", str(state), "--result", str(result)]
    ) == 2
    assert not result.exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("AGENTCLI_INTERCODE_TEST_DATA"),
    reason="set AGENTCLI_INTERCODE_TEST_DATA to the locked official JSON",
)
def test_real_intercode_unsolved_and_gold_command_environments(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[4]
    source = json.loads(
        (repo_root / "configs" / "memory_benchmark" / "source-lock.json").read_text(
            encoding="utf-8"
        )
    )["sources"]["intercode"]
    data_path = Path(os.environ["AGENTCLI_INTERCODE_TEST_DATA"])
    records = json.loads(data_path.read_text(encoding="utf-8"))
    adapter = InterCodeBashAdapter(
        task_data_path=data_path,
        source=source,
        run_id="intercode-real-integration",
        runtime_root=tmp_path / "runtime",
    )
    task = adapter.load_tasks(limit=1)[0]

    unsolved_dir = tmp_path / "unsolved"
    unsolved_dir.mkdir()
    unsolved = adapter.prepare_task(task, task_dir=unsolved_dir, seed=42)
    try:
        assert intercode_bash_cli_main(
            ["initialize", "--state", str(unsolved.adapter_state_path)]
        ) == 0
        assert intercode_bash_cli_main(
            [
                "score",
                "--state",
                str(unsolved.adapter_state_path),
                "--result",
                str(unsolved.official_result_path),
            ]
        ) == 1
    finally:
        adapter.cleanup_task(unsolved)

    solved_dir = tmp_path / "solved"
    solved_dir.mkdir()
    solved = adapter.prepare_task(task, task_dir=solved_dir, seed=43)
    try:
        assert intercode_bash_cli_main(
            ["initialize", "--state", str(solved.adapter_state_path)]
        ) == 0
        assert benchmark_action_main(
            ["--command", str(records[0]["gold"])], cwd=solved.repo_path
        ) == 0
        assert intercode_bash_cli_main(
            [
                "score",
                "--state",
                str(solved.adapter_state_path),
                "--result",
                str(solved.official_result_path),
            ]
        ) == 0
        official = load_official_evaluator_result(
            solved.official_result_path,
            expected_task_id=task.task_id,
            expected_evaluator_hash=str(task.evaluator_spec["hash"]),
            returncode=0,
        )
        assert official.resolved is True
        assert math.isclose(official.reward, 1.0)
    finally:
        adapter.cleanup_task(solved)
