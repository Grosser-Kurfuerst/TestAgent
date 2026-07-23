from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
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
    benchmark_container_name,
)
from my_agent.evaluation.memory_benchmark.adapters.lifelong_os import (
    LIFELONG_OS_EVALUATOR_NAME,
    LifelongOSAdapter,
    lifelong_os_cli_main,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    load_official_evaluator_result,
)
from my_agent.policy.identity import canonical_sha256


FIXTURE = (
    Path(__file__).parents[3]
    / "data"
    / "memory_benchmark"
    / "lifelong_os_sample.jsonl"
)
SOURCE_REVISION = "d6f19b42eb358d9150379f0c68c2985c5a867520"
DATA_REVISION = "75054b60177d4dcddb93b984413ff799b0a1fdbc"
IMAGE_DIGEST = canonical_sha256({"image": "lifelong-os"})


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


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _source(path: Path) -> dict[str, str]:
    return {
        "revision": SOURCE_REVISION,
        "task_data_revision": DATA_REVISION,
        "task_data_sha256": _sha256_file(path),
        "container_image": "local-os/default:locked",
        "container_digest": IMAGE_DIGEST,
        "evaluator_entrypoint": "src.tasks.instance.os_interaction.task.OSInteraction._complete",
    }


def _adapter(
    tmp_path: Path,
    *,
    data_path: Path = FIXTURE,
    runtime: _FakeDockerRuntime | None = None,
) -> LifelongOSAdapter:
    return LifelongOSAdapter(
        task_data_path=data_path,
        source=_source(data_path),
        run_id="run-1",
        runtime_root=tmp_path / "runtime",
        docker_runtime=runtime or _FakeDockerRuntime(),
    )


def _row(index: int) -> dict[str, Any]:
    return {
        "sample_index": index,
        "instruction": f"Create /tmp/result-{index}.",
        "initialization_command_item": {
            "command_name": "bash",
            "script": f"rm -f /tmp/result-{index}",
        },
        "evaluation_info": {
            "evaluation_command_item": {
                "command_name": "bash",
                "script": f"test -f /tmp/result-{index}",
            },
            "ground_truth_command_item": {
                "command_name": "bash",
                "script": f"touch /tmp/result-{index}",
            },
        },
        "skill_list": ["touch"],
        "raw_entry_hash": index,
    }


def test_lifelong_os_fixture_conversion_preserves_source_order_and_hides_answers(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)

    first = tuple(adapter.load_tasks(limit=2))
    second = tuple(adapter.load_tasks(limit=2))

    assert first == second
    assert [task.task_id for task in first] == ["7", "3"]
    assert [task.order_index for task in first] == [1, 2]
    assert all(task.source_revision == SOURCE_REVISION for task in first)
    assert all(task.evaluator_spec["name"] == LIFELONG_OS_EVALUATOR_NAME for task in first)
    rendered = json.dumps([task.to_dict() for task in first])
    assert "GROUND_TRUTH_SECRET" not in rendered
    assert "SECOND_GROUND_TRUTH_SECRET" not in rendered
    assert "evaluation_script" not in rendered
    assert len({task.content_hash for task in first}) == 2


def test_lifelong_os_limit_returns_exact_first_forty_tasks(tmp_path: Path) -> None:
    data_path = tmp_path / "forty-five.jsonl"
    data_path.write_text(
        "".join(json.dumps(_row(index), sort_keys=True) + "\n" for index in range(45)),
        encoding="utf-8",
    )

    tasks = tuple(_adapter(tmp_path, data_path=data_path).load_tasks(limit=40))

    assert len(tasks) == 40
    assert [task.task_id for task in tasks] == [str(index) for index in range(40)]
    assert [task.order_index for task in tasks] == list(range(1, 41))


def test_lifelong_os_allows_official_empty_initialization_script(tmp_path: Path) -> None:
    data_path = tmp_path / "empty-init.jsonl"
    row = _row(0)
    row["initialization_command_item"]["script"] = ""
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    adapter = _adapter(tmp_path, data_path=data_path)
    task = adapter.load_tasks(limit=1)[0]
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    prepared = adapter.prepare_task(task, task_dir=task_dir, seed=42)
    commands: list[str] = []

    def command_runner(
        argv: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        commands.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    assert (
        lifelong_os_cli_main(
            ["initialize", "--state", str(prepared.adapter_state_path)],
            command_runner=command_runner,
        )
        == 0
    )
    assert commands == [":"]
    adapter.cleanup_task(prepared)


def test_lifelong_os_prepare_initializes_scores_and_cleans_container(
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
    calls: list[str] = []
    shell_flags: list[str] = []

    assert "GROUND_TRUTH_SECRET" not in private_state
    assert "initialization_script" not in public_state
    assert "evaluation_script" not in public_state
    assert runtime.created[0]["image"] == "local-os/default:locked"
    assert runtime.created[0]["expected_digest"] == IMAGE_DIGEST

    def command_runner(
        argv: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        shell_flags.append(argv[-2])
        calls.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    assert (
        lifelong_os_cli_main(
            ["initialize", "--state", str(prepared.adapter_state_path)],
            command_runner=command_runner,
        )
        == 0
    )
    assert (
        lifelong_os_cli_main(
            [
                "score",
                "--state",
                str(prepared.adapter_state_path),
                "--result",
                str(prepared.official_result_path),
            ],
            command_runner=command_runner,
        )
        == 0
    )
    official = load_official_evaluator_result(
        prepared.official_result_path,
        expected_task_id=task.task_id,
        expected_evaluator_hash=str(task.evaluator_spec["hash"]),
        returncode=0,
    )

    assert official.resolved is True
    assert calls == [
        "rm -f /tmp/public-result.txt",
        'test "$(cat /tmp/public-result.txt 2>/dev/null)" = ready',
    ]
    assert shell_flags == ["-c", "-c"]
    adapter.cleanup_task(prepared)
    assert len(runtime.cleaned) == 1


def test_lifelong_os_legal_failure_and_scorer_error_are_distinct(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    task = adapter.load_tasks(limit=1)[0]
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    prepared = adapter.prepare_task(task, task_dir=task_dir, seed=42)

    def task_failure(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    assert (
        lifelong_os_cli_main(
            [
                "score",
                "--state",
                str(prepared.adapter_state_path),
                "--result",
                str(prepared.official_result_path),
            ],
            command_runner=task_failure,
        )
        == 1
    )
    official = load_official_evaluator_result(
        prepared.official_result_path,
        expected_task_id=task.task_id,
        expected_evaluator_hash=str(task.evaluator_spec["hash"]),
        returncode=1,
    )
    assert official.resolved is False

    def scorer_error(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 125, stdout="", stderr="docker failed")

    assert (
        lifelong_os_cli_main(
            [
                "score",
                "--state",
                str(prepared.adapter_state_path),
                "--result",
                str(prepared.official_result_path),
            ],
            command_runner=scorer_error,
        )
        == 2
    )
    assert not prepared.official_result_path.exists()
    adapter.cleanup_task(prepared)


def test_lifelong_os_unknown_evaluator_state_fails_closed(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    result = tmp_path / "official_result.json"
    state.write_text('{"schema_version":"unknown"}\n', encoding="utf-8")
    result.write_text("stale\n", encoding="utf-8")

    assert (
        lifelong_os_cli_main(
            ["score", "--state", str(state), "--result", str(result)],
        )
        == 2
    )
    assert not result.exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("AGENTCLI_LIFELONG_OS_TEST_DATA"),
    reason="set AGENTCLI_LIFELONG_OS_TEST_DATA to the locked official parquet",
)
def test_real_lifelong_os_environment_starts_and_official_evaluator_fails_before_solution(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).parents[4]
    source = json.loads(
        (repo_root / "configs" / "memory_benchmark" / "source-lock.json").read_text(
            encoding="utf-8"
        )
    )["sources"]["lifelong_agent_bench"]
    adapter = LifelongOSAdapter(
        task_data_path=os.environ["AGENTCLI_LIFELONG_OS_TEST_DATA"],
        source=source,
        run_id="lifelong-real-integration",
        runtime_root=tmp_path / "runtime",
    )
    task = adapter.load_tasks(limit=1)[0]
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    prepared = adapter.prepare_task(task, task_dir=task_dir, seed=42)
    try:
        assert (
            lifelong_os_cli_main(
                ["initialize", "--state", str(prepared.adapter_state_path)]
            )
            == 0
        )
        assert (
            lifelong_os_cli_main(
                [
                    "score",
                    "--state",
                    str(prepared.adapter_state_path),
                    "--result",
                    str(prepared.official_result_path),
                ]
            )
            == 1
        )
        official = load_official_evaluator_result(
            prepared.official_result_path,
            expected_task_id=task.task_id,
            expected_evaluator_hash=str(task.evaluator_spec["hash"]),
            returncode=1,
        )
        assert official.resolved is False
    finally:
        adapter.cleanup_task(prepared)
