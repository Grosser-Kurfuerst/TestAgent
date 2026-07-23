from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import sys

import pytest

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import (
    CommandResult,
    ManifestBenchmarkResult,
    ManifestEvalResult,
    run_manifest_benchmark,
)
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    ACTION_LOG_SCHEMA_VERSION,
    BenchmarkActionState,
    finalize_action_log,
    write_benchmark_action_files,
)
from my_agent.evaluation.memory_benchmark.backends import (
    AgentCliFourTierBackend,
    Mem0Backend,
    NoMemoryBackend,
    memory_stream_project_key,
)
from my_agent.evaluation.memory_benchmark.contracts import (
    BenchmarkTask,
    ExternalMemoryItem,
    MemoryBenchmarkTaskResult,
    Mem0SearchResult,
    Mem0WriteResult,
    OfficialEvaluatorResult,
    PreparedBenchmarkTask,
    ProviderUsage,
    PublicEpisode,
    write_official_result_atomic,
)
from my_agent.evaluation.memory_benchmark.runner import (
    MemoryBenchmarkInfrastructureError,
    run_memory_benchmark_stream,
)
from my_agent.memory.experience import (
    ExperienceCreatedBy,
    ExperienceMemory,
    ExperienceStore,
    ExperienceTier,
    TipPayload,
)
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import MemoryScope, content_fingerprint
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.runtime import run_agent
from my_agent.tools import RepoTools


HASH = canonical_sha256({"fixture": "runner"})


def _tip_memory(memory_id: str, content: str, *, project_key: str, source_task: str):
    return ExperienceMemory(
        id=memory_id,
        content=content,
        tier=ExperienceTier.TIP,
        payload=TipPayload(category="fixture", severity="info", trigger=content),
        scope=MemoryScope.PROJECT,
        project_key=project_key,
        created_at=datetime.now(timezone.utc),
        token_count=estimate_tokens(content),
        fingerprint=content_fingerprint(content),
        source_task=source_task,
        created_by=ExperienceCreatedBy.MANUAL,
        writer_confidence=1.0,
    )


def _task(index: int) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="lifelong_os",
        subset="os",
        task_id=f"task-{index}",
        order_index=index,
        task_group="lifelong_os:os",
        instruction=f"Perform task {index}.",
        split="test",
        source_revision="a" * 40,
        content_hash=canonical_sha256({"task": index}),
        environment_spec={"image": "fixture"},
        evaluator_spec={"name": "fixture", "version": "v1", "hash": HASH},
        tags=("fixture",),
    )


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        use_fake_llm=True,
        max_steps=4,
        command_timeout=20,
        trace_dir=tmp_path / "traces",
        memory_dir=tmp_path / "base-memory",
    )


def _project_key(arm: str) -> str:
    return memory_stream_project_key(
        run_id="run-1",
        seed=42,
        benchmark="lifelong_os",
        arm=arm,
    )


class FakeAdapter:
    name = "fake"

    def __init__(self, *, cleanup_failure_task: str = "") -> None:
        self.cleanup_failure_task = cleanup_failure_task
        self.prepared_ids: list[str] = []
        self.finalized_ids: list[str] = []
        self.cleaned_ids: list[str] = []

    def load_tasks(self, *, limit: int) -> list[BenchmarkTask]:
        return [_task(index) for index in range(1, limit + 1)]

    def prepare_task(
        self,
        task: BenchmarkTask,
        *,
        task_dir: Path,
        seed: int,
    ) -> PreparedBenchmarkTask:
        self.prepared_ids.append(task.task_id)
        repo = task_dir / "repo"
        repo.mkdir()
        (repo / "README.md").write_text(task.instruction, encoding="utf-8")
        runtime_log = task_dir / "runtime-actions.jsonl"
        action = {
            "schema_version": ACTION_LOG_SCHEMA_VERSION,
            "sequence": 1,
            "command": f"echo {task.task_id}",
            "returncode": 0,
            "stdout": f"{task.task_id}\n",
            "stderr": "",
            "timed_out": False,
            "elapsed_sec": 0.01,
        }
        runtime_log.write_bytes(canonical_json_bytes(action) + b"\n")
        state = BenchmarkActionState(
            container_name=f"fixture-{seed}-{task.task_id}",
            runtime_action_log_path=runtime_log,
            timeout_seconds=120,
            max_output_chars=4000,
        )
        write_benchmark_action_files(repo, state)
        adapter_state = task_dir / "adapter_state.json"
        adapter_state.write_text("{}\n", encoding="utf-8")
        return PreparedBenchmarkTask(
            task=task,
            repo_path=repo,
            public_prompt=task.instruction,
            agent_test_command=None,
            initial_environment_command=(sys.executable, "-c", "raise SystemExit(0)"),
            hidden_evaluator_command=(sys.executable, "-c", "raise SystemExit(0)"),
            env_overrides={"FIXTURE_TASK": task.task_id},
            action_log_path=task_dir / "actions.jsonl",
            runtime_action_log_path=runtime_log,
            adapter_state_path=adapter_state,
            public_tool_state_path=repo / ".agentcli" / "benchmark_state.json",
            official_result_path=task_dir / "official_result.json",
        )

    def finalize_task_artifacts(self, prepared: PreparedBenchmarkTask) -> None:
        finalize_action_log(prepared.runtime_action_log_path, prepared.action_log_path)
        if prepared.task.task_id not in self.finalized_ids:
            self.finalized_ids.append(prepared.task.task_id)

    def cleanup_task(self, prepared: PreparedBenchmarkTask) -> None:
        self.finalize_task_artifacts(prepared)
        self.cleaned_ids.append(prepared.task.task_id)
        if prepared.task.task_id == self.cleanup_failure_task:
            raise RuntimeError("cleanup failed")


class FakeManifestRunner:
    def __init__(
        self,
        *,
        formal: bool,
        outcomes: dict[str, bool] | None = None,
        infrastructure_task: str = "",
    ) -> None:
        self.formal = formal
        self.outcomes = dict(outcomes or {})
        self.infrastructure_task = infrastructure_task
        self.calls: list[dict[str, Any]] = []
        self.visible_counts_before: list[int] = []

    def __call__(self, **kwargs: Any) -> ManifestBenchmarkResult:
        manifest_path = Path(kwargs["tasks_path"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        task_payload = payload["tasks"][0]
        task_id = str(task_payload["id"])
        config = kwargs["config"]
        assert isinstance(config, AgentConfig)
        store = ExperienceStore.from_dir(config.memory_dir)
        visible_before = store.all(project_key=config.memory_project_key)
        self.visible_counts_before.append(len(visible_before))
        written_ids: list[str] = []
        writer_status = ""
        infrastructure = task_id == self.infrastructure_task
        if self.formal and not infrastructure:
            if task_id == "task-1":
                memory = _tip_memory(
                    "memory-task-1",
                    "Reusable task one memory",
                    project_key=config.memory_project_key,
                    source_task=task_id,
                )
                appended = store.append_all_atomically((memory,))
                written_ids = [item.id for item in appended.appended]
                writer_status = "committed"
            else:
                assert any(memory.source_task == "task-1" for memory in visible_before)
                writer_status = "no_write"
        resolved = self.outcomes.get(task_id, True)
        official = OfficialEvaluatorResult(
            task_id=task_id,
            evaluator_hash=HASH,
            resolved=resolved,
            reward=1.0 if resolved else 0.0,
        )
        write_official_result_atomic(task_payload["official_result_path"], official)
        result = ManifestEvalResult(
            task_id=task_id,
            status="passed" if resolved else "failed",
            resolved=resolved,
            task_valid=True,
            failure_type=(
                "evaluator_error"
                if infrastructure
                else ("" if resolved else "official_evaluator_failed")
            ),
            initial_visible=CommandResult("", True, 0, skipped=True),
            evaluation_kind="external_state",
            agent_final_answer=f"finished {task_id}",
            official_result_path=str(task_payload["official_result_path"]),
            task_group=str(task_payload["task_group"]),
            reward=official.reward,
            evaluator_name="fixture",
            evaluator_version="v1",
            evaluator_hash=HASH,
            outcome_finalized=not infrastructure,
            evolver_writer_status=writer_status,
            written_memory_ids=written_ids,
            repository_revision_after_writer=store.revision(),
            mode=str(kwargs["mode"]),
            env_overrides=dict(task_payload["env_overrides"]),
            memory_mode="shared_stream",
            stream_id=str(payload["stream_id"]),
            memory_dir=str(config.memory_dir),
            memory_project_key=config.memory_project_key,
            memory_entries_before=len(visible_before),
            memory_entries_after=len(store.all(project_key=config.memory_project_key)),
            memory_growth=len(written_ids),
            agent_steps=1,
            agent_done=True,
            agent_stop_reason="assistant_final",
            error="fixture infrastructure error" if infrastructure else "",
        )
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.jsonl"
        summary_path = output_dir / "summary.json"
        results_path.write_text(json.dumps(result.to_dict()) + "\n", encoding="utf-8")
        summary_path.write_text("{}\n", encoding="utf-8")
        self.calls.append(
            {
                "task_id": task_id,
                "payload": payload,
                "config": config,
                "mode": kwargs["mode"],
                "agent_runner": kwargs["agent_runner"],
            }
        )
        return ManifestBenchmarkResult(
            results=[result],
            summary={},
            output_dir=output_dir,
            results_path=results_path,
            summary_path=summary_path,
        )


def _run(
    tmp_path: Path,
    *,
    backend: NoMemoryBackend | AgentCliFourTierBackend | Mem0Backend,
    adapter: FakeAdapter,
    manifest_runner: Callable[..., ManifestBenchmarkResult],
    tasks: list[BenchmarkTask] | None = None,
):
    output = tmp_path / "stream"
    return run_memory_benchmark_stream(
        tasks=tasks or [_task(1), _task(2)],
        adapter=adapter,
        backend=backend,
        base_config=_config(tmp_path),
        output_dir=output,
        run_id="run-1",
        seed=42,
        stream_memory_dir=output / "memory",
        stream_project_key=backend.stream_project_key,
        protocol_hash=HASH,
        actor_identity_hash=HASH,
        tools_hash=HASH,
        backend_config_hash=HASH,
        manifest_runner=manifest_runner,
    )


def test_no_memory_stream_preserves_order_react_mode_and_zero_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    user_tools = home / ".config" / "agentcli" / "tools.json"
    user_tools.parent.mkdir(parents=True)
    user_tools.write_text(
        json.dumps(
            {
                "version": 1,
                "tools": [
                    {
                        "kind": "command",
                        "name": "user_echo",
                        "description": "Must not load.",
                        "risk": "execute",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        "command": {"argv": [sys.executable, "-c", "print('bad')"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    output = tmp_path / "stream"
    project_key = _project_key("no_memory")
    backend = NoMemoryBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=project_key,
    )
    adapter = FakeAdapter()
    manifest = FakeManifestRunner(formal=False)

    result = _run(
        tmp_path,
        backend=backend,
        adapter=adapter,
        manifest_runner=manifest,
    )

    assert [execution.task.task_id for execution in result.executions] == ["task-1", "task-2"]
    assert adapter.prepared_ids == ["task-1", "task-2"]
    assert adapter.cleaned_ids == ["task-1", "task-2"]
    assert [call["mode"] for call in manifest.calls] == ["react", "react"]
    assert {call["config"].memory_dir for call in manifest.calls} == {(output / "memory").resolve()}
    assert {call["config"].memory_project_key for call in manifest.calls} == {project_key}
    assert all(execution.memory_after.entry_count == 0 for execution in result.executions)
    assert all(execution.backend_finalize.written_ids == () for execution in result.executions)
    rows = [
        MemoryBenchmarkTaskResult.from_dict(json.loads(line))
        for line in result.results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row.task_id for row in rows] == ["task-1", "task-2"]
    assert all(row.protocol_hash == HASH for row in rows)
    assert all(row.actor_identity_hash == HASH for row in rows)
    assert all(row.tools_hash == HASH for row in rows)
    assert all(row.backend_config_hash == HASH for row in rows)
    assert all(row.memory_usage_available and row.memory_total_tokens == 0 for row in rows)
    for execution, call in zip(result.executions, manifest.calls, strict=True):
        tools = RepoTools(execution.prepared.repo_path, config=call["config"])
        assert "benchmark_action" in tools.tool_names
        assert "user_echo" not in tools.tool_names
        assert execution.public_episode_path.exists()


def test_four_tier_stream_reuses_repository_and_original_agent_runner(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    project_key = _project_key("agentcli_four_tier")
    backend = AgentCliFourTierBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=project_key,
    )
    adapter = FakeAdapter()
    manifest = FakeManifestRunner(formal=True)

    result = _run(
        tmp_path,
        backend=backend,
        adapter=adapter,
        manifest_runner=manifest,
    )

    assert manifest.visible_counts_before == [0, 1]
    assert all(call["agent_runner"] is run_agent for call in manifest.calls)
    assert result.executions[0].backend_finalize.written_ids == ("memory-task-1",)
    assert result.executions[0].memory_after.entry_count == 1
    assert result.executions[1].memory_before.revision == result.executions[0].memory_after.revision
    assert result.executions[1].memory_after.entry_count == 1
    rows = [
        MemoryBenchmarkTaskResult.from_dict(json.loads(line))
        for line in result.results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row.order_index for row in rows] == [1, 2]
    assert all(row.arm == "agentcli_four_tier" for row in rows)
    assert all(not row.memory_usage_available for row in rows)
    derived = json.loads(
        (output / "tasks" / "0002_task-2" / "derived_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    env = derived["tasks"][0]["env_overrides"]
    assert env["AGENTCLI_MEMORY_DIR"] == str((output / "memory").resolve())
    assert env["AGENTCLI_MEMORY_PROJECT_KEY"] == project_key


def test_no_memory_runner_integrates_with_real_manifest_path(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    seen_tools: list[list[str]] = []

    class RealManifestAdapter(FakeAdapter):
        def prepare_task(self, task: BenchmarkTask, *, task_dir: Path, seed: int):
            prepared = super().prepare_task(task, task_dir=task_dir, seed=seed)
            official = OfficialEvaluatorResult(
                task_id=task.task_id,
                evaluator_hash=HASH,
                resolved=True,
                reward=1.0,
            )
            official_bytes = canonical_json_bytes(official.to_dict()) + b"\n"
            scorer = (
                "from pathlib import Path; "
                f"Path({str(prepared.official_result_path)!r}).write_bytes({official_bytes!r}); "
                "raise SystemExit(0)"
            )
            return replace(
                prepared,
                hidden_evaluator_command=(sys.executable, "-c", scorer),
            )

    class FakeAgentBackend(NoMemoryBackend):
        def build_agent_runner(self, *, context):
            self._validate_context(context)

            def agent_runner(**kwargs: Any):
                assert kwargs["mode"] == "react"
                tools = RepoTools(kwargs["repo_path"], config=kwargs["config"])
                seen_tools.append(tools.tool_names)
                return SimpleNamespace(
                    trace_path=None,
                    run_id="fake-agent",
                    steps=1,
                    done=True,
                    stop_reason="assistant_final",
                    final_answer="done",
                )

            return agent_runner

    backend = FakeAgentBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=_project_key("no_memory"),
    )

    result = _run(
        tmp_path,
        backend=backend,
        adapter=RealManifestAdapter(),
        manifest_runner=run_manifest_benchmark,
        tasks=[_task(1)],
    )

    assert result.executions[0].manifest_result.resolved is True
    assert seen_tools and "benchmark_action" in seen_tools[0]


def test_normal_task_failure_continues_the_stream(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    backend = NoMemoryBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=_project_key("no_memory"),
    )
    adapter = FakeAdapter()
    manifest = FakeManifestRunner(formal=False, outcomes={"task-1": False})

    result = _run(
        tmp_path,
        backend=backend,
        adapter=adapter,
        manifest_runner=manifest,
    )

    assert [execution.manifest_result.resolved for execution in result.executions] == [False, True]
    assert adapter.cleaned_ids == ["task-1", "task-2"]
    rows = [json.loads(line) for line in result.results_path.read_text().splitlines()]
    assert [row["failure_type"] for row in rows] == ["official_evaluator_failed", ""]


def test_fake_eight_task_ab_stream_keeps_expected_repository_behavior(tmp_path: Path) -> None:
    tasks = [_task(index) for index in range(1, 9)]
    no_root = tmp_path / "no-memory"
    no_output = no_root / "stream"
    no_backend = NoMemoryBackend(
        stream_memory_dir=no_output / "memory",
        stream_project_key=_project_key("no_memory"),
    )
    no_result = _run(
        no_root,
        backend=no_backend,
        adapter=FakeAdapter(),
        manifest_runner=FakeManifestRunner(formal=False),
        tasks=tasks,
    )

    formal_root = tmp_path / "four-tier"
    formal_output = formal_root / "stream"
    formal_backend = AgentCliFourTierBackend(
        stream_memory_dir=formal_output / "memory",
        stream_project_key=_project_key("agentcli_four_tier"),
    )
    formal_result = _run(
        formal_root,
        backend=formal_backend,
        adapter=FakeAdapter(),
        manifest_runner=FakeManifestRunner(formal=True),
        tasks=tasks,
    )

    assert len(no_result.executions) == 8
    assert all(execution.memory_after.entry_count == 0 for execution in no_result.executions)
    assert len(formal_result.executions) == 8
    assert [execution.memory_after.entry_count for execution in formal_result.executions] == [1] * 8
    assert all(
        current.memory_before.revision == previous.memory_after.revision
        for previous, current in zip(
            formal_result.executions,
            formal_result.executions[1:],
            strict=False,
        )
    )


def test_infrastructure_failure_aborts_before_next_task_and_cleans_up(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    backend = NoMemoryBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=_project_key("no_memory"),
    )
    adapter = FakeAdapter()
    manifest = FakeManifestRunner(formal=False, infrastructure_task="task-1")

    with pytest.raises(MemoryBenchmarkInfrastructureError) as raised:
        _run(
            tmp_path,
            backend=backend,
            adapter=adapter,
            manifest_runner=manifest,
        )

    assert raised.value.manifest_result is not None
    assert raised.value.manifest_result.failure_type == "evaluator_error"
    assert adapter.prepared_ids == ["task-1"]
    assert adapter.cleaned_ids == ["task-1"]
    assert (output / "tasks" / "0001_task-1" / "actions.jsonl").exists()
    assert (output / "results.jsonl").read_text(encoding="utf-8") == ""


def test_evaluator_infrastructure_failure_never_calls_formal_backend_finalize(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stream"

    class CountingBackend(AgentCliFourTierBackend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.finalize_calls = 0

        def finalize_task(self, episode, result):
            self.finalize_calls += 1
            return super().finalize_task(episode, result)

    backend = CountingBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=_project_key("agentcli_four_tier"),
    )
    adapter = FakeAdapter()

    with pytest.raises(MemoryBenchmarkInfrastructureError):
        _run(
            tmp_path,
            backend=backend,
            adapter=adapter,
            manifest_runner=FakeManifestRunner(
                formal=True,
                infrastructure_task="task-1",
            ),
        )

    assert backend.finalize_calls == 0
    assert ExperienceStore.from_dir(output / "memory").all() == []


def test_cleanup_failure_aborts_stream_after_preserving_primary_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    backend = NoMemoryBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=_project_key("no_memory"),
    )
    adapter = FakeAdapter(cleanup_failure_task="task-1")
    manifest = FakeManifestRunner(formal=False)

    with pytest.raises(MemoryBenchmarkInfrastructureError) as raised:
        _run(
            tmp_path,
            backend=backend,
            adapter=adapter,
            manifest_runner=manifest,
        )

    assert [stage for stage, _ in raised.value.failures] == ["adapter_cleanup"]
    assert adapter.prepared_ids == ["task-1"]
    assert (output / "tasks" / "0001_task-1" / "public_episode.json").exists()
    assert (output / "results.jsonl").read_text(encoding="utf-8") == ""


def test_runner_rejects_out_of_order_tasks_before_preparing_any_task(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    backend = NoMemoryBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=_project_key("no_memory"),
    )
    adapter = FakeAdapter()

    with pytest.raises(ValueError, match="consecutive"):
        _run(
            tmp_path,
            backend=backend,
            adapter=adapter,
            manifest_runner=FakeManifestRunner(formal=False),
            tasks=[_task(2), _task(1)],
        )

    assert adapter.prepared_ids == []


def test_runner_rejects_adapter_task_identity_mismatch_and_still_cleans_up(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stream"

    class WrongTaskAdapter(FakeAdapter):
        def prepare_task(self, task: BenchmarkTask, *, task_dir: Path, seed: int):
            prepared = super().prepare_task(task, task_dir=task_dir, seed=seed)
            return replace(prepared, task=_task(2))

    adapter = WrongTaskAdapter()
    backend = NoMemoryBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=_project_key("no_memory"),
    )

    with pytest.raises(MemoryBenchmarkInfrastructureError, match="wrong task"):
        _run(
            tmp_path,
            backend=backend,
            adapter=adapter,
            manifest_runner=FakeManifestRunner(formal=False),
            tasks=[_task(1)],
        )

    assert adapter.prepared_ids == ["task-1"]
    assert adapter.cleaned_ids == ["task-2"]


def test_runner_refuses_to_overwrite_existing_stream_directory(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    output.mkdir()
    backend = NoMemoryBackend(
        stream_memory_dir=output / "memory",
        stream_project_key=_project_key("no_memory"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        run_memory_benchmark_stream(
            tasks=[_task(1)],
            adapter=FakeAdapter(),
            backend=backend,
            base_config=_config(tmp_path),
            output_dir=output,
            run_id="run-1",
            seed=42,
            stream_memory_dir=output / "memory",
            stream_project_key=_project_key("no_memory"),
            protocol_hash=HASH,
            actor_identity_hash=HASH,
            tools_hash=HASH,
            backend_config_hash=HASH,
            manifest_runner=FakeManifestRunner(formal=False),
        )


def test_runner_refuses_nonempty_memory_directory_outside_stream(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    wrong_memory = tmp_path / "existing-memory"
    wrong_memory.mkdir()
    (wrong_memory / "experience_memory.jsonl").write_text("stale\n", encoding="utf-8")
    backend = NoMemoryBackend(
        stream_memory_dir=wrong_memory,
        stream_project_key=_project_key("no_memory"),
    )

    with pytest.raises(ValueError, match="stream output memory directory"):
        run_memory_benchmark_stream(
            tasks=[_task(1)],
            adapter=FakeAdapter(),
            backend=backend,
            base_config=_config(tmp_path),
            output_dir=output,
            run_id="run-1",
            seed=42,
            stream_memory_dir=wrong_memory,
            stream_project_key=_project_key("no_memory"),
            protocol_hash=HASH,
            actor_identity_hash=HASH,
            tools_hash=HASH,
            backend_config_hash=HASH,
            manifest_runner=FakeManifestRunner(formal=False),
        )


class FakeMem0StreamClient:
    def __init__(
        self,
        persistence_dir: Path,
        *,
        fail_search_call: int = 0,
        fail_add_call: int = 0,
    ) -> None:
        self.persistence_dir = persistence_dir
        self.items: list[ExternalMemoryItem] = []
        self.episodes: list[PublicEpisode] = []
        self.search_calls = 0
        self.add_calls = 0
        self.fail_search_call = fail_search_call
        self.fail_add_call = fail_add_call
        self.closed = False

    def search(self, query: str, *, stream_key: str, limit: int) -> Mem0SearchResult:
        del query, stream_key
        self.search_calls += 1
        if self.search_calls == self.fail_search_call:
            raise RuntimeError("fixture Mem0 search failed")
        return Mem0SearchResult(
            items=tuple(self.items[:limit]),
            llm_usage=ProviderUsage(),
            embedding_calls=1,
            embedding_elapsed_sec=0.01,
            elapsed_sec=0.02,
        )

    def add(self, episode: PublicEpisode, *, stream_key: str) -> Mem0WriteResult:
        del stream_key
        self.add_calls += 1
        if self.add_calls == self.fail_add_call:
            raise RuntimeError("fixture Mem0 add failed")
        self.episodes.append(episode)
        item = ExternalMemoryItem(
            memory_id=f"memory-{self.add_calls}",
            text=f"Reusable public outcome from {episode.task_id}: {episode.final_response}",
        )
        self.items.append(item)
        return Mem0WriteResult(
            written_ids=(item.memory_id,),
            llm_usage=ProviderUsage(),
            embedding_calls=1,
            embedding_elapsed_sec=0.01,
            elapsed_sec=0.02,
        )

    def count(self, *, stream_key: str) -> int:
        del stream_key
        return len(self.items)

    def close(self) -> None:
        self.closed = True


def test_mem0_fake_eight_task_stream_retrieves_prior_task_and_records_unknown_usage(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stream"
    project_key = _project_key("mem0")
    client = FakeMem0StreamClient(output / "memory" / "mem0")
    backend = Mem0Backend(
        stream_memory_dir=output / "memory",
        stream_project_key=project_key,
        client=client,
    )
    adapter = FakeAdapter()
    manifest = FakeManifestRunner(formal=False, outcomes={"task-2": False})

    result = _run(
        tmp_path,
        backend=backend,
        adapter=adapter,
        manifest_runner=manifest,
        tasks=[_task(index) for index in range(1, 9)],
    )

    assert len(result.executions) == 8
    assert result.executions[0].context.candidate_count == 0
    assert result.executions[1].context.candidate_count == 1
    assert "task-1" in result.executions[1].context.injected_text
    assert client.search_calls == 8
    assert client.add_calls == 8
    assert client.episodes[1].resolved is False
    assert client.episodes[1].failure_type == "official_evaluator_failed"
    assert client.closed is True
    rows = [
        MemoryBenchmarkTaskResult.from_dict(json.loads(line))
        for line in result.results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 8
    assert all(row.arm == "mem0" for row in rows)
    assert all(row.memory_usage_available is False for row in rows)
    assert all(row.memory_total_tokens is None for row in rows)
    assert all(row.system_total_tokens is None for row in rows)
    assert rows[1].memory["candidate_count"] == 1
    assert rows[1].memory["selected_count"] == 1


@pytest.mark.parametrize(
    ("fail_search_call", "fail_add_call", "expected_rows"),
    [(2, 0, 1), (0, 1, 0)],
)
def test_mem0_search_or_add_failure_aborts_stream_without_pseudo_result(
    tmp_path: Path,
    fail_search_call: int,
    fail_add_call: int,
    expected_rows: int,
) -> None:
    output = tmp_path / "stream"
    project_key = _project_key("mem0")
    client = FakeMem0StreamClient(
        output / "memory" / "mem0",
        fail_search_call=fail_search_call,
        fail_add_call=fail_add_call,
    )
    backend = Mem0Backend(
        stream_memory_dir=output / "memory",
        stream_project_key=project_key,
        client=client,
    )
    adapter = FakeAdapter()

    with pytest.raises(MemoryBenchmarkInfrastructureError, match="Mem0"):
        _run(
            tmp_path,
            backend=backend,
            adapter=adapter,
            manifest_runner=FakeManifestRunner(formal=False),
            tasks=[_task(1), _task(2)],
        )

    rows = [
        line
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == expected_rows
    assert client.closed is True
