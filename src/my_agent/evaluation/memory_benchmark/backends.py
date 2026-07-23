"""Memory backends used by the streaming benchmark runner."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import AgentRunnerFn, ManifestEvalResult
from my_agent.evaluation.memory_benchmark.contracts import (
    BackendFinalizeResult,
    BenchmarkTask,
    MemoryContextSelection,
    MemoryRepositorySnapshot,
    ProviderUsage,
    PublicEpisode,
)
from my_agent.memory.experience import ExperienceStore, ExperienceTier
from my_agent.runtime import run_agent


class MemoryBenchmarkBackend(Protocol):
    name: str

    def prepare_context(self, task: BenchmarkTask) -> MemoryContextSelection: ...

    def configure_task(
        self,
        base_config: AgentConfig,
        *,
        stream_memory_dir: Path,
        stream_project_key: str,
        context: MemoryContextSelection,
    ) -> AgentConfig: ...

    def build_agent_runner(
        self,
        *,
        context: MemoryContextSelection,
    ) -> AgentRunnerFn: ...

    def finalize_task(
        self,
        episode: PublicEpisode,
        result: ManifestEvalResult,
    ) -> BackendFinalizeResult: ...

    def snapshot(self) -> MemoryRepositorySnapshot: ...

    def close(self) -> None: ...


class _LocalExperienceBackend:
    name = ""

    def __init__(self, *, stream_memory_dir: str | Path, stream_project_key: str) -> None:
        self.stream_memory_dir = Path(stream_memory_dir).expanduser().resolve()
        self.stream_project_key = _required_stream_project_key(stream_project_key)

    def _validate_stream(self, memory_dir: Path, project_key: str) -> None:
        if Path(memory_dir).expanduser().resolve() != self.stream_memory_dir:
            raise ValueError("backend stream_memory_dir does not match the configured stream")
        if project_key != self.stream_project_key:
            raise ValueError("backend stream_project_key does not match the configured stream")

    def _validate_context(self, context: MemoryContextSelection) -> None:
        if context.backend != self.name:
            raise ValueError(
                f"memory context backend mismatch: expected {self.name!r}, got {context.backend!r}"
            )

    def build_agent_runner(
        self,
        *,
        context: MemoryContextSelection,
    ) -> AgentRunnerFn:
        self._validate_context(context)
        return run_agent

    def snapshot(self) -> MemoryRepositorySnapshot:
        store = ExperienceStore.from_dir(self.stream_memory_dir)
        snapshot = store.load_strict_snapshot()
        tier_counts = Counter(memory.tier.value for memory in snapshot.memories)
        return MemoryRepositorySnapshot(
            revision=snapshot.revision,
            entry_count=len(snapshot.memories),
            repository_bytes=_directory_size(self.stream_memory_dir),
            tier_counts={tier.value: tier_counts[tier.value] for tier in ExperienceTier},
            repository_path=str(self.stream_memory_dir),
        )

    def close(self) -> None:
        return None


class NoMemoryBackend(_LocalExperienceBackend):
    """Keep normal per-task memory services but disable persistent evolver state."""

    name = "no_memory"

    def prepare_context(self, task: BenchmarkTask) -> MemoryContextSelection:
        if not isinstance(task, BenchmarkTask):
            raise ValueError("task must be a BenchmarkTask")
        return MemoryContextSelection(
            backend=self.name,
            candidate_count=0,
            selected_ids=(),
            selected_texts=(),
            selected_content_tokens=0,
            injected_text="",
            estimated_tokens=0,
            retrieval_elapsed_sec=0.0,
        )

    def configure_task(
        self,
        base_config: AgentConfig,
        *,
        stream_memory_dir: Path,
        stream_project_key: str,
        context: MemoryContextSelection,
    ) -> AgentConfig:
        self._validate_stream(stream_memory_dir, stream_project_key)
        self._validate_context(context)
        return _common_benchmark_config(
            base_config,
            memory_dir=self.stream_memory_dir,
            memory_project_key=self.stream_project_key,
            memory_evolver_mode="off",
        )

    def finalize_task(
        self,
        episode: PublicEpisode,
        result: ManifestEvalResult,
    ) -> BackendFinalizeResult:
        if not isinstance(episode, PublicEpisode) or not isinstance(result, ManifestEvalResult):
            raise ValueError("No Memory finalize requires public episode and manifest result")
        if result.written_memory_ids or result.evolver_writer_status:
            raise RuntimeError("No Memory task unexpectedly reported persistent memory writes")
        return BackendFinalizeResult(
            status="not_applicable",
            llm_usage=ProviderUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


class AgentCliFourTierBackend(_LocalExperienceBackend):
    """Delegate native four-tier selection and finalization to formal runtime."""

    name = "agentcli_four_tier"

    def prepare_context(self, task: BenchmarkTask) -> MemoryContextSelection:
        if not isinstance(task, BenchmarkTask):
            raise ValueError("task must be a BenchmarkTask")
        # Formal runtime owns the one and only selector call for this task.
        return MemoryContextSelection(
            backend=self.name,
            candidate_count=0,
            selected_ids=(),
            selected_texts=(),
            selected_content_tokens=0,
            injected_text="",
            estimated_tokens=0,
            retrieval_elapsed_sec=0.0,
        )

    def configure_task(
        self,
        base_config: AgentConfig,
        *,
        stream_memory_dir: Path,
        stream_project_key: str,
        context: MemoryContextSelection,
    ) -> AgentConfig:
        self._validate_stream(stream_memory_dir, stream_project_key)
        self._validate_context(context)
        common = _common_benchmark_config(
            base_config,
            memory_dir=self.stream_memory_dir,
            memory_project_key=self.stream_project_key,
            memory_evolver_mode="formal",
        )
        return replace(
            common,
            memory_evolver_candidate_top_k_per_tier=50,
            memory_evolver_selected_max_items=20,
            memory_evolver_selection_prompt_tokens=1_800,
            memory_evolver_maintenance_interval_tasks=30,
            memory_evolver_maintenance_enabled=True,
        )

    def build_agent_runner(
        self,
        *,
        context: MemoryContextSelection,
    ) -> AgentRunnerFn:
        self._validate_context(context)
        # Identity is significant: manifest runner uses it to share formal resources.
        return run_agent

    def finalize_task(
        self,
        episode: PublicEpisode,
        result: ManifestEvalResult,
    ) -> BackendFinalizeResult:
        if not isinstance(episode, PublicEpisode) or not isinstance(result, ManifestEvalResult):
            raise ValueError("four-tier finalize requires public episode and manifest result")
        if not result.outcome_finalized:
            raise RuntimeError("four-tier backend cannot finalize an unfinalized outcome")
        if result.failure_type == "evolver_finalize_failed":
            raise RuntimeError("formal memory finalize failed inside manifest runner")
        if not result.evolver_writer_status:
            raise RuntimeError("formal manifest result is missing evolver_writer_status")
        return BackendFinalizeResult(
            status=result.evolver_writer_status,
            written_ids=tuple(result.written_memory_ids),
        )


def memory_stream_project_key(
    *,
    run_id: str,
    seed: int,
    benchmark: str,
    arm: str,
) -> str:
    parts = {
        "run_id": str(run_id).strip(),
        "benchmark": str(benchmark).strip(),
        "arm": str(arm).strip(),
    }
    if any(not value for value in parts.values()):
        raise ValueError("run_id, benchmark, and arm must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return f"memory-benchmark:{parts['run_id']}:{seed}:{parts['benchmark']}:{parts['arm']}"


def _common_benchmark_config(
    base_config: AgentConfig,
    *,
    memory_dir: Path,
    memory_project_key: str,
    memory_evolver_mode: str,
) -> AgentConfig:
    if not isinstance(base_config, AgentConfig):
        raise ValueError("base_config must be an AgentConfig")
    return replace(
        base_config,
        agent_mode="react",
        memory_enabled=True,
        memory_dir=memory_dir,
        memory_project_key=memory_project_key,
        memory_evolver_mode=memory_evolver_mode,
        memory_evolver_writer_enabled=False,
        enable_project_tools=True,
        tool_config_paths=(),
        enable_project_plugins=False,
        mcp_enabled=False,
        mcp_enable_project_servers=False,
        hitl_enabled=False,
        hitl_non_interactive="reject",
    )


def _required_stream_project_key(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("stream_project_key must be non-empty")
    return normalized


def _directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


__all__ = [
    "AgentCliFourTierBackend",
    "MemoryBenchmarkBackend",
    "NoMemoryBackend",
    "memory_stream_project_key",
]
