"""Memory backends used by the streaming benchmark runner."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
import os

from my_agent.config import AgentConfig
from my_agent.evaluation.manifest_benchmark import AgentRunnerFn, ManifestEvalResult
from my_agent.evaluation.memory_benchmark.contracts import (
    BackendFinalizeResult,
    BenchmarkTask,
    MemoryContextSelection,
    MemoryRepositorySnapshot,
    Mem0SearchResult,
    ProviderUsage,
    PublicEpisode,
)
from my_agent.evaluation.memory_benchmark.api_embedding import (
    ApiEmbeddingMetrics,
    MemoryBenchmarkApiEmbeddingEncoder,
)
from my_agent.evaluation.memory_benchmark.api_policy import (
    ApiPolicyMetrics,
    ApiPolicyRoleMetrics,
    MemoryBenchmarkApiPolicy,
)
from my_agent.evaluation.memory_benchmark.external_memory import (
    ExternalContextMemoryManager,
    Mem0ClientAdapter,
)
from my_agent.llm import build_llm
from my_agent.memory.experience import ExperienceStore, ExperienceTier
from my_agent.memory.experience.retrieval.embedding import EmbeddingRetriever
from my_agent.memory.evolver.coordinator import EvolverCoordinator
from my_agent.memory.evolver.task_session import AgentEpisodeArtifact, TaskEvolverSession
from my_agent.memory.evolver.writing.contracts import ExperienceWriteStep
from my_agent.memory.manager import MemoryManager
from my_agent.memory.token import estimate_tokens
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.runtime import run_agent
from my_agent.training.contracts import AuthoritativeTaskOutcome, EvaluatorIdentity


MEM0_CONTEXT_HEADER = "Relevant selected external memory:"
BACKEND_EVENT_SCHEMA_VERSION = "memory-benchmark-backend-event-v1"


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

    def build_agent_runner(
        self,
        *,
        context: MemoryContextSelection,
    ) -> AgentRunnerFn:
        self._validate_context(context)
        return _build_task_agent_runner()

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
    """Own the API-backed four-tier lifecycle outside the manifest runtime."""

    name = "agentcli_four_tier"

    def __init__(
        self,
        *,
        stream_memory_dir: str | Path,
        stream_project_key: str,
        policy: MemoryBenchmarkApiPolicy,
        embedding_encoder: MemoryBenchmarkApiEmbeddingEncoder,
        candidate_top_k_per_tier: int = 50,
        selected_max_items: int = 20,
        selected_content_max_tokens: int = 1_800,
        generation_temperature: float = 1.0,
        generation_top_p: float = 0.95,
        maintenance_interval_tasks: int = 30,
        maintenance_enabled: bool = True,
    ) -> None:
        super().__init__(
            stream_memory_dir=stream_memory_dir,
            stream_project_key=stream_project_key,
        )
        if self.stream_memory_dir.exists() and any(self.stream_memory_dir.iterdir()):
            raise FileExistsError(
                "four-tier stream memory directory must be empty before initialization"
            )
        for field_name, value in (
            ("candidate_top_k_per_tier", candidate_top_k_per_tier),
            ("selected_max_items", selected_max_items),
            ("selected_content_max_tokens", selected_content_max_tokens),
            ("maintenance_interval_tasks", maintenance_interval_tasks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if selected_max_items > 20:
            raise ValueError("selected_max_items cannot exceed 20")
        if selected_content_max_tokens > 1_800:
            raise ValueError("selected_content_max_tokens cannot exceed 1800")
        if not isinstance(maintenance_enabled, bool):
            raise ValueError("maintenance_enabled must be a bool")
        if not isinstance(policy, MemoryBenchmarkApiPolicy):
            raise ValueError("four-tier backend requires MemoryBenchmarkApiPolicy")
        if not isinstance(embedding_encoder, MemoryBenchmarkApiEmbeddingEncoder):
            raise ValueError(
                "four-tier backend requires MemoryBenchmarkApiEmbeddingEncoder"
            )
        self.policy = policy
        self.embedding_encoder = embedding_encoder
        self.maintenance_interval_tasks = maintenance_interval_tasks
        self.backend_events_path = self.stream_memory_dir.parent / "backend_events.jsonl"
        self.store = ExperienceStore.from_dir(self.stream_memory_dir)
        self.embedding_retriever = EmbeddingRetriever(embedding_encoder)
        self.coordinator = EvolverCoordinator(
            store=self.store,
            project_key=self.stream_project_key,
            policy_identity=policy.identity(),
            retriever=self.embedding_retriever,
            policy=policy,
            dataset_dir=None,
            top_k_per_tier=candidate_top_k_per_tier,
            selected_max_items=selected_max_items,
            selection_token_budget=selected_content_max_tokens,
            generation_temperature=generation_temperature,
            generation_top_p=generation_top_p,
            maintenance_interval_tasks=maintenance_interval_tasks,
            maintenance_enabled=maintenance_enabled,
        )
        self.coordinator.require_formal_role_bindings(policy)
        self._pending_session: TaskEvolverSession | None = None
        self._pending_policy_snapshot: ApiPolicyMetrics | None = None
        self._pending_embedding_snapshot: ApiEmbeddingMetrics | None = None
        self._pending_events: list[tuple[str, Mapping[str, Any]]] = []
        self._pending_retrieval_elapsed_sec = 0.0

    def prepare_context(self, task: BenchmarkTask) -> MemoryContextSelection:
        if not isinstance(task, BenchmarkTask):
            raise ValueError("task must be a BenchmarkTask")
        if self._pending_session is not None:
            raise RuntimeError("previous four-tier task has not been finalized")
        self._pending_events = []
        self.coordinator.set_trace_sink(
            lambda event, payload: self._pending_events.append(
                (event, dict(payload))
            )
        )
        self._pending_policy_snapshot = self.policy.metrics_snapshot()
        self._pending_embedding_snapshot = self.embedding_encoder.metrics_snapshot()
        trajectory_id = canonical_sha256(
            {
                "stream_project_key": self.stream_project_key,
                "task_id": task.task_id,
                "order_index": task.order_index,
            }
        )
        started = perf_counter()
        try:
            session = self.coordinator.begin_task(
                task=task.instruction,
                task_id=task.task_id,
                task_group=task.task_group,
                trajectory_id=trajectory_id,
                stream_id=f"{task.benchmark}:{task.subset}",
            )
            memory_context = self.coordinator.context_for_session(session)
        except Exception:
            self._clear_pending()
            raise
        elapsed = max(0.0, perf_counter() - started)
        self._pending_session = session
        self._pending_retrieval_elapsed_sec = elapsed
        candidates = {item.memory_id: item for item in session.candidate_snapshot}
        selected = tuple(candidates[memory_id] for memory_id in session.selected_memory_ids)
        return MemoryContextSelection(
            backend=self.name,
            candidate_count=len(session.candidate_snapshot),
            selected_ids=session.selected_memory_ids,
            selected_texts=tuple(item.content for item in selected),
            selected_content_tokens=sum(item.token_count for item in selected),
            injected_text=memory_context.injected_text,
            estimated_tokens=memory_context.estimated_tokens,
            retrieval_elapsed_sec=elapsed,
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
            memory_evolver_mode="off",
        )
        return replace(
            common,
            policy_adapter_path=None,
            policy_identity_manifest=None,
        )

    def build_agent_runner(
        self,
        *,
        context: MemoryContextSelection,
    ) -> AgentRunnerFn:
        self._validate_context(context)
        return _build_task_agent_runner(external_context=context)

    def finalize_task(
        self,
        episode: PublicEpisode,
        result: ManifestEvalResult,
    ) -> BackendFinalizeResult:
        if not isinstance(episode, PublicEpisode) or not isinstance(result, ManifestEvalResult):
            raise ValueError("four-tier finalize requires public episode and manifest result")
        if not result.outcome_finalized:
            raise RuntimeError("four-tier backend cannot finalize an unfinalized outcome")
        session = self._pending_session
        if session is None:
            raise RuntimeError("four-tier finalize requires a pending task session")
        if episode.task_id != session.task_id or result.task_id != session.task_id:
            raise RuntimeError("four-tier finalize task does not match pending session")
        if result.task_group != session.task_group:
            raise RuntimeError("four-tier finalize task group does not match pending session")
        policy_snapshot = self._pending_policy_snapshot
        embedding_snapshot = self._pending_embedding_snapshot
        if policy_snapshot is None or embedding_snapshot is None:
            raise RuntimeError("four-tier pending metrics snapshots are missing")
        started = perf_counter()
        try:
            finalize_result = self.coordinator.finalize_task(
                AgentEpisodeArtifact(
                    session=session,
                    trace_path=Path(result.trace_path),
                    stop_reason=result.agent_stop_reason,
                    final_answer=episode.final_response,
                    tool_history=_public_episode_steps(episode),
                    task=episode.instruction,
                ),
                AuthoritativeTaskOutcome(
                    task_id=result.task_id,
                    task_group=result.task_group,
                    task_valid=result.task_valid,
                    resolved=result.resolved,
                    reward=result.reward,
                    evaluator=EvaluatorIdentity(
                        result.evaluator_name,
                        result.evaluator_version,
                        result.evaluator_hash,
                    ),
                    outcome_finalized=True,
                ),
            )
            policy_metrics = self.policy.metrics_since(policy_snapshot)
            embedding_metrics = self.embedding_encoder.metrics_since(
                embedding_snapshot
            )
            usage_by_role = _policy_usages(policy_metrics)
            combined_usage = _combine_provider_usage(*usage_by_role.values())
            maintenance_metrics = _four_tier_maintenance_metrics(
                self._pending_events,
                status=finalize_result.maintenance_status,
            )
            _append_backend_events(
                self.backend_events_path,
                task_id=session.task_id,
                events=self._pending_events,
            )
            return BackendFinalizeResult(
                status=finalize_result.writer_status,
                written_ids=finalize_result.written_memory_ids,
                llm_usage=combined_usage,
                usage_by_role=usage_by_role,
                embedding_calls=embedding_metrics.calls,
                embedding_elapsed_sec=embedding_metrics.elapsed_sec,
                elapsed_sec=(
                    self._pending_retrieval_elapsed_sec
                    + max(0.0, perf_counter() - started)
                ),
                metrics=maintenance_metrics,
            )
        finally:
            self._clear_pending()

    def close(self) -> None:
        self.coordinator.set_trace_sink(None)
        self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending_session = None
        self._pending_policy_snapshot = None
        self._pending_embedding_snapshot = None
        self._pending_events = []
        self._pending_retrieval_elapsed_sec = 0.0


class Mem0Backend:
    """Search before each task and add only the finalized public episode."""

    name = "mem0"

    def __init__(
        self,
        *,
        stream_memory_dir: str | Path,
        stream_project_key: str,
        client: Any | None = None,
        mem0_config: Mapping[str, Any] | None = None,
        search_limit: int = 50,
        selected_max_items: int = 20,
        selected_content_max_tokens: int = 1_800,
    ) -> None:
        self.stream_memory_dir = Path(stream_memory_dir).expanduser().resolve()
        self.stream_project_key = _required_stream_project_key(stream_project_key)
        if isinstance(search_limit, bool) or not isinstance(search_limit, int) or search_limit <= 0:
            raise ValueError("Mem0 search_limit must be a positive integer")
        if (
            isinstance(selected_max_items, bool)
            or not isinstance(selected_max_items, int)
            or not 1 <= selected_max_items <= 20
        ):
            raise ValueError("Mem0 selected_max_items must be between 1 and 20")
        if (
            isinstance(selected_content_max_tokens, bool)
            or not isinstance(selected_content_max_tokens, int)
            or not 1 <= selected_content_max_tokens <= 1_800
        ):
            raise ValueError(
                "Mem0 selected_content_max_tokens must be between 1 and 1800"
            )
        self.search_limit = search_limit
        self.selected_max_items = selected_max_items
        self.selected_content_max_tokens = selected_content_max_tokens
        self.mem0_dir = self.stream_memory_dir / "mem0"
        self._mem0_config = dict(mem0_config or {})
        if client is not None:
            client_persistence = getattr(client, "persistence_dir", None)
            if client_persistence is not None and (
                Path(client_persistence).expanduser().resolve() != self.mem0_dir
            ):
                raise ValueError("Mem0 client persistence must use the arm-local mem0 directory")
        self._client_instance = client
        self._pending_search: Mem0SearchResult | None = None

    def prepare_context(self, task: BenchmarkTask) -> MemoryContextSelection:
        if not isinstance(task, BenchmarkTask):
            raise ValueError("task must be a BenchmarkTask")
        if self._pending_search is not None:
            raise RuntimeError("previous Mem0 task has not been finalized")
        search = self._client().search(
            task.instruction,
            stream_key=self.stream_project_key,
            limit=self.search_limit,
        )
        if not isinstance(search, Mem0SearchResult):
            raise RuntimeError("Mem0 client search must return Mem0SearchResult")
        self._pending_search = search
        candidates = search.items[: self.search_limit]
        selected_ids: list[str] = []
        selected_texts: list[str] = []
        selected_tokens = 0
        for item in candidates:
            if len(selected_ids) >= self.selected_max_items:
                break
            item_tokens = estimate_tokens(item.text)
            if selected_tokens + item_tokens > self.selected_content_max_tokens:
                continue
            selected_ids.append(item.memory_id)
            selected_texts.append(item.text)
            selected_tokens += item_tokens
        injected_text = _render_mem0_context(selected_ids, selected_texts)
        return MemoryContextSelection(
            backend=self.name,
            candidate_count=len(candidates),
            selected_ids=tuple(selected_ids),
            selected_texts=tuple(selected_texts),
            selected_content_tokens=selected_tokens,
            injected_text=injected_text,
            estimated_tokens=estimate_tokens(injected_text),
            retrieval_elapsed_sec=search.elapsed_sec,
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

    def build_agent_runner(
        self,
        *,
        context: MemoryContextSelection,
    ) -> AgentRunnerFn:
        self._validate_context(context)
        return _build_task_agent_runner(external_context=context)

    def finalize_task(
        self,
        episode: PublicEpisode,
        result: ManifestEvalResult,
    ) -> BackendFinalizeResult:
        if not isinstance(episode, PublicEpisode) or not isinstance(result, ManifestEvalResult):
            raise ValueError("Mem0 finalize requires public episode and manifest result")
        if not result.outcome_finalized:
            raise RuntimeError("Mem0 cannot write an unfinalized outcome")
        search = self._pending_search
        if search is None:
            raise RuntimeError("Mem0 finalize requires a completed task search")
        write = self._client().add(episode, stream_key=self.stream_project_key)
        combined_usage = _combine_provider_usage(search.llm_usage, write.llm_usage)
        self._pending_search = None
        return BackendFinalizeResult(
            status="committed" if write.written_ids else "no_write",
            written_ids=write.written_ids,
            llm_usage=combined_usage,
            usage_by_role={
                "search": search.llm_usage,
                "add": write.llm_usage,
            },
            embedding_calls=search.embedding_calls + write.embedding_calls,
            embedding_elapsed_sec=(
                search.embedding_elapsed_sec + write.embedding_elapsed_sec
            ),
            elapsed_sec=search.elapsed_sec + write.elapsed_sec,
        )

    def snapshot(self) -> MemoryRepositorySnapshot:
        count = self._client().count(stream_key=self.stream_project_key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError("Mem0 client count must return a non-negative integer")
        repository_bytes = _directory_size(self.mem0_dir)
        return MemoryRepositorySnapshot(
            revision=canonical_sha256(
                {"entry_count": count, "repository_bytes": repository_bytes}
            ),
            entry_count=count,
            repository_bytes=repository_bytes,
            tier_counts={"external": count},
            repository_path=str(self.mem0_dir),
        )

    def close(self) -> None:
        if self._client_instance is not None:
            self._client_instance.close()

    def _validate_stream(self, memory_dir: Path, project_key: str) -> None:
        if Path(memory_dir).expanduser().resolve() != self.stream_memory_dir:
            raise ValueError("backend stream_memory_dir does not match the configured stream")
        if project_key != self.stream_project_key:
            raise ValueError("backend stream_project_key does not match the configured stream")

    def _validate_context(self, context: MemoryContextSelection) -> None:
        if context.backend != self.name:
            raise ValueError(
                f"memory context backend mismatch: expected {self.name!r}, "
                f"got {context.backend!r}"
            )

    def _client(self) -> Any:
        if self._client_instance is None:
            self._client_instance = Mem0ClientAdapter(
                persistence_dir=self.mem0_dir,
                config=self._mem0_config,
            )
        return self._client_instance


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


def _build_task_agent_runner(
    *,
    external_context: MemoryContextSelection | None = None,
) -> AgentRunnerFn:
    def task_agent_runner(**runner_kwargs: Any) -> Any:
        call_kwargs = dict(runner_kwargs)
        config = call_kwargs.get("config")
        if not isinstance(config, AgentConfig):
            raise ValueError("benchmark agent runner requires AgentConfig")
        task_config = replace(config, memory_evolver_mode="off")
        repo_path = Path(call_kwargs["repo_path"]).resolve()
        supplied_llm = call_kwargs.pop("llm", None)
        task_llm = (
            supplied_llm
            if supplied_llm is not None
            else _build_benchmark_actor(task_config)
        )
        inner_memory = MemoryManager.from_config(
            config=task_config,
            llm=task_llm,
            repo_path=repo_path,
        )
        memory_manager: Any = inner_memory
        if external_context is not None:
            memory_manager = ExternalContextMemoryManager(inner_memory, external_context)
        call_kwargs.pop("memory_manager", None)
        call_kwargs["config"] = task_config
        call_kwargs["llm"] = task_llm
        call_kwargs["memory_manager"] = memory_manager
        return run_agent(**call_kwargs)

    return task_agent_runner


def _build_benchmark_actor(config: AgentConfig) -> Any:
    return build_llm(config)


def _render_mem0_context(ids: list[str], texts: list[str]) -> str:
    if not ids:
        return ""
    blocks = [MEM0_CONTEXT_HEADER]
    for memory_id, text in zip(ids, texts, strict=True):
        blocks.append(f"[mem0:{memory_id}]\n{text.strip()}")
    return "\n\n".join(blocks)


def _combine_provider_usage(*usages: ProviderUsage) -> ProviderUsage:
    if not usages or any(not usage.available for usage in usages):
        return ProviderUsage()
    prompt_tokens = (
        sum(usage.prompt_tokens for usage in usages if usage.prompt_tokens is not None)
        if all(usage.prompt_tokens is not None for usage in usages)
        else None
    )
    completion_tokens = (
        sum(
            usage.completion_tokens
            for usage in usages
            if usage.completion_tokens is not None
        )
        if all(usage.completion_tokens is not None for usage in usages)
        else None
    )
    return ProviderUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=sum(usage.resolved_total_tokens or 0 for usage in usages),
    )


def _policy_usages(metrics: ApiPolicyMetrics) -> dict[str, ProviderUsage]:
    usages: dict[str, ProviderUsage] = {}
    for role, role_metrics in metrics.by_role.items():
        if role_metrics.calls < 1:
            continue
        usages[role] = _policy_role_usage(role_metrics)
    return usages


def _policy_role_usage(metrics: ApiPolicyRoleMetrics) -> ProviderUsage:
    if not metrics.usage_available:
        return ProviderUsage()
    return ProviderUsage(
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        total_tokens=metrics.total_tokens,
    )


def _public_episode_steps(
    episode: PublicEpisode,
) -> tuple[ExperienceWriteStep, ...]:
    steps: list[ExperienceWriteStep] = []
    for action in episode.actions:
        sequence = int(action["sequence"])
        returncode = int(action["returncode"])
        timed_out = bool(action["timed_out"])
        stdout = str(action.get("stdout", ""))
        stderr = str(action.get("stderr", ""))
        output = "\n".join(part for part in (stdout, stderr) if part)[:4_000]
        steps.append(
            ExperienceWriteStep(
                step_num=sequence,
                tool="benchmark_action",
                arguments={"command": str(action["command"])},
                ok=returncode == 0 and not timed_out,
                output=output,
                blocked=False,
                error_code=(
                    "timeout"
                    if timed_out
                    else "" if returncode == 0 else f"returncode_{returncode}"
                ),
            )
        )
    return tuple(steps)


def _four_tier_maintenance_metrics(
    events: list[tuple[str, Mapping[str, Any]]],
    *,
    status: str | None,
) -> dict[str, Any]:
    cadence_events = [
        payload
        for event, payload in events
        if event == "memory.evolver_maintenance_cadence"
    ]
    operations = Counter()
    for event, payload in events:
        if event != "opd.decision" or payload.get("role") != "maintenance":
            continue
        parsed_output = payload.get("parsed_output")
        if not isinstance(parsed_output, Mapping):
            continue
        tool_call = parsed_output.get("tool_call")
        if isinstance(tool_call, Mapping):
            name = str(tool_call.get("name") or "")
            if name in {"keep", "delete", "merge", "promote"}:
                operations[name] += 1
    maintenance_status = status or "not_due"
    latest_cadence = cadence_events[-1] if cadence_events else {}
    applied = sum(
        1 for payload in cadence_events if payload.get("status") in {"committed", "noop"}
    )
    failures = sum(
        1
        for payload in cadence_events
        if payload.get("status") not in {"committed", "noop"}
    )
    return {
        "maintenance_status": maintenance_status,
        "maintenance_error": str(latest_cadence.get("error") or ""),
        "maintenance_turns": int(latest_cadence.get("turns") or 0),
        "maintenance_operation_ids": list(latest_cadence.get("operation_ids") or ()),
        "maintenance_runs": len(cadence_events),
        "maintenance_applied_runs": applied,
        "maintenance_failures": failures,
        "maintenance_keep": operations["keep"],
        "maintenance_delete": operations["delete"],
        "maintenance_merge": operations["merge"],
        "maintenance_promote": operations["promote"],
    }


def _append_backend_events(
    path: Path,
    *,
    task_id: str,
    events: list[tuple[str, Mapping[str, Any]]],
) -> None:
    if not events:
        return
    payload = b"".join(
        canonical_json_bytes(
            {
                "schema_version": BACKEND_EVENT_SCHEMA_VERSION,
                "task_id": task_id,
                "event": event,
                "payload": dict(event_payload),
            }
        )
        + b"\n"
        for event, event_payload in events
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(
                f"short append to {path}: wrote {written} of {len(payload)} bytes"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    "MEM0_CONTEXT_HEADER",
    "Mem0Backend",
    "MemoryBenchmarkBackend",
    "NoMemoryBackend",
    "memory_stream_project_key",
]
