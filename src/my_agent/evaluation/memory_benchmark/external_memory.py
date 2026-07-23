"""Evaluation-only external memory integration for the Mem0 benchmark arm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Iterator
import os

from my_agent.evaluation.memory_benchmark.contracts import (
    ExternalMemoryItem,
    Mem0SearchResult,
    Mem0WriteResult,
    MemoryContextSelection,
    ProviderUsage,
    PublicEpisode,
)
from my_agent.memory.manager import MemoryManager
from my_agent.memory.types import MemoryContext


_LOCAL_VECTOR_STORES = frozenset({"faiss", "qdrant"})
_REMOTE_QDRANT_FIELDS = ("client", "host", "port", "url", "api_key")


class ExternalContextMemoryManager:
    """Delegate normal memory behavior while freezing one external context block."""

    def __init__(
        self,
        inner: MemoryManager,
        context: MemoryContextSelection,
    ) -> None:
        if not isinstance(inner, MemoryManager):
            raise ValueError("inner must be a MemoryManager")
        if inner.config.memory_evolver_mode != "off":
            raise ValueError("external context requires memory_evolver_mode='off'")
        if not isinstance(context, MemoryContextSelection):
            raise ValueError("context must be a MemoryContextSelection")
        self._inner = inner
        self._context = context

    @property
    def inner(self) -> MemoryManager:
        return self._inner

    @property
    def external_context(self) -> MemoryContextSelection:
        return self._context

    @property
    def config(self) -> Any:
        return self._inner.config

    @property
    def context_profile(self) -> Any:
        return self._inner.context_profile

    @property
    def project_key(self) -> str:
        return self._inner.project_key

    @property
    def session_id(self) -> str:
        return self._inner.session_id

    @property
    def evolver_coordinator(self) -> Any | None:
        return self._inner.evolver_coordinator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def set_trace_sink(self, trace_sink: Any | None) -> tuple[Any | None, Any | None]:
        return self._inner.set_trace_sink(trace_sink)

    def restore_trace_sink(self, snapshot: tuple[Any | None, Any | None]) -> None:
        self._inner.restore_trace_sink(snapshot)

    def append_task_goal(self, goal: str, *, run_id: str = "") -> Any:
        return self._inner.append_task_goal(goal, run_id=run_id)

    def append_user_message(self, content: str, *, run_id: str = "") -> Any:
        return self._inner.append_user_message(content, run_id=run_id)

    def append_assistant_response(self, response: Any, *, run_id: str = "") -> Any:
        return self._inner.append_assistant_response(response, run_id=run_id)

    def append_tool_result(self, result: Any, *, run_id: str = "") -> Any:
        return self._inner.append_tool_result(result, run_id=run_id)

    def append_summary(
        self,
        content: str,
        *,
        source: str = "summary",
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self._inner.append_summary(
            content,
            source=source,
            run_id=run_id,
            metadata=metadata,
        )

    def save_experience(self, **kwargs: Any) -> Any:
        return self._inner.save_experience(**kwargs)

    def build_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        limit: int | None = None,
        include_short_term: bool = False,
    ) -> MemoryContext[Any]:
        del query, max_tokens, limit, include_short_term
        return self._frozen_context()

    def build_evolver_context_for_query(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        top_k_per_tier: int | None = None,
        max_items: int | None = None,
    ) -> MemoryContext[Any]:
        del query, max_tokens, top_k_per_tier, max_items
        return self._frozen_context()

    def render_short_term_messages(
        self,
        *,
        max_tokens: int | None = None,
    ) -> list[Any]:
        return self._inner.render_short_term_messages(max_tokens=max_tokens)

    def compact_short_term(self, **kwargs: Any) -> Any:
        return self._inner.compact_short_term(**kwargs)

    def clear_short_term(
        self,
        *,
        extract_first: bool = True,
        reason: str = "clear",
    ) -> Any:
        return self._inner.clear_short_term(
            extract_first=extract_first,
            reason=reason,
        )

    def status(self, *, include_entries: bool = True) -> Any:
        return self._inner.status(include_entries=include_entries)

    def fork_for_task(
        self,
        *,
        session_id: str,
        run_id: str = "",
    ) -> "ExternalContextMemoryManager":
        return ExternalContextMemoryManager(
            self._inner.fork_for_task(session_id=session_id, run_id=run_id),
            self._context,
        )

    def trace_context_event(self, event: str, payload: dict[str, Any]) -> None:
        self._inner.trace_context_event(event, payload)

    def begin_formal_evolver_task(self, **kwargs: Any) -> Any:
        return self._inner.begin_formal_evolver_task(**kwargs)

    def write_experiences_from_run(self, **kwargs: Any) -> Any:
        return self._inner.write_experiences_from_run(**kwargs)

    def require_formal_runtime_binding(self, **kwargs: Any) -> None:
        self._inner.require_formal_runtime_binding(**kwargs)

    def _frozen_context(self) -> MemoryContext[Any]:
        return MemoryContext(
            injected_text=self._context.injected_text,
            hits=[],
            estimated_tokens=self._context.estimated_tokens,
        )


class Mem0ClientAdapter:
    """Normalize Mem0 2.x APIs and keep all persistent state arm-local."""

    def __init__(
        self,
        *,
        persistence_dir: str | Path,
        config: Mapping[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self.persistence_dir = Path(persistence_dir).expanduser().resolve()
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.config = localize_mem0_config(config or {}, self.persistence_dir)
        if client is None:
            _require_explicit_provider_identity(self.config)
            self._client = self._build_client()
        else:
            self._client = client
        self._closed = False

    def search(
        self,
        query: str,
        *,
        stream_key: str,
        limit: int,
    ) -> Mem0SearchResult:
        normalized_query = str(query).strip()
        if not normalized_query:
            raise ValueError("Mem0 search query must be non-empty")
        normalized_stream = _required_stream_key(stream_key)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("Mem0 search limit must be a positive integer")
        self._require_open()
        started = perf_counter()
        with _track_embedding_calls(self._client) as embedding_metrics:
            raw = self._client.search(
                normalized_query,
                top_k=limit,
                filters={"run_id": normalized_stream},
            )
        items = _normalize_memory_items(raw, operation="search")[:limit]
        return Mem0SearchResult(
            items=tuple(items),
            llm_usage=_provider_usage(raw),
            embedding_calls=embedding_metrics.calls,
            embedding_elapsed_sec=embedding_metrics.elapsed_sec,
            elapsed_sec=perf_counter() - started,
        )

    def add(
        self,
        episode: PublicEpisode,
        *,
        stream_key: str,
    ) -> Mem0WriteResult:
        if not isinstance(episode, PublicEpisode):
            raise ValueError("Mem0 add requires a PublicEpisode")
        normalized_stream = _required_stream_key(stream_key)
        self._require_open()
        started = perf_counter()
        with _track_embedding_calls(self._client) as embedding_metrics:
            raw = self._client.add(
                _episode_messages(episode),
                run_id=normalized_stream,
                metadata={
                    "task_id": episode.task_id,
                    "resolved": episode.resolved,
                    "reward": episode.reward,
                    "failure_type": episode.failure_type,
                },
            )
        written_ids = tuple(
            item.memory_id for item in _normalize_memory_items(raw, operation="add")
        )
        return Mem0WriteResult(
            written_ids=written_ids,
            llm_usage=_provider_usage(raw),
            embedding_calls=embedding_metrics.calls,
            embedding_elapsed_sec=embedding_metrics.elapsed_sec,
            elapsed_sec=perf_counter() - started,
        )

    def count(self, *, stream_key: str) -> int:
        normalized_stream = _required_stream_key(stream_key)
        self._require_open()
        raw = self._client.get_all(
            filters={"run_id": normalized_stream},
            top_k=10_000,
        )
        return len(_normalize_memory_items(raw, operation="get_all"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        vector_store = getattr(self._client, "vector_store", None)
        vector_client = getattr(vector_store, "client", None)
        vector_close = getattr(vector_client, "close", None)
        if callable(vector_close):
            vector_close()

    def _build_client(self) -> Any:
        runtime_dir = self.persistence_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        previous_dir = os.environ.get("MEM0_DIR")
        previous_telemetry = os.environ.get("MEM0_TELEMETRY")
        os.environ["MEM0_DIR"] = str(runtime_dir)
        os.environ["MEM0_TELEMETRY"] = "false"
        try:
            from mem0 import Memory
            import mem0.configs.base as mem0_base
            import mem0.memory.main as mem0_main
            import mem0.memory.setup as mem0_setup
            import mem0.memory.telemetry as mem0_telemetry

            if mem0_telemetry.MEM0_TELEMETRY:
                raise RuntimeError("Mem0 telemetry must be disabled for benchmark isolation")
            mem0_setup.mem0_dir = str(runtime_dir)
            mem0_base.mem0_dir = str(runtime_dir)
            mem0_main.mem0_dir = str(runtime_dir)
            return Memory.from_config(deepcopy(self.config))
        finally:
            _restore_environment("MEM0_DIR", previous_dir)
            _restore_environment("MEM0_TELEMETRY", previous_telemetry)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Mem0 client adapter is closed")


def localize_mem0_config(
    config: Mapping[str, Any],
    persistence_dir: str | Path,
) -> dict[str, Any]:
    """Return a Mem0 config whose vector and history state cannot leave the arm."""

    if not isinstance(config, Mapping):
        raise ValueError("Mem0 config must be a mapping")
    root = Path(persistence_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    normalized = deepcopy(dict(config))
    vector_store = normalized.get("vector_store") or {}
    if not isinstance(vector_store, Mapping):
        raise ValueError("Mem0 vector_store config must be a mapping")
    vector_store = deepcopy(dict(vector_store))
    provider = str(vector_store.get("provider") or "qdrant").strip().lower()
    if provider not in _LOCAL_VECTOR_STORES:
        raise ValueError(
            "Mem0 benchmark requires a local qdrant or faiss vector store"
        )
    provider_config = vector_store.get("config") or {}
    if not isinstance(provider_config, Mapping):
        raise ValueError("Mem0 vector_store.config must be a mapping")
    provider_config = deepcopy(dict(provider_config))
    if provider == "qdrant":
        remote_fields = [
            field for field in _REMOTE_QDRANT_FIELDS if provider_config.get(field) is not None
        ]
        if remote_fields:
            raise ValueError(
                "Mem0 benchmark does not allow remote qdrant fields: "
                + ", ".join(sorted(remote_fields))
            )
    provider_config["path"] = str(root / "vector_store")
    vector_store["provider"] = provider
    vector_store["config"] = provider_config
    normalized["vector_store"] = vector_store
    history_path = root / "history" / "history.db"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    normalized["history_db_path"] = str(history_path)
    return normalized


def _episode_messages(episode: PublicEpisode) -> list[dict[str, str]]:
    action_lines: list[str] = []
    for index, action in enumerate(episode.actions, start=1):
        command = str(action.get("command", ""))
        stdout = str(action.get("stdout", ""))
        stderr = str(action.get("stderr", ""))
        returncode = action.get("returncode")
        action_lines.append(
            f"Action {index}: command={command!r} returncode={returncode!r}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    outcome = "success" if episode.resolved else "failure"
    assistant_parts = [
        "Public actions and observations:",
        "\n\n".join(action_lines) if action_lines else "(none)",
        f"Final response:\n{episode.final_response}",
        (
            f"Authoritative outcome: {outcome}; reward={episode.reward}; "
            f"failure_type={episode.failure_type or 'none'}"
        ),
    ]
    return [
        {"role": "user", "content": episode.instruction},
        {"role": "assistant", "content": "\n\n".join(assistant_parts)},
    ]


def _require_explicit_provider_identity(config: Mapping[str, Any]) -> None:
    for component in ("llm", "embedder"):
        raw_component = config.get(component)
        if not isinstance(raw_component, Mapping):
            raise ValueError(f"Mem0 {component} provider and model must be explicit")
        provider = str(raw_component.get("provider") or "").strip()
        provider_config = raw_component.get("config")
        if not provider or not isinstance(provider_config, Mapping):
            raise ValueError(f"Mem0 {component} provider and model must be explicit")
        model = str(
            provider_config.get("model")
            or provider_config.get("model_name")
            or ""
        ).strip()
        if not model:
            raise ValueError(f"Mem0 {component} provider and model must be explicit")


def _normalize_memory_items(raw: Any, *, operation: str) -> list[ExternalMemoryItem]:
    results: Any
    if isinstance(raw, Mapping):
        results = raw.get("results")
    else:
        results = raw
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise ValueError(f"Mem0 {operation} response must contain a results array")
    normalized: list[ExternalMemoryItem] = []
    for index, item in enumerate(results):
        if not isinstance(item, Mapping):
            raise ValueError(f"Mem0 {operation} result {index} must be an object")
        memory_id = str(item.get("id") or item.get("memory_id") or "").strip()
        text = str(item.get("memory") or item.get("text") or "").strip()
        if not memory_id or not text:
            raise ValueError(
                f"Mem0 {operation} result {index} requires non-empty id and memory text"
            )
        raw_score = item.get("score")
        score = None if raw_score is None else float(raw_score)
        raw_metadata = item.get("metadata") or {}
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"Mem0 {operation} result {index} metadata must be an object")
        normalized.append(
            ExternalMemoryItem(
                memory_id=memory_id,
                text=text,
                score=score,
                metadata=dict(raw_metadata),
            )
        )
    return normalized


def _provider_usage(raw: Any) -> ProviderUsage:
    if not isinstance(raw, Mapping):
        return ProviderUsage()
    usage: Any = raw.get("llm_usage")
    if not isinstance(usage, Mapping):
        usage = raw.get("usage")
    if not isinstance(usage, Mapping):
        return ProviderUsage()
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    return ProviderUsage(
        prompt_tokens=_optional_usage_int(prompt, "prompt_tokens"),
        completion_tokens=_optional_usage_int(completion, "completion_tokens"),
        total_tokens=_optional_usage_int(total, "total_tokens"),
    )


def _optional_usage_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Mem0 {field_name} must be a non-negative integer")
    return value


@dataclass
class _EmbeddingMetrics:
    calls: int = 0
    elapsed_sec: float = 0.0


@contextmanager
def _track_embedding_calls(client: Any) -> Iterator[_EmbeddingMetrics]:
    metrics = _EmbeddingMetrics()
    embedding_model = getattr(client, "embedding_model", None)
    originals = {
        name: method
        for name in ("embed", "embed_batch")
        if callable(method := getattr(embedding_model, name, None))
    }
    if not originals:
        yield metrics
        return
    lock = Lock()

    try:
        for name, original in originals.items():
            def tracked(
                *args: Any,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                started = perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    with lock:
                        metrics.calls += 1
                        metrics.elapsed_sec += perf_counter() - started

            setattr(embedding_model, name, tracked)
    except (AttributeError, TypeError) as exc:
        for name, original in originals.items():
            try:
                setattr(embedding_model, name, original)
            except (AttributeError, TypeError):
                pass
        raise RuntimeError("Mem0 embedding provider cannot be instrumented") from exc
    try:
        yield metrics
    finally:
        for name, original in originals.items():
            setattr(embedding_model, name, original)


def _required_stream_key(value: str) -> str:
    normalized = str(value).strip()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError("Mem0 stream_key must be non-empty and contain no whitespace")
    return normalized


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


__all__ = [
    "ExternalContextMemoryManager",
    "Mem0ClientAdapter",
    "localize_mem0_config",
]
