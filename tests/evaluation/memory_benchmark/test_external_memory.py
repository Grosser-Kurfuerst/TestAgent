from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from my_agent.config import AgentConfig
from my_agent.context import AgentContextManager
from my_agent.evaluation.memory_benchmark.api_config import ApiEndpoint
from my_agent.evaluation.memory_benchmark.contracts import (
    MemoryContextSelection,
    Mem0SearchResult,
    Mem0WriteResult,
    PublicEpisode,
)
from my_agent.evaluation.memory_benchmark.external_memory import (
    ExternalContextMemoryManager,
    Mem0ClientAdapter,
    build_memory_benchmark_mem0_config,
    localize_mem0_config,
    memory_benchmark_mem0_backend_identity,
)
from my_agent.evaluation.memory_benchmark.protocol import load_memory_benchmark_config
from my_agent.llm import FakeLLM
from my_agent.llm.types import Message
from my_agent.memory.api import MemoryService
from my_agent.memory.manager import MemoryManager
from my_agent.policy.identity import canonical_sha256


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        provider="fake",
        api_key="",
        base_url=None,
        model="fake",
        temperature=0.0,
        max_steps=4,
        command_timeout=20,
        trace_dir=tmp_path / "traces",
        use_fake_llm=True,
        memory_dir=tmp_path / "memory",
        memory_project_key="fixture-project",
        memory_evolver_mode="off",
        context_window=8_192,
        memory_context_tokens=1_800,
        memory_context_tokens_explicit=True,
    )


def _selection(text: str = "Remember the public command sequence.") -> MemoryContextSelection:
    return MemoryContextSelection(
        backend="mem0",
        candidate_count=1,
        selected_ids=("memory-1",),
        selected_texts=(text,),
        selected_content_tokens=10,
        injected_text=f"Relevant selected external memory:\n\n[mem0:memory-1]\n{text}",
        estimated_tokens=20,
        retrieval_elapsed_sec=0.01,
    )


def _endpoint(*, model: str, key: str, label: str) -> ApiEndpoint:
    return ApiEndpoint(
        api_key=key,
        base_url=f"https://{label}.example.test/v1",
        model=model,
        endpoint_hash=canonical_sha256({"endpoint": label}),
    )


def test_mem0_config_is_derived_from_shared_api_endpoints() -> None:
    actor = _endpoint(model="qwen-plus", key="actor-secret", label="actor")
    embedding = _endpoint(
        model="text-embedding-v4",
        key="embedding-secret",
        label="embedding",
    )

    config = build_memory_benchmark_mem0_config(
        actor_endpoint=actor,
        embedding_endpoint=embedding,
        embedding_dimension=1_024,
    )

    assert config["llm"]["config"] == {
        "model": actor.model,
        "api_key": actor.api_key,
        "openai_base_url": actor.base_url,
    }
    assert config["embedder"]["config"] == {
        "model": embedding.model,
        "api_key": embedding.api_key,
        "openai_base_url": embedding.base_url,
    }
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1_024


def test_mem0_backend_identity_is_public_and_matches_shared_embedding() -> None:
    actor = _endpoint(model="qwen-plus", key="actor-secret", label="actor")
    embedding = _endpoint(
        model="text-embedding-v4",
        key="embedding-secret",
        label="embedding",
    )

    identity = memory_benchmark_mem0_backend_identity(
        actor_endpoint=actor,
        embedding_endpoint=embedding,
        embedding_dimension=1_024,
        search_limit=50,
        selected_max_items=20,
        selected_content_max_tokens=1_800,
        package_version="2.0.13",
    )

    assert identity["llm_model"] == actor.model
    assert identity["embedding_model"] == embedding.model
    assert identity["embedding_endpoint_hash"] == embedding.endpoint_hash
    assert "actor-secret" not in str(identity)
    assert "embedding-secret" not in str(identity)


def test_v2_benchmark_config_has_no_mem0_config_path() -> None:
    config = load_memory_benchmark_config("configs/memory_benchmark/v2.json")

    assert "mem0_config" not in config.environment


def _episode() -> PublicEpisode:
    return PublicEpisode(
        task_id="task-1",
        instruction="Create the requested public file.",
        actions=(
            {
                "command": "printf public > result.txt",
                "returncode": 0,
                "stdout": "public",
                "stderr": "",
            },
        ),
        final_response="Created result.txt.",
        resolved=False,
        reward=0.0,
        failure_type="official_evaluator_failed",
    )


def test_external_context_manager_delegates_and_freezes_system_memory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    inner = MemoryManager.from_config(
        config=config,
        llm=FakeLLM(),
        repo_path=tmp_path,
    )
    external = ExternalContextMemoryManager(inner, _selection())

    assert isinstance(external, MemoryService)
    inner.append_user_message("short-term note")
    messages, context, _ = AgentContextManager(inner.context_profile).prepare_messages(
        base_messages=[Message(role="system", content="base system")],
        query="different query",
        tools=[],
        memory=external,
    )

    assert external.config is inner.config
    assert external.status().short_term_entries == 1
    assert context.injected_text == _selection().injected_text
    assert any(
        message.role == "system" and _selection().injected_text in (message.content or "")
        for message in messages
    )
    assert all(_selection().injected_text not in (message.content or "") for message in messages if message.role == "user")
    assert external.build_context_for_query("another", max_tokens=1).injected_text == _selection().injected_text
    forked = external.fork_for_task(session_id="task-session")
    assert forked.external_context is external.external_context
    assert forked.inner is not inner


class _FakeEmbedding:
    def embed(self, text: str, operation: str) -> list[float]:
        del text, operation
        return [0.1, 0.2]


class _FakeMem0:
    def __init__(self) -> None:
        self.embedding_model = _FakeEmbedding()
        self.memories: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.closed = False

    def search(self, query: str, *, top_k: int, filters: dict[str, str]) -> dict[str, Any]:
        self.embedding_model.embed(query, "search")
        rows = [row for row in self.memories if row["run_id"] == filters["run_id"]]
        return {"results": rows[:top_k]}

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        run_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self.embedding_model.embed(messages[-1]["content"], "add")
        memory_id = f"memory-{len(self.memories) + 1}"
        row = {
            "id": memory_id,
            "memory": messages[-1]["content"],
            "run_id": run_id,
            "metadata": metadata,
        }
        self.memories.append(row)
        self.add_calls.append({"messages": messages, "run_id": run_id, "metadata": metadata})
        return {
            "results": [{"id": memory_id, "memory": row["memory"], "event": "ADD"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }

    def get_all(self, *, filters: dict[str, str], top_k: int) -> dict[str, Any]:
        rows = [row for row in self.memories if row["run_id"] == filters["run_id"]]
        return {"results": rows[:top_k]}

    def close(self) -> None:
        self.closed = True


def test_mem0_adapter_normalizes_api_tracks_usage_and_uses_stream_filter(tmp_path: Path) -> None:
    fake = _FakeMem0()
    adapter = Mem0ClientAdapter(
        persistence_dir=tmp_path / "stream" / "memory" / "mem0",
        client=fake,
    )

    empty = adapter.search("first task", stream_key="stream-a", limit=50)
    written = adapter.add(_episode(), stream_key="stream-a")
    found = adapter.search("second task", stream_key="stream-a", limit=50)

    assert isinstance(empty, Mem0SearchResult)
    assert empty.items == ()
    assert empty.llm_usage.available is False
    assert empty.embedding_calls == 1
    assert isinstance(written, Mem0WriteResult)
    assert written.written_ids == ("memory-1",)
    assert written.llm_usage.resolved_total_tokens == 10
    assert written.embedding_calls == 1
    assert [item.memory_id for item in found.items] == ["memory-1"]
    assert found.embedding_calls == 1
    assert adapter.count(stream_key="stream-a") == 1
    assert adapter.count(stream_key="stream-b") == 0
    payload = fake.add_calls[0]
    rendered_payload = str(payload)
    assert payload["run_id"] == "stream-a"
    assert "printf public" in rendered_payload
    assert "Created result.txt" in rendered_payload
    assert "official_evaluator_failed" in rendered_payload
    assert "hidden evaluator" not in rendered_payload
    assert written == Mem0WriteResult.from_dict(written.to_dict())
    assert found == Mem0SearchResult.from_dict(found.to_dict())
    adapter.close()
    assert fake.closed is True


def test_mem0_config_forces_vector_and_history_state_under_arm(tmp_path: Path) -> None:
    home = tmp_path / "home"
    arm_dir = tmp_path / "run" / "arms" / "mem0" / "seed_42" / "os" / "memory" / "mem0"
    config = localize_mem0_config(
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {"collection_name": "benchmark"},
            },
            "llm": {"provider": "openai", "config": {"model": "recorded-model"}},
        },
        arm_dir,
    )

    vector_path = Path(config["vector_store"]["config"]["path"])
    history_path = Path(config["history_db_path"])
    assert vector_path.is_relative_to(arm_dir)
    assert history_path.is_relative_to(arm_dir)
    assert not vector_path.is_relative_to(home)
    assert config["llm"]["config"]["model"] == "recorded-model"

    with pytest.raises(ValueError, match="remote qdrant"):
        localize_mem0_config(
            {"vector_store": {"provider": "qdrant", "config": {"url": "https://qdrant"}}},
            arm_dir,
        )


def test_mem0_response_without_explicit_usage_remains_unknown(tmp_path: Path) -> None:
    fake = _FakeMem0()
    adapter = Mem0ClientAdapter(persistence_dir=tmp_path / "mem0", client=fake)

    result = adapter.search("query", stream_key="stream-a", limit=50)

    assert result.llm_usage.prompt_tokens is None
    assert result.llm_usage.completion_tokens is None
    assert result.llm_usage.total_tokens is None
    assert result.llm_usage.available is False


def test_real_mem0_adapter_rejects_unrecorded_default_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="llm provider and model must be explicit"):
        Mem0ClientAdapter(persistence_dir=tmp_path / "mem0", config={})
