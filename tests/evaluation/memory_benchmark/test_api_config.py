from __future__ import annotations

from pathlib import Path

import pytest

from my_agent.config import AgentConfig
from my_agent.evaluation.memory_benchmark.api_config import (
    actor_api_identity_hash,
    resolve_actor_endpoint,
    resolve_embedding_endpoint,
)
from my_agent.evaluation.memory_benchmark.protocol import load_memory_benchmark_config


CONFIG_PATH = Path("configs/memory_benchmark/v2.json")


def _agent_config(**overrides: object) -> AgentConfig:
    values: dict[str, object] = {
        "provider": "openai",
        "api_key": "actor-secret",
        "base_url": "https://example.test/compatible-mode/v1/",
        "model": " Qwen-Plus ",
        "temperature": 1.0,
        "max_steps": 5,
        "command_timeout": 30,
        "trace_dir": Path("traces"),
        "use_fake_llm": False,
    }
    values.update(overrides)
    return AgentConfig(**values)  # type: ignore[arg-type]


def test_actor_endpoint_is_normalized_and_secret_free() -> None:
    endpoint = resolve_actor_endpoint(_agent_config())

    assert endpoint.base_url == "https://example.test/compatible-mode/v1"
    assert endpoint.model == "Qwen-Plus"
    assert endpoint.api_key == "actor-secret"
    assert "actor-secret" not in endpoint.endpoint_hash


@pytest.mark.parametrize(
    ("env", "expected_key"),
    [
        ({"MY_AGENT_EMBEDDING_API_KEY": "embedding-secret"}, "embedding-secret"),
        ({"DASHSCOPE_API_KEY": "dashscope-secret"}, "dashscope-secret"),
        ({}, "actor-secret"),
    ],
)
def test_embedding_endpoint_key_fallbacks(
    env: dict[str, str], expected_key: str
) -> None:
    actor = resolve_actor_endpoint(_agent_config())
    endpoint = resolve_embedding_endpoint(
        actor=actor,
        benchmark_config=load_memory_benchmark_config(CONFIG_PATH),
        env=env,
    )

    assert endpoint.api_key == expected_key
    assert endpoint.model == "text-embedding-v4"


def test_embedding_endpoint_url_falls_back_to_actor() -> None:
    actor = resolve_actor_endpoint(_agent_config())

    inherited = resolve_embedding_endpoint(
        actor=actor,
        benchmark_config=load_memory_benchmark_config(CONFIG_PATH),
        env={},
    )
    separate = resolve_embedding_endpoint(
        actor=actor,
        benchmark_config=load_memory_benchmark_config(CONFIG_PATH),
        env={"MY_AGENT_EMBEDDING_BASE_URL": "https://embedding.example/v1/"},
    )

    assert inherited.base_url == actor.base_url
    assert separate.base_url == "https://embedding.example/v1"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"provider": "fake"}, "openai provider"),
        ({"api_key": ""}, "API key"),
        ({"base_url": ""}, "base URL"),
        ({"base_url": "ftp://example.test/v1"}, "http or https"),
        ({"base_url": "https://{WorkspaceId}.example/v1"}, "WorkspaceId"),
        ({"model": ""}, "model"),
    ],
)
def test_actor_endpoint_rejects_invalid_configuration(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_actor_endpoint(_agent_config(**overrides))


def test_actor_identity_is_stable_and_model_sensitive() -> None:
    first = resolve_actor_endpoint(_agent_config())
    same = resolve_actor_endpoint(_agent_config(api_key="different-secret"))
    changed = resolve_actor_endpoint(_agent_config(model="qwen-max"))

    assert actor_api_identity_hash(first) == actor_api_identity_hash(same)
    assert actor_api_identity_hash(first) != actor_api_identity_hash(changed)
