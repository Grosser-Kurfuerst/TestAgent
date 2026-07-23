"""Secret-safe OpenAI-compatible API configuration for memory benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from my_agent.config import AgentConfig
from my_agent.evaluation.memory_benchmark.protocol import MemoryBenchmarkConfig
from my_agent.policy.identity import canonical_sha256


@dataclass(frozen=True)
class ApiEndpoint:
    api_key: str
    base_url: str
    model: str
    endpoint_hash: str


def resolve_actor_endpoint(config: AgentConfig) -> ApiEndpoint:
    """Resolve the single actor endpoint used by every benchmark arm."""

    if config.provider.strip().casefold() != "openai":
        raise ValueError("memory benchmark requires the openai provider")
    return _build_endpoint(
        api_key=config.api_key,
        base_url=config.base_url or "",
        model=config.model,
        label="actor",
    )


def resolve_embedding_endpoint(
    *,
    actor: ApiEndpoint,
    benchmark_config: MemoryBenchmarkConfig,
    env: Mapping[str, str],
) -> ApiEndpoint:
    """Resolve the benchmark embedding endpoint with documented fallbacks."""

    api_key = (
        str(env.get("MY_AGENT_EMBEDDING_API_KEY", "")).strip()
        or str(env.get("DASHSCOPE_API_KEY", "")).strip()
        or actor.api_key
    )
    base_url = (
        str(env.get("MY_AGENT_EMBEDDING_BASE_URL", "")).strip()
        or actor.base_url
    )
    return _build_endpoint(
        api_key=api_key,
        base_url=base_url,
        model=benchmark_config.embedding["model"],
        label="embedding",
    )


def actor_api_identity_hash(endpoint: ApiEndpoint) -> str:
    """Return the public, secret-free identity shared by all actor roles."""

    return canonical_sha256(
        {
            "provider": "openai_compatible",
            "model": endpoint.model,
            "endpoint_hash": endpoint.endpoint_hash,
        }
    )


def _build_endpoint(
    *, api_key: str, base_url: str, model: str, label: str
) -> ApiEndpoint:
    normalized_key = str(api_key).strip()
    normalized_model = str(model).strip()
    if not normalized_key:
        raise ValueError(f"{label} API key must be non-empty")
    if not normalized_model:
        raise ValueError(f"{label} model must be non-empty")
    normalized_url = _normalize_base_url(base_url, label=label)
    return ApiEndpoint(
        api_key=normalized_key,
        base_url=normalized_url,
        model=normalized_model,
        endpoint_hash=canonical_sha256(
            {
                "provider": "openai_compatible",
                "base_url": normalized_url,
            }
        ),
    )


def _normalize_base_url(value: str, *, label: str) -> str:
    raw = str(value).strip().rstrip("/")
    if not raw:
        raise ValueError(f"{label} base URL must be non-empty")
    if "{WorkspaceId}" in raw:
        raise ValueError(f"{label} base URL contains unresolved {{WorkspaceId}}")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} base URL must use http or https")
    return urlunsplit(parsed)


__all__ = [
    "ApiEndpoint",
    "actor_api_identity_hash",
    "resolve_actor_endpoint",
    "resolve_embedding_endpoint",
]
