"""Configuration and immutable protocol identity for memory benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any
import json
import re

from my_agent.policy.identity import (
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
)


MEMORY_BENCHMARK_CONFIG_SCHEMA_VERSION = "memory-benchmark-config-v1"
MEMORY_BENCHMARK_PROTOCOL_SCHEMA_VERSION = "memory-benchmark-protocol-v1"

REQUIRED_ENVIRONMENT_VARIABLES = {
    "lifelong_agent_bench_root": "AGENTCLI_LIFELONG_AGENT_BENCH_ROOT",
    "intercode_root": "AGENTCLI_INTERCODE_ROOT",
    "data_root": "AGENTCLI_MEMORY_BENCHMARK_DATA_ROOT",
    "mem0_config": "AGENTCLI_MEM0_CONFIG_PATH",
}
REQUIRED_ARMS = frozenset({"no_memory", "agentcli_four_tier", "mem0"})
REQUIRED_BENCHMARKS = frozenset({"lifelong_os", "intercode_bash"})

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SECRET_FIELD_MARKERS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "password",
    "client_secret",
    "credential",
)


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    source: str
    subset: str
    source_split: str
    evaluation_split: str
    limit: int
    order_policy: str

    def __post_init__(self) -> None:
        for field_name in ("source", "subset", "source_split", "evaluation_split"):
            _require_non_empty(getattr(self, field_name), field_name=field_name)
        if self.evaluation_split != "test":
            raise ValueError("benchmark evaluation_split must be 'test'")
        _require_positive_int(self.limit, field_name="limit")
        if self.order_policy not in {"source_order", "stable_task_id"}:
            raise ValueError("order_policy must be source_order or stable_task_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "subset": self.subset,
            "source_split": self.source_split,
            "evaluation_split": self.evaluation_split,
            "limit": self.limit,
            "order_policy": self.order_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkSuiteConfig":
        return cls(
            source=_required_string(data.get("source"), field_name="source"),
            subset=_required_string(data.get("subset"), field_name="subset"),
            source_split=_required_string(
                data.get("source_split"), field_name="source_split"
            ),
            evaluation_split=_required_string(
                data.get("evaluation_split"), field_name="evaluation_split"
            ),
            limit=_required_int(data.get("limit"), field_name="limit"),
            order_policy=_required_string(
                data.get("order_policy"), field_name="order_policy"
            ),
        )


@dataclass(frozen=True)
class MemoryBenchmarkConfig:
    source_lock_path: str
    environment: Mapping[str, str]
    benchmarks: Mapping[str, BenchmarkSuiteConfig]
    arms: Mapping[str, Mapping[str, Any]]
    runtime: Mapping[str, Any]
    embedding: Mapping[str, str]
    memory: Mapping[str, Any]
    output_root: str
    seeds: tuple[int, ...]
    actor_sampling_seed_supported: bool
    smoke: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_relative_path(self.source_lock_path, field_name="source_lock_path")
        _require_relative_path(self.output_root, field_name="output_root")
        environment = _string_mapping(self.environment, field_name="environment")
        if environment != REQUIRED_ENVIRONMENT_VARIABLES:
            raise ValueError("memory benchmark environment variables do not match v1 contract")
        object.__setattr__(self, "environment", environment)

        benchmarks = dict(self.benchmarks)
        if set(benchmarks) != REQUIRED_BENCHMARKS:
            raise ValueError("config must define exactly lifelong_os and intercode_bash")
        if not all(isinstance(item, BenchmarkSuiteConfig) for item in benchmarks.values()):
            raise ValueError("benchmarks must contain BenchmarkSuiteConfig values")
        object.__setattr__(self, "benchmarks", benchmarks)

        arms = {str(name): dict(settings) for name, settings in self.arms.items()}
        if set(arms) != REQUIRED_ARMS:
            raise ValueError("config must define exactly the three comparison arms")
        for name, settings in arms.items():
            if not isinstance(settings.get("enabled"), bool):
                raise ValueError(f"arm {name} requires a boolean enabled field")
        if not any(settings["enabled"] for settings in arms.values()):
            raise ValueError("at least one memory benchmark arm must be enabled")
        object.__setattr__(self, "arms", arms)

        runtime = dict(self.runtime)
        _validate_runtime_config(runtime)
        object.__setattr__(self, "runtime", runtime)
        embedding = _string_mapping(self.embedding, field_name="embedding")
        if set(embedding) != {"model", "revision_env"}:
            raise ValueError("embedding config requires model and revision_env")
        object.__setattr__(self, "embedding", embedding)
        memory = dict(self.memory)
        _validate_memory_config(memory)
        object.__setattr__(self, "memory", memory)

        seeds = tuple(self.seeds)
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be non-empty and unique")
        for seed in seeds:
            _require_non_negative_int(seed, field_name="seed")
        object.__setattr__(self, "seeds", seeds)
        if not isinstance(self.actor_sampling_seed_supported, bool):
            raise ValueError("actor_sampling_seed_supported must be a bool")

        smoke = dict(self.smoke)
        _require_positive_int(smoke.get("task_count"), field_name="smoke.task_count")
        _require_positive_int(
            smoke.get("maintenance_interval_tasks"),
            field_name="smoke.maintenance_interval_tasks",
        )
        object.__setattr__(self, "smoke", smoke)

    @property
    def enabled_arms(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, settings in self.arms.items() if settings["enabled"]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEMORY_BENCHMARK_CONFIG_SCHEMA_VERSION,
            "source_lock_path": self.source_lock_path,
            "environment": dict(sorted(self.environment.items())),
            "benchmarks": {
                name: suite.to_dict() for name, suite in sorted(self.benchmarks.items())
            },
            "arms": {name: dict(settings) for name, settings in sorted(self.arms.items())},
            "runtime": dict(self.runtime),
            "embedding": dict(self.embedding),
            "memory": dict(self.memory),
            "output_root": self.output_root,
            "seeds": list(self.seeds),
            "actor_sampling_seed_supported": self.actor_sampling_seed_supported,
            "smoke": dict(self.smoke),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryBenchmarkConfig":
        if data.get("schema_version") != MEMORY_BENCHMARK_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported memory benchmark config schema")
        _reject_secret_or_absolute_values(data)
        expected_fields = {
            "schema_version",
            "source_lock_path",
            "environment",
            "benchmarks",
            "arms",
            "runtime",
            "embedding",
            "memory",
            "output_root",
            "seeds",
            "actor_sampling_seed_supported",
            "smoke",
        }
        if set(data) != expected_fields:
            raise ValueError("memory benchmark config fields do not match the v1 schema")
        raw_benchmarks = _required_mapping(data.get("benchmarks"), field_name="benchmarks")
        raw_arms = _required_mapping(data.get("arms"), field_name="arms")
        raw_seeds = data.get("seeds")
        if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
            raise ValueError("seeds must be an array")
        return cls(
            source_lock_path=_required_string(
                data.get("source_lock_path"), field_name="source_lock_path"
            ),
            environment=_string_mapping(
                _required_mapping(data.get("environment"), field_name="environment"),
                field_name="environment",
            ),
            benchmarks={
                name: BenchmarkSuiteConfig.from_dict(
                    _required_mapping(value, field_name=f"benchmarks.{name}")
                )
                for name, value in raw_benchmarks.items()
            },
            arms={
                name: _required_mapping(value, field_name=f"arms.{name}")
                for name, value in raw_arms.items()
            },
            runtime=_required_mapping(data.get("runtime"), field_name="runtime"),
            embedding=_string_mapping(
                _required_mapping(data.get("embedding"), field_name="embedding"),
                field_name="embedding",
            ),
            memory=_required_mapping(data.get("memory"), field_name="memory"),
            output_root=_required_string(data.get("output_root"), field_name="output_root"),
            seeds=tuple(_required_int(seed, field_name="seed") for seed in raw_seeds),
            actor_sampling_seed_supported=_required_bool(
                data.get("actor_sampling_seed_supported"),
                field_name="actor_sampling_seed_supported",
            ),
            smoke=_required_mapping(data.get("smoke"), field_name="smoke"),
        )


@dataclass(frozen=True)
class MemoryBenchmarkProtocol:
    ordered_task_ids_by_benchmark: Mapping[str, tuple[str, ...]]
    task_manifest_hashes: Mapping[str, str]
    source_lock_hash: str
    actor_identity_hash: str
    tools_hash: str
    evaluator_hashes: Mapping[str, str]
    docker_image_digests: Mapping[str, str]
    backend_config_hashes: Mapping[str, str]
    agentcli_commit: str
    uv_lock_hash: str
    python_version: str
    runtime_environment_hash: str
    repetition_ids: tuple[int, ...]
    agent_mode: str
    context_window: int
    response_reserve_tokens: int
    compression_buffer_tokens: int
    repo_context_budget_tokens: int
    tool_schema_budget_tokens: int
    memory_short_term_tokens: int
    memory_context_tokens: int
    memory_tool_result_chars: int
    max_steps: int
    command_timeout: int
    actor_temperature: float
    memory_generation_temperature: float
    memory_generation_top_p: float
    selected_max_items: int
    selected_content_max_tokens: int
    maintenance_interval_tasks: int
    actor_sampling_seed_supported: bool
    pilot: bool = False

    def __post_init__(self) -> None:
        task_ids = {
            str(name): _string_tuple(ids, field_name=f"ordered task IDs for {name}")
            for name, ids in self.ordered_task_ids_by_benchmark.items()
        }
        if not task_ids or any(not ids for ids in task_ids.values()):
            raise ValueError("each benchmark requires at least one ordered task ID")
        for name, ids in task_ids.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f"benchmark {name} contains duplicate task IDs")
        object.__setattr__(
            self,
            "ordered_task_ids_by_benchmark",
            MappingProxyType(task_ids),
        )

        benchmark_names = set(task_ids)
        for field_name in (
            "task_manifest_hashes",
            "evaluator_hashes",
            "docker_image_digests",
        ):
            mapping = _hash_mapping(getattr(self, field_name), field_name=field_name)
            if set(mapping) != benchmark_names:
                raise ValueError(f"{field_name} keys must match ordered benchmarks")
            object.__setattr__(self, field_name, MappingProxyType(mapping))
        backend_hashes = _hash_mapping(
            self.backend_config_hashes,
            field_name="backend_config_hashes",
        )
        if not backend_hashes:
            raise ValueError("backend_config_hashes must not be empty")
        if not set(backend_hashes).issubset(REQUIRED_ARMS):
            raise ValueError("backend_config_hashes contains an unknown arm")
        object.__setattr__(
            self,
            "backend_config_hashes",
            MappingProxyType(backend_hashes),
        )

        for field_name in (
            "source_lock_hash",
            "actor_identity_hash",
            "tools_hash",
            "uv_lock_hash",
            "runtime_environment_hash",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if _COMMIT_RE.fullmatch(self.agentcli_commit) is None:
            raise ValueError("agentcli_commit must be a full lowercase Git commit")
        _require_non_empty(self.python_version, field_name="python_version")

        repetitions = tuple(self.repetition_ids)
        if not repetitions or len(repetitions) != len(set(repetitions)):
            raise ValueError("repetition_ids must be non-empty and unique")
        for repetition in repetitions:
            _require_non_negative_int(repetition, field_name="repetition_id")
        object.__setattr__(self, "repetition_ids", repetitions)

        if self.agent_mode != "react":
            raise ValueError("memory benchmark agent_mode must be 'react'")
        for field_name in (
            "context_window",
            "response_reserve_tokens",
            "compression_buffer_tokens",
            "repo_context_budget_tokens",
            "tool_schema_budget_tokens",
            "memory_short_term_tokens",
            "memory_context_tokens",
            "memory_tool_result_chars",
            "max_steps",
            "command_timeout",
            "selected_max_items",
            "selected_content_max_tokens",
            "maintenance_interval_tasks",
        ):
            _require_positive_int(getattr(self, field_name), field_name=field_name)
        if self.selected_max_items > 20:
            raise ValueError("selected_max_items cannot exceed 20")
        if self.selected_content_max_tokens > 1800:
            raise ValueError("selected_content_max_tokens cannot exceed 1800")
        if self.selected_content_max_tokens > self.memory_context_tokens:
            raise ValueError("selected content budget cannot exceed memory context budget")
        actor_temperature = _required_float(
            self.actor_temperature, field_name="actor_temperature"
        )
        memory_temperature = _required_float(
            self.memory_generation_temperature,
            field_name="memory_generation_temperature",
        )
        memory_top_p = _required_float(
            self.memory_generation_top_p,
            field_name="memory_generation_top_p",
        )
        object.__setattr__(self, "actor_temperature", actor_temperature)
        object.__setattr__(self, "memory_generation_temperature", memory_temperature)
        object.__setattr__(self, "memory_generation_top_p", memory_top_p)
        if actor_temperature < 0 or memory_temperature < 0:
            raise ValueError("temperatures must be non-negative")
        if not 0 < memory_top_p <= 1:
            raise ValueError("memory_generation_top_p must be in (0, 1]")
        if not isinstance(self.actor_sampling_seed_supported, bool):
            raise ValueError("actor_sampling_seed_supported must be a bool")
        if not isinstance(self.pilot, bool):
            raise ValueError("pilot must be a bool")

    @property
    def protocol_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEMORY_BENCHMARK_PROTOCOL_SCHEMA_VERSION,
            "ordered_task_ids_by_benchmark": {
                name: list(ids)
                for name, ids in sorted(self.ordered_task_ids_by_benchmark.items())
            },
            "task_manifest_hashes": dict(sorted(self.task_manifest_hashes.items())),
            "source_lock_hash": self.source_lock_hash,
            "actor_identity_hash": self.actor_identity_hash,
            "tools_hash": self.tools_hash,
            "evaluator_hashes": dict(sorted(self.evaluator_hashes.items())),
            "docker_image_digests": dict(sorted(self.docker_image_digests.items())),
            "backend_config_hashes": dict(sorted(self.backend_config_hashes.items())),
            "agentcli_commit": self.agentcli_commit,
            "uv_lock_hash": self.uv_lock_hash,
            "python_version": self.python_version,
            "runtime_environment_hash": self.runtime_environment_hash,
            "repetition_ids": list(self.repetition_ids),
            "agent_mode": self.agent_mode,
            "context_window": self.context_window,
            "response_reserve_tokens": self.response_reserve_tokens,
            "compression_buffer_tokens": self.compression_buffer_tokens,
            "repo_context_budget_tokens": self.repo_context_budget_tokens,
            "tool_schema_budget_tokens": self.tool_schema_budget_tokens,
            "memory_short_term_tokens": self.memory_short_term_tokens,
            "memory_context_tokens": self.memory_context_tokens,
            "memory_tool_result_chars": self.memory_tool_result_chars,
            "max_steps": self.max_steps,
            "command_timeout": self.command_timeout,
            "actor_temperature": self.actor_temperature,
            "memory_generation_temperature": self.memory_generation_temperature,
            "memory_generation_top_p": self.memory_generation_top_p,
            "selected_max_items": self.selected_max_items,
            "selected_content_max_tokens": self.selected_content_max_tokens,
            "maintenance_interval_tasks": self.maintenance_interval_tasks,
            "actor_sampling_seed_supported": self.actor_sampling_seed_supported,
            "pilot": self.pilot,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryBenchmarkProtocol":
        if data.get("schema_version") != MEMORY_BENCHMARK_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("unsupported memory benchmark protocol schema")
        raw_task_ids = _required_mapping(
            data.get("ordered_task_ids_by_benchmark"),
            field_name="ordered_task_ids_by_benchmark",
        )
        return cls(
            ordered_task_ids_by_benchmark={
                name: _sequence_of_strings(ids, field_name=f"task IDs for {name}")
                for name, ids in raw_task_ids.items()
            },
            task_manifest_hashes=_string_mapping(
                _required_mapping(
                    data.get("task_manifest_hashes"),
                    field_name="task_manifest_hashes",
                ),
                field_name="task_manifest_hashes",
            ),
            source_lock_hash=_required_string(
                data.get("source_lock_hash"), field_name="source_lock_hash"
            ),
            actor_identity_hash=_required_string(
                data.get("actor_identity_hash"), field_name="actor_identity_hash"
            ),
            tools_hash=_required_string(data.get("tools_hash"), field_name="tools_hash"),
            evaluator_hashes=_string_mapping(
                _required_mapping(
                    data.get("evaluator_hashes"), field_name="evaluator_hashes"
                ),
                field_name="evaluator_hashes",
            ),
            docker_image_digests=_string_mapping(
                _required_mapping(
                    data.get("docker_image_digests"),
                    field_name="docker_image_digests",
                ),
                field_name="docker_image_digests",
            ),
            backend_config_hashes=_string_mapping(
                _required_mapping(
                    data.get("backend_config_hashes"),
                    field_name="backend_config_hashes",
                ),
                field_name="backend_config_hashes",
            ),
            agentcli_commit=_required_string(
                data.get("agentcli_commit"), field_name="agentcli_commit"
            ),
            uv_lock_hash=_required_string(data.get("uv_lock_hash"), field_name="uv_lock_hash"),
            python_version=_required_string(
                data.get("python_version"), field_name="python_version"
            ),
            runtime_environment_hash=_required_string(
                data.get("runtime_environment_hash"),
                field_name="runtime_environment_hash",
            ),
            repetition_ids=_sequence_of_ints(
                data.get("repetition_ids"), field_name="repetition_ids"
            ),
            agent_mode=_required_string(data.get("agent_mode"), field_name="agent_mode"),
            context_window=_required_int(data.get("context_window"), field_name="context_window"),
            response_reserve_tokens=_required_int(
                data.get("response_reserve_tokens"),
                field_name="response_reserve_tokens",
            ),
            compression_buffer_tokens=_required_int(
                data.get("compression_buffer_tokens"),
                field_name="compression_buffer_tokens",
            ),
            repo_context_budget_tokens=_required_int(
                data.get("repo_context_budget_tokens"),
                field_name="repo_context_budget_tokens",
            ),
            tool_schema_budget_tokens=_required_int(
                data.get("tool_schema_budget_tokens"),
                field_name="tool_schema_budget_tokens",
            ),
            memory_short_term_tokens=_required_int(
                data.get("memory_short_term_tokens"),
                field_name="memory_short_term_tokens",
            ),
            memory_context_tokens=_required_int(
                data.get("memory_context_tokens"),
                field_name="memory_context_tokens",
            ),
            memory_tool_result_chars=_required_int(
                data.get("memory_tool_result_chars"),
                field_name="memory_tool_result_chars",
            ),
            max_steps=_required_int(data.get("max_steps"), field_name="max_steps"),
            command_timeout=_required_int(
                data.get("command_timeout"), field_name="command_timeout"
            ),
            actor_temperature=_required_float(
                data.get("actor_temperature"), field_name="actor_temperature"
            ),
            memory_generation_temperature=_required_float(
                data.get("memory_generation_temperature"),
                field_name="memory_generation_temperature",
            ),
            memory_generation_top_p=_required_float(
                data.get("memory_generation_top_p"),
                field_name="memory_generation_top_p",
            ),
            selected_max_items=_required_int(
                data.get("selected_max_items"), field_name="selected_max_items"
            ),
            selected_content_max_tokens=_required_int(
                data.get("selected_content_max_tokens"),
                field_name="selected_content_max_tokens",
            ),
            maintenance_interval_tasks=_required_int(
                data.get("maintenance_interval_tasks"),
                field_name="maintenance_interval_tasks",
            ),
            actor_sampling_seed_supported=_required_bool(
                data.get("actor_sampling_seed_supported"),
                field_name="actor_sampling_seed_supported",
            ),
            pilot=_required_bool(data.get("pilot", False), field_name="pilot"),
        )


def load_memory_benchmark_config(path: str | Path) -> MemoryBenchmarkConfig:
    """Load a tracked, secret-free memory benchmark configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid memory benchmark config JSON: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("memory benchmark config must be a JSON object")
    return MemoryBenchmarkConfig.from_dict(payload)


def canonical_config_bytes(config: MemoryBenchmarkConfig) -> bytes:
    if not isinstance(config, MemoryBenchmarkConfig):
        raise ValueError("config must be a MemoryBenchmarkConfig")
    return canonical_json_bytes(config.to_dict())


def backend_config_hash(config: Mapping[str, Any]) -> str:
    _reject_secret_or_absolute_values(config)
    return canonical_sha256(dict(config))


def _validate_runtime_config(runtime: Mapping[str, Any]) -> None:
    for field_name in (
        "context_window",
        "response_reserve_tokens",
        "compression_buffer_tokens",
        "repo_context_budget_tokens",
        "tool_schema_budget_tokens",
        "memory_short_term_tokens",
        "memory_context_tokens",
        "memory_tool_result_chars",
        "max_steps",
        "command_timeout_seconds",
    ):
        _require_positive_int(runtime.get(field_name), field_name=f"runtime.{field_name}")
    if runtime.get("agent_mode") != "react":
        raise ValueError("runtime.agent_mode must be 'react'")
    temperature = _required_float(
        runtime.get("actor_temperature"), field_name="runtime.actor_temperature"
    )
    if temperature < 0:
        raise ValueError("runtime.actor_temperature must be non-negative")


def _validate_memory_config(memory: Mapping[str, Any]) -> None:
    for field_name in (
        "selected_max_items",
        "selected_content_max_tokens",
        "agentcli_candidate_top_k_per_tier",
        "agentcli_maintenance_interval_tasks",
        "mem0_search_limit",
    ):
        _require_positive_int(memory.get(field_name), field_name=f"memory.{field_name}")
    if memory["selected_max_items"] > 20:
        raise ValueError("memory.selected_max_items cannot exceed 20")
    if memory["selected_content_max_tokens"] > 1800:
        raise ValueError("memory.selected_content_max_tokens cannot exceed 1800")
    generation_temperature = _required_float(
        memory.get("generation_temperature"),
        field_name="memory.generation_temperature",
    )
    if generation_temperature < 0:
        raise ValueError("memory.generation_temperature must be non-negative")
    generation_top_p = _required_float(
        memory.get("generation_top_p"), field_name="memory.generation_top_p"
    )
    if not 0 < generation_top_p <= 1:
        raise ValueError("memory.generation_top_p must be in (0, 1]")


def _reject_secret_or_absolute_values(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.casefold().replace("-", "_")
            if any(marker in normalized for marker in _SECRET_FIELD_MARKERS):
                raise ValueError(f"tracked config cannot contain secret field: {path}.{key_text}")
            _reject_secret_or_absolute_values(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_secret_or_absolute_values(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _looks_like_absolute_path(value):
        raise ValueError(f"tracked config cannot contain an absolute path: {path}")


def _looks_like_absolute_path(value: str) -> bool:
    if value.startswith("~"):
        return True
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _hash_mapping(value: Mapping[str, str], *, field_name: str) -> dict[str, str]:
    normalized = _string_mapping(value, field_name=field_name)
    for key, item in normalized.items():
        require_sha256(item, field_name=f"{field_name}[{key}]")
    return normalized


def _require_relative_path(value: str, *, field_name: str) -> None:
    _require_non_empty(value, field_name=field_name)
    if _looks_like_absolute_path(value):
        raise ValueError(f"{field_name} must be repo-relative")


def _require_non_empty(value: Any, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _required_string(value: Any, *, field_name: str) -> str:
    _require_non_empty(value, field_name=field_name)
    return str(value)


def _required_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_positive_int(value: Any, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative_int(value: Any, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _required_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _required_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _required_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _string_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} values must be non-empty strings")
        normalized[str(key)] = item
    return normalized


def _string_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return normalized


def _sequence_of_strings(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of strings")
    return _string_tuple(value, field_name=field_name)


def _sequence_of_ints(value: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of integers")
    return tuple(_required_int(item, field_name=field_name) for item in value)


__all__ = [
    "BenchmarkSuiteConfig",
    "MEMORY_BENCHMARK_CONFIG_SCHEMA_VERSION",
    "MEMORY_BENCHMARK_PROTOCOL_SCHEMA_VERSION",
    "MemoryBenchmarkConfig",
    "MemoryBenchmarkProtocol",
    "REQUIRED_ARMS",
    "REQUIRED_BENCHMARKS",
    "REQUIRED_ENVIRONMENT_VARIABLES",
    "backend_config_hash",
    "canonical_config_bytes",
    "load_memory_benchmark_config",
]
