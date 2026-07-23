from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest

from my_agent.evaluation.memory_benchmark.protocol import (
    MemoryBenchmarkConfig,
    MemoryBenchmarkProtocol,
    backend_config_hash,
    canonical_config_bytes,
    load_memory_benchmark_config,
)
from my_agent.policy.identity import canonical_sha256


CONFIG_PATH = Path("configs/memory_benchmark/v1.json")


def _hash(label: str) -> str:
    return canonical_sha256({"label": label})


def _protocol() -> MemoryBenchmarkProtocol:
    return MemoryBenchmarkProtocol(
        ordered_task_ids_by_benchmark={
            "lifelong_os": ("os-1", "os-2"),
            "intercode_bash": ("bash-1", "bash-2"),
        },
        task_manifest_hashes={
            "lifelong_os": _hash("lifelong manifest"),
            "intercode_bash": _hash("intercode manifest"),
        },
        source_lock_hash=_hash("source lock"),
        actor_identity_hash=_hash("actor"),
        tools_hash=_hash("tools"),
        evaluator_hashes={
            "lifelong_os": _hash("lifelong evaluator"),
            "intercode_bash": _hash("intercode evaluator"),
        },
        docker_image_digests={
            "lifelong_os": _hash("lifelong image"),
            "intercode_bash": _hash("intercode image"),
        },
        backend_config_hashes={
            "no_memory": _hash("no memory"),
            "agentcli_four_tier": _hash("agentcli"),
            "mem0": _hash("mem0"),
        },
        agentcli_commit="a" * 40,
        uv_lock_hash=_hash("uv.lock"),
        python_version="3.12.4",
        runtime_environment_hash=_hash("runtime"),
        repetition_ids=(42, 43, 44),
        agent_mode="react",
        context_window=32768,
        response_reserve_tokens=4096,
        compression_buffer_tokens=2048,
        repo_context_budget_tokens=6000,
        tool_schema_budget_tokens=8000,
        memory_short_term_tokens=16000,
        memory_context_tokens=1800,
        memory_tool_result_chars=4000,
        max_steps=40,
        command_timeout=120,
        actor_temperature=1.0,
        memory_generation_temperature=1.0,
        memory_generation_top_p=0.95,
        selected_max_items=20,
        selected_content_max_tokens=1800,
        maintenance_interval_tasks=30,
        actor_sampling_seed_supported=False,
    )


def test_same_protocol_has_stable_hash_and_round_trips() -> None:
    protocol = _protocol()
    restored = MemoryBenchmarkProtocol.from_dict(protocol.to_dict())

    assert restored == protocol
    assert restored.protocol_hash == protocol.protocol_hash
    assert "run_seed" not in protocol.to_dict()


def test_protocol_mappings_cannot_mutate_after_hashing() -> None:
    protocol = _protocol()
    original_hash = protocol.protocol_hash

    with pytest.raises(TypeError):
        protocol.backend_config_hashes["mem0"] = _hash("mutated")  # type: ignore[index]

    assert protocol.protocol_hash == original_hash


def test_task_order_changes_protocol_hash() -> None:
    protocol = _protocol()
    changed = replace(
        protocol,
        ordered_task_ids_by_benchmark={
            **protocol.ordered_task_ids_by_benchmark,
            "lifelong_os": ("os-2", "os-1"),
        },
    )

    assert changed.protocol_hash != protocol.protocol_hash


def test_task_manifest_hash_changes_protocol_hash() -> None:
    protocol = _protocol()
    changed = replace(
        protocol,
        task_manifest_hashes={
            **protocol.task_manifest_hashes,
            "lifelong_os": _hash("changed manifest"),
        },
    )

    assert changed.protocol_hash != protocol.protocol_hash


def test_enabled_arm_set_and_backend_config_change_protocol_hash() -> None:
    protocol = _protocol()
    removed_arm = replace(
        protocol,
        backend_config_hashes={
            key: value
            for key, value in protocol.backend_config_hashes.items()
            if key != "mem0"
        },
    )
    changed_arm = replace(
        protocol,
        backend_config_hashes={
            **protocol.backend_config_hashes,
            "mem0": _hash("changed mem0"),
        },
    )

    assert removed_arm.protocol_hash != protocol.protocol_hash
    assert changed_arm.protocol_hash != protocol.protocol_hash


def test_duplicate_task_ids_and_invalid_budgets_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate task IDs"):
        replace(
            _protocol(),
            ordered_task_ids_by_benchmark={
                "lifelong_os": ("os-1", "os-1"),
                "intercode_bash": ("bash-1",),
            },
        )
    with pytest.raises(ValueError, match="selected_max_items"):
        replace(_protocol(), selected_max_items=21)
    with pytest.raises(ValueError, match="max_steps"):
        replace(_protocol(), max_steps=0)


def test_tracked_v1_config_loads_and_canonical_round_trips() -> None:
    config = load_memory_benchmark_config(CONFIG_PATH)
    canonical_payload = json.loads(canonical_config_bytes(config))

    assert config.enabled_arms == ("agentcli_four_tier", "mem0", "no_memory")
    assert config.benchmarks["lifelong_os"].source_split == "train"
    assert MemoryBenchmarkConfig.from_dict(canonical_payload) == config


def test_config_rejects_absolute_paths_and_secret_fields() -> None:
    payload = load_memory_benchmark_config(CONFIG_PATH).to_dict()
    absolute = deepcopy(payload)
    absolute["output_root"] = "/home/example/results"
    with pytest.raises(ValueError, match="absolute path"):
        MemoryBenchmarkConfig.from_dict(absolute)

    secret = deepcopy(payload)
    secret["arms"]["mem0"]["api_key"] = "not-allowed"
    with pytest.raises(ValueError, match="secret field"):
        MemoryBenchmarkConfig.from_dict(secret)


def test_config_rejects_invalid_budget() -> None:
    payload = load_memory_benchmark_config(CONFIG_PATH).to_dict()
    payload["memory"]["selected_content_max_tokens"] = 1801

    with pytest.raises(ValueError, match="1800"):
        MemoryBenchmarkConfig.from_dict(payload)


def test_backend_config_hash_is_canonical_and_secret_free() -> None:
    first = backend_config_hash({"mode": "formal", "top_k": 50})
    second = backend_config_hash({"top_k": 50, "mode": "formal"})

    assert first == second
    with pytest.raises(ValueError, match="secret field"):
        backend_config_hash({"api_key": "not-allowed"})
