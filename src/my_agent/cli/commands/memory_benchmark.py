from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from my_agent.cli.common import CliContext
from my_agent.config import AgentConfig
from my_agent.context import AgentContextManager
from my_agent.evaluation.memory_benchmark.adapters.docker_runtime import (
    BenchmarkActionState,
    DockerRuntime,
    benchmark_action_tool_config,
    benchmark_action_tools_hash,
    write_benchmark_action_files,
)
from my_agent.evaluation.memory_benchmark.adapters.intercode_bash import (
    InterCodeBashAdapter,
)
from my_agent.evaluation.memory_benchmark.adapters.lifelong_os import LifelongOSAdapter
from my_agent.evaluation.memory_benchmark.adapters.smoke import run_smoke_benchmark
from my_agent.evaluation.memory_benchmark.backends import (
    AgentCliFourTierBackend,
    Mem0Backend,
    NoMemoryBackend,
    memory_stream_project_key,
)
from my_agent.evaluation.memory_benchmark.contracts import BenchmarkTask
from my_agent.evaluation.memory_benchmark.external_memory import localize_mem0_config
from my_agent.evaluation.memory_benchmark.protocol import (
    MemoryBenchmarkConfig,
    MemoryBenchmarkProtocol,
    backend_config_hash,
    canonical_config_bytes,
    load_memory_benchmark_config,
)
from my_agent.evaluation.memory_benchmark.runner import (
    MemoryBenchmarkStreamResult,
    run_memory_benchmark_stream,
)
from my_agent.evaluation.memory_benchmark.source_lock import load_source_lock
from my_agent.evaluation.policy_config import (
    configure_evaluation_policy,
    validate_evaluation_policy_identity,
)
from my_agent.llm.types import Message
from my_agent.memory.token import estimate_tokens
from my_agent.policy.identity import (
    PolicyIdentity,
    canonical_json_bytes,
    canonical_sha256,
    load_policy_identity_manifest,
    require_matching_policy_identity,
)
from my_agent.tools import RepoTools


PREPARED_SUITE_SCHEMA_VERSION = "memory-benchmark-prepared-suite-v1"
PREFLIGHT_SCHEMA_VERSION = "memory-benchmark-preflight-v1"
STREAM_SUMMARY_SCHEMA_VERSION = "memory-benchmark-stream-summary-v1"

_BENCHMARK_ORDER = ("lifelong_os", "intercode_bash")
_ARM_ORDER = ("no_memory", "agentcli_four_tier", "mem0")
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "password",
    "client_secret",
    "credential",
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "memory-benchmark",
        help="Run the long-term memory comparison workflow.",
    )
    commands = parser.add_subparsers(dest="memory_benchmark_command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Build deterministic local task manifests from locked benchmark sources.",
    )
    prepare.add_argument("--config", required=True)
    prepare.set_defaults(_handler=handle)

    preflight = commands.add_parser(
        "preflight",
        help="Freeze and validate one immutable memory benchmark run protocol.",
    )
    _add_policy_arguments(preflight)
    preflight.add_argument("--run-dir", required=True)
    _add_development_selection_arguments(preflight)
    preflight.set_defaults(_handler=handle)

    smoke = commands.add_parser(
        "smoke",
        help="Run the deterministic eight-task memory functionality smoke stream.",
    )
    _add_policy_arguments(smoke)
    smoke.add_argument("--output-dir")
    smoke.set_defaults(_handler=handle)

    run = commands.add_parser(
        "run",
        help="Run preflighted benchmark streams without overwriting prior results.",
    )
    _add_policy_arguments(run)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--arms", required=True)
    _add_development_selection_arguments(run)
    run.set_defaults(_handler=handle)

    report = commands.add_parser(
        "report",
        help="Generate the paired comparison report for a completed run.",
    )
    report.add_argument("--run-dir", required=True)
    report.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    try:
        command = str(args.memory_benchmark_command)
        if command == "prepare":
            result = prepare_memory_benchmark_data(
                config_path=args.config,
                env=ctx.env,
            )
        elif command == "preflight":
            result = preflight_memory_benchmark(
                config_path=args.config,
                run_dir=args.run_dir,
                checkpoint=args.checkpoint,
                identity_manifest=args.identity_manifest,
                base_config=ctx.config_from_env(require_env_file=False),
                env=ctx.env,
                benchmarks=args.benchmarks,
                limit=args.limit,
            )
        elif command == "smoke":
            result = _run_smoke(args, ctx)
        elif command == "run":
            result = run_preflighted_memory_benchmark(
                config_path=args.config,
                run_dir=args.run_dir,
                checkpoint=args.checkpoint,
                identity_manifest=args.identity_manifest,
                seed=args.seed,
                arms=args.arms,
                base_config=ctx.config_from_env(require_env_file=False),
                env=ctx.env,
                benchmarks=args.benchmarks,
                limit=args.limit,
            )
        else:
            raise RuntimeError(
                "memory-benchmark report is available after Iteration 11"
            )
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        ctx.close_mcp_servers()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    status = result.get("status")
    return 0 if status in {None, "passed", "completed"} else 1


def prepare_memory_benchmark_data(
    *,
    config_path: str | Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    config_file, repo_root, config = _load_config_context(config_path)
    checkout_roots = _checkout_roots(config, env)
    source_lock = load_source_lock(
        repo_root / config.source_lock_path,
        checkout_roots=checkout_roots,
    )
    data_root = _data_root(config, env, repo_root)
    tasks_by_benchmark = _load_official_tasks(
        config=config,
        source_lock=source_lock,
        checkout_roots=checkout_roots,
        limits={name: config.benchmarks[name].limit for name in _BENCHMARK_ORDER},
        run_id="prepare",
        runtime_root=data_root / ".prepare-runtime",
    )
    config_hash = _config_hash(config)
    source_lock_hash = canonical_sha256(source_lock)
    suite = _suite_manifest_payload(
        config=config,
        tasks_by_benchmark=tasks_by_benchmark,
        config_hash=config_hash,
        source_lock_hash=source_lock_hash,
        pilot=False,
    )
    for benchmark, tasks in tasks_by_benchmark.items():
        _write_jsonl_atomic(
            data_root / benchmark / "tasks.jsonl",
            (task.to_dict() for task in tasks),
        )
    _write_json_atomic(data_root / "suite_manifest.json", suite)
    return {
        "status": "completed",
        "config": str(config_file),
        "data_root": str(data_root),
        "suite_manifest": str(data_root / "suite_manifest.json"),
        "source_lock_hash": source_lock_hash,
        "benchmarks": {
            name: {
                "task_count": len(tasks),
                "task_manifest_hash": _task_manifest_hash(tasks),
            }
            for name, tasks in tasks_by_benchmark.items()
        },
    }


def preflight_memory_benchmark(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    checkpoint: str | Path,
    identity_manifest: str | Path,
    base_config: AgentConfig,
    env: Mapping[str, str],
    benchmarks: str | Sequence[str] | None = None,
    limit: int | None = None,
    docker_runtime: DockerRuntime | None = None,
    policy_identity_validator: Callable[
        [AgentConfig, PolicyIdentity], PolicyIdentity
    ] = validate_evaluation_policy_identity,
) -> dict[str, Any]:
    config_file, repo_root, config = _load_config_context(config_path)
    selected_benchmarks, limits, pilot = _selection(config, benchmarks, limit)
    checkout_roots = _checkout_roots(config, env)
    source_lock = load_source_lock(
        repo_root / config.source_lock_path,
        checkout_roots=checkout_roots,
    )
    prepared_suite = _load_prepared_suite(config, env, repo_root)
    config_hash = _config_hash(config)
    if prepared_suite.get("config_hash") != config_hash:
        raise ValueError("prepared suite config hash does not match the current config")
    source_lock_hash = canonical_sha256(source_lock)
    if prepared_suite.get("source_lock_hash") != source_lock_hash:
        raise ValueError("prepared suite source lock hash does not match the current lock")

    official_tasks = _load_official_tasks(
        config=config,
        source_lock=source_lock,
        checkout_roots=checkout_roots,
        limits=limits,
        run_id="preflight",
        runtime_root=Path(run_dir).expanduser().resolve().parent / ".preflight-runtime",
    )
    prepared_tasks = _selected_prepared_tasks(
        prepared_suite,
        selected_benchmarks=selected_benchmarks,
        limits=limits,
    )
    if official_tasks != prepared_tasks:
        raise ValueError("prepared benchmark tasks do not match the locked official sources")

    configured_agent = _configure_benchmark_agent(
        base_config,
        config,
        checkpoint=checkpoint,
        identity_manifest=identity_manifest,
        env=env,
    )
    expected_identity = load_policy_identity_manifest(identity_manifest)
    identity = policy_identity_validator(configured_agent, expected_identity)
    require_matching_policy_identity(expected_identity, identity)
    mem0_enabled = "mem0" in config.enabled_arms
    mem0_config = _load_mem0_config(config, env) if mem0_enabled else {}
    mem0_version = _validate_mem0_config(mem0_config) if mem0_enabled else ""
    backend_configs = _backend_configs(
        config,
        embedding_revision=configured_agent.embedding_revision,
        mem0_config=mem0_config,
        mem0_version=mem0_version,
        arms=config.enabled_arms,
    )
    backend_hashes = {
        arm: backend_config_hash(payload)
        for arm, payload in backend_configs.items()
    }

    runtime = docker_runtime or DockerRuntime()
    docker_version = runtime.preflight()
    sources = _mapping(source_lock.get("sources"), "source lock sources")
    docker_digests: dict[str, str] = {}
    evaluator_hashes: dict[str, str] = {}
    for benchmark in selected_benchmarks:
        suite = config.benchmarks[benchmark]
        source = _mapping(sources.get(suite.source), f"source {suite.source}")
        image = _required_string(source.get("container_image"), "container_image")
        expected_digest = _required_string(
            source.get("container_digest"), "container_digest"
        )
        actual_digest = runtime.inspect_image(image)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"Docker image digest mismatch for {benchmark}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        docker_digests[benchmark] = actual_digest
        hashes = {
            _required_string(task.evaluator_spec.get("hash"), "evaluator hash")
            for task in official_tasks[benchmark]
        }
        if len(hashes) != 1:
            raise ValueError(f"benchmark {benchmark} has inconsistent evaluator hashes")
        evaluator_hashes[benchmark] = hashes.pop()
        _validate_evaluator_entrypoint(
            _required_string(source.get("evaluator_entrypoint"), "evaluator_entrypoint"),
            checkout_root=checkout_roots[suite.source],
        )

    context_check = _validate_context_budget(configured_agent, config)
    agentcli_commit = _git_clean_commit(repo_root)
    uv_lock_hash = _sha256_file(repo_root / "uv.lock")
    python_version = ".".join(str(item) for item in sys.version_info[:3])
    runtime_environment_hash = canonical_sha256(
        {
            "schema_version": "memory-benchmark-runtime-v1",
            "python_version": python_version,
            "agentcli_commit": agentcli_commit,
            "uv_lock_hash": uv_lock_hash,
        }
    )
    protocol = MemoryBenchmarkProtocol(
        ordered_task_ids_by_benchmark={
            name: tuple(task.task_id for task in tasks)
            for name, tasks in official_tasks.items()
        },
        task_manifest_hashes={
            name: _task_manifest_hash(tasks)
            for name, tasks in official_tasks.items()
        },
        source_lock_hash=source_lock_hash,
        actor_identity_hash=identity.identity_hash,
        tools_hash=benchmark_action_tools_hash(),
        evaluator_hashes=evaluator_hashes,
        docker_image_digests=docker_digests,
        backend_config_hashes=backend_hashes,
        agentcli_commit=agentcli_commit,
        uv_lock_hash=uv_lock_hash,
        python_version=python_version,
        runtime_environment_hash=runtime_environment_hash,
        repetition_ids=config.seeds,
        agent_mode=str(config.runtime["agent_mode"]),
        context_window=int(config.runtime["context_window"]),
        response_reserve_tokens=int(config.runtime["response_reserve_tokens"]),
        compression_buffer_tokens=int(config.runtime["compression_buffer_tokens"]),
        repo_context_budget_tokens=int(config.runtime["repo_context_budget_tokens"]),
        tool_schema_budget_tokens=int(config.runtime["tool_schema_budget_tokens"]),
        memory_short_term_tokens=int(config.runtime["memory_short_term_tokens"]),
        memory_context_tokens=int(config.runtime["memory_context_tokens"]),
        memory_tool_result_chars=int(config.runtime["memory_tool_result_chars"]),
        max_steps=int(config.runtime["max_steps"]),
        command_timeout=int(config.runtime["command_timeout_seconds"]),
        actor_temperature=float(config.runtime["actor_temperature"]),
        memory_generation_temperature=float(config.memory["generation_temperature"]),
        memory_generation_top_p=float(config.memory["generation_top_p"]),
        selected_max_items=int(config.memory["selected_max_items"]),
        selected_content_max_tokens=int(
            config.memory["selected_content_max_tokens"]
        ),
        maintenance_interval_tasks=int(
            config.memory["agentcli_maintenance_interval_tasks"]
        ),
        actor_sampling_seed_supported=config.actor_sampling_seed_supported,
        pilot=pilot,
    )
    run_path = Path(run_dir).expanduser().resolve()
    _validate_preflight_run_dir(run_path, enabled_arms=config.enabled_arms)
    _require_disk_space(run_path)
    run_id = _run_id(run_path, protocol.protocol_hash)
    selected_suite = _suite_manifest_payload(
        config=config,
        tasks_by_benchmark=official_tasks,
        config_hash=config_hash,
        source_lock_hash=source_lock_hash,
        pilot=pilot,
    )
    preflight_payload = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "run_id": run_id,
        "protocol_hash": protocol.protocol_hash,
        "config_hash": config_hash,
        "benchmarks": list(selected_benchmarks),
        "limits": dict(limits),
        "pilot": pilot,
        "checkpoint_identity_hash": identity.identity_hash,
        "checks": {
            "prepared_suite": "passed",
            "source_revisions": "passed",
            "docker": "passed",
            "docker_server_version": docker_version,
            "mem0": "passed",
            "context_budget": context_check,
            "disk_space": "passed",
        },
    }
    _write_preflight_artifacts(
        run_path,
        protocol=protocol,
        source_lock=source_lock,
        suite_manifest=selected_suite,
        backend_configs=backend_configs,
        backend_hashes=backend_hashes,
        preflight=preflight_payload,
    )
    return dict(preflight_payload)


def run_preflighted_memory_benchmark(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    checkpoint: str | Path,
    identity_manifest: str | Path,
    seed: int,
    arms: str | Sequence[str],
    base_config: AgentConfig,
    env: Mapping[str, str],
    benchmarks: str | Sequence[str] | None = None,
    limit: int | None = None,
    stream_runner: Callable[..., MemoryBenchmarkStreamResult] = run_memory_benchmark_stream,
    policy_identity_validator: Callable[
        [AgentConfig, PolicyIdentity], PolicyIdentity
    ] = validate_evaluation_policy_identity,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    config_file, repo_root, config = _load_config_context(config_path)
    run_path = Path(run_dir).expanduser().resolve()
    preflight = _load_mapping(run_path / "preflight.json")
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported memory benchmark preflight schema")
    if preflight.get("status") != "passed":
        raise ValueError("memory benchmark preflight has not passed")
    protocol = MemoryBenchmarkProtocol.from_dict(
        _load_mapping(run_path / "protocol.json")
    )
    protocol_hash_text = (run_path / "protocol_hash.txt").read_text(
        encoding="utf-8"
    ).strip()
    if protocol_hash_text != protocol.protocol_hash:
        raise ValueError("protocol hash file does not match protocol.json")
    if preflight.get("protocol_hash") != protocol.protocol_hash:
        raise ValueError("preflight protocol hash does not match protocol.json")
    if preflight.get("config_hash") != _config_hash(config):
        raise ValueError("run config no longer matches preflight")
    if protocol.agentcli_commit != _git_clean_commit(repo_root):
        raise ValueError("AgentCli commit no longer matches preflight protocol")
    if protocol.uv_lock_hash != _sha256_file(repo_root / "uv.lock"):
        raise ValueError("uv.lock no longer matches preflight protocol")
    if seed not in protocol.repetition_ids:
        raise ValueError(f"seed {seed} is not part of the preflight protocol")

    expected_benchmarks = tuple(str(item) for item in preflight.get("benchmarks", ()))
    requested_benchmarks, requested_limits, requested_pilot = _selection(
        config,
        benchmarks,
        limit,
        default_benchmarks=expected_benchmarks,
        default_limits={
            str(name): int(value)
            for name, value in _mapping(preflight.get("limits"), "preflight limits").items()
        },
    )
    if requested_benchmarks != expected_benchmarks:
        raise ValueError("run benchmark selection does not match preflight")
    expected_limits = {
        str(name): int(value)
        for name, value in _mapping(preflight.get("limits"), "preflight limits").items()
    }
    if requested_limits != expected_limits or requested_pilot != protocol.pilot:
        raise ValueError("run limit/pilot selection does not match preflight")
    requested_arms = _parse_named_selection(arms, allowed=config.enabled_arms, label="arm")

    checkout_roots = _checkout_roots(config, env)
    source_lock = load_source_lock(
        repo_root / config.source_lock_path,
        checkout_roots=checkout_roots,
    )
    if canonical_sha256(source_lock) != protocol.source_lock_hash:
        raise ValueError("source lock no longer matches preflight protocol")
    tasks_by_benchmark = _load_official_tasks(
        config=config,
        source_lock=source_lock,
        checkout_roots=checkout_roots,
        limits=requested_limits,
        run_id=str(preflight["run_id"]),
        runtime_root=run_path / ".runtime",
    )
    _validate_tasks_against_protocol(tasks_by_benchmark, protocol)
    configured_agent = _configure_benchmark_agent(
        base_config,
        config,
        checkpoint=checkpoint,
        identity_manifest=identity_manifest,
        env=env,
    )
    expected_identity = load_policy_identity_manifest(identity_manifest)
    identity = policy_identity_validator(configured_agent, expected_identity)
    require_matching_policy_identity(expected_identity, identity)
    if identity.identity_hash != protocol.actor_identity_hash:
        raise ValueError("policy identity no longer matches preflight protocol")
    mem0_requested = "mem0" in requested_arms
    mem0_config = _load_mem0_config(config, env) if mem0_requested else {}
    mem0_version = _validate_mem0_config(mem0_config) if mem0_requested else ""
    current_backend_configs = _backend_configs(
        config,
        embedding_revision=configured_agent.embedding_revision,
        mem0_config=mem0_config,
        mem0_version=mem0_version,
        arms=requested_arms,
    )

    streams: list[dict[str, Any]] = []
    for arm in requested_arms:
        backend_hash = _load_backend_config_hash(run_path, arm)
        if protocol.backend_config_hashes.get(arm) != backend_hash:
            raise ValueError(f"backend config hash mismatch for arm {arm}")
        current_backend_hash = backend_config_hash(current_backend_configs[arm])
        if current_backend_hash != backend_hash:
            raise ValueError(f"current backend config no longer matches preflight for arm {arm}")
        for benchmark in requested_benchmarks:
            target = run_path / "arms" / arm / f"seed_{seed}" / benchmark
            if target.exists():
                raise FileExistsError(
                    f"memory benchmark stream output already exists: {target}"
                )
            stream_memory_dir = target / "memory"
            stream_project_key = memory_stream_project_key(
                run_id=str(preflight["run_id"]),
                seed=seed,
                benchmark=benchmark,
                arm=arm,
            )
            backend = _build_backend(
                arm,
                config=config,
                stream_memory_dir=stream_memory_dir,
                stream_project_key=stream_project_key,
                mem0_config=mem0_config,
            )
            adapter = _build_adapter(
                benchmark,
                config=config,
                source_lock=source_lock,
                checkout_roots=checkout_roots,
                run_id=str(preflight["run_id"]),
                runtime_root=run_path / ".runtime",
            )
            result = stream_runner(
                tasks=tasks_by_benchmark[benchmark],
                adapter=adapter,
                backend=backend,
                base_config=configured_agent,
                output_dir=target,
                run_id=str(preflight["run_id"]),
                seed=seed,
                stream_memory_dir=stream_memory_dir,
                stream_project_key=stream_project_key,
                protocol_hash=protocol.protocol_hash,
                actor_identity_hash=protocol.actor_identity_hash,
                tools_hash=protocol.tools_hash,
                backend_config_hash=backend_hash,
                actor_sampling_seed_supported=protocol.actor_sampling_seed_supported,
                actor_sampling_seed_effective=(
                    seed if protocol.actor_sampling_seed_supported else None
                ),
                max_steps=protocol.max_steps,
                command_timeout=protocol.command_timeout,
            )
            summary = _stream_summary(result, protocol_hash=protocol.protocol_hash)
            _write_json_atomic(target / "summary.json", summary)
            streams.append(summary)
    return {
        "status": "completed",
        "config": str(config_file),
        "run_dir": str(run_path),
        "protocol_hash": protocol.protocol_hash,
        "seed": seed,
        "pilot": protocol.pilot,
        "streams": streams,
    }


def _run_smoke(args: argparse.Namespace, ctx: CliContext) -> dict[str, Any]:
    config_file, repo_root, config = _load_config_context(args.config)
    base_config = _configure_benchmark_agent(
        ctx.config_from_env(require_env_file=False),
        config,
        checkpoint=args.checkpoint,
        identity_manifest=args.identity_manifest,
        env=ctx.env,
    )
    identity = load_policy_identity_manifest(args.identity_manifest)
    data_root = _data_root(config, ctx.env, repo_root)
    mem0_config = _load_mem0_config(config, ctx.env)
    source_lock = load_source_lock(repo_root / config.source_lock_path)
    sources = _mapping(source_lock.get("sources"), "source lock sources")
    smoke_source_name = str(config.smoke.get("container_source", "lifelong_agent_bench"))
    smoke_source = _mapping(
        sources.get(smoke_source_name),
        f"source lock source {smoke_source_name}",
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else repo_root
        / config.output_root
        / f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    return run_smoke_benchmark(
        base_config=base_config,
        output_dir=output_dir,
        data_root=data_root,
        actor_identity_hash=identity.identity_hash,
        mem0_config=mem0_config,
        container_image=_required_string(
            smoke_source.get("container_image"), "smoke container image"
        ),
        container_digest=_required_string(
            smoke_source.get("container_digest"), "smoke container digest"
        ),
        seed=42,
        maintenance_interval_tasks=int(
            config.smoke.get("maintenance_interval_tasks", 4)
        ),
    )


def _configure_benchmark_agent(
    base_config: AgentConfig,
    config: MemoryBenchmarkConfig,
    *,
    checkpoint: str | Path,
    identity_manifest: str | Path,
    env: Mapping[str, str],
) -> AgentConfig:
    configured = configure_evaluation_policy(
        base_config,
        checkpoint=checkpoint,
        identity_manifest=identity_manifest,
    )
    embedding_revision_env = config.embedding["revision_env"]
    embedding_revision = str(env.get(embedding_revision_env, "")).strip()
    if not embedding_revision:
        raise ValueError(
            f"memory benchmark requires frozen embedding revision via "
            f"{embedding_revision_env}"
        )
    runtime = config.runtime
    memory = config.memory
    return replace(
        configured,
        context_window=int(runtime["context_window"]),
        response_reserve_tokens=int(runtime["response_reserve_tokens"]),
        compression_buffer_tokens=int(runtime["compression_buffer_tokens"]),
        repo_context_budget_tokens=int(runtime["repo_context_budget_tokens"]),
        tool_schema_budget_tokens=int(runtime["tool_schema_budget_tokens"]),
        memory_short_term_tokens=int(runtime["memory_short_term_tokens"]),
        memory_context_tokens=int(runtime["memory_context_tokens"]),
        memory_tool_result_chars=int(runtime["memory_tool_result_chars"]),
        max_steps=int(runtime["max_steps"]),
        command_timeout=int(runtime["command_timeout_seconds"]),
        temperature=float(runtime["actor_temperature"]),
        memory_evolver_generation_temperature=float(memory["generation_temperature"]),
        memory_evolver_generation_top_p=float(memory["generation_top_p"]),
        memory_evolver_candidate_top_k_per_tier=int(
            memory["agentcli_candidate_top_k_per_tier"]
        ),
        memory_evolver_selected_max_items=int(memory["selected_max_items"]),
        memory_evolver_selection_prompt_tokens=int(
            memory["selected_content_max_tokens"]
        ),
        memory_evolver_maintenance_interval_tasks=int(
            memory["agentcli_maintenance_interval_tasks"]
        ),
        memory_evolver_maintenance_enabled=bool(
            memory["agentcli_maintenance_enabled"]
        ),
        memory_evolver_retrieval_backend=str(memory["agentcli_retrieval_backend"]),
        memory_evolver_selection_backend=str(memory["agentcli_selection_backend"]),
        embedding_model=config.embedding["model"],
        embedding_revision=embedding_revision,
        context_window_explicit=True,
        response_reserve_tokens_explicit=True,
        compression_buffer_tokens_explicit=True,
        repo_context_budget_tokens_explicit=True,
        tool_schema_budget_tokens_explicit=True,
        memory_short_term_tokens_explicit=True,
        memory_context_tokens_explicit=True,
        memory_tool_result_chars_explicit=True,
    )


def _load_config_context(
    config_path: str | Path,
) -> tuple[Path, Path, MemoryBenchmarkConfig]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_memory_benchmark_config(config_file)
    if len(config_file.parents) < 3:
        raise ValueError("memory benchmark config must live under configs/<name>")
    repo_root = config_file.parents[2]
    return config_file, repo_root, config


def _checkout_roots(
    config: MemoryBenchmarkConfig,
    env: Mapping[str, str],
) -> dict[str, Path]:
    mappings = {
        "lifelong_agent_bench": config.environment["lifelong_agent_bench_root"],
        "intercode": config.environment["intercode_root"],
    }
    roots: dict[str, Path] = {}
    for source, env_name in mappings.items():
        value = str(env.get(env_name, "")).strip()
        if not value:
            raise ValueError(f"memory benchmark requires {env_name}")
        roots[source] = Path(value).expanduser().resolve()
    return roots


def _data_root(
    config: MemoryBenchmarkConfig,
    env: Mapping[str, str],
    repo_root: Path,
) -> Path:
    env_name = config.environment["data_root"]
    value = str(env.get(env_name, "")).strip()
    return (
        Path(value).expanduser().resolve()
        if value
        else (repo_root / "data" / "memory_benchmark").resolve()
    )


def _load_mem0_config(
    config: MemoryBenchmarkConfig,
    env: Mapping[str, str],
) -> Mapping[str, Any]:
    env_name = config.environment["mem0_config"]
    value = str(env.get(env_name, "")).strip()
    if not value:
        raise ValueError(f"memory benchmark requires Mem0 config via {env_name}")
    return _load_mapping(Path(value).expanduser().resolve())


def _load_official_tasks(
    *,
    config: MemoryBenchmarkConfig,
    source_lock: Mapping[str, Any],
    checkout_roots: Mapping[str, Path],
    limits: Mapping[str, int],
    run_id: str,
    runtime_root: Path,
) -> dict[str, tuple[BenchmarkTask, ...]]:
    tasks_by_benchmark: dict[str, tuple[BenchmarkTask, ...]] = {}
    for benchmark in limits:
        adapter = _build_adapter(
            benchmark,
            config=config,
            source_lock=source_lock,
            checkout_roots=checkout_roots,
            run_id=run_id,
            runtime_root=runtime_root,
        )
        tasks = tuple(adapter.load_tasks(limit=int(limits[benchmark])))
        tasks_by_benchmark[benchmark] = tasks
    return tasks_by_benchmark


def _build_adapter(
    benchmark: str,
    *,
    config: MemoryBenchmarkConfig,
    source_lock: Mapping[str, Any],
    checkout_roots: Mapping[str, Path],
    run_id: str,
    runtime_root: Path,
) -> LifelongOSAdapter | InterCodeBashAdapter:
    suite = config.benchmarks[benchmark]
    sources = _mapping(source_lock.get("sources"), "source lock sources")
    source = _mapping(sources.get(suite.source), f"source {suite.source}")
    data_path = checkout_roots[suite.source] / _required_string(
        source.get("task_data_path"), "task_data_path"
    )
    kwargs = {
        "task_data_path": data_path,
        "source": source,
        "run_id": run_id,
        "runtime_root": runtime_root,
        "command_timeout_seconds": int(config.runtime["command_timeout_seconds"]),
        "max_output_chars": int(config.runtime["memory_tool_result_chars"]),
    }
    if benchmark == "lifelong_os":
        return LifelongOSAdapter(**kwargs)
    if benchmark == "intercode_bash":
        return InterCodeBashAdapter(**kwargs)
    raise ValueError(f"unsupported memory benchmark: {benchmark}")


def _build_backend(
    arm: str,
    *,
    config: MemoryBenchmarkConfig,
    stream_memory_dir: Path,
    stream_project_key: str,
    mem0_config: Mapping[str, Any],
) -> NoMemoryBackend | AgentCliFourTierBackend | Mem0Backend:
    if arm == "no_memory":
        return NoMemoryBackend(
            stream_memory_dir=stream_memory_dir,
            stream_project_key=stream_project_key,
        )
    if arm == "agentcli_four_tier":
        return AgentCliFourTierBackend(
            stream_memory_dir=stream_memory_dir,
            stream_project_key=stream_project_key,
            maintenance_interval_tasks=int(
                config.memory["agentcli_maintenance_interval_tasks"]
            ),
        )
    if arm == "mem0":
        return Mem0Backend(
            stream_memory_dir=stream_memory_dir,
            stream_project_key=stream_project_key,
            mem0_config=mem0_config,
            search_limit=int(config.memory["mem0_search_limit"]),
            selected_max_items=int(config.memory["selected_max_items"]),
            selected_content_max_tokens=int(
                config.memory["selected_content_max_tokens"]
            ),
        )
    raise ValueError(f"unsupported memory benchmark arm: {arm}")


def _selection(
    config: MemoryBenchmarkConfig,
    benchmarks: str | Sequence[str] | None,
    limit: int | None,
    *,
    default_benchmarks: Sequence[str] | None = None,
    default_limits: Mapping[str, int] | None = None,
) -> tuple[tuple[str, ...], dict[str, int], bool]:
    selected = (
        _parse_named_selection(
            benchmarks,
            allowed=_BENCHMARK_ORDER,
            label="benchmark",
        )
        if benchmarks is not None
        else tuple(default_benchmarks or _BENCHMARK_ORDER)
    )
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer")
    limits: dict[str, int] = {}
    for benchmark in selected:
        formal_limit = config.benchmarks[benchmark].limit
        chosen = (
            limit
            if limit is not None
            else int((default_limits or {}).get(benchmark, formal_limit))
        )
        if chosen > formal_limit:
            raise ValueError(
                f"limit {chosen} exceeds configured limit {formal_limit} for {benchmark}"
            )
        limits[benchmark] = chosen
    pilot = selected != _BENCHMARK_ORDER or any(
        limits[name] != config.benchmarks[name].limit for name in selected
    )
    return selected, limits, pilot


def _parse_named_selection(
    value: str | Sequence[str],
    *,
    allowed: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    selected = tuple(str(item).strip() for item in raw if str(item).strip())
    if not selected:
        raise ValueError(f"{label} selection must not be empty")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{label} selection contains duplicates")
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label} selection: {unknown}")
    return tuple(name for name in allowed if name in selected)


def _suite_manifest_payload(
    *,
    config: MemoryBenchmarkConfig,
    tasks_by_benchmark: Mapping[str, Sequence[BenchmarkTask]],
    config_hash: str,
    source_lock_hash: str,
    pilot: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PREPARED_SUITE_SCHEMA_VERSION,
        "config_hash": config_hash,
        "source_lock_hash": source_lock_hash,
        "pilot": pilot,
        "benchmarks": {
            name: {
                "source": config.benchmarks[name].source,
                "subset": config.benchmarks[name].subset,
                "limit": len(tasks),
                "task_ids": [task.task_id for task in tasks],
                "task_manifest_hash": _task_manifest_hash(tasks),
                "tasks": [task.to_dict() for task in tasks],
            }
            for name, tasks in tasks_by_benchmark.items()
        },
    }


def _load_prepared_suite(
    config: MemoryBenchmarkConfig,
    env: Mapping[str, str],
    repo_root: Path,
) -> Mapping[str, Any]:
    payload = _load_mapping(_data_root(config, env, repo_root) / "suite_manifest.json")
    if payload.get("schema_version") != PREPARED_SUITE_SCHEMA_VERSION:
        raise ValueError("unsupported prepared suite manifest schema")
    return payload


def _selected_prepared_tasks(
    suite: Mapping[str, Any],
    *,
    selected_benchmarks: Sequence[str],
    limits: Mapping[str, int],
) -> dict[str, tuple[BenchmarkTask, ...]]:
    raw_benchmarks = _mapping(suite.get("benchmarks"), "prepared suite benchmarks")
    result: dict[str, tuple[BenchmarkTask, ...]] = {}
    for benchmark in selected_benchmarks:
        payload = _mapping(raw_benchmarks.get(benchmark), f"prepared {benchmark}")
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
            raise ValueError(f"prepared {benchmark} tasks must be an array")
        all_tasks = tuple(
            BenchmarkTask.from_dict(_mapping(item, f"prepared {benchmark} task"))
            for item in raw_tasks
        )
        if payload.get("task_manifest_hash") != _task_manifest_hash(all_tasks):
            raise ValueError(f"prepared {benchmark} task manifest hash mismatch")
        if limits[benchmark] > len(all_tasks):
            raise ValueError(f"prepared {benchmark} does not contain enough tasks")
        result[benchmark] = all_tasks[: limits[benchmark]]
    return result


def _backend_configs(
    config: MemoryBenchmarkConfig,
    *,
    embedding_revision: str,
    mem0_config: Mapping[str, Any],
    mem0_version: str,
    arms: Sequence[str],
) -> dict[str, dict[str, Any]]:
    shared_limits = {
        "selected_max_items": int(config.memory["selected_max_items"]),
        "selected_content_max_tokens": int(
            config.memory["selected_content_max_tokens"]
        ),
    }
    configs = {
        "no_memory": {
            "schema_version": "memory-benchmark-backend-config-v1",
            "arm": "no_memory",
            **dict(config.arms["no_memory"]),
            **shared_limits,
        },
        "agentcli_four_tier": {
            "schema_version": "memory-benchmark-backend-config-v1",
            "arm": "agentcli_four_tier",
            **dict(config.arms["agentcli_four_tier"]),
            **shared_limits,
            "embedding_model": config.embedding["model"],
            "embedding_revision": embedding_revision,
            "candidate_top_k_per_tier": int(
                config.memory["agentcli_candidate_top_k_per_tier"]
            ),
            "retrieval_backend": config.memory["agentcli_retrieval_backend"],
            "selection_backend": config.memory["agentcli_selection_backend"],
            "generation_temperature": float(config.memory["generation_temperature"]),
            "generation_top_p": float(config.memory["generation_top_p"]),
            "maintenance_interval_tasks": int(
                config.memory["agentcli_maintenance_interval_tasks"]
            ),
            "maintenance_enabled": bool(
                config.memory["agentcli_maintenance_enabled"]
            ),
        },
        "mem0": {
            "schema_version": "memory-benchmark-backend-config-v1",
            "arm": "mem0",
            **dict(config.arms["mem0"]),
            **shared_limits,
            "search_limit": int(config.memory["mem0_search_limit"]),
            "package_version": mem0_version,
            "config": _sanitize_external_config(mem0_config),
        },
    }
    unknown = sorted(set(arms) - set(configs))
    if unknown:
        raise ValueError(f"unknown backend config arms: {unknown}")
    return {arm: configs[arm] for arm in _ARM_ORDER if arm in arms}


def _validate_mem0_config(config: Mapping[str, Any]) -> str:
    version = importlib.metadata.version("mem0ai")
    for component in ("llm", "embedder"):
        payload = _mapping(config.get(component), f"Mem0 {component}")
        provider = _required_string(payload.get("provider"), f"Mem0 {component} provider")
        provider_config = _mapping(
            payload.get("config"), f"Mem0 {component} config"
        )
        model = str(
            provider_config.get("model") or provider_config.get("model_name") or ""
        ).strip()
        if not provider or not model:
            raise ValueError(f"Mem0 {component} provider and model must be explicit")
    with tempfile.TemporaryDirectory() as tmp:
        localize_mem0_config(config, Path(tmp) / "mem0")
    return version


def _validate_evaluator_entrypoint(
    entrypoint: str,
    *,
    checkout_root: Path,
) -> None:
    parts = entrypoint.split(".")
    original_sys_path = list(sys.path)
    sys.path.insert(0, str(checkout_root))
    try:
        for module_length in range(len(parts), 0, -1):
            module_name = ".".join(parts[:module_length])
            try:
                target: Any = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                missing = str(exc.name or "")
                if missing and not (
                    module_name == missing or module_name.startswith(f"{missing}.")
                ):
                    raise ImportError(
                        f"evaluator entrypoint dependency is unavailable: {missing}"
                    ) from exc
                continue
            for attribute in parts[module_length:]:
                if not hasattr(target, attribute):
                    raise ImportError(
                        f"evaluator entrypoint attribute not found: {entrypoint}"
                    )
                target = getattr(target, attribute)
            if not callable(target):
                raise ImportError(f"evaluator entrypoint is not callable: {entrypoint}")
            return
    finally:
        sys.path[:] = original_sys_path
    raise ImportError(f"evaluator entrypoint cannot be imported: {entrypoint}")


def _sanitize_external_config(value: Any, *, key: str = "") -> Any:
    normalized_key = key.casefold().replace("-", "_")
    if any(marker in normalized_key for marker in _SECRET_KEY_MARKERS):
        return None
    if isinstance(value, Mapping):
        return {
            str(item_key): sanitized
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
            if (
                sanitized := _sanitize_external_config(
                    item_value,
                    key=str(item_key),
                )
            )
            is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sanitize_external_config(item) for item in value]
    if isinstance(value, str) and (Path(value).is_absolute() or value.startswith("~")):
        return "<run-local>"
    return value


def _validate_context_budget(
    configured_agent: AgentConfig,
    config: MemoryBenchmarkConfig,
) -> dict[str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = root / "repo"
        repo.mkdir()
        write_benchmark_action_files(
            repo,
            BenchmarkActionState(
                container_name="preflight-container",
                runtime_action_log_path=root / "runtime" / "actions.jsonl",
                timeout_seconds=int(config.runtime["command_timeout_seconds"]),
                max_output_chars=int(config.runtime["memory_tool_result_chars"]),
            ),
        )
        task_config = replace(
            configured_agent,
            agent_mode="react",
            enable_project_tools=True,
            tool_config_paths=(),
            enable_project_plugins=False,
            mcp_enabled=False,
            mcp_enable_project_servers=False,
            hitl_enabled=False,
        )
        tools = RepoTools(repo, timeout=task_config.command_timeout, config=task_config).tool_definitions()
        manager = AgentContextManager.from_config(task_config)
        memory_block = _synthetic_memory_block(
            int(config.memory["selected_max_items"]),
            int(config.memory["selected_content_max_tokens"]),
        )
        messages = [
            Message(role="system", content="AgentCli memory benchmark actor."),
            Message(role="system", content=memory_block),
            Message(role="user", content="Execute the benchmark task using benchmark_action."),
        ]
        fixed_with_memory = manager.estimate_tokens(messages, tools)
        trigger = manager.profile.compression_trigger_tokens
        if fixed_with_memory >= trigger:
            raise ValueError(
                "memory benchmark context budget is invalid: "
                f"fixed_with_memory_tokens={fixed_with_memory}, trigger={trigger}"
            )
        project_tools = [
            tool for tool in tools if tool.get("function", {}).get("name") == "benchmark_action"
        ]
        if len(project_tools) != 1:
            raise ValueError("preflight must load exactly one benchmark_action project tool")
        if benchmark_action_tool_config()["tools"][0]["name"] != "benchmark_action":
            raise ValueError("benchmark_action tool config is invalid")
        return {
            "fixed_with_memory_tokens": fixed_with_memory,
            "compression_trigger_tokens": trigger,
            "tool_count": len(tools),
        }


def _synthetic_memory_block(max_items: int, content_budget: int) -> str:
    words: list[str] = []
    while estimate_tokens(" ".join((*words, "memory"))) <= content_budget:
        words.append("memory")
    chunks = [words[index::max_items] for index in range(max_items)]
    blocks = ["Relevant selected experience:"]
    for index, chunk in enumerate(chunks, 1):
        blocks.append(f"[tip:preflight-{index}]\n{' '.join(chunk)}")
    return "\n\n".join(blocks)


def _validate_preflight_run_dir(run_dir: Path, *, enabled_arms: Sequence[str]) -> None:
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise ValueError(f"run dir is not a directory: {run_dir}")
    if any((run_dir / "arms").glob("*/seed_*")) if (run_dir / "arms").exists() else False:
        raise FileExistsError("preflight refuses a run dir containing scored seed output")
    allowed = {
        "protocol.json",
        "protocol_hash.txt",
        "source-lock.json",
        "suite_manifest.json",
        "preflight.json",
    }
    allowed.update(
        f"arms/{arm}/{name}"
        for arm in enabled_arms
        for name in ("backend_config.json", "backend_config_hash.txt")
    )
    unexpected = sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.relative_to(run_dir).as_posix() not in allowed
    )
    if unexpected:
        raise FileExistsError(f"preflight run dir contains unexpected files: {unexpected}")


def _write_preflight_artifacts(
    run_dir: Path,
    *,
    protocol: MemoryBenchmarkProtocol,
    source_lock: Mapping[str, Any],
    suite_manifest: Mapping[str, Any],
    backend_configs: Mapping[str, Mapping[str, Any]],
    backend_hashes: Mapping[str, str],
    preflight: Mapping[str, Any],
) -> None:
    _write_json_immutable(run_dir / "protocol.json", protocol.to_dict())
    _write_bytes_immutable(
        run_dir / "protocol_hash.txt",
        f"{protocol.protocol_hash}\n".encode("utf-8"),
    )
    _write_json_immutable(run_dir / "source-lock.json", source_lock)
    _write_json_immutable(run_dir / "suite_manifest.json", suite_manifest)
    for arm, payload in backend_configs.items():
        _write_json_immutable(run_dir / "arms" / arm / "backend_config.json", payload)
        _write_bytes_immutable(
            run_dir / "arms" / arm / "backend_config_hash.txt",
            f"{backend_hashes[arm]}\n".encode("utf-8"),
        )
    _write_json_immutable(run_dir / "preflight.json", preflight)


def _load_backend_config_hash(run_dir: Path, arm: str) -> str:
    payload = _load_mapping(run_dir / "arms" / arm / "backend_config.json")
    actual = backend_config_hash(payload)
    recorded = (run_dir / "arms" / arm / "backend_config_hash.txt").read_text(
        encoding="utf-8"
    ).strip()
    if actual != recorded:
        raise ValueError(f"backend config artifact hash mismatch for arm {arm}")
    return actual


def _validate_tasks_against_protocol(
    tasks_by_benchmark: Mapping[str, Sequence[BenchmarkTask]],
    protocol: MemoryBenchmarkProtocol,
) -> None:
    if set(tasks_by_benchmark) != set(protocol.ordered_task_ids_by_benchmark):
        raise ValueError("run benchmark set does not match protocol")
    for benchmark, tasks in tasks_by_benchmark.items():
        ids = tuple(task.task_id for task in tasks)
        if ids != protocol.ordered_task_ids_by_benchmark[benchmark]:
            raise ValueError(f"task order no longer matches protocol for {benchmark}")
        if _task_manifest_hash(tasks) != protocol.task_manifest_hashes[benchmark]:
            raise ValueError(f"task manifest hash no longer matches protocol for {benchmark}")


def _stream_summary(
    result: MemoryBenchmarkStreamResult,
    *,
    protocol_hash: str,
) -> dict[str, Any]:
    executions = tuple(result.executions)
    return {
        "schema_version": STREAM_SUMMARY_SCHEMA_VERSION,
        "arm": result.arm,
        "seed": result.seed,
        "benchmark": result.benchmark,
        "task_count": len(executions),
        "resolved": sum(
            1 for execution in executions if execution.task_result.resolved
        ),
        "protocol_hash": protocol_hash,
        "results_path": str(result.results_path),
    }


def _task_manifest_hash(tasks: Sequence[BenchmarkTask]) -> str:
    return canonical_sha256([task.to_dict() for task in tasks])


def _config_hash(config: MemoryBenchmarkConfig) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(canonical_config_bytes(config)).hexdigest()}"


def _git_clean_commit(repo_root: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        raise RuntimeError(f"AgentCli repo is not a Git checkout: {repo_root}")
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError("could not inspect AgentCli worktree status")
    if status.stdout.strip():
        raise RuntimeError("formal memory benchmark preflight requires a clean tracked worktree")
    return commit.stdout.strip()


def _require_disk_space(run_dir: Path) -> None:
    ancestor = run_dir.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free <= 0:
        raise RuntimeError("memory benchmark output filesystem has no free space")


def _run_id(run_dir: Path, protocol_hash: str) -> str:
    base = "".join(
        character if character.isalnum() or character in "_.-" else "-"
        for character in run_dir.name
    ).strip("-.")
    return f"{base or 'memory-benchmark'}-{protocol_hash.removeprefix('sha256:')[:8]}"


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_atomic(path, canonical_json_bytes(dict(payload)) + b"\n")


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]] | Any) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _write_bytes_atomic(path, payload)


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_immutable(path, canonical_json_bytes(dict(payload)) + b"\n")


def _write_bytes_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable preflight artifact differs: {path}")
        return
    _write_bytes_atomic(path, payload)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--identity-manifest", required=True)


def _add_development_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmarks")
    parser.add_argument("--limit", type=int)


__all__ = [
    "add_parser",
    "handle",
    "preflight_memory_benchmark",
    "prepare_memory_benchmark_data",
    "run_preflighted_memory_benchmark",
]
