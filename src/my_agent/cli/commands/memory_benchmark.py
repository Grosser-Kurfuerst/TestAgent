from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from my_agent.cli.common import CliContext
from my_agent.config import AgentConfig
from my_agent.evaluation.memory_benchmark.adapters.smoke import run_smoke_benchmark
from my_agent.evaluation.memory_benchmark.source_lock import load_source_lock
from my_agent.policy.identity import load_policy_identity_manifest


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "memory-benchmark",
        help="Run the long-term memory comparison workflow.",
    )
    commands = parser.add_subparsers(dest="memory_benchmark_command", required=True)
    smoke = commands.add_parser(
        "smoke",
        help="Run the deterministic eight-task memory functionality smoke stream.",
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--checkpoint", required=True)
    smoke.add_argument("--identity-manifest", required=True)
    smoke.add_argument("--output-dir")
    smoke.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    try:
        config_path = Path(args.config).expanduser().resolve()
        benchmark_config = _load_mapping(config_path)
        base_config = _configure_smoke_agent(
            ctx.config_from_env(require_env_file=False),
            benchmark_config,
            checkpoint=args.checkpoint,
            identity_manifest=args.identity_manifest,
            env=ctx.env,
        )
        identity = load_policy_identity_manifest(args.identity_manifest)
        repo_root = config_path.parents[2]
        environment = _mapping(benchmark_config.get("environment"), "environment")
        data_env = str(environment.get("data_root", ""))
        data_root = (
            Path(str(ctx.env[data_env])).expanduser().resolve()
            if data_env and ctx.env.get(data_env)
            else repo_root / "data" / "memory_benchmark"
        )
        mem0_env = str(environment.get("mem0_config", ""))
        if not mem0_env or not ctx.env.get(mem0_env):
            raise ValueError(f"smoke requires Mem0 config via {mem0_env or 'configured environment'}")
        mem0_config = _load_mapping(Path(str(ctx.env[mem0_env])).expanduser().resolve())
        smoke_config = _mapping(benchmark_config.get("smoke"), "smoke")
        source_lock = load_source_lock(repo_root / str(benchmark_config["source_lock_path"]))
        sources = _mapping(source_lock.get("sources"), "source lock sources")
        smoke_source_name = str(
            smoke_config.get("container_source", "lifelong_agent_bench")
        )
        smoke_source = _mapping(
            sources.get(smoke_source_name),
            f"source lock source {smoke_source_name}",
        )
        container_image = str(smoke_source.get("container_image") or "")
        container_digest = str(smoke_source.get("container_digest") or "")
        if not container_image or not container_digest:
            raise ValueError("smoke source lock requires a container image and digest")
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else repo_root
            / str(benchmark_config.get("output_root", "evaluationResults/memory_benchmark"))
            / f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        report = run_smoke_benchmark(
            base_config=base_config,
            output_dir=output_dir,
            data_root=data_root,
            actor_identity_hash=identity.identity_hash,
            mem0_config=mem0_config,
            container_image=container_image,
            container_digest=container_digest,
            seed=42,
            maintenance_interval_tasks=int(
                smoke_config.get("maintenance_interval_tasks", 4)
            ),
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        ctx.close_mcp_servers()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") == "passed" else 1


def _configure_smoke_agent(
    config: AgentConfig,
    benchmark_config: Mapping[str, Any],
    *,
    checkpoint: str | Path,
    identity_manifest: str | Path,
    env: Mapping[str, str],
) -> AgentConfig:
    identity_path = Path(identity_manifest).expanduser().resolve()
    identity = load_policy_identity_manifest(identity_path)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"evaluation checkpoint not found: {checkpoint_path}")
    runtime = _mapping(benchmark_config.get("runtime"), "runtime")
    memory = _mapping(benchmark_config.get("memory"), "memory")
    embedding = _mapping(benchmark_config.get("embedding"), "embedding")
    revision_env = str(embedding.get("revision_env", ""))
    embedding_revision = str(env.get(revision_env, "")).strip()
    if not embedding_revision:
        raise ValueError(f"smoke requires frozen embedding revision via {revision_env}")
    return replace(
        config,
        policy_backend="transformers",
        policy_base_model=identity.base_model,
        policy_base_revision=identity.base_revision,
        policy_tokenizer_revision=identity.tokenizer_revision,
        policy_adapter_path=(
            checkpoint_path if identity.adapter_hash is not None else None
        ),
        policy_identity_manifest=identity_path,
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
        memory_evolver_retrieval_backend=str(memory["agentcli_retrieval_backend"]),
        memory_evolver_selection_backend=str(memory["agentcli_selection_backend"]),
        embedding_model=str(embedding["model"]),
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


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


__all__ = ["add_parser", "handle"]
