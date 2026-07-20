from __future__ import annotations

import argparse
import json
import sys

from my_agent.cli.common import CliContext


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("config", help="Print resolved local configuration.")
    parser.add_argument("--check-api-key", action="store_true", help="Validate provider and API key settings.")
    parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    try:
        config = ctx.config_from_env(require_env_file=True)
        if args.check_api_key:
            config.require_api_key()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_config_payload(config), ensure_ascii=False, indent=2))
    return 0


def _config_payload(config: object) -> dict[str, object]:
    return {
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "reasoning_effort": config.reasoning_effort,
        "max_steps": config.max_steps,
        "command_timeout": config.command_timeout,
        "trace_dir": str(config.trace_dir),
        "use_fake_llm": config.use_fake_llm,
        "api_key_configured": bool(config.api_key),
        "tool_config_paths": [str(path) for path in config.tool_config_paths],
        "enable_project_tools": config.enable_project_tools,
        "enable_project_plugins": config.enable_project_plugins,
        "max_iterations": config.max_iterations,
        "max_tool_calls": config.max_tool_calls,
        "max_elapsed_seconds": config.max_elapsed_seconds,
        "token_budget": config.token_budget,
        "context_window": config.context_window,
        "repo_context_budget_tokens": config.repo_context_budget_tokens,
        "tool_schema_budget_tokens": config.tool_schema_budget_tokens,
        "max_tool_result_chars": config.max_tool_result_chars,
        "plan_task_max_steps": config.plan_task_max_steps,
        "plan_max_tasks": config.plan_max_tasks,
        "plan_max_replans": config.plan_max_replans,
        "agent_mode": config.agent_mode,
        "team_worker_count": config.team_worker_count,
        "team_max_steps": config.team_max_steps,
        "team_max_retries": config.team_max_retries,
        "team_step_max_steps": config.team_step_max_steps,
        "team_dependency_context_chars": config.team_dependency_context_chars,
        "team_parallel_enabled": config.team_parallel_enabled,
        "team_allow_unapproved_results": config.team_allow_unapproved_results,
        "memory_enabled": config.memory_enabled,
        "memory_dir": str(config.memory_dir),
        "memory_project_key": config.memory_project_key,
        "memory_short_term_tokens": config.memory_short_term_tokens,
        "memory_short_term_entries": config.memory_short_term_entries,
        "memory_context_tokens": config.memory_context_tokens,
        "memory_retrieval_limit": config.memory_retrieval_limit,
        "memory_compression_trigger_ratio": config.memory_compression_trigger_ratio,
        "memory_retain_recent_turns": config.memory_retain_recent_turns,
        "memory_map_chunk_size": config.memory_map_chunk_size,
        "memory_tool_result_chars": config.memory_tool_result_chars,
        "memory_evolver_mode": config.memory_evolver_mode,
        "memory_evolver_top_k_per_tier": config.memory_evolver_top_k_per_tier,
        "memory_evolver_selected_max_items": config.memory_evolver_selected_max_items,
        "memory_evolver_min_score": config.memory_evolver_min_score,
        "memory_evolver_min_experience_entries": config.memory_evolver_min_experience_entries,
        "memory_evolver_tier_caps": dict(config.memory_evolver_tier_caps),
        "memory_evolver_tier_weights": dict(config.memory_evolver_tier_weights),
        "hitl_enabled": config.hitl_enabled,
        "hitl_audit_dir": str(config.hitl_audit_dir),
        "hitl_non_interactive": config.hitl_non_interactive,
        "hitl_medium_risk_mode": config.hitl_medium_risk_mode,
        "hitl_llm_judge_enabled": config.hitl_llm_judge_enabled,
        "max_parallel_tools": config.max_parallel_tools,
        "tool_batch_timeout_seconds": config.tool_batch_timeout_seconds,
        "tool_shutdown_grace_seconds": config.tool_shutdown_grace_seconds,
        "max_process_output_chars": config.max_process_output_chars,
        "plan_parallel_enabled": config.plan_parallel_enabled,
        "plan_max_parallel_tasks": config.plan_max_parallel_tasks,
        "plan_task_batch_timeout_seconds": config.plan_task_batch_timeout_seconds,
        "team_step_batch_timeout_seconds": config.team_step_batch_timeout_seconds,
        "mcp_enabled": config.mcp_enabled,
        "mcp_startup_wait_seconds": config.mcp_startup_wait_seconds,
        "mcp_initialize_timeout_seconds": config.mcp_initialize_timeout_seconds,
        "mcp_call_timeout_seconds": config.mcp_call_timeout_seconds,
        "mcp_max_startup_workers": config.mcp_max_startup_workers,
        "mcp_require_approval": config.mcp_require_approval,
        "mcp_enable_project_servers": config.mcp_enable_project_servers,
    }
