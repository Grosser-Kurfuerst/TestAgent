from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Mapping, Sequence

from my_agent.config import AgentConfig
from my_agent.cancellation import CancellationToken
from my_agent.data import (
    build_humaneval,
    build_mbpp,
    build_swebench_lite,
    export_alpaca,
    local_tasks_to_sft,
    swebench_to_sft,
    traces_to_sft,
)
from my_agent.indexer import RepoIndexer
from my_agent.mcp.manager import McpServerManagerPool
from my_agent.mcp.observability import format_mcp_disabled, format_mcp_logs, format_mcp_status
from my_agent.evaluation.manifest_benchmark import run_manifest_benchmark
from my_agent.evaluation.trace_metrics import collect_trace_metrics, format_trace_metrics
from my_agent.runtime import run_agent
from my_agent.stats import collect_trace_stats, format_trace_stats
from my_agent.tools import RepoTools
from my_agent.ui import AgentRepl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_FILE = PROJECT_ROOT / "examples" / "tasks" / "sample_task.json"


def load_task(path: str | Path = DEFAULT_TASK_FILE) -> dict[str, Any]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Task file must contain one JSON object.")
    for key in ("repo", "task"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Task file requires a non-empty {key!r} field.")
    return payload


def format_task(task: dict[str, Any]) -> str:
    lines = [
        f"id: {task.get('id', 'unknown')}",
        f"source: {task.get('source', 'local')}",
        f"repo: {task['repo']}",
        f"task: {task['task']}",
        f"test_command: {task.get('test_command') or 'not configured'}",
    ]
    return "\n".join(lines)


def _positive_top_k(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top_k must be >= 1.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("top_k must be >= 1.")
    return parsed


def _positive_max_steps(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max_steps must be >= 1.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("max_steps must be >= 1.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal coding-agent scaffold.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_task_parser = subparsers.add_parser("load-task", help="Load and print a task manifest.")
    load_task_parser.add_argument("--task-file", default=str(DEFAULT_TASK_FILE), help="Path to a task JSON file.")

    index_parser = subparsers.add_parser("index", help="Preview repository context without calling an LLM.")
    index_parser.add_argument("--repo", required=True, help="Target repository path.")
    index_parser.add_argument("--query", default="", help="Optional retrieval query.")
    index_parser.add_argument("--top-k", type=_positive_top_k, default=8, help="Number of retrieved files.")

    retrieve_parser = subparsers.add_parser("retrieve", help="Run lightweight lexical retrieval over a repository.")
    retrieve_parser.add_argument("--repo", required=True, help="Target repository path.")
    retrieve_parser.add_argument("--query", required=True, help="Search query.")
    retrieve_parser.add_argument("--top-k", type=_positive_top_k, default=5, help="Number of retrieved files.")

    run_parser = subparsers.add_parser("run", help="Run the coding agent runtime.")
    run_parser.add_argument("--task-file", default=str(DEFAULT_TASK_FILE), help="Path to a task JSON file.")
    run_parser.add_argument("--repo", help="Override repository path from the task file.")
    run_parser.add_argument("--task", help="Override task text from the task file.")
    run_parser.add_argument("--test-command", help="Override test command from the task file.")
    run_parser.add_argument("--max-steps", type=_positive_max_steps, help="Maximum actor tool calls.")
    run_parser.add_argument("--trace-dir", help="Directory for JSONL traces.")
    run_hitl = run_parser.add_mutually_exclusive_group()
    run_hitl.add_argument("--hitl", dest="hitl", action="store_true", default=None, help="Enable HITL approvals.")
    run_hitl.add_argument("--no-hitl", dest="hitl", action="store_false", help="Disable HITL approvals.")
    run_parser.add_argument(
        "--mode",
        choices=("react", "plan", "team", "auto"),
        default=None,
        help="Execution mode. Default: AGENTCLI_AGENT_MODE or auto.",
    )

    chat_parser = subparsers.add_parser("chat", help="Start the interactive ReAct shell.")
    chat_parser.add_argument("--repo", required=True, help="Target repository path.")
    chat_parser.add_argument("--trace-dir", help="Directory for JSONL traces.")
    chat_parser.add_argument("--test-command", help="Default test command for /plan and task runs.")
    chat_hitl = chat_parser.add_mutually_exclusive_group()
    chat_hitl.add_argument("--hitl", dest="hitl", action="store_true", default=None, help="Enable HITL approvals.")
    chat_hitl.add_argument("--no-hitl", dest="hitl", action="store_false", help="Disable HITL approvals.")
    chat_parser.add_argument(
        "--mode",
        choices=("react", "plan", "team", "auto"),
        default=None,
        help="Interactive execution mode. Default: AGENTCLI_AGENT_MODE or auto.",
    )
    chat_parser.add_argument("--no-banner", action="store_true", help="Do not print the startup banner.")

    stats_parser = subparsers.add_parser("stats", help="Summarize one trace file or a directory of JSONL traces.")
    stats_parser.add_argument("--trace", required=True, help="Trace JSONL file or directory.")
    stats_parser.add_argument("--recursive", action="store_true", help="Recursively aggregate child trace files.")
    stats_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON stats.")

    manifest_parser = subparsers.add_parser("eval-manifest", help="Run manifest-based agent capability evaluation.")
    manifest_parser.add_argument("--tasks", required=True, help="Task manifest JSONL or JSON path.")
    manifest_parser.add_argument("--output-dir", required=True, help="Directory for results, work repos, and traces.")
    manifest_parser.add_argument(
        "--mode",
        choices=("react", "plan", "team", "auto"),
        default="auto",
        help="Execution mode for evaluated tasks.",
    )
    manifest_parser.add_argument("--max-steps", type=_positive_max_steps, help="Maximum agent steps per task.")
    manifest_parser.add_argument("--command-timeout", type=int, help="Timeout in seconds for evaluator test commands.")
    manifest_parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override for evaluator commands. May be repeated.",
    )

    config_parser = subparsers.add_parser("config", help="Print resolved local configuration.")
    config_parser.add_argument("--check-api-key", action="store_true", help="Validate provider and API key settings.")

    tools_parser = subparsers.add_parser("tools", help="Inspect and validate dynamically registered tools.")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command", required=True)
    tools_list_parser = tools_subparsers.add_parser("list", help="List tools available for a repository.")
    tools_list_parser.add_argument("--repo", required=True, help="Target repository path.")
    tools_validate_parser = tools_subparsers.add_parser("validate", help="Validate tool configuration for a repository.")
    tools_validate_parser.add_argument("--repo", required=True, help="Target repository path.")

    mcp_parser = subparsers.add_parser("mcp", help="Inspect MCP server status and logs.")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", required=True)
    mcp_status_parser = mcp_subparsers.add_parser("status", help="Show MCP server status for a repository.")
    mcp_status_parser.add_argument("--repo", required=True, help="Target repository path.")
    mcp_logs_parser = mcp_subparsers.add_parser("logs", help="Show recent MCP server stderr lines.")
    mcp_logs_parser.add_argument("server", help="MCP server name.")
    mcp_logs_parser.add_argument("--repo", required=True, help="Target repository path.")
    mcp_reload_parser = mcp_subparsers.add_parser("reload", help="Reload MCP servers for a repository.")
    mcp_reload_parser.add_argument("--repo", required=True, help="Target repository path.")

    _add_data_commands(subparsers)

    return parser


def _add_data_commands(subparsers: Any) -> None:
    def add_limit_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--limit", type=int, default=50, help="Maximum number of dataset rows to process.")

    def add_output_dir_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", required=True, help="Output directory for generated data.")

    mbpp_parser = subparsers.add_parser("build-mbpp", help="Build MBPP task repos and SFT samples.")
    add_limit_arg(mbpp_parser)
    add_output_dir_arg(mbpp_parser)
    mbpp_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")

    he_parser = subparsers.add_parser("build-humaneval", help="Build HumanEval task repos and SFT samples.")
    add_limit_arg(he_parser)
    add_output_dir_arg(he_parser)
    he_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")

    swe_parser = subparsers.add_parser("build-swebench", help="Build SWE-bench Lite task manifests.")
    add_limit_arg(swe_parser)
    add_output_dir_arg(swe_parser)
    swe_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")

    swe2sft_parser = subparsers.add_parser("swebench-to-sft", help="Convert SWE-bench manifests to SFT samples.")
    swe2sft_parser.add_argument("--input", required=True, help="SWE-bench Lite task JSONL file.")
    swe2sft_parser.add_argument("--output", required=True, help="Output SFT JSONL file.")
    swe2sft_parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip malformed input records and include them in the report.",
    )

    tasks2sft_parser = subparsers.add_parser(
        "tasks-to-sft",
        help="Convert local task manifests to SFT strategy samples.",
    )
    tasks2sft_parser.add_argument(
        "--input",
        required=True,
        help="Task manifest JSONL file (can also be a single JSON object).",
    )
    tasks2sft_parser.add_argument("--output", required=True, help="Output SFT JSONL file.")
    tasks2sft_parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip malformed input records and include them in the report.",
    )

    trace2sft_parser = subparsers.add_parser("traces-to-sft", help="Convert agent trace JSONL files to SFT samples.")
    trace2sft_parser.add_argument("--input", required=True, help="Trace JSONL file or directory of trace files.")
    trace2sft_parser.add_argument("--output", required=True, help="Output SFT JSONL file.")
    trace2sft_parser.add_argument("--strict", action="store_true", help="Fail on malformed trace JSONL records.")

    alpaca_parser = subparsers.add_parser(
        "export-alpaca",
        help="Convert SFT JSONL files to LLaMA-Factory alpaca format.",
    )
    alpaca_parser.add_argument("--inputs", nargs="*", default=[], help="SFT JSONL files to include.")
    alpaca_parser.add_argument("--output-dir", required=True, help="Directory for alpaca outputs.")
    alpaca_parser.add_argument("--train-ratio", type=float, default=0.95, help="Train split ratio (default 0.95).")
    alpaca_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    alpaca_parser.add_argument("--system-prompt", help="Override the default system prompt.")
    alpaca_parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip malformed SFT records and include them in dataset_stats.json.",
    )


def _section(title: str, body: str) -> str:
    return f"# {title}\n{body}"


DATA_COMMANDS = {
    "build-mbpp",
    "build-humaneval",
    "build-swebench",
    "swebench-to-sft",
    "tasks-to-sft",
    "traces-to-sft",
    "export-alpaca",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "load-task":
        print(format_task(load_task(args.task_file)))
        return 0

    if args.command == "index":
        snapshot = RepoIndexer(args.repo).snapshot(query=args.query, top_k=args.top_k)
        print(_section("Repository tree", snapshot.tree))
        print()
        print(_section("Symbol index", snapshot.symbols))
        print()
        print(_section("Retrieval notes", snapshot.retrieval_notes))
        print()
        print(_section("Project rules", snapshot.project_rules))
        print()
        print(_section("Important file previews", snapshot.file_summaries))
        return 0

    if args.command == "retrieve":
        print(RepoIndexer(args.repo).retrieve(query=args.query, top_k=args.top_k))
        return 0

    if args.command == "run":
        task_payload = load_task(args.task_file)
        try:
            config = AgentConfig.from_env(env=_tool_environment_overrides(os.environ), require_env_file=False)
            config = _with_hitl_flag(config, args.hitl)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        repo_path = _resolve_repo_path(args.repo or task_payload["repo"])
        test_command = args.test_command if args.test_command is not None else task_payload.get("test_command")
        trace_dir = _resolve_trace_dir(args.trace_dir, config.trace_dir)
        cancellation_token = CancellationToken()
        interrupted = False
        result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

        def run_in_background() -> None:
            try:
                result_queue.put(
                    (
                        "ok",
                        run_agent(
                            repo_path=repo_path,
                            task=args.task or task_payload["task"],
                            test_command=test_command,
                            config=config,
                            max_steps=args.max_steps,
                            trace_dir=trace_dir,
                            mode=args.mode,
                            cancellation_token=cancellation_token,
                            event_sink=_run_event_sink,
                        ),
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - thread boundary relays errors to main.
                result_queue.put(("error", exc))

        worker = threading.Thread(target=run_in_background, name="agentcli-run", daemon=True)
        worker.start()
        try:
            try:
                result_kind, result_payload = result_queue.get()
            except KeyboardInterrupt:
                interrupted = True
                cancellation_token.cancel("keyboard_interrupt")
                try:
                    result_kind, result_payload = result_queue.get(timeout=max(0, config.tool_shutdown_grace_seconds))
                except (KeyboardInterrupt, Empty):
                    print("Cancelled.", file=sys.stderr)
                    return 130
            if result_kind == "error":
                if isinstance(result_payload, KeyboardInterrupt):
                    cancellation_token.cancel("keyboard_interrupt")
                    print("Cancelled.", file=sys.stderr)
                    return 130
                if isinstance(result_payload, (RuntimeError, ValueError)):
                    print(f"Error: {result_payload}", file=sys.stderr)
                    return 1
                if isinstance(result_payload, BaseException):
                    raise result_payload
                raise RuntimeError(result_payload)
            final_state = result_payload
            print(_section("Plan", final_state.plan))
            print()
            print(_section("Review", final_state.review))
            print()
            print(_section("Final summary", final_state.final_answer))
            print()
            print(f"Trace: {final_state.trace_path}")
            if interrupted:
                return 130
            if getattr(final_state, "stop_reason", "") in {
                "plan_failed",
                "plan_validation_failed",
                "plan_cancelled",
                "team_failed",
                "team_validation_failed",
                "team_cancelled",
                "team_planner_failed",
                "context_over_budget",
            }:
                return 1
            return 0
        finally:
            _close_mcp_servers()

    if args.command == "chat":
        try:
            config = AgentConfig.from_env(env=_tool_environment_overrides(os.environ), require_env_file=False)
            config = _with_hitl_flag(config, args.hitl)
            repo_path = _resolve_repo_path(args.repo)
            trace_dir = _resolve_trace_dir(args.trace_dir, config.trace_dir)
            try:
                repl = AgentRepl(
                    repo_path=repo_path,
                    config=config,
                    trace_dir=trace_dir,
                    mode=args.mode or config.agent_mode,
                    test_command=args.test_command,
                )
                return repl.run(show_banner=not args.no_banner)
            finally:
                _close_mcp_servers()
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.command == "stats":
        try:
            stats = (
                collect_trace_metrics(args.trace, recursive=True)
                if args.recursive
                else collect_trace_stats(args.trace)
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(format_trace_metrics(stats) if args.recursive else format_trace_stats(stats))
        return 0

    if args.command == "eval-manifest":
        try:
            config = AgentConfig.from_env(env=_tool_environment_overrides(os.environ), require_env_file=False)
            result = run_manifest_benchmark(
                tasks_path=args.tasks,
                output_dir=args.output_dir,
                config=config,
                mode=args.mode,
                max_steps=args.max_steps,
                command_timeout=args.command_timeout,
                env=_parse_env_overrides(args.env),
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        finally:
            _close_mcp_servers()
        print(result.render())
        return 0

    if args.command == "config":
        try:
            config = AgentConfig.from_env(env=_tool_environment_overrides(os.environ))
            if args.check_api_key:
                config.require_api_key()
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "provider": config.provider,
                    "base_url": config.base_url,
                    "model": config.model,
                    "temperature": config.temperature,
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
                    "memory_short_term_tokens": config.memory_short_term_tokens,
                    "memory_short_term_entries": config.memory_short_term_entries,
                    "memory_context_tokens": config.memory_context_tokens,
                    "memory_retrieval_limit": config.memory_retrieval_limit,
                    "memory_compression_trigger_ratio": config.memory_compression_trigger_ratio,
                    "memory_retain_recent_turns": config.memory_retain_recent_turns,
                    "memory_map_chunk_size": config.memory_map_chunk_size,
                    "memory_tool_result_chars": config.memory_tool_result_chars,
                    "memory_auto_extract": config.memory_auto_extract,
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
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "tools":
        return _handle_tools_command(args)

    if args.command == "mcp":
        return _handle_mcp_command(args)

    data_exit = _handle_data_command(args)
    if data_exit is not None:
        return data_exit

    parser.error(f"Unknown command: {args.command}")
    return 2


def _handle_data_command(args: argparse.Namespace) -> int | None:
    if args.command not in DATA_COMMANDS:
        return None

    try:
        if args.command == "build-mbpp":
            result = build_mbpp(output_dir=args.output_dir, limit=args.limit, split=args.split)
        elif args.command == "build-humaneval":
            result = build_humaneval(output_dir=args.output_dir, limit=args.limit, split=args.split)
        elif args.command == "build-swebench":
            result = build_swebench_lite(output_dir=args.output_dir, limit=args.limit, split=args.split)
        elif args.command == "swebench-to-sft":
            result = swebench_to_sft(input_path=args.input, output_path=args.output, strict=not args.non_strict)
        elif args.command == "tasks-to-sft":
            result = local_tasks_to_sft(input_path=args.input, output_path=args.output, strict=not args.non_strict)
        elif args.command == "traces-to-sft":
            result = traces_to_sft(trace_path=args.input, output_path=args.output, strict=args.strict)
        elif args.command == "export-alpaca":
            inputs = args.inputs if args.inputs else []
            if not inputs:
                print("Error: --inputs is required (one or more SFT JSONL files).", file=sys.stderr)
                return 1
            kwargs: dict[str, object] = {
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "strict": not args.non_strict,
            }
            if args.system_prompt:
                kwargs["system_prompt"] = args.system_prompt
            result = export_alpaca(input_files=inputs, output_dir=args.output_dir, **kwargs)
        else:
            return None
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.command == "build-swebench":
            print("Hint: install data dependencies: uv sync --extra data", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result.render())
    return 0


def _handle_tools_command(args: argparse.Namespace) -> int:
    try:
        config = AgentConfig.from_env(env=_tool_environment_overrides(os.environ), require_env_file=False)
        repo_path = _resolve_repo_path(args.repo)
        tools = RepoTools(repo_path, timeout=config.command_timeout, config=config)

        if args.tools_command == "list":
            print("name\tsource\trisk\tenabled\tdescription")
            for tool in tools.registry.tools:
                print(
                    "\t".join(
                        [
                            tool.spec.name,
                            tool.spec.source,
                            tool.spec.risk.value,
                            "yes" if tool.spec.enabled else "no",
                            tool.spec.description,
                        ]
                    )
                )
            return 0

        if args.tools_command == "validate":
            print(f"Tools validation OK: {len(tools.registry.tools)} tools loaded.")
            return 0

        raise ValueError(f"Unknown tools command: {args.tools_command}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_mcp_servers()


def _handle_mcp_command(args: argparse.Namespace) -> int:
    try:
        config = AgentConfig.from_env(env=_tool_environment_overrides(os.environ), require_env_file=False)
        repo_path = _resolve_repo_path(args.repo)
        if not config.mcp_enabled:
            if args.mcp_command == "logs":
                print(format_mcp_logs(args.server, ["MCP is disabled."]))
            else:
                print(format_mcp_disabled())
            return 0
        manager = McpServerManagerPool.get(repo_path, config, start=args.mcp_command != "reload")

        if args.mcp_command == "status":
            print(format_mcp_status(manager.status_rows()))
            return 0

        if args.mcp_command == "logs":
            print(format_mcp_logs(args.server, manager.logs(args.server)))
            return 0

        if args.mcp_command == "reload":
            manager.reload(max_wait_seconds=config.mcp_startup_wait_seconds)
            print("Reloaded MCP servers.")
            print(format_mcp_status(manager.status_rows()))
            return 0

        raise ValueError(f"Unknown MCP command: {args.mcp_command}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_mcp_servers()


def _tool_environment_overrides(env: Mapping[str, str]) -> dict[str, str]:
    keys = {
        "MY_AGENT_LLM_PROVIDER",
        "MY_AGENT_USE_FAKE_LLM",
        "MY_AGENT_API_KEY",
        "OPENAI_API_KEY",
        "MY_AGENT_BASE_URL",
        "OPENAI_BASE_URL",
        "MY_AGENT_MODEL",
        "MY_AGENT_TEMPERATURE",
        "MY_AGENT_MAX_STEPS",
        "MY_AGENT_COMMAND_TIMEOUT",
        "MY_AGENT_TRACE_DIR",
        "AGENTCLI_ENABLE_PROJECT_TOOLS",
        "AGENTCLI_ENABLE_PROJECT_PLUGINS",
        "AGENTCLI_TOOL_CONFIGS",
        "MY_AGENT_ENABLE_PROJECT_TOOLS",
        "MY_AGENT_ENABLE_PROJECT_PLUGINS",
        "MY_AGENT_TOOL_CONFIGS",
        "MY_AGENT_MAX_ITERATIONS",
        "MY_AGENT_MAX_TOOL_CALLS",
        "MY_AGENT_MAX_ELAPSED_SECONDS",
        "MY_AGENT_TOKEN_BUDGET",
        "MY_AGENT_STAGNATION_WINDOW",
        "MY_AGENT_REPEATED_FAILURE_WINDOW",
        "MY_AGENT_CONTEXT_WINDOW",
        "MY_AGENT_RESPONSE_RESERVE_TOKENS",
        "MY_AGENT_COMPRESSION_BUFFER_TOKENS",
        "AGENTCLI_REPO_CONTEXT_BUDGET_TOKENS",
        "MY_AGENT_REPO_CONTEXT_BUDGET_TOKENS",
        "AGENTCLI_TOOL_SCHEMA_BUDGET_TOKENS",
        "MY_AGENT_TOOL_SCHEMA_BUDGET_TOKENS",
        "MY_AGENT_RETAIN_RECENT_TURNS",
        "MY_AGENT_MAX_TOOL_RESULT_CHARS",
        "MY_AGENT_MAX_SUMMARY_INPUT_CHARS",
        "AGENTCLI_PLAN_TASK_MAX_STEPS",
        "AGENTCLI_PLAN_MAX_TASKS",
        "AGENTCLI_PLAN_MAX_REPLANS",
        "AGENTCLI_AGENT_MODE",
        "MY_AGENT_PLAN_TASK_MAX_STEPS",
        "MY_AGENT_PLAN_MAX_TASKS",
        "MY_AGENT_PLAN_MAX_REPLANS",
        "MY_AGENT_AGENT_MODE",
        "AGENTCLI_TEAM_WORKERS",
        "AGENTCLI_TEAM_MAX_STEPS",
        "AGENTCLI_TEAM_MAX_RETRIES",
        "AGENTCLI_TEAM_STEP_MAX_STEPS",
        "AGENTCLI_TEAM_DEPENDENCY_CONTEXT_CHARS",
        "AGENTCLI_TEAM_PARALLEL",
        "AGENTCLI_TEAM_ALLOW_UNAPPROVED_RESULTS",
        "MY_AGENT_TEAM_WORKERS",
        "MY_AGENT_TEAM_MAX_STEPS",
        "MY_AGENT_TEAM_MAX_RETRIES",
        "MY_AGENT_TEAM_STEP_MAX_STEPS",
        "MY_AGENT_TEAM_DEPENDENCY_CONTEXT_CHARS",
        "MY_AGENT_TEAM_PARALLEL",
        "MY_AGENT_TEAM_ALLOW_UNAPPROVED_RESULTS",
        "AGENTCLI_MEMORY",
        "MY_AGENT_MEMORY",
        "AGENTCLI_MEMORY_DIR",
        "MY_AGENT_MEMORY_DIR",
        "AGENTCLI_MEMORY_SHORT_TERM_TOKENS",
        "MY_AGENT_MEMORY_SHORT_TERM_TOKENS",
        "AGENTCLI_MEMORY_SHORT_TERM_ENTRIES",
        "MY_AGENT_MEMORY_SHORT_TERM_ENTRIES",
        "AGENTCLI_MEMORY_CONTEXT_TOKENS",
        "MY_AGENT_MEMORY_CONTEXT_TOKENS",
        "AGENTCLI_MEMORY_RETRIEVAL_LIMIT",
        "MY_AGENT_MEMORY_RETRIEVAL_LIMIT",
        "AGENTCLI_MEMORY_COMPRESSION_TRIGGER_RATIO",
        "MY_AGENT_MEMORY_COMPRESSION_TRIGGER_RATIO",
        "AGENTCLI_MEMORY_RETAIN_RECENT_TURNS",
        "MY_AGENT_MEMORY_RETAIN_RECENT_TURNS",
        "AGENTCLI_MEMORY_MAP_CHUNK_SIZE",
        "MY_AGENT_MEMORY_MAP_CHUNK_SIZE",
        "AGENTCLI_MEMORY_TOOL_RESULT_CHARS",
        "MY_AGENT_MEMORY_TOOL_RESULT_CHARS",
        "AGENTCLI_MEMORY_AUTO_EXTRACT",
        "MY_AGENT_MEMORY_AUTO_EXTRACT",
        "AGENTCLI_HITL",
        "MY_AGENT_HITL",
        "AGENTCLI_HITL_AUDIT_DIR",
        "MY_AGENT_HITL_AUDIT_DIR",
        "AGENTCLI_HITL_NON_INTERACTIVE",
        "MY_AGENT_HITL_NON_INTERACTIVE",
        "AGENTCLI_HITL_MEDIUM_RISK_MODE",
        "MY_AGENT_HITL_MEDIUM_RISK_MODE",
        "AGENTCLI_HITL_LLM_JUDGE",
        "MY_AGENT_HITL_LLM_JUDGE",
        "AGENTCLI_MAX_PARALLEL_TOOLS",
        "MY_AGENT_MAX_PARALLEL_TOOLS",
        "AGENTCLI_TOOL_BATCH_TIMEOUT_SECONDS",
        "MY_AGENT_TOOL_BATCH_TIMEOUT_SECONDS",
        "AGENTCLI_TOOL_SHUTDOWN_GRACE_SECONDS",
        "MY_AGENT_TOOL_SHUTDOWN_GRACE_SECONDS",
        "AGENTCLI_MAX_PROCESS_OUTPUT_CHARS",
        "MY_AGENT_MAX_PROCESS_OUTPUT_CHARS",
        "AGENTCLI_PLAN_PARALLEL",
        "MY_AGENT_PLAN_PARALLEL",
        "AGENTCLI_PLAN_MAX_PARALLEL_TASKS",
        "MY_AGENT_PLAN_MAX_PARALLEL_TASKS",
        "AGENTCLI_PLAN_TASK_BATCH_TIMEOUT_SECONDS",
        "MY_AGENT_PLAN_TASK_BATCH_TIMEOUT_SECONDS",
        "AGENTCLI_TEAM_STEP_BATCH_TIMEOUT_SECONDS",
        "MY_AGENT_TEAM_STEP_BATCH_TIMEOUT_SECONDS",
        "AGENTCLI_MCP",
        "MY_AGENT_MCP",
        "AGENTCLI_MCP_STARTUP_WAIT_SECONDS",
        "MY_AGENT_MCP_STARTUP_WAIT_SECONDS",
        "AGENTCLI_MCP_INITIALIZE_TIMEOUT_SECONDS",
        "MY_AGENT_MCP_INITIALIZE_TIMEOUT_SECONDS",
        "AGENTCLI_MCP_CALL_TIMEOUT_SECONDS",
        "MY_AGENT_MCP_CALL_TIMEOUT_SECONDS",
        "AGENTCLI_MCP_MAX_STARTUP_WORKERS",
        "MY_AGENT_MCP_MAX_STARTUP_WORKERS",
        "AGENTCLI_MCP_REQUIRE_APPROVAL",
        "MY_AGENT_MCP_REQUIRE_APPROVAL",
        "AGENTCLI_MCP_ENABLE_PROJECT_SERVERS",
        "MY_AGENT_MCP_ENABLE_PROJECT_SERVERS",
    }
    return {key: env[key] for key in keys if key in env}


def _parse_env_overrides(items: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}.")
        key, value = item.split("=", 1)
        if not key.strip():
            raise ValueError("--env key must not be empty.")
        overrides[key.strip()] = value
    return overrides


def _run_event_sink(event: object) -> None:
    event_name = getattr(event, "event", "")
    payload = getattr(event, "payload", {})
    if event_name != "tools.schema_capped" or not isinstance(payload, dict):
        return
    print(_format_tool_schema_capped_status(payload), file=sys.stderr)


def _format_tool_schema_capped_status(payload: dict[str, object]) -> str:
    included = _safe_int(payload.get("included_count"))
    omitted = _safe_int(payload.get("omitted_count"))
    omitted_names = payload.get("omitted")
    names: list[str] = []
    if isinstance(omitted_names, list):
        names = [str(name) for name in omitted_names[:5]]
    suffix = f": {', '.join(names)}" if names else ""
    return f"Tool schema budget applied: {included} exposed, {omitted} omitted{suffix}."


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _close_mcp_servers() -> None:
    from my_agent.mcp.manager import McpServerManagerPool

    McpServerManagerPool.close_all()


def _with_hitl_flag(config: AgentConfig, enabled: bool | None) -> AgentConfig:
    if enabled is None:
        return config
    return replace(config, hitl_enabled=enabled)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_trace_dir(value: str | None, default: Path) -> Path:
    path = Path(value) if value is not None else default
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
