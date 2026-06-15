from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from my_agent.config import AgentConfig
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

    chat_parser = subparsers.add_parser("chat", help="Start the interactive ReAct shell.")
    chat_parser.add_argument("--repo", required=True, help="Target repository path.")
    chat_parser.add_argument("--trace-dir", help="Directory for JSONL traces.")
    chat_parser.add_argument("--no-banner", action="store_true", help="Do not print the startup banner.")

    stats_parser = subparsers.add_parser("stats", help="Summarize one trace file or a directory of JSONL traces.")
    stats_parser.add_argument("--trace", required=True, help="Trace JSONL file or directory.")
    stats_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON stats.")

    config_parser = subparsers.add_parser("config", help="Print resolved local configuration.")
    config_parser.add_argument("--check-api-key", action="store_true", help="Validate provider and API key settings.")

    tools_parser = subparsers.add_parser("tools", help="Inspect and validate dynamically registered tools.")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command", required=True)
    tools_list_parser = tools_subparsers.add_parser("list", help="List tools available for a repository.")
    tools_list_parser.add_argument("--repo", required=True, help="Target repository path.")
    tools_validate_parser = tools_subparsers.add_parser("validate", help="Validate tool configuration for a repository.")
    tools_validate_parser.add_argument("--repo", required=True, help="Target repository path.")

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
            config = AgentConfig.from_env(env=_tool_environment_overrides(os.environ))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        repo_path = _resolve_repo_path(args.repo or task_payload["repo"])
        test_command = args.test_command if args.test_command is not None else task_payload.get("test_command")
        trace_dir = _resolve_trace_dir(args.trace_dir, config.trace_dir)
        try:
            final_state = run_agent(
                repo_path=repo_path,
                task=args.task or task_payload["task"],
                test_command=test_command,
                config=config,
                max_steps=args.max_steps if args.max_steps is not None else config.max_steps,
                trace_dir=trace_dir,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(_section("Plan", final_state.plan))
        print()
        print(_section("Review", final_state.review))
        print()
        print(_section("Final summary", final_state.final_answer))
        print()
        print(f"Trace: {final_state.trace_path}")
        return 0

    if args.command == "chat":
        try:
            config = AgentConfig.from_env(env=_tool_environment_overrides(os.environ))
            repo_path = _resolve_repo_path(args.repo)
            trace_dir = _resolve_trace_dir(args.trace_dir, config.trace_dir)
            return AgentRepl(repo_path=repo_path, config=config, trace_dir=trace_dir).run(show_banner=not args.no_banner)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.command == "stats":
        try:
            stats = collect_trace_stats(args.trace)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(format_trace_stats(stats))
        return 0

    if args.command == "config":
        try:
            config = AgentConfig.from_env()
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
                    "max_tool_result_chars": config.max_tool_result_chars,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "tools":
        return _handle_tools_command(args)

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
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

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


def _tool_environment_overrides(env: Mapping[str, str]) -> dict[str, str]:
    keys = {
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
        "MY_AGENT_RETAIN_RECENT_TURNS",
        "MY_AGENT_MAX_TOOL_RESULT_CHARS",
        "MY_AGENT_MAX_SUMMARY_INPUT_CHARS",
    }
    return {key: env[key] for key in keys if key in env}


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_trace_dir(value: str | None, default: Path) -> Path:
    path = Path(value) if value is not None else default
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
