from __future__ import annotations

import argparse
import sys

from my_agent.cli.common import (
    CliContext,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("chat", help="Start the interactive ReAct shell.")
    parser.add_argument("--repo", required=True, help="Target repository path.")
    parser.add_argument("--trace-dir", help="Directory for JSONL traces.")
    parser.add_argument("--test-command", help="Default test command for /plan and task runs.")
    chat_hitl = parser.add_mutually_exclusive_group()
    chat_hitl.add_argument("--hitl", dest="hitl", action="store_true", default=None, help="Enable HITL approvals.")
    chat_hitl.add_argument("--no-hitl", dest="hitl", action="store_false", help="Disable HITL approvals.")
    parser.add_argument(
        "--mode",
        choices=("react", "plan", "team", "auto"),
        default=None,
        help="Interactive execution mode. Default: AGENTCLI_AGENT_MODE or auto.",
    )
    parser.add_argument("--no-banner", action="store_true", help="Do not print the startup banner.")
    parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    try:
        config = ctx.config_from_env(require_env_file=False)
        config = ctx.with_hitl_flag(config, args.hitl)
        repo_path = ctx.resolve_repo_path(args.repo)
        trace_dir = ctx.resolve_trace_dir(args.trace_dir, config.trace_dir)
        try:
            repl = ctx.agent_repl_cls(
                repo_path=repo_path,
                config=config,
                trace_dir=trace_dir,
                mode=args.mode or config.agent_mode,
                test_command=args.test_command,
            )
            return repl.run(show_banner=not args.no_banner)
        finally:
            ctx.close_mcp_servers()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
