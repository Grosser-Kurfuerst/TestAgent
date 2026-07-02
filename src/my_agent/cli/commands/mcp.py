from __future__ import annotations

import argparse
import sys

from my_agent.cli.common import CliContext
from my_agent.mcp.manager import McpServerManagerPool
from my_agent.mcp.observability import format_mcp_disabled, format_mcp_logs, format_mcp_status


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("mcp", help="Inspect MCP server status and logs.")
    mcp_subparsers = parser.add_subparsers(dest="mcp_command", required=True)
    status_parser = mcp_subparsers.add_parser("status", help="Show MCP server status for a repository.")
    status_parser.add_argument("--repo", required=True, help="Target repository path.")
    status_parser.set_defaults(_handler=handle)
    logs_parser = mcp_subparsers.add_parser("logs", help="Show recent MCP server stderr lines.")
    logs_parser.add_argument("server", help="MCP server name.")
    logs_parser.add_argument("--repo", required=True, help="Target repository path.")
    logs_parser.set_defaults(_handler=handle)
    reload_parser = mcp_subparsers.add_parser("reload", help="Reload MCP servers for a repository.")
    reload_parser.add_argument("--repo", required=True, help="Target repository path.")
    reload_parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    try:
        config = ctx.config_from_env(require_env_file=False)
        repo_path = ctx.resolve_repo_path(args.repo)
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
        ctx.close_mcp_servers()
