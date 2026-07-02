from __future__ import annotations

import argparse
import sys

from my_agent.cli.common import CliContext
from my_agent.tools import RepoTools


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("tools", help="Inspect and validate dynamically registered tools.")
    tools_subparsers = parser.add_subparsers(dest="tools_command", required=True)
    list_parser = tools_subparsers.add_parser("list", help="List tools available for a repository.")
    list_parser.add_argument("--repo", required=True, help="Target repository path.")
    list_parser.set_defaults(_handler=handle)
    validate_parser = tools_subparsers.add_parser("validate", help="Validate tool configuration for a repository.")
    validate_parser.add_argument("--repo", required=True, help="Target repository path.")
    validate_parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    try:
        config = ctx.config_from_env(require_env_file=False)
        repo_path = ctx.resolve_repo_path(args.repo)
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
        ctx.close_mcp_servers()
