from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

from my_agent.cli.commands import (
    chat,
    config,
    data,
    eval as eval_manifest,
    mcp,
    memory,
    repo,
    run,
    stats,
    task,
    tools,
)
from my_agent.cli.common import CliContext

CommandHandler = Callable[[argparse.Namespace, CliContext], int]

COMMAND_MODULES = (
    task,
    repo,
    run,
    chat,
    stats,
    eval_manifest,
    config,
    tools,
    mcp,
    memory,
    data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal coding-agent scaffold.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in COMMAND_MODULES:
        module.add_parser(subparsers)
    return parser


def _build_context() -> CliContext:
    import my_agent.cli as cli_package

    return CliContext(
        run_agent=cli_package.run_agent,
        agent_repl_cls=cli_package.AgentRepl,
    )


def main(argv: Sequence[str] | None = None, *, ctx: CliContext | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _handler_from_args(args)
    return handler(args, ctx or _build_context())


def _handler_from_args(args: argparse.Namespace) -> CommandHandler:
    handler = getattr(args, "_handler", None)
    if handler is None:
        raise ValueError(f"Unknown command: {getattr(args, 'command', '')}")
    return handler


if __name__ == "__main__":
    raise SystemExit(main())
