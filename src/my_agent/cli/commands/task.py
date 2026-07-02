from __future__ import annotations

import argparse

from my_agent.cli.common import CliContext, DEFAULT_TASK_FILE, format_task, load_task


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("load-task", help="Load and print a task manifest.")
    parser.add_argument("--task-file", default=str(DEFAULT_TASK_FILE), help="Path to a task JSON file.")
    parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    _ = ctx
    print(format_task(load_task(args.task_file)))
    return 0
