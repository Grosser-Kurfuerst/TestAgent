from __future__ import annotations

import argparse
import sys

from my_agent.cli.common import CliContext, positive_max_steps
from my_agent.evaluation.manifest_benchmark import run_manifest_benchmark
from my_agent.evaluation.policy_config import configure_evaluation_policy


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval-manifest", help="Run manifest-based agent capability evaluation.")
    parser.add_argument("--tasks", required=True, help="Task manifest JSONL or JSON path.")
    parser.add_argument("--output-dir", required=True, help="Directory for results, work repos, and traces.")
    parser.add_argument(
        "--mode",
        choices=("react", "plan", "team", "auto"),
        default="auto",
        help="Execution mode for evaluated tasks.",
    )
    parser.add_argument("--max-steps", type=positive_max_steps, help="Maximum agent steps per task.")
    parser.add_argument("--command-timeout", type=int, help="Timeout in seconds for evaluator test commands.")
    parser.add_argument("--checkpoint", help="Frozen policy checkpoint or adapter directory.")
    parser.add_argument("--identity-manifest", help="Frozen policy identity manifest.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override for evaluator commands. May be repeated.",
    )
    parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    try:
        config = ctx.config_from_env(require_env_file=False)
        config = configure_evaluation_policy(
            config,
            checkpoint=args.checkpoint,
            identity_manifest=args.identity_manifest,
        )
        result = run_manifest_benchmark(
            tasks_path=args.tasks,
            output_dir=args.output_dir,
            config=config,
            mode=args.mode,
            max_steps=args.max_steps,
            command_timeout=args.command_timeout,
            env=ctx.parse_env_overrides(args.env),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        ctx.close_mcp_servers()
    print(result.render())
    return 0
