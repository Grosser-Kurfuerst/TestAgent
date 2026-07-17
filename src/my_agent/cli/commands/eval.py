from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from my_agent.cli.common import CliContext, positive_max_steps
from my_agent.evaluation.manifest_benchmark import run_manifest_benchmark
from my_agent.policy.identity import load_policy_identity_manifest


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
        if args.identity_manifest:
            identity_path = Path(args.identity_manifest).expanduser().resolve()
            identity = load_policy_identity_manifest(identity_path)
            checkpoint = (
                Path(args.checkpoint).expanduser().resolve()
                if args.checkpoint
                else identity_path.parent
            )
            if identity.adapter_hash is not None and not checkpoint.exists():
                raise FileNotFoundError(f"evaluation checkpoint not found: {checkpoint}")
            config = replace(
                config,
                policy_backend="transformers",
                policy_base_model=identity.base_model,
                policy_base_revision=identity.base_revision,
                policy_tokenizer_revision=identity.tokenizer_revision,
                policy_adapter_path=checkpoint if identity.adapter_hash is not None else None,
                policy_identity_manifest=identity_path,
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
