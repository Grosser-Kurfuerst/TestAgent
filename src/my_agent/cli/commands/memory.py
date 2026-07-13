"""CLI registration for deterministic OPD-Evolver memory maintenance."""

from __future__ import annotations

import argparse
import math

from my_agent.cli.common import CliContext
from my_agent.cli.memory_maintenance import run_maintenance_command

def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage OPD-Evolver memory artifacts.",
    )
    memory_subparsers = memory_parser.add_subparsers(
        dest="memory_command",
        required=True,
    )
    parser = memory_subparsers.add_parser(
        "maintain",
        help="Plan or apply deterministic single-project memory maintenance.",
    )
    parser.add_argument("--memory-dir", required=True, help="Memory store directory.")
    parser.add_argument(
        "--memory-project-key",
        required=True,
        type=_nonempty_text,
        help="Exact project or stream key to maintain.",
    )
    parser.add_argument(
        "--attribution",
        help="Attribution JSONL (default: <memory-dir>/memory_attribution.jsonl when present).",
    )
    parser.add_argument(
        "--output",
        help="Plan JSON output (default: <memory-dir>/maintenance_plan.json).",
    )
    parser.add_argument("--plan", help="Existing reviewed plan JSON to apply.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan without mutating memory (default).")
    mode.add_argument("--apply", action="store_true", help="Apply the reviewed or generated plan.")
    parser.add_argument(
        "--as-of",
        help="ISO-8601 planning time (default: current UTC date at 00:00:00).",
    )
    parser.add_argument(
        "--trace-output",
        help="Trace JSONL (default: <memory-dir>/maintenance_trace.jsonl).",
    )
    parser.add_argument(
        "--history-output",
        help="Audit history JSONL (default: <memory-dir>/maintenance_history.jsonl).",
    )
    parser.add_argument(
        "--backup-dir",
        help="Backup directory (default: <memory-dir>/maintenance_backups).",
    )
    parser.add_argument(
        "--lock-timeout-seconds",
        type=_nonnegative_float,
        default=30.0,
        help="Process-lock timeout in seconds (default: 30).",
    )
    parser.add_argument("--delete-value-threshold", type=float)
    parser.add_argument("--delete-min-confidence", type=float)
    parser.add_argument("--delete-min-candidate-count", type=int)
    parser.add_argument("--stale-after-days", type=int)
    parser.add_argument("--merge-threshold", type=float)
    parser.add_argument("--merge-max-cluster-size", type=int)
    parser.add_argument("--promote-value-threshold", type=float)
    parser.add_argument("--promote-min-confidence", type=float)
    parser.add_argument("--promote-min-selected-count", type=int)
    parser.add_argument("--max-promotions", type=int)
    parser.set_defaults(_handler=handle)


def _nonempty_text(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("value must not be empty")
    return normalized


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return parsed


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    _ = ctx
    return run_maintenance_command(args)
