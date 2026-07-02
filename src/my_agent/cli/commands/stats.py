from __future__ import annotations

import argparse
import json
import sys

from my_agent.cli.common import CliContext
from my_agent.observability.stats import collect_trace_stats, format_trace_stats
from my_agent.observability.trace_metrics import collect_trace_metrics, format_trace_metrics


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("stats", help="Summarize one trace file or a directory of JSONL traces.")
    parser.add_argument("--trace", required=True, help="Trace JSONL file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively aggregate child trace files.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON stats.")
    parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    _ = ctx
    try:
        stats = collect_trace_metrics(args.trace, recursive=True) if args.recursive else collect_trace_stats(args.trace)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_trace_metrics(stats) if args.recursive else format_trace_stats(stats))
    return 0
