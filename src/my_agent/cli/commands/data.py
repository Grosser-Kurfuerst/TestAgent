from __future__ import annotations

import argparse
import sys

from my_agent.cli.common import CliContext
from my_agent.data import (
    build_humaneval,
    build_mbpp,
    build_swebench_lite,
    export_alpaca,
    local_tasks_to_sft,
    swebench_to_sft,
    traces_to_sft,
)

DATA_COMMANDS = {
    "build-mbpp",
    "build-humaneval",
    "build-swebench",
    "swebench-to-sft",
    "tasks-to-sft",
    "traces-to-sft",
    "export-alpaca",
}


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    def add_limit_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--limit", type=int, default=50, help="Maximum number of dataset rows to process.")

    def add_output_dir_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", required=True, help="Output directory for generated data.")

    mbpp_parser = subparsers.add_parser("build-mbpp", help="Build MBPP task repos and SFT samples.")
    add_limit_arg(mbpp_parser)
    add_output_dir_arg(mbpp_parser)
    mbpp_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")
    mbpp_parser.set_defaults(_handler=handle)

    he_parser = subparsers.add_parser("build-humaneval", help="Build HumanEval task repos and SFT samples.")
    add_limit_arg(he_parser)
    add_output_dir_arg(he_parser)
    he_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")
    he_parser.set_defaults(_handler=handle)

    swe_parser = subparsers.add_parser("build-swebench", help="Build SWE-bench Lite task manifests.")
    add_limit_arg(swe_parser)
    add_output_dir_arg(swe_parser)
    swe_parser.add_argument("--split", default="test", help="HuggingFace dataset split name.")
    swe_parser.set_defaults(_handler=handle)

    swe2sft_parser = subparsers.add_parser("swebench-to-sft", help="Convert SWE-bench manifests to SFT samples.")
    swe2sft_parser.add_argument("--input", required=True, help="SWE-bench Lite task JSONL file.")
    swe2sft_parser.add_argument("--output", required=True, help="Output SFT JSONL file.")
    swe2sft_parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip malformed input records and include them in the report.",
    )
    swe2sft_parser.set_defaults(_handler=handle)

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
    tasks2sft_parser.set_defaults(_handler=handle)

    trace2sft_parser = subparsers.add_parser("traces-to-sft", help="Convert agent trace JSONL files to SFT samples.")
    trace2sft_parser.add_argument("--input", required=True, help="Trace JSONL file or directory of trace files.")
    trace2sft_parser.add_argument("--output", required=True, help="Output SFT JSONL file.")
    trace2sft_parser.add_argument("--strict", action="store_true", help="Fail on malformed trace JSONL records.")
    trace2sft_parser.set_defaults(_handler=handle)

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
    alpaca_parser.set_defaults(_handler=handle)


def handle(args: argparse.Namespace, ctx: CliContext) -> int:
    _ = ctx
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
            raise ValueError(f"Unknown data command: {args.command}")
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
