#!/usr/bin/env python3
from __future__ import annotations

"""CLI wrapper for base-vs-SFT protocol evaluation."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from my_agent.evaluation.protocol_runner import print_comparison, run_evaluation  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base vs SFT model on protocol metrics.")
    parser.add_argument("--val-data", required=True, help="Validation data in LLaMA-Factory alpaca JSON format.")
    parser.add_argument("--output-dir", default="outputs/sft_protocol_eval", help="Directory for evaluation artifacts.")
    parser.add_argument("--base-model", help="Base model path or HuggingFace model id.")
    parser.add_argument("--adapter-dir", help="LoRA adapter directory or LLaMA-Factory output directory.")
    parser.add_argument("--base-responses", help="Optional JSON response file for base model.")
    parser.add_argument("--sft-responses", help="Optional JSON response file for SFT model.")
    parser.add_argument("--limit", type=int, help="Limit validation samples.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_evaluation(
        val_data=args.val_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        base_responses_path=args.base_responses,
        sft_responses_path=args.sft_responses,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    print_comparison(summary)
    print(f"Results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
