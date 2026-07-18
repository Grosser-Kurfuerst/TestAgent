#!/usr/bin/env python3
"""Verify patched LLaMA-Factory preprocessing against AgentCli rendered SFT."""

from __future__ import annotations

import argparse

from my_agent.sft.parity import verify_llamafactory_parity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--lock", default="integrations/llamafactory/lock.json")
    parser.add_argument("--llamafactory-checkout", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--rendered", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--python", default="python")
    args = parser.parse_args()
    report = verify_llamafactory_parity(
        repository_root=args.repository_root,
        lock_path=args.lock,
        checkout=args.llamafactory_checkout,
        tokenizer_path=args.tokenizer_path,
        rendered_dir=args.rendered,
        export_dir=args.export,
        output_path=args.output,
        sample_limit=args.sample_limit,
        python_executable=args.python,
    )
    print(report.parity_report_hash)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
