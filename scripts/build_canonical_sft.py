#!/usr/bin/env python3
"""Build evidence-joined canonical SFT warm-start data."""

from __future__ import annotations

import argparse

from my_agent.sft.build import (
    build_canonical_sft_dataset,
    load_expert_corrections,
    load_held_out_hashes,
    load_synthetic_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="formal runtime evidence directory")
    parser.add_argument("--output", required=True, help="canonical dataset output directory")
    parser.add_argument("--expert-corrections")
    parser.add_argument("--synthetic-samples")
    parser.add_argument("--held-out-hashes")
    args = parser.parse_args()

    result = build_canonical_sft_dataset(
        source_dir=args.source,
        output_dir=args.output,
        corrections=(
            load_expert_corrections(args.expert_corrections)
            if args.expert_corrections
            else ()
        ),
        synthetic_records=(
            load_synthetic_records(args.synthetic_samples)
            if args.synthetic_samples
            else ()
        ),
        held_out_hashes=(
            load_held_out_hashes(args.held_out_hashes)
            if args.held_out_hashes
            else ()
        ),
    )
    print(result.manifest.dataset_manifest_hash)


if __name__ == "__main__":
    main()
