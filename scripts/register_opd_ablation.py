#!/usr/bin/env python3
"""Bind a trained checkpoint to one verified paper-ablation recipe."""

from __future__ import annotations

import argparse
import json

from my_agent.evaluation.opd_evaluation import write_ablation_manifest
from my_agent.opd_ablation import PAPER_ABLATIONS
from my_agent.training.recollection import load_trained_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ablation", required=True, choices=PAPER_ABLATIONS)
    args = parser.parse_args()
    checkpoint = load_trained_checkpoint(args.checkpoint, label=args.ablation)
    path = write_ablation_manifest(
        checkpoint,
        ablation=args.ablation,
    )
    print(json.dumps({
        "ablation": args.ablation,
        "checkpoint_identity_hash": checkpoint.identity.identity_hash,
        "ablation_manifest": str(path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
