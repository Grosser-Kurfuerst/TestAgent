#!/usr/bin/env python3
"""Export canonical AgentCli SFT data to the locked LLaMA-Factory format."""

from __future__ import annotations

import argparse
from pathlib import Path

from my_agent.sft.export import export_llamafactory_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--rendered", required=True)
    parser.add_argument("--lock", default="integrations/llamafactory/lock.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = export_llamafactory_dataset(
        canonical_dir=args.canonical,
        rendered_dir=args.rendered,
        output_dir=args.output,
        lock_path=args.lock,
        repository_root=root,
    )
    print(manifest.export_manifest_hash)


if __name__ == "__main__":
    main()
