#!/usr/bin/env python3
"""Register a legacy LLaMA-Factory SFT adapter as OPD M0."""

from __future__ import annotations

import argparse
import json

from my_agent.training.sft_registration import register_sft_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--opd-config", required=True)
    parser.add_argument("--chat-template", default="model_default")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = register_sft_checkpoint(
        trainer_output=args.trainer_output,
        output=args.output,
        base_model=args.base_model,
        base_revision=args.base_revision,
        tokenizer_revision=args.tokenizer_revision,
        opd_config=args.opd_config,
        chat_template=args.chat_template,
        dtype=args.dtype,
        device=args.device,
    )
    print(json.dumps({
        "adapter": str(result.adapter_dir),
        "identity_manifest": str(result.identity_manifest_path),
        "training_manifest": str(result.training_manifest_path),
        "policy_identity_hash": result.identity.identity_hash,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
