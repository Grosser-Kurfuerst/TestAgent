#!/usr/bin/env python3
"""Render canonical SFT samples with the immutable runtime tokenizer/template."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

import yaml

from my_agent.policy.chat_template import CanonicalChatTemplate
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256, hash_artifact_path
from my_agent.sft.rendered import RenderedSFTManifest, RenderedSFTSample
from my_agent.sft.semantic import SemanticSFTSample


_TOKENIZER_NAMES = {
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
_TOKENIZER_PREFIXES = ("added_tokens", "merges", "spiece", "tokenizer", "vocab")
_TOKENIZER_SUFFIXES = (".model", ".tiktoken")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = _load_mapping(config_path, kind="SFT config", yaml_input=True)
    if config.get("schema_version") != "agentcli-sft-warm-start-config-v1":
        raise ValueError("unsupported SFT warm-start config schema")
    model = _mapping(config.get("model"), "model")
    data = _mapping(config.get("data"), "data")
    base_model = _required_string(model.get("name_or_path"), "model.name_or_path")
    tokenizer_revision = _required_string(
        model.get("tokenizer_revision"),
        "model.tokenizer_revision",
    )
    cutoff_len = _positive_int(data.get("cutoff_len"), "data.cutoff_len")

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("SFT rendering requires the 'opd-train' extra") from exc

    tokenizer_snapshot = Path(snapshot_download(
        repo_id=base_model,
        revision=tokenizer_revision,
    ))
    resolved_revision = _resolved_snapshot_revision(tokenizer_snapshot, tokenizer_revision)
    if resolved_revision != tokenizer_revision:
        raise ValueError("downloaded tokenizer revision does not match the immutable config")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=True)
    template = CanonicalChatTemplate(
        tokenizer,
        configured_template=_required_string(model.get("chat_template"), "model.chat_template"),
    )
    tokenizer_hash = hash_artifact_path(
        tokenizer_snapshot,
        include=_is_tokenizer_artifact,
    )

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    semantic_manifest = _load_mapping(
        input_dir / "dataset_manifest.json",
        kind="canonical dataset manifest",
        yaml_input=False,
    )
    semantic_manifest_hash = _manifest_hash(semantic_manifest)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"rendered SFT output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_hashes: dict[str, tuple[str, ...]] = {}
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source = input_dir / f"{split}.jsonl"
        if not source.is_file():
            continue
        samples = _read_semantic_samples(source)
        rendered = tuple(
            RenderedSFTSample.from_semantic(
                sample,
                chat_template=template,
                tokenizer_revision=tokenizer_revision,
                tokenizer_hash=tokenizer_hash,
                cutoff_len=cutoff_len,
            )
            for sample in samples
        )
        destination = output_dir / f"{split}.jsonl"
        destination.write_bytes(b"".join(
            canonical_json_bytes(sample.to_dict()) + b"\n"
            for sample in rendered
        ))
        split_hashes[split] = tuple(sample.rendered_sample_hash for sample in rendered)
        split_counts[split] = len(rendered)
    if not split_hashes:
        raise ValueError("canonical SFT input contains no train/validation/test JSONL files")

    manifest = RenderedSFTManifest.create(
        semantic_dataset_manifest_hash=semantic_manifest_hash,
        tokenizer_revision=tokenizer_revision,
        tokenizer_hash=tokenizer_hash,
        chat_template_hash=template.template_hash,
        cutoff_len=cutoff_len,
        split_rendered_sample_hashes=split_hashes,
    )
    manifest_path = output_dir / "rendered_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest.to_dict()) + b"\n")
    print(json.dumps({
        "rendered_manifest": str(manifest_path),
        "rendered_manifest_hash": manifest.rendered_manifest_hash,
        "semantic_dataset_manifest_hash": semantic_manifest_hash,
        "split_counts": split_counts,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _read_semantic_samples(path: Path) -> tuple[SemanticSFTSample, ...]:
    samples: list[SemanticSFTSample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid canonical SFT JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number}: canonical SFT row must be an object")
        samples.append(SemanticSFTSample.from_dict(payload))
    return tuple(samples)


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    declared = payload.get("dataset_manifest_hash")
    if declared is None:
        return canonical_sha256(dict(payload))
    body = {key: value for key, value in payload.items() if key != "dataset_manifest_hash"}
    actual = canonical_sha256(body)
    if declared != actual:
        raise ValueError("canonical dataset manifest hash mismatch")
    return actual


def _load_mapping(path: Path, *, kind: str, yaml_input: bool) -> Mapping[str, Any]:
    try:
        payload = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_input
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {kind}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{kind} must be an object")
    return payload


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"SFT config {field_name} must be an object")
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _resolved_snapshot_revision(path: Path, fallback: str) -> str:
    if path.parent.name == "snapshots" and path.name:
        return path.name
    return fallback


def _is_tokenizer_artifact(path: Path) -> bool:
    return (
        path.name in _TOKENIZER_NAMES
        or path.name.startswith(_TOKENIZER_PREFIXES)
        or path.name.endswith(_TOKENIZER_SUFFIXES)
    )


if __name__ == "__main__":
    raise SystemExit(main())
