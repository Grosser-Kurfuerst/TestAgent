from __future__ import annotations

from pathlib import Path
import json

import pytest

from my_agent.policy.chat_template import CanonicalChatTemplate
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.sft.export import (
    LlamaFactoryExportManifest,
    export_llamafactory_dataset,
    load_llamafactory_lock,
    project_semantic_sample,
)
from my_agent.sft.manifest import SFTDatasetManifest
from my_agent.sft.parity import _validate_export_artifacts
from my_agent.sft.rendered import RenderedSFTManifest, RenderedSFTSample
from tests.sft.test_rendered import TOKENIZER_HASH, _NativeToolTokenizer
from tests.sft.test_semantic import _action_sample


ROOT = Path(__file__).resolve().parents[2]


def test_project_semantic_sample_preserves_object_arguments_and_call_ids() -> None:
    sample = _action_sample()
    row, normalization_hash = project_semantic_sample(sample)

    call = row["messages"][-1]["tool_calls"][0]
    assert call["id"] == sample.target.tool_calls[0].call_id
    assert call["function"]["arguments"] == {"path": "src/foo.py"}
    assert isinstance(call["function"]["arguments"], dict)
    assert row["tools"][0]["function"]["parameters"]["type"] == "object"
    assert normalization_hash == canonical_sha256({
        "messages": row["messages"][:-1],
        "target": row["messages"][-1],
        "tools": row["tools"],
    })


def test_export_binds_canonical_rendered_lock_and_dataset_info(tmp_path: Path) -> None:
    sample = _action_sample()
    tokenizer = _NativeToolTokenizer()
    template = CanonicalChatTemplate(tokenizer)
    rendered = RenderedSFTSample.from_semantic(
        sample,
        chat_template=template,
        tokenizer_revision="tokenizer-commit",
        tokenizer_hash=TOKENIZER_HASH,
        cutoff_len=20_000,
    )
    canonical_dir = tmp_path / "canonical"
    rendered_dir = tmp_path / "rendered"
    canonical_dir.mkdir()
    rendered_dir.mkdir()
    samples_by_split = {"train": (sample,), "validation": (), "test": ()}
    dataset_manifest = SFTDatasetManifest.create(
        samples_by_split=samples_by_split,
        quality_counts={"accepted": 1},
        filter_reason_counts={"accepted_fixture": 1},
        source_evidence_hashes={"fixture": canonical_sha256({"fixture": 1})},
        group_splits={"fixture-group": "train"},
    )
    rendered_manifest = RenderedSFTManifest.create(
        semantic_dataset_manifest_hash=dataset_manifest.dataset_manifest_hash,
        tokenizer_revision="tokenizer-commit",
        tokenizer_hash=TOKENIZER_HASH,
        chat_template_hash=template.template_hash,
        cutoff_len=20_000,
        split_rendered_sample_hashes={
            "train": (rendered.rendered_sample_hash,),
            "validation": (),
            "test": (),
        },
    )
    for split, samples in samples_by_split.items():
        _write_jsonl(canonical_dir / f"{split}.jsonl", samples)
        _write_jsonl(
            rendered_dir / f"{split}.jsonl",
            (rendered,) if split == "train" else (),
        )
    (canonical_dir / "dataset_manifest.json").write_bytes(
        canonical_json_bytes(dataset_manifest.to_dict()) + b"\n"
    )
    (rendered_dir / "rendered_manifest.json").write_bytes(
        canonical_json_bytes(rendered_manifest.to_dict()) + b"\n"
    )

    output = tmp_path / "llamafactory"
    manifest = export_llamafactory_dataset(
        canonical_dir=canonical_dir,
        rendered_dir=rendered_dir,
        output_dir=output,
        lock_path="integrations/llamafactory/lock.json",
        repository_root=ROOT,
    )

    assert LlamaFactoryExportManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.rendered_manifest_hash == rendered_manifest.rendered_manifest_hash
    assert manifest.semantic_dataset_manifest_hash == dataset_manifest.dataset_manifest_hash
    assert manifest.split_counts == {"test": 0, "train": 1, "validation": 0}
    rows = json.loads((output / "train_messages.json").read_text())
    raw_rows = (output / "train_messages.json").read_text()
    assert '"id":"call_770ef254227f","type":"function","function":{"name":"read_file","arguments":{"path":"src/foo.py"}}' in raw_rows
    assert rows[0]["messages"][-1]["tool_calls"][0]["function"]["arguments"] == {
        "path": "src/foo.py"
    }
    dataset_info = json.loads((output / "dataset_info.json").read_text())
    locked_dataset_info = json.loads(
        (ROOT / "integrations/llamafactory/fixtures/dataset_info.json").read_text()
    )
    assert dataset_info == locked_dataset_info
    assert load_llamafactory_lock(
        "integrations/llamafactory/lock.json",
        repository_root=ROOT,
    ).lock_hash == manifest.lock_hash

    (output / "train_messages.json").write_text(
        raw_rows.replace("src/foo.py", "src/bar.py"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="file hash"):
        _validate_export_artifacts(output, manifest)


def _write_jsonl(path: Path, records) -> None:
    path.write_bytes(b"".join(
        canonical_json_bytes(record.to_dict()) + b"\n" for record in records
    ))
