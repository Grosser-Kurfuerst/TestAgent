from __future__ import annotations

from pathlib import Path

from my_agent.policy.chat_template import CanonicalChatTemplate
from my_agent.policy.identity import canonical_sha256
from my_agent.sft.export import LlamaFactoryExportManifest, load_llamafactory_lock
from my_agent.sft.parity import (
    TemplateParityReport,
    _ExpectedParityCase,
    compare_parity_cases,
)
from my_agent.sft.rendered import RenderedSFTManifest
from tests.sft.test_rendered import TOKENIZER_HASH, _NativeToolTokenizer


ROOT = Path(__file__).resolve().parents[2]


def test_compare_parity_cases_reports_first_token_mismatch() -> None:
    case = _ExpectedParityCase(
        sample_id="sample-1",
        input_ids=(1, 2, 3),
        labels=(-100, 2, 3),
        normalization_input_hash=canonical_sha256({"sample": 1}),
        fixture=False,
    )
    mismatch = compare_parity_cases((case,), ({
        "sample_id": "sample-1",
        "input_ids": [1, 9, 3],
        "labels": [-100, 2, 3],
        "normalized_template_input_hash": case.normalization_input_hash,
        "arguments_are_objects": True,
    },))

    assert mismatch is not None
    assert mismatch["component"] == "input_ids"
    assert mismatch["first_index"] == 1


def test_parity_report_round_trip_binds_lock_rendered_and_export() -> None:
    lock = load_llamafactory_lock(
        "integrations/llamafactory/lock.json",
        repository_root=ROOT,
    )
    template = CanonicalChatTemplate(_NativeToolTokenizer())
    rendered = RenderedSFTManifest.create(
        semantic_dataset_manifest_hash=canonical_sha256({"semantic": 1}),
        tokenizer_revision="tokenizer-commit",
        tokenizer_hash=TOKENIZER_HASH,
        chat_template_hash=template.template_hash,
        cutoff_len=20_000,
        split_rendered_sample_hashes={"train": ()},
    )
    export = LlamaFactoryExportManifest.create(
        semantic_dataset_manifest_hash=rendered.semantic_dataset_manifest_hash,
        rendered_manifest_hash=rendered.rendered_manifest_hash,
        lock=lock,
        split_rows={"train": ()},
        normalized_template_input_hashes={},
        dataset_info_hash=lock.dataset_info_fixture_hash,
    )
    normalization = {
        "sample-1": canonical_sha256({"normalized": 1}),
        "fixture-1": canonical_sha256({"normalized": 2}),
    }
    report = TemplateParityReport.create(
        lock=lock,
        rendered_manifest=rendered,
        export_manifest=export,
        sample_count=1,
        fixture_count=1,
        normalization_input_hashes=normalization,
        first_mismatch=None,
    )

    assert report.passed
    assert TemplateParityReport.from_dict(report.to_dict()) == report
