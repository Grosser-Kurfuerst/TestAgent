"""Token-level parity between AgentCli rendering and patched LLaMA-Factory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import random
import subprocess
import tempfile

from my_agent.policy.chat_template import (
    CanonicalChatTemplate,
    canonicalize_messages,
    canonicalize_tools,
)
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256, hash_artifact_path
from my_agent.sft.export import (
    PARITY_REPORT_SCHEMA_VERSION,
    LlamaFactoryExportManifest,
    LlamaFactoryLock,
    load_export_manifest,
    load_llamafactory_lock,
    validate_llamafactory_checkout,
)
from my_agent.sft.rendered import RenderedSFTManifest, RenderedSFTSample


_TOKENIZER_NAMES = {
    "chat_template.jinja", "special_tokens_map.json", "tokenizer.json",
    "tokenizer.model", "tokenizer_config.json", "vocab.json",
}
_TOKENIZER_PREFIXES = ("added_tokens", "merges", "spiece", "tokenizer", "vocab")
_TOKENIZER_SUFFIXES = (".model", ".tiktoken")


@dataclass(frozen=True)
class TemplateParityReport:
    lock_revision: str
    template_artifact_hash: str
    lock_hash: str
    rendered_manifest_hash: str
    export_manifest_hash: str
    tokenizer_revision: str
    tokenizer_hash: str
    chat_template_hash: str
    sample_count: int
    fixture_count: int
    normalization_input_hashes: Mapping[str, str]
    passed: bool
    first_mismatch: Mapping[str, Any] | None
    parity_report_hash: str
    schema_version: str = PARITY_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARITY_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported SFT parity report schema")
        if self.sample_count < 0 or self.fixture_count < 0:
            raise ValueError("SFT parity counts must be non-negative")
        if self.passed != (self.first_mismatch is None):
            raise ValueError("SFT parity passed flag and mismatch do not align")
        if len(self.normalization_input_hashes) != self.sample_count + self.fixture_count:
            raise ValueError("SFT parity normalization hashes do not cover all cases")
        if self.parity_report_hash != canonical_sha256(self.payload_without_hash()):
            raise ValueError("SFT parity report hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        lock: LlamaFactoryLock,
        rendered_manifest: RenderedSFTManifest,
        export_manifest: LlamaFactoryExportManifest,
        sample_count: int,
        fixture_count: int,
        normalization_input_hashes: Mapping[str, str],
        first_mismatch: Mapping[str, Any] | None,
    ) -> "TemplateParityReport":
        values = {
            "lock_revision": lock.revision,
            "template_artifact_hash": lock.template_artifact_hash,
            "lock_hash": lock.lock_hash,
            "rendered_manifest_hash": rendered_manifest.rendered_manifest_hash,
            "export_manifest_hash": export_manifest.export_manifest_hash,
            "tokenizer_revision": rendered_manifest.tokenizer_revision,
            "tokenizer_hash": rendered_manifest.tokenizer_hash,
            "chat_template_hash": rendered_manifest.chat_template_hash,
            "sample_count": sample_count,
            "fixture_count": fixture_count,
            "normalization_input_hashes": dict(sorted(normalization_input_hashes.items())),
            "passed": first_mismatch is None,
            "first_mismatch": None if first_mismatch is None else dict(first_mismatch),
        }
        return cls(
            **values,
            parity_report_hash=canonical_sha256(_report_payload(**values)),
        )

    def payload_without_hash(self) -> dict[str, Any]:
        return _report_payload(
            lock_revision=self.lock_revision,
            template_artifact_hash=self.template_artifact_hash,
            lock_hash=self.lock_hash,
            rendered_manifest_hash=self.rendered_manifest_hash,
            export_manifest_hash=self.export_manifest_hash,
            tokenizer_revision=self.tokenizer_revision,
            tokenizer_hash=self.tokenizer_hash,
            chat_template_hash=self.chat_template_hash,
            sample_count=self.sample_count,
            fixture_count=self.fixture_count,
            normalization_input_hashes=self.normalization_input_hashes,
            passed=self.passed,
            first_mismatch=self.first_mismatch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"parity_report_hash": self.parity_report_hash, **self.payload_without_hash()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TemplateParityReport":
        expected = {
            "schema_version", "lock_revision", "template_artifact_hash", "lock_hash",
            "rendered_manifest_hash", "export_manifest_hash", "tokenizer_revision",
            "tokenizer_hash", "chat_template_hash", "sample_count", "fixture_count",
            "normalization_input_hashes", "passed", "first_mismatch",
            "parity_report_hash",
        }
        if set(data) != expected:
            raise ValueError("SFT parity report fields do not match schema")
        mismatch = data["first_mismatch"]
        if mismatch is not None and not isinstance(mismatch, Mapping):
            raise ValueError("SFT parity first_mismatch must be object or null")
        passed = data["passed"]
        if not isinstance(passed, bool):
            raise ValueError("SFT parity passed must be boolean")
        return cls(
            schema_version=_required_string(data["schema_version"], "schema_version"),
            lock_revision=_required_string(data["lock_revision"], "lock_revision"),
            template_artifact_hash=_required_string(
                data["template_artifact_hash"], "template_artifact_hash"
            ),
            lock_hash=_required_string(data["lock_hash"], "lock_hash"),
            rendered_manifest_hash=_required_string(
                data["rendered_manifest_hash"], "rendered_manifest_hash"
            ),
            export_manifest_hash=_required_string(
                data["export_manifest_hash"], "export_manifest_hash"
            ),
            tokenizer_revision=_required_string(
                data["tokenizer_revision"], "tokenizer_revision"
            ),
            tokenizer_hash=_required_string(data["tokenizer_hash"], "tokenizer_hash"),
            chat_template_hash=_required_string(
                data["chat_template_hash"], "chat_template_hash"
            ),
            sample_count=_integer(data["sample_count"], "sample_count"),
            fixture_count=_integer(data["fixture_count"], "fixture_count"),
            normalization_input_hashes=_string_mapping(
                data["normalization_input_hashes"], "normalization_input_hashes"
            ),
            passed=passed,
            first_mismatch=None if mismatch is None else dict(mismatch),
            parity_report_hash=_required_string(
                data["parity_report_hash"], "parity_report_hash"
            ),
        )


@dataclass(frozen=True)
class _ExpectedParityCase:
    sample_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    normalization_input_hash: str
    fixture: bool


def verify_llamafactory_parity(
    *,
    repository_root: str | Path,
    lock_path: str | Path,
    checkout: str | Path,
    tokenizer_path: str | Path,
    rendered_dir: str | Path,
    export_dir: str | Path,
    output_path: str | Path,
    sample_limit: int = 128,
    python_executable: str = "python",
) -> TemplateParityReport:
    root = Path(repository_root)
    lock = load_llamafactory_lock(lock_path, repository_root=root)
    patch_path = root / lock.patch_path
    validate_llamafactory_checkout(
        checkout=checkout,
        lock=lock,
        patch_path=patch_path,
        require_patch_applied=True,
    )
    rendered_root = Path(rendered_dir)
    rendered_manifest = RenderedSFTManifest.from_dict(
        _load_mapping(rendered_root / "rendered_manifest.json", "rendered manifest")
    )
    export_root = Path(export_dir)
    export_manifest = load_export_manifest(export_root / "export_manifest.json")
    if export_manifest.rendered_manifest_hash != rendered_manifest.rendered_manifest_hash:
        raise ValueError("export manifest does not bind the rendered manifest")
    if export_manifest.lock_hash != lock.lock_hash:
        raise ValueError("export manifest does not bind the LLaMA-Factory lock")
    if export_manifest.dataset_info_hash != lock.dataset_info_fixture_hash:
        raise ValueError("export manifest dataset info does not match the lock")
    split_rows = _validate_export_artifacts(export_root, export_manifest)
    tokenizer_root = Path(tokenizer_path)
    actual_tokenizer_hash = hash_artifact_path(tokenizer_root, include=_is_tokenizer_artifact)
    if actual_tokenizer_hash != rendered_manifest.tokenizer_hash:
        raise ValueError("parity tokenizer artifacts do not match the rendered manifest")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("SFT parity requires the 'opd-train' extra") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    template = CanonicalChatTemplate(tokenizer)
    if template.template_hash != rendered_manifest.chat_template_hash:
        raise ValueError("parity chat template does not match rendered samples")

    expected, rows, fixture_count = _collect_cases(
        root=root,
        rendered_root=rendered_root,
        export_manifest=export_manifest,
        split_rows=split_rows,
        template=template,
        sample_limit=sample_limit,
    )
    first_mismatch: Mapping[str, Any] | None = None
    try:
        actual = _run_bridge(
            root=root,
            checkout=Path(checkout),
            tokenizer_root=tokenizer_root,
            rows=rows,
            dataset_info_path=export_root / "dataset_info.json",
            cutoff_len=rendered_manifest.cutoff_len,
            python_executable=python_executable,
        )
        first_mismatch = compare_parity_cases(expected, actual)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        first_mismatch = {
            "sample_id": "",
            "component": "llamafactory_bridge",
            "detail": str(exc),
        }
    report = TemplateParityReport.create(
        lock=lock,
        rendered_manifest=rendered_manifest,
        export_manifest=export_manifest,
        sample_count=len(expected) - fixture_count,
        fixture_count=fixture_count,
        normalization_input_hashes={
            item.sample_id: item.normalization_input_hash for item in expected
        },
        first_mismatch=first_mismatch,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report.to_dict()) + b"\n")
    return report


def compare_parity_cases(
    expected: Sequence[_ExpectedParityCase],
    actual_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    actual_by_id = {
        _required_string(row.get("sample_id"), "sample_id"): row for row in actual_rows
    }
    if len(actual_by_id) != len(actual_rows):
        return {"sample_id": "", "component": "duplicate_result", "detail": "duplicate IDs"}
    for case in expected:
        actual = actual_by_id.get(case.sample_id)
        if actual is None:
            return {"sample_id": case.sample_id, "component": "missing_result"}
        checks = (
            ("input_ids", list(case.input_ids), actual.get("input_ids")),
            ("labels", list(case.labels), actual.get("labels")),
            (
                "normalized_template_input_hash",
                case.normalization_input_hash,
                actual.get("normalized_template_input_hash"),
            ),
            ("arguments_are_objects", True, actual.get("arguments_are_objects")),
        )
        for component, expected_value, actual_value in checks:
            if expected_value != actual_value:
                mismatch = {
                    "sample_id": case.sample_id,
                    "component": component,
                    "expected": expected_value,
                    "actual": actual_value,
                }
                if isinstance(expected_value, list) and isinstance(actual_value, list):
                    mismatch["first_index"] = _first_sequence_mismatch(
                        expected_value,
                        actual_value,
                    )
                return mismatch
    extras = sorted(set(actual_by_id) - {item.sample_id for item in expected})
    if extras:
        return {"sample_id": extras[0], "component": "unexpected_result"}
    return None


def _collect_cases(
    *,
    root: Path,
    rendered_root: Path,
    export_manifest: LlamaFactoryExportManifest,
    split_rows: Mapping[str, tuple[dict[str, Any], ...]],
    template: CanonicalChatTemplate,
    sample_limit: int,
) -> tuple[tuple[_ExpectedParityCase, ...], tuple[dict[str, Any], ...], int]:
    if sample_limit < 1:
        raise ValueError("parity sample_limit must be positive")
    rendered_by_id: dict[str, RenderedSFTSample] = {}
    export_rows: list[dict[str, Any]] = []
    for split in sorted(export_manifest.split_sample_ids):
        export_rows.extend(split_rows[split])
        for item in _read_rendered(rendered_root / f"{split}.jsonl"):
            rendered_by_id[item.sample_id] = item
    ordered_rows = sorted(export_rows, key=lambda item: item["sample_id"])
    randomizer = random.Random(export_manifest.export_manifest_hash)
    selected_rows = tuple(
        randomizer.sample(ordered_rows, min(sample_limit, len(ordered_rows)))
    )
    expected: list[_ExpectedParityCase] = []
    for row in selected_rows:
        sample_id = _required_string(row.get("sample_id"), "sample_id")
        rendered = rendered_by_id.get(sample_id)
        if rendered is None:
            raise ValueError("exported parity sample is absent from rendered data")
        expected.append(_ExpectedParityCase(
            sample_id=sample_id,
            input_ids=rendered.input_ids,
            labels=rendered.labels,
            normalization_input_hash=export_manifest.normalized_template_input_hashes[sample_id],
            fixture=False,
        ))

    fixture_manifest = _load_mapping(
        root / "integrations/llamafactory/fixtures/manifest.json",
        "fixture manifest",
    )
    fixture_names = sorted(_mapping(fixture_manifest.get("fixtures"), "fixtures"))
    fixture_rows: list[dict[str, Any]] = []
    for name in fixture_names:
        row = dict(_load_mapping(
            root / "integrations/llamafactory/fixtures" / name,
            f"fixture {name}",
        ))
        messages = row.get("messages")
        tools = row.get("tools")
        if not isinstance(messages, list) or not messages:
            raise ValueError("parity fixture messages must be a non-empty array")
        canonical_messages = canonicalize_messages(messages)
        canonical_tools = canonicalize_tools(tools if isinstance(tools, list) else [])
        target = canonical_messages[-1]
        turn = template.render_training_turn(
            canonical_messages[:-1],
            canonical_tools,
            target,
        )
        full_mask = (0,) * len(turn.prompt_token_ids) + turn.assistant_loss_mask
        labels = tuple(
            token if active else -100
            for token, active in zip(turn.input_ids, full_mask)
        )
        sample_id = _required_string(row.get("sample_id"), "fixture sample_id")
        expected.append(_ExpectedParityCase(
            sample_id=sample_id,
            input_ids=turn.input_ids,
            labels=labels,
            normalization_input_hash=turn.normalized_template_input_hash,
            fixture=True,
        ))
        fixture_rows.append(row)
    return tuple(expected), (*selected_rows, *fixture_rows), len(fixture_rows)


def _run_bridge(
    *,
    root: Path,
    checkout: Path,
    tokenizer_root: Path,
    rows: Sequence[Mapping[str, Any]],
    dataset_info_path: Path,
    cutoff_len: int,
    python_executable: str,
) -> tuple[Mapping[str, Any], ...]:
    with tempfile.TemporaryDirectory(prefix="agentcli-sft-parity-") as temp:
        temp_root = Path(temp)
        input_path = temp_root / "input.json"
        output_path = temp_root / "output.json"
        input_path.write_bytes(_ordered_json_bytes(list(rows)) + b"\n")
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(checkout / "src") + (os.pathsep + existing if existing else "")
        completed = subprocess.run(
            [
                python_executable,
                str(root / "integrations/llamafactory/parity_bridge.py"),
                "--input", str(input_path),
                "--output", str(output_path),
                "--tokenizer", str(tokenizer_root),
                "--checkout", str(checkout),
                "--dataset-info",
                str(dataset_info_path),
                "--cutoff-len", str(cutoff_len),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"LLaMA-Factory parity bridge failed: {detail}")
        values = _load_array(output_path, "parity bridge output")
        return tuple(values)


def _report_payload(**values: Any) -> dict[str, Any]:
    return {
        "schema_version": PARITY_REPORT_SCHEMA_VERSION,
        "lock_revision": values["lock_revision"],
        "template_artifact_hash": values["template_artifact_hash"],
        "lock_hash": values["lock_hash"],
        "rendered_manifest_hash": values["rendered_manifest_hash"],
        "export_manifest_hash": values["export_manifest_hash"],
        "tokenizer_revision": values["tokenizer_revision"],
        "tokenizer_hash": values["tokenizer_hash"],
        "chat_template_hash": values["chat_template_hash"],
        "sample_count": values["sample_count"],
        "fixture_count": values["fixture_count"],
        "normalization_input_hashes": dict(
            sorted(values["normalization_input_hashes"].items())
        ),
        "passed": values["passed"],
        "first_mismatch": values["first_mismatch"],
    }


def _validate_export_artifacts(
    export_root: Path,
    manifest: LlamaFactoryExportManifest,
) -> dict[str, tuple[dict[str, Any], ...]]:
    dataset_info = _load_mapping(export_root / "dataset_info.json", "export dataset info")
    if canonical_sha256(dataset_info) != manifest.dataset_info_hash:
        raise ValueError("export dataset_info.json hash does not match its manifest")
    split_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    for split in sorted(manifest.split_sample_ids):
        path = export_root / f"{split}_messages.json"
        raw = path.read_bytes()
        if _bytes_sha256(raw) != manifest.split_file_hashes[split]:
            raise ValueError(f"export {split} file hash does not match its manifest")
        rows = tuple(_load_array(path, f"export {split} split"))
        if len(rows) != manifest.split_counts[split]:
            raise ValueError(f"export {split} count does not match its manifest")
        sample_ids = tuple(
            _required_string(row.get("sample_id"), "sample_id") for row in rows
        )
        if sample_ids != manifest.split_sample_ids[split]:
            raise ValueError(f"export {split} sample IDs do not match its manifest")
        split_rows[split] = rows
    return split_rows


def _first_sequence_mismatch(expected: list[Any], actual: list[Any]) -> int:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    return min(len(expected), len(actual))


def _read_rendered(path: Path) -> tuple[RenderedSFTSample, ...]:
    samples: list[RenderedSFTSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid rendered JSON at {path}:{line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError("rendered parity row must be an object")
            samples.append(RenderedSFTSample.from_dict(payload))
    return tuple(samples)


def _load_mapping(path: Path, kind: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind} must be an object")
    return value


def _ordered_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return "sha256:" + sha256(value).hexdigest()


def _load_array(path: Path, kind: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{kind} must be an array of objects")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    return {
        _required_string(key, field_name): _required_string(item, field_name)
        for key, item in _mapping(value, field_name).items()
    }


def _is_tokenizer_artifact(path: Path) -> bool:
    return (
        path.name in _TOKENIZER_NAMES
        or path.name.startswith(_TOKENIZER_PREFIXES)
        or path.name.endswith(_TOKENIZER_SUFFIXES)
    )


__all__ = [
    "TemplateParityReport",
    "compare_parity_cases",
    "verify_llamafactory_parity",
]
