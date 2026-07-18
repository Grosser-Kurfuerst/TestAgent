"""Deterministic OpenAI-tools projection for the locked LLaMA-Factory path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import subprocess

from my_agent.policy.chat_template import canonical_messages_to_hf, canonical_tools_to_hf
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256, hash_artifact_path
from my_agent.sft.manifest import SFTDatasetManifest
from my_agent.sft.rendered import RenderedSFTManifest, RenderedSFTSample
from my_agent.sft.semantic import SemanticSFTSample


LLAMAFACTORY_LOCK_SCHEMA_VERSION = "agentcli-llamafactory-lock-v1"
LLAMAFACTORY_EXPORT_SCHEMA_VERSION = "agentcli-llamafactory-export-v1"
PARITY_REPORT_SCHEMA_VERSION = "agentcli-sft-template-parity-v1"
_LOCK_FIELDS = {
    "schema_version", "repository", "revision", "version", "dataset_format",
    "template_mode", "template_name", "patch_path", "patch_apply_args",
    "patched_file_hashes",
    "template_artifact_hash", "tool_fixture_manifest_hash",
    "dataset_info_fixture_hash",
}
_PATCHED_FILES = frozenset({
    "src/llamafactory/data/converter.py",
    "src/llamafactory/data/parser.py",
    "src/llamafactory/data/processor/supervised.py",
})


@dataclass(frozen=True)
class LlamaFactoryLock:
    repository: str
    revision: str
    version: str
    dataset_format: str
    template_mode: str
    template_name: str
    patch_path: str
    patch_apply_args: tuple[str, ...]
    patched_file_hashes: Mapping[str, str]
    template_artifact_hash: str
    tool_fixture_manifest_hash: str
    dataset_info_fixture_hash: str
    schema_version: str = LLAMAFACTORY_LOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LLAMAFACTORY_LOCK_SCHEMA_VERSION:
            raise ValueError("unsupported LLaMA-Factory lock schema")
        for field_name in (
            "repository", "revision", "version", "dataset_format", "template_mode",
            "template_name", "patch_path", "template_artifact_hash",
            "tool_fixture_manifest_hash", "dataset_info_fixture_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LLaMA-Factory lock {field_name} must be non-empty")
        if self.template_mode != "pinned_patch":
            raise ValueError("formal SFT requires the locked pinned-patch mode")
        if self.dataset_format != "agentcli_openai_tools":
            raise ValueError("unsupported LLaMA-Factory dataset format")
        if self.patch_apply_args != ("--unidiff-zero",):
            raise ValueError("unexpected LLaMA-Factory patch apply arguments")
        if set(self.patched_file_hashes) != _PATCHED_FILES:
            raise ValueError("locked patched-file hashes do not cover the pinned patch")

    @property
    def lock_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "revision": self.revision,
            "version": self.version,
            "dataset_format": self.dataset_format,
            "template_mode": self.template_mode,
            "template_name": self.template_name,
            "patch_path": self.patch_path,
            "patch_apply_args": list(self.patch_apply_args),
            "patched_file_hashes": dict(sorted(self.patched_file_hashes.items())),
            "template_artifact_hash": self.template_artifact_hash,
            "tool_fixture_manifest_hash": self.tool_fixture_manifest_hash,
            "dataset_info_fixture_hash": self.dataset_info_fixture_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LlamaFactoryLock":
        if set(data) != _LOCK_FIELDS:
            raise ValueError("LLaMA-Factory lock fields do not match schema")
        args = data["patch_apply_args"]
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise ValueError("LLaMA-Factory patch_apply_args must be a string array")
        return cls(
            schema_version=_required_string(data["schema_version"], "schema_version"),
            repository=_required_string(data["repository"], "repository"),
            revision=_required_string(data["revision"], "revision"),
            version=_required_string(data["version"], "version"),
            dataset_format=_required_string(data["dataset_format"], "dataset_format"),
            template_mode=_required_string(data["template_mode"], "template_mode"),
            template_name=_required_string(data["template_name"], "template_name"),
            patch_path=_required_string(data["patch_path"], "patch_path"),
            patch_apply_args=tuple(args),
            patched_file_hashes=_string_mapping(
                data["patched_file_hashes"], "patched_file_hashes"
            ),
            template_artifact_hash=_required_string(
                data["template_artifact_hash"], "template_artifact_hash"
            ),
            tool_fixture_manifest_hash=_required_string(
                data["tool_fixture_manifest_hash"], "tool_fixture_manifest_hash"
            ),
            dataset_info_fixture_hash=_required_string(
                data["dataset_info_fixture_hash"], "dataset_info_fixture_hash"
            ),
        )


@dataclass(frozen=True)
class LlamaFactoryExportManifest:
    semantic_dataset_manifest_hash: str
    rendered_manifest_hash: str
    lock_hash: str
    dataset_format: str
    template_name: str
    split_counts: Mapping[str, int]
    split_sample_ids: Mapping[str, tuple[str, ...]]
    split_file_hashes: Mapping[str, str]
    normalized_template_input_hashes: Mapping[str, str]
    dataset_info_hash: str
    export_manifest_hash: str
    schema_version: str = LLAMAFACTORY_EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LLAMAFACTORY_EXPORT_SCHEMA_VERSION:
            raise ValueError("unsupported LLaMA-Factory export manifest schema")
        if set(self.split_counts) != set(self.split_sample_ids):
            raise ValueError("export split counts and sample IDs do not align")
        if set(self.split_counts) != set(self.split_file_hashes):
            raise ValueError("export split counts and file hashes do not align")
        all_ids = [sample_id for ids in self.split_sample_ids.values() for sample_id in ids]
        if set(all_ids) != set(self.normalized_template_input_hashes):
            raise ValueError("export normalization hashes do not cover every sample")
        for split, count in self.split_counts.items():
            if not split or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("export split counts are invalid")
            if count != len(self.split_sample_ids[split]):
                raise ValueError("export split count does not match sample IDs")
        if self.export_manifest_hash != canonical_sha256(self.payload_without_hash()):
            raise ValueError("LLaMA-Factory export manifest hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        semantic_dataset_manifest_hash: str,
        rendered_manifest_hash: str,
        lock: LlamaFactoryLock,
        split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
        normalized_template_input_hashes: Mapping[str, str],
        dataset_info_hash: str,
    ) -> "LlamaFactoryExportManifest":
        values = {
            "semantic_dataset_manifest_hash": semantic_dataset_manifest_hash,
            "rendered_manifest_hash": rendered_manifest_hash,
            "lock_hash": lock.lock_hash,
            "dataset_format": lock.dataset_format,
            "template_name": lock.template_name,
            "split_counts": {
                split: len(rows) for split, rows in sorted(split_rows.items())
            },
            "split_sample_ids": {
                split: tuple(_required_string(row.get("sample_id"), "sample_id") for row in rows)
                for split, rows in sorted(split_rows.items())
            },
            "split_file_hashes": {
                split: _bytes_sha256(_ordered_json_bytes(list(rows)) + b"\n")
                for split, rows in sorted(split_rows.items())
            },
            "normalized_template_input_hashes": dict(
                sorted(normalized_template_input_hashes.items())
            ),
            "dataset_info_hash": dataset_info_hash,
        }
        return cls(
            **values,
            export_manifest_hash=canonical_sha256(_export_manifest_payload(**values)),
        )

    def payload_without_hash(self) -> dict[str, Any]:
        return _export_manifest_payload(
            semantic_dataset_manifest_hash=self.semantic_dataset_manifest_hash,
            rendered_manifest_hash=self.rendered_manifest_hash,
            lock_hash=self.lock_hash,
            dataset_format=self.dataset_format,
            template_name=self.template_name,
            split_counts=self.split_counts,
            split_sample_ids=self.split_sample_ids,
            split_file_hashes=self.split_file_hashes,
            normalized_template_input_hashes=self.normalized_template_input_hashes,
            dataset_info_hash=self.dataset_info_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"export_manifest_hash": self.export_manifest_hash, **self.payload_without_hash()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LlamaFactoryExportManifest":
        expected = {
            "schema_version", "semantic_dataset_manifest_hash", "rendered_manifest_hash",
            "lock_hash", "dataset_format", "template_name", "split_counts",
            "split_sample_ids", "split_file_hashes",
            "normalized_template_input_hashes", "dataset_info_hash",
            "export_manifest_hash",
        }
        if set(data) != expected:
            raise ValueError("LLaMA-Factory export manifest fields do not match schema")
        return cls(
            schema_version=_required_string(data["schema_version"], "schema_version"),
            semantic_dataset_manifest_hash=_required_string(
                data["semantic_dataset_manifest_hash"], "semantic_dataset_manifest_hash"
            ),
            rendered_manifest_hash=_required_string(
                data["rendered_manifest_hash"], "rendered_manifest_hash"
            ),
            lock_hash=_required_string(data["lock_hash"], "lock_hash"),
            dataset_format=_required_string(data["dataset_format"], "dataset_format"),
            template_name=_required_string(data["template_name"], "template_name"),
            split_counts=_int_mapping(data["split_counts"], "split_counts"),
            split_sample_ids=_tuple_mapping(data["split_sample_ids"], "split_sample_ids"),
            split_file_hashes=_string_mapping(
                data["split_file_hashes"], "split_file_hashes"
            ),
            normalized_template_input_hashes=_string_mapping(
                data["normalized_template_input_hashes"],
                "normalized_template_input_hashes",
            ),
            dataset_info_hash=_required_string(data["dataset_info_hash"], "dataset_info_hash"),
            export_manifest_hash=_required_string(
                data["export_manifest_hash"], "export_manifest_hash"
            ),
        )


def load_llamafactory_lock(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> LlamaFactoryLock:
    root = Path(repository_root)
    lock_path = Path(path)
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    lock = LlamaFactoryLock.from_dict(_load_mapping(lock_path, "LLaMA-Factory lock"))
    patch = root / lock.patch_path
    if hash_artifact_path(patch) != lock.template_artifact_hash:
        raise ValueError("locked LLaMA-Factory patch hash mismatch")
    fixtures = root / "integrations/llamafactory/fixtures"
    fixture_manifest = _load_mapping(fixtures / "manifest.json", "tool fixture manifest")
    if canonical_sha256(fixture_manifest) != lock.tool_fixture_manifest_hash:
        raise ValueError("locked tool fixture manifest hash mismatch")
    dataset_info = _load_mapping(fixtures / "dataset_info.json", "dataset info fixture")
    if canonical_sha256(dataset_info) != lock.dataset_info_fixture_hash:
        raise ValueError("locked dataset info fixture hash mismatch")
    return lock


def project_semantic_sample(sample: SemanticSFTSample) -> tuple[dict[str, Any], str]:
    messages = canonical_messages_to_hf((*sample.messages, sample.target))
    tools = canonical_tools_to_hf(sample.tools)
    _validate_openai_tools_payload(messages, tools)
    row = {
        "sample_id": sample.sample_id,
        "messages": messages,
        "tools": tools,
    }
    normalization_hash = canonical_sha256({
        "messages": messages[:-1],
        "target": messages[-1],
        "tools": tools,
    })
    return row, normalization_hash


def export_llamafactory_dataset(
    *,
    canonical_dir: str | Path,
    rendered_dir: str | Path,
    output_dir: str | Path,
    lock_path: str | Path,
    repository_root: str | Path,
) -> LlamaFactoryExportManifest:
    root = Path(repository_root)
    lock = load_llamafactory_lock(lock_path, repository_root=root)
    canonical_root = Path(canonical_dir)
    rendered_root = Path(rendered_dir)
    semantic_manifest = SFTDatasetManifest.from_dict(
        _load_mapping(canonical_root / "dataset_manifest.json", "canonical SFT manifest")
    )
    rendered_manifest = RenderedSFTManifest.from_dict(
        _load_mapping(rendered_root / "rendered_manifest.json", "rendered SFT manifest")
    )
    if rendered_manifest.semantic_dataset_manifest_hash != semantic_manifest.dataset_manifest_hash:
        raise ValueError("rendered manifest does not bind the canonical dataset manifest")

    split_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    normalization_hashes: dict[str, str] = {}
    for split in sorted(semantic_manifest.split_sample_ids):
        semantic_samples = _read_jsonl_objects(
            canonical_root / f"{split}.jsonl",
            SemanticSFTSample.from_dict,
        )
        rendered_samples = _read_jsonl_objects(
            rendered_root / f"{split}.jsonl",
            RenderedSFTSample.from_dict,
        )
        sample_ids = tuple(sample.sample_id for sample in semantic_samples)
        if sample_ids != semantic_manifest.split_sample_ids[split]:
            raise ValueError("canonical split order does not match its manifest")
        if tuple(sample.sample_id for sample in rendered_samples) != sample_ids:
            raise ValueError("rendered split sample IDs do not match canonical samples")
        if tuple(sample.rendered_sample_hash for sample in rendered_samples) != (
            rendered_manifest.split_rendered_sample_hashes[split]
        ):
            raise ValueError("rendered split hashes do not match rendered manifest")
        rows: list[dict[str, Any]] = []
        for sample in semantic_samples:
            row, normalization_hash = project_semantic_sample(sample)
            rows.append(row)
            normalization_hashes[sample.sample_id] = normalization_hash
        split_rows[split] = tuple(rows)

    dataset_info = _load_mapping(
        root / "integrations/llamafactory/fixtures/dataset_info.json",
        "dataset info fixture",
    )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"LLaMA-Factory export directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        (output / f"{split}_messages.json").write_bytes(
            _ordered_json_bytes(list(rows)) + b"\n"
        )
    (output / "dataset_info.json").write_bytes(canonical_json_bytes(dataset_info) + b"\n")
    manifest = LlamaFactoryExportManifest.create(
        semantic_dataset_manifest_hash=semantic_manifest.dataset_manifest_hash,
        rendered_manifest_hash=rendered_manifest.rendered_manifest_hash,
        lock=lock,
        split_rows=split_rows,
        normalized_template_input_hashes=normalization_hashes,
        dataset_info_hash=canonical_sha256(dataset_info),
    )
    (output / "export_manifest.json").write_bytes(
        canonical_json_bytes(manifest.to_dict()) + b"\n"
    )
    return manifest


def validate_llamafactory_checkout(
    *,
    checkout: str | Path,
    lock: LlamaFactoryLock,
    patch_path: str | Path,
    require_patch_applied: bool,
) -> None:
    root = Path(checkout)
    revision = _git(root, "rev-parse", "HEAD").strip()
    if revision != lock.revision:
        raise ValueError("LLaMA-Factory checkout revision does not match the lock")
    patch = str(Path(patch_path).resolve())
    apply_args = list(lock.patch_apply_args)
    if require_patch_applied:
        _git(root, "apply", "--reverse", "--check", *apply_args, patch)
        changed = frozenset(
            line for line in _git(root, "diff", "--name-only", "HEAD").splitlines() if line
        )
        if changed != _PATCHED_FILES:
            raise ValueError("LLaMA-Factory checkout contains changes outside the locked patch")
        for relative_path, expected_hash in lock.patched_file_hashes.items():
            if hash_artifact_path(root / relative_path) != expected_hash:
                raise ValueError("LLaMA-Factory patched file hash mismatch")
    else:
        if _git(root, "status", "--short", "--untracked-files=no").strip():
            raise ValueError("LLaMA-Factory checkout must be clean before applying the patch")
        _git(root, "apply", "--check", *apply_args, patch)


def apply_locked_llamafactory_patch(
    *,
    checkout: str | Path,
    lock: LlamaFactoryLock,
    patch_path: str | Path,
) -> None:
    validate_llamafactory_checkout(
        checkout=checkout,
        lock=lock,
        patch_path=patch_path,
        require_patch_applied=False,
    )
    _git(
        Path(checkout),
        "apply",
        *lock.patch_apply_args,
        str(Path(patch_path).resolve()),
    )
    validate_llamafactory_checkout(
        checkout=checkout,
        lock=lock,
        patch_path=patch_path,
        require_patch_applied=True,
    )


def load_export_manifest(path: str | Path) -> LlamaFactoryExportManifest:
    return LlamaFactoryExportManifest.from_dict(
        _load_mapping(Path(path), "LLaMA-Factory export manifest")
    )


def _validate_openai_tools_payload(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> None:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("LLaMA-Factory export target must be the final assistant message")
    call_ids: list[str] = []
    observed_ids: list[str] = []
    for message in messages:
        if not isinstance(message.get("content"), str):
            raise ValueError("OpenAI-style message content must be a string")
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("OpenAI-style tool_calls must be an array")
        for call in calls:
            if not isinstance(call, Mapping):
                raise ValueError("OpenAI-style tool call must be an object")
            function = call.get("function")
            if not isinstance(function, Mapping) or not isinstance(
                function.get("arguments"), Mapping
            ):
                raise ValueError("OpenAI-style function.arguments must remain an object")
            call_ids.append(_required_string(call.get("id"), "tool call ID"))
        if message.get("role") == "tool":
            observed_ids.append(_required_string(message.get("tool_call_id"), "tool_call_id"))
    if any(observation_id not in call_ids for observation_id in observed_ids):
        raise ValueError("tool observation references an unknown call ID")
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping) or not isinstance(
            function.get("parameters"), Mapping
        ):
            raise ValueError("OpenAI-style tool parameters must remain an object")


def _export_manifest_payload(**values: Any) -> dict[str, Any]:
    return {
        "schema_version": LLAMAFACTORY_EXPORT_SCHEMA_VERSION,
        "semantic_dataset_manifest_hash": values["semantic_dataset_manifest_hash"],
        "rendered_manifest_hash": values["rendered_manifest_hash"],
        "lock_hash": values["lock_hash"],
        "dataset_format": values["dataset_format"],
        "template_name": values["template_name"],
        "split_counts": dict(sorted(values["split_counts"].items())),
        "split_sample_ids": {
            split: list(ids) for split, ids in sorted(values["split_sample_ids"].items())
        },
        "split_file_hashes": dict(sorted(values["split_file_hashes"].items())),
        "normalized_template_input_hashes": dict(
            sorted(values["normalized_template_input_hashes"].items())
        ),
        "dataset_info_hash": values["dataset_info_hash"],
    }


def _read_jsonl_objects(path: Path, loader) -> tuple[Any, ...]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(loader(payload))
    return tuple(records)


def _load_mapping(path: Path, kind: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {kind}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{kind} must be an object")
    return payload


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


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"LLaMA-Factory git check failed: {detail}")
    return completed.stdout


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    payload = _mapping(value, field_name)
    return {
        _required_string(key, field_name): _required_string(item, field_name)
        for key, item in payload.items()
    }


def _int_mapping(value: Any, field_name: str) -> dict[str, int]:
    payload = _mapping(value, field_name)
    result: dict[str, int] = {}
    for key, item in payload.items():
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field_name} values must be integers")
        result[_required_string(key, field_name)] = item
    return result


def _tuple_mapping(value: Any, field_name: str) -> dict[str, tuple[str, ...]]:
    payload = _mapping(value, field_name)
    result: dict[str, tuple[str, ...]] = {}
    for key, item in payload.items():
        if not isinstance(item, list) or any(not isinstance(entry, str) for entry in item):
            raise ValueError(f"{field_name} values must be string arrays")
        result[_required_string(key, field_name)] = tuple(item)
    return result


__all__ = [
    "LLAMAFACTORY_EXPORT_SCHEMA_VERSION",
    "LLAMAFACTORY_LOCK_SCHEMA_VERSION",
    "PARITY_REPORT_SCHEMA_VERSION",
    "LlamaFactoryExportManifest",
    "LlamaFactoryLock",
    "apply_locked_llamafactory_patch",
    "export_llamafactory_dataset",
    "load_export_manifest",
    "load_llamafactory_lock",
    "project_semantic_sample",
    "validate_llamafactory_checkout",
]
