"""Tokenized completion-only SFT samples and rendered dataset manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from my_agent.policy.chat_template import CanonicalChatTemplate
from my_agent.policy.identity import canonical_sha256, require_sha256
from my_agent.policy.transformers_policy import parse_tool_calls
from my_agent.sft.contracts import (
    RENDERED_MANIFEST_SCHEMA_VERSION,
    RENDERED_SFT_SCHEMA_VERSION,
    validate_expected_output_contract,
)
from my_agent.sft.semantic import SemanticSFTSample


_FIELDS = {
    "schema_version",
    "sample_id",
    "semantic_sample_hash",
    "rendered_sample_hash",
    "expected_output_kind",
    "expected_tool_call_count",
    "tokenizer_revision",
    "tokenizer_hash",
    "chat_template_hash",
    "raw_completion",
    "prompt_token_ids",
    "completion_token_ids",
    "assistant_loss_mask",
    "input_ids",
    "full_sequence_loss_mask",
    "labels",
    "cutoff_len",
    "truncated",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "semantic_dataset_manifest_hash",
    "tokenizer_revision",
    "tokenizer_hash",
    "chat_template_hash",
    "cutoff_len",
    "split_rendered_sample_hashes",
    "rendered_manifest_hash",
}


@dataclass(frozen=True)
class RenderedSFTSample:
    sample_id: str
    semantic_sample_hash: str
    rendered_sample_hash: str
    expected_output_kind: str
    expected_tool_call_count: int | None
    tokenizer_revision: str
    tokenizer_hash: str
    chat_template_hash: str
    raw_completion: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    assistant_loss_mask: tuple[int, ...]
    input_ids: tuple[int, ...]
    full_sequence_loss_mask: tuple[int, ...]
    labels: tuple[int, ...]
    cutoff_len: int
    truncated: bool = False
    schema_version: str = RENDERED_SFT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RENDERED_SFT_SCHEMA_VERSION:
            raise ValueError("unsupported rendered SFT schema")
        for field_name in (
            "sample_id",
            "semantic_sample_hash",
            "rendered_sample_hash",
            "tokenizer_hash",
            "chat_template_hash",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if self.sample_id != self.semantic_sample_hash:
            raise ValueError("rendered sample must preserve its semantic sample ID")
        if not self.tokenizer_revision.strip():
            raise ValueError("rendered SFT tokenizer_revision must not be blank")
        validate_expected_output_contract(
            self.expected_output_kind,
            self.expected_tool_call_count,
        )
        if isinstance(self.cutoff_len, bool) or self.cutoff_len < 1:
            raise ValueError("rendered SFT cutoff_len must be positive")
        if self.truncated:
            raise ValueError("formal rendered SFT samples must not be truncated")
        if not self.prompt_token_ids or not self.completion_token_ids:
            raise ValueError("rendered SFT prompt and completion tokens must not be empty")
        if self.input_ids != self.prompt_token_ids + self.completion_token_ids:
            raise ValueError("rendered SFT input_ids do not match prompt + completion")
        if len(self.input_ids) > self.cutoff_len:
            raise ValueError("rendered SFT sample exceeds cutoff_len")
        if self.assistant_loss_mask != (1,) * len(self.completion_token_ids):
            raise ValueError("assistant_loss_mask must cover only the full current completion")
        expected_mask = (0,) * len(self.prompt_token_ids) + self.assistant_loss_mask
        if self.full_sequence_loss_mask != expected_mask:
            raise ValueError("full_sequence_loss_mask does not match completion-only masking")
        expected_labels = tuple(
            token if mask else -100
            for token, mask in zip(self.input_ids, self.full_sequence_loss_mask)
        )
        if self.labels != expected_labels:
            raise ValueError("rendered SFT labels do not match the loss mask")
        if self.rendered_sample_hash != canonical_sha256(self.payload_without_rendered_hash()):
            raise ValueError("rendered_sample_hash does not match its payload")

    @classmethod
    def from_semantic(
        cls,
        sample: SemanticSFTSample,
        *,
        chat_template: CanonicalChatTemplate,
        tokenizer_revision: str,
        tokenizer_hash: str,
        cutoff_len: int,
    ) -> "RenderedSFTSample":
        turn = chat_template.render_training_turn(
            sample.messages,
            sample.tools,
            sample.target,
        )
        if len(turn.input_ids) > cutoff_len:
            raise ValueError("semantic SFT sample exceeds cutoff_len and cannot be truncated")
        if sample.expected_output_kind in {"tool_call", "maintenance_tool_call"}:
            parsed_calls = parse_tool_calls(turn.raw_completion)
            if parsed_calls != sample.target.tool_calls:
                raise ValueError("rendered SFT tool calls do not round-trip through runtime parser")
            if sample.expected_output_kind == "maintenance_tool_call":
                from my_agent.memory.evolver.maintenance.formal.tools import (
                    parse_maintenance_tool_call,
                )

                parse_maintenance_tool_call(parsed_calls)
        elif parse_tool_calls(turn.raw_completion):
            raise ValueError("rendered non-tool SFT target unexpectedly parses as a tool call")
        prompt = turn.prompt_token_ids
        completion = turn.completion_token_ids
        mask = turn.assistant_loss_mask
        input_ids = turn.input_ids
        full_mask = (0,) * len(prompt) + mask
        labels = tuple(token if active else -100 for token, active in zip(input_ids, full_mask))
        payload = _payload_without_rendered_hash(
            sample_id=sample.sample_id,
            semantic_sample_hash=sample.sample_id,
            expected_output_kind=sample.expected_output_kind,
            expected_tool_call_count=sample.expected_tool_call_count,
            tokenizer_revision=tokenizer_revision,
            tokenizer_hash=tokenizer_hash,
            chat_template_hash=chat_template.template_hash,
            raw_completion=turn.raw_completion,
            prompt_token_ids=prompt,
            completion_token_ids=completion,
            assistant_loss_mask=mask,
            input_ids=input_ids,
            full_sequence_loss_mask=full_mask,
            labels=labels,
            cutoff_len=cutoff_len,
            truncated=False,
        )
        return cls(
            rendered_sample_hash=canonical_sha256(payload),
            sample_id=sample.sample_id,
            semantic_sample_hash=sample.sample_id,
            expected_output_kind=sample.expected_output_kind,
            expected_tool_call_count=sample.expected_tool_call_count,
            tokenizer_revision=tokenizer_revision,
            tokenizer_hash=tokenizer_hash,
            chat_template_hash=chat_template.template_hash,
            raw_completion=turn.raw_completion,
            prompt_token_ids=prompt,
            completion_token_ids=completion,
            assistant_loss_mask=mask,
            input_ids=input_ids,
            full_sequence_loss_mask=full_mask,
            labels=labels,
            cutoff_len=cutoff_len,
        )

    def payload_without_rendered_hash(self) -> dict[str, Any]:
        return _payload_without_rendered_hash(
            sample_id=self.sample_id,
            semantic_sample_hash=self.semantic_sample_hash,
            expected_output_kind=self.expected_output_kind,
            expected_tool_call_count=self.expected_tool_call_count,
            tokenizer_revision=self.tokenizer_revision,
            tokenizer_hash=self.tokenizer_hash,
            chat_template_hash=self.chat_template_hash,
            raw_completion=self.raw_completion,
            prompt_token_ids=self.prompt_token_ids,
            completion_token_ids=self.completion_token_ids,
            assistant_loss_mask=self.assistant_loss_mask,
            input_ids=self.input_ids,
            full_sequence_loss_mask=self.full_sequence_loss_mask,
            labels=self.labels,
            cutoff_len=self.cutoff_len,
            truncated=self.truncated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendered_sample_hash": self.rendered_sample_hash,
            **self.payload_without_rendered_hash(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderedSFTSample":
        if set(data) != _FIELDS:
            raise ValueError("rendered SFT sample fields do not match the schema")
        count = data["expected_tool_call_count"]
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int)
        ):
            raise ValueError("expected_tool_call_count must be an integer or null")
        cutoff_len = data["cutoff_len"]
        truncated = data["truncated"]
        if isinstance(cutoff_len, bool) or not isinstance(cutoff_len, int):
            raise ValueError("rendered SFT cutoff_len must be an integer")
        if not isinstance(truncated, bool):
            raise ValueError("rendered SFT truncated must be a boolean")
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            sample_id=_string(data["sample_id"], "sample_id"),
            semantic_sample_hash=_string(
                data["semantic_sample_hash"], "semantic_sample_hash"
            ),
            rendered_sample_hash=_string(
                data["rendered_sample_hash"], "rendered_sample_hash"
            ),
            expected_output_kind=_string(
                data["expected_output_kind"], "expected_output_kind"
            ),
            expected_tool_call_count=count,
            tokenizer_revision=_string(data["tokenizer_revision"], "tokenizer_revision"),
            tokenizer_hash=_string(data["tokenizer_hash"], "tokenizer_hash"),
            chat_template_hash=_string(data["chat_template_hash"], "chat_template_hash"),
            raw_completion=_string(data["raw_completion"], "raw_completion", allow_blank=False),
            prompt_token_ids=_int_tuple(data["prompt_token_ids"], "prompt_token_ids"),
            completion_token_ids=_int_tuple(
                data["completion_token_ids"], "completion_token_ids"
            ),
            assistant_loss_mask=_int_tuple(
                data["assistant_loss_mask"], "assistant_loss_mask"
            ),
            input_ids=_int_tuple(data["input_ids"], "input_ids"),
            full_sequence_loss_mask=_int_tuple(
                data["full_sequence_loss_mask"], "full_sequence_loss_mask"
            ),
            labels=_int_tuple(data["labels"], "labels"),
            cutoff_len=cutoff_len,
            truncated=truncated,
        )


@dataclass(frozen=True)
class RenderedSFTManifest:
    semantic_dataset_manifest_hash: str
    tokenizer_revision: str
    tokenizer_hash: str
    chat_template_hash: str
    cutoff_len: int
    split_rendered_sample_hashes: Mapping[str, tuple[str, ...]]
    rendered_manifest_hash: str
    schema_version: str = RENDERED_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RENDERED_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported rendered SFT manifest schema")
        for field_name in (
            "semantic_dataset_manifest_hash",
            "tokenizer_hash",
            "chat_template_hash",
            "rendered_manifest_hash",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if not self.tokenizer_revision.strip() or self.cutoff_len < 1:
            raise ValueError("rendered SFT manifest tokenizer/cutoff are invalid")
        if not self.split_rendered_sample_hashes:
            raise ValueError("rendered SFT manifest must contain at least one split")
        for split, hashes in self.split_rendered_sample_hashes.items():
            if not split or len(hashes) != len(set(hashes)):
                raise ValueError("rendered SFT manifest split hashes are invalid")
            for value in hashes:
                require_sha256(value, field_name="rendered sample hash")
        if self.rendered_manifest_hash != canonical_sha256(self.payload_without_hash()):
            raise ValueError("rendered SFT manifest hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        semantic_dataset_manifest_hash: str,
        tokenizer_revision: str,
        tokenizer_hash: str,
        chat_template_hash: str,
        cutoff_len: int,
        split_rendered_sample_hashes: Mapping[str, tuple[str, ...]],
    ) -> "RenderedSFTManifest":
        payload = _manifest_payload_without_hash(
            semantic_dataset_manifest_hash=semantic_dataset_manifest_hash,
            tokenizer_revision=tokenizer_revision,
            tokenizer_hash=tokenizer_hash,
            chat_template_hash=chat_template_hash,
            cutoff_len=cutoff_len,
            split_rendered_sample_hashes=split_rendered_sample_hashes,
        )
        return cls(
            semantic_dataset_manifest_hash=semantic_dataset_manifest_hash,
            tokenizer_revision=tokenizer_revision,
            tokenizer_hash=tokenizer_hash,
            chat_template_hash=chat_template_hash,
            cutoff_len=cutoff_len,
            split_rendered_sample_hashes={
                split: tuple(hashes)
                for split, hashes in split_rendered_sample_hashes.items()
            },
            rendered_manifest_hash=canonical_sha256(payload),
        )

    def payload_without_hash(self) -> dict[str, Any]:
        return _manifest_payload_without_hash(
            semantic_dataset_manifest_hash=self.semantic_dataset_manifest_hash,
            tokenizer_revision=self.tokenizer_revision,
            tokenizer_hash=self.tokenizer_hash,
            chat_template_hash=self.chat_template_hash,
            cutoff_len=self.cutoff_len,
            split_rendered_sample_hashes=self.split_rendered_sample_hashes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"rendered_manifest_hash": self.rendered_manifest_hash, **self.payload_without_hash()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderedSFTManifest":
        if set(data) != _MANIFEST_FIELDS:
            raise ValueError("rendered SFT manifest fields do not match the schema")
        split_hashes = data["split_rendered_sample_hashes"]
        if not isinstance(split_hashes, Mapping):
            raise ValueError("rendered SFT manifest splits must be an object")
        if any(not isinstance(split, str) for split in split_hashes):
            raise ValueError("rendered SFT manifest split names must be strings")
        cutoff_len = data["cutoff_len"]
        if isinstance(cutoff_len, bool) or not isinstance(cutoff_len, int):
            raise ValueError("rendered SFT manifest cutoff_len must be an integer")
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            semantic_dataset_manifest_hash=_string(
                data["semantic_dataset_manifest_hash"],
                "semantic_dataset_manifest_hash",
            ),
            tokenizer_revision=_string(data["tokenizer_revision"], "tokenizer_revision"),
            tokenizer_hash=_string(data["tokenizer_hash"], "tokenizer_hash"),
            chat_template_hash=_string(data["chat_template_hash"], "chat_template_hash"),
            cutoff_len=cutoff_len,
            split_rendered_sample_hashes={
                split: _string_tuple(hashes, "rendered sample hashes")
                for split, hashes in split_hashes.items()
            },
            rendered_manifest_hash=_string(
                data["rendered_manifest_hash"], "rendered_manifest_hash"
            ),
        )


def _payload_without_rendered_hash(
    *,
    sample_id: str,
    semantic_sample_hash: str,
    expected_output_kind: str,
    expected_tool_call_count: int | None,
    tokenizer_revision: str,
    tokenizer_hash: str,
    chat_template_hash: str,
    raw_completion: str,
    prompt_token_ids: tuple[int, ...],
    completion_token_ids: tuple[int, ...],
    assistant_loss_mask: tuple[int, ...],
    input_ids: tuple[int, ...],
    full_sequence_loss_mask: tuple[int, ...],
    labels: tuple[int, ...],
    cutoff_len: int,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "schema_version": RENDERED_SFT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "semantic_sample_hash": semantic_sample_hash,
        "expected_output_kind": expected_output_kind,
        "expected_tool_call_count": expected_tool_call_count,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_hash": tokenizer_hash,
        "chat_template_hash": chat_template_hash,
        "raw_completion": raw_completion,
        "prompt_token_ids": list(prompt_token_ids),
        "completion_token_ids": list(completion_token_ids),
        "assistant_loss_mask": list(assistant_loss_mask),
        "input_ids": list(input_ids),
        "full_sequence_loss_mask": list(full_sequence_loss_mask),
        "labels": list(labels),
        "cutoff_len": cutoff_len,
        "truncated": truncated,
    }


def _manifest_payload_without_hash(
    *,
    semantic_dataset_manifest_hash: str,
    tokenizer_revision: str,
    tokenizer_hash: str,
    chat_template_hash: str,
    cutoff_len: int,
    split_rendered_sample_hashes: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    return {
        "schema_version": RENDERED_MANIFEST_SCHEMA_VERSION,
        "semantic_dataset_manifest_hash": semantic_dataset_manifest_hash,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_hash": tokenizer_hash,
        "chat_template_hash": chat_template_hash,
        "cutoff_len": cutoff_len,
        "split_rendered_sample_hashes": {
            split: list(hashes)
            for split, hashes in sorted(split_rendered_sample_hashes.items())
        },
    }


def _int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"rendered SFT {field_name} must be an integer array")
    return tuple(value)


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a string array")
    return tuple(value)


def _string(value: Any, field_name: str, *, allow_blank: bool = False) -> str:
    if not isinstance(value, str) or (not allow_blank and not value):
        raise ValueError(f"rendered SFT {field_name} must be a string")
    return value


__all__ = ["RenderedSFTManifest", "RenderedSFTSample"]
