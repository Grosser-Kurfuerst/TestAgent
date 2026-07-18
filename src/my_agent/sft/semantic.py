"""Tokenizer-independent semantic samples for formal SFT warm start."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import json

from my_agent.policy.identity import canonical_sha256, require_sha256
from my_agent.sft.contracts import (
    CANONICAL_SFT_SCHEMA_VERSION,
    deterministic_tool_call_id,
    validate_expected_output_contract,
)
from my_agent.training.role_views import CanonicalMessage, CanonicalTool


_FIELDS = {
    "schema_version",
    "sample_id",
    "role",
    "purpose",
    "expected_output_kind",
    "expected_tool_call_count",
    "messages",
    "tools",
    "target",
    "metadata",
}
_ROLE_OUTPUT_KINDS = {
    "action": {"tool_call", "assistant_text"},
    "selection": {"selection_json"},
    "writing": {"writing_json"},
    "maintenance": {"maintenance_tool_call"},
}
_REQUIRED_METADATA = {
    "source",
    "source_id",
    "task_group",
    "repository_key",
    "quality_status",
}


@dataclass(frozen=True)
class SemanticSFTSample:
    sample_id: str
    role: str
    purpose: str
    expected_output_kind: str
    expected_tool_call_count: int | None
    messages: tuple[CanonicalMessage, ...]
    tools: tuple[CanonicalTool, ...]
    target: CanonicalMessage
    metadata: Mapping[str, Any]
    schema_version: str = CANONICAL_SFT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_SFT_SCHEMA_VERSION:
            raise ValueError("unsupported canonical SFT schema")
        require_sha256(self.sample_id, field_name="sample_id")
        if self.role not in _ROLE_OUTPUT_KINDS:
            raise ValueError(f"unsupported SFT role: {self.role!r}")
        if self.purpose != "sft_warm_start":
            raise ValueError("canonical SFT purpose must be sft_warm_start")
        validate_expected_output_contract(
            self.expected_output_kind,
            self.expected_tool_call_count,
        )
        if self.expected_output_kind not in _ROLE_OUTPUT_KINDS[self.role]:
            raise ValueError("SFT role and expected_output_kind do not match")
        if not self.messages:
            raise ValueError("canonical SFT messages must not be empty")
        if self.target.role != "assistant":
            raise ValueError("canonical SFT target must be an assistant message")
        tool_names = {tool.name for tool in self.tools}
        if len(tool_names) != len(self.tools):
            raise ValueError("canonical SFT tools must have unique names")
        _validate_history_bindings(self.messages, tool_names)
        _validate_target(self, tool_names)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("canonical SFT metadata must be an object")
        missing_metadata = sorted(_REQUIRED_METADATA - set(self.metadata))
        if missing_metadata:
            raise ValueError(
                "canonical SFT metadata is missing: " + ", ".join(missing_metadata)
            )
        if self.sample_id != canonical_sha256(self.payload_without_sample_id()):
            raise ValueError("canonical SFT sample_id does not match its payload")

    @classmethod
    def create(
        cls,
        *,
        role: str,
        expected_output_kind: str,
        expected_tool_call_count: int | None,
        messages: tuple[CanonicalMessage, ...],
        tools: tuple[CanonicalTool, ...],
        target: CanonicalMessage,
        metadata: Mapping[str, Any],
    ) -> "SemanticSFTSample":
        payload = _payload_without_sample_id(
            role=role,
            purpose="sft_warm_start",
            expected_output_kind=expected_output_kind,
            expected_tool_call_count=expected_tool_call_count,
            messages=messages,
            tools=tools,
            target=target,
            metadata=metadata,
        )
        return cls(
            sample_id=canonical_sha256(payload),
            role=role,
            purpose="sft_warm_start",
            expected_output_kind=expected_output_kind,
            expected_tool_call_count=expected_tool_call_count,
            messages=messages,
            tools=tools,
            target=target,
            metadata=dict(metadata),
        )

    def payload_without_sample_id(self) -> dict[str, Any]:
        return _payload_without_sample_id(
            role=self.role,
            purpose=self.purpose,
            expected_output_kind=self.expected_output_kind,
            expected_tool_call_count=self.expected_tool_call_count,
            messages=self.messages,
            tools=self.tools,
            target=self.target,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, **self.payload_without_sample_id()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticSFTSample":
        if set(data) != _FIELDS:
            raise ValueError("canonical SFT sample fields do not match the schema")
        messages = _object_array(data["messages"], "messages")
        tools = _object_array(data["tools"], "tools")
        target = data["target"]
        metadata = data["metadata"]
        if not isinstance(target, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("canonical SFT target and metadata must be objects")
        count = data["expected_tool_call_count"]
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int)
        ):
            raise ValueError("expected_tool_call_count must be an integer or null")
        return cls(
            schema_version=str(data["schema_version"]),
            sample_id=str(data["sample_id"]),
            role=str(data["role"]),
            purpose=str(data["purpose"]),
            expected_output_kind=str(data["expected_output_kind"]),
            expected_tool_call_count=count,
            messages=tuple(CanonicalMessage.from_dict(item) for item in messages),
            tools=tuple(CanonicalTool.from_dict(item) for item in tools),
            target=CanonicalMessage.from_dict(target),
            metadata=dict(metadata),
        )


def _payload_without_sample_id(
    *,
    role: str,
    purpose: str,
    expected_output_kind: str,
    expected_tool_call_count: int | None,
    messages: tuple[CanonicalMessage, ...],
    tools: tuple[CanonicalTool, ...],
    target: CanonicalMessage,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_SFT_SCHEMA_VERSION,
        "role": role,
        "purpose": purpose,
        "expected_output_kind": expected_output_kind,
        "expected_tool_call_count": expected_tool_call_count,
        "messages": [message.to_dict() for message in messages],
        "tools": [tool.to_dict() for tool in tools],
        "target": target.to_dict(),
        "metadata": dict(metadata),
    }


def _validate_history_bindings(
    messages: tuple[CanonicalMessage, ...],
    tool_names: set[str],
) -> None:
    pending_call_ids: list[str] = []
    for message in messages:
        if message.role == "assistant":
            if pending_call_ids:
                raise ValueError("assistant tool calls require bound observations before next turn")
            call_ids = [call.call_id for call in message.tool_calls]
            if len(call_ids) != len(set(call_ids)):
                raise ValueError("assistant tool call IDs must be unique within one turn")
            for call in message.tool_calls:
                if call.name not in tool_names:
                    raise ValueError(f"assistant history references unknown tool: {call.name}")
            pending_call_ids.extend(call_ids)
        elif message.role == "tool":
            if not pending_call_ids or message.tool_call_id != pending_call_ids[0]:
                raise ValueError("tool observation does not bind the next pending call ID")
            pending_call_ids.pop(0)
        elif pending_call_ids:
            raise ValueError("tool observations must immediately follow their assistant call")
    if pending_call_ids:
        raise ValueError("canonical SFT history contains unbound tool calls")


def _validate_target(sample: SemanticSFTSample, tool_names: set[str]) -> None:
    target = sample.target
    if sample.expected_output_kind in {"tool_call", "maintenance_tool_call"}:
        if target.content:
            raise ValueError("tool-call SFT targets must not contain hidden reasoning text")
        if len(target.tool_calls) != sample.expected_tool_call_count:
            raise ValueError("SFT target tool-call count does not match the contract")
        call_ids = [call.call_id for call in target.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("target tool call IDs must be unique")
        for index, call in enumerate(target.tool_calls):
            if call.name not in tool_names:
                raise ValueError(f"SFT target references unknown tool: {call.name}")
            expected_id = deterministic_tool_call_id(
                call_index=index,
                name=call.name,
                arguments=call.arguments_json,
            )
            if call.call_id != expected_id:
                raise ValueError("SFT target call_id does not match the deterministic rule")
        return
    if target.tool_calls:
        raise ValueError("non-tool SFT targets must not contain tool calls")
    if sample.expected_output_kind == "assistant_text":
        if not target.content.strip():
            raise ValueError("assistant_text target must not be blank")
        return
    try:
        payload = json.loads(target.content)
    except json.JSONDecodeError as exc:
        raise ValueError("structured SFT target must contain valid JSON") from exc
    expected_type = dict if sample.expected_output_kind == "selection_json" else list
    if not isinstance(payload, expected_type):
        raise ValueError("structured SFT target has the wrong JSON container type")


def _object_array(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"canonical SFT {field_name} must be an array of objects")
    return tuple(value)


__all__ = ["SemanticSFTSample"]
