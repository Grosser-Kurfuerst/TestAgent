"""Immutable semantic, rendered, evaluation, and call-ID SFT contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import json

from my_agent.policy.identity import canonical_json_bytes, canonical_sha256


CANONICAL_SFT_SCHEMA_VERSION = "agentcli-canonical-sft-v1"
RENDERED_SFT_SCHEMA_VERSION = "agentcli-rendered-sft-v1"
DATASET_MANIFEST_SCHEMA_VERSION = "agentcli-sft-dataset-manifest-v1"
RENDERED_MANIFEST_SCHEMA_VERSION = "agentcli-rendered-sft-manifest-v1"
SFT_RUN_MANIFEST_SCHEMA_VERSION = "agentcli-sft-run-manifest-v1"
EXPECTED_OUTPUT_KINDS = frozenset({
    "tool_call",
    "assistant_text",
    "selection_json",
    "writing_json",
    "maintenance_tool_call",
})
TOOL_CALL_OUTPUT_KINDS = frozenset({"tool_call", "maintenance_tool_call"})
ENVIRONMENT_EXCLUSION_CODES = frozenset({
    "sandbox_unavailable",
    "fixture_setup_failed",
    "required_dependency_unavailable",
})


def deterministic_tool_call_id(
    *,
    call_index: int,
    name: str,
    arguments: Mapping[str, Any] | str,
) -> str:
    if isinstance(call_index, bool) or not isinstance(call_index, int) or call_index < 0:
        raise ValueError("call_index must be a non-negative integer")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call name must not be blank")
    if isinstance(arguments, str):
        try:
            arguments_object = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool call arguments must be valid JSON") from exc
        if canonical_json_bytes(arguments_object).decode("utf-8") != arguments:
            raise ValueError("tool call arguments string must use canonical JSON")
    else:
        arguments_object = dict(arguments)
    if not isinstance(arguments_object, dict):
        raise ValueError("tool call arguments must be a JSON object")
    digest = canonical_sha256({
        "index": call_index,
        "name": name,
        "arguments": arguments_object,
    })
    return "call_" + digest[7:19]


def validate_expected_output_contract(
    expected_output_kind: str,
    expected_tool_call_count: int | None,
) -> None:
    if expected_output_kind not in EXPECTED_OUTPUT_KINDS:
        raise ValueError(f"unsupported expected_output_kind: {expected_output_kind!r}")
    if expected_output_kind in TOOL_CALL_OUTPUT_KINDS:
        if (
            isinstance(expected_tool_call_count, bool)
            or not isinstance(expected_tool_call_count, int)
            or expected_tool_call_count < 1
        ):
            raise ValueError("tool-call outputs require a positive expected_tool_call_count")
    elif expected_tool_call_count is not None:
        raise ValueError("non-tool outputs require expected_tool_call_count=null")


__all__ = [
    "CANONICAL_SFT_SCHEMA_VERSION",
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "ENVIRONMENT_EXCLUSION_CODES",
    "EXPECTED_OUTPUT_KINDS",
    "RENDERED_MANIFEST_SCHEMA_VERSION",
    "RENDERED_SFT_SCHEMA_VERSION",
    "SFT_RUN_MANIFEST_SCHEMA_VERSION",
    "TOOL_CALL_OUTPUT_KINDS",
    "deterministic_tool_call_id",
    "validate_expected_output_contract",
]
