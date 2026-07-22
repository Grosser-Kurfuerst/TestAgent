"""Formal maintenance tool schemas and safety-neutral operation translation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
import json

from my_agent.memory.evolver.maintenance.contracts import (
    MaintenanceAction,
    MaintenanceOperation,
    MaintenancePlanError,
    _operation_id,
    _source_precondition,
)
from my_agent.memory.experience.serialization import experience_payload_from_dict
from my_agent.memory.experience.models import ExperienceCreatedBy, ExperienceMemory
from my_agent.memory.token import estimate_tokens
from my_agent.memory.types import content_fingerprint
from my_agent.policy.identity import canonical_json_bytes, canonical_sha256
from my_agent.training.role_views import CanonicalTool, CanonicalToolCall


FORMAL_MAINTENANCE_TOOL_NAMES = frozenset({"lookup", "merge", "delete", "finish"})


@dataclass(frozen=True)
class MaintenanceToolCommand:
    call_id: str
    name: str
    arguments: Mapping[str, Any]


def formal_maintenance_tools() -> tuple[CanonicalTool, ...]:
    return tuple(_tool(name, description, parameters) for name, description, parameters in (
        (
            "lookup",
            "Inspect matching repository memories by content query or exact memory ID without mutating the staged plan.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "tiers": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        (
            "merge",
            "Stage a merge of same-tier project memories; the first source is the anchor.",
            {
                "type": "object",
                "properties": {
                    "source_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "replacement": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "payload": {"type": "object"},
                        },
                        "required": ["content", "payload"],
                        "additionalProperties": False,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["source_ids", "replacement", "reason"],
                "additionalProperties": False,
            },
        ),
        (
            "delete",
            "Stage deletion of repository memories with an explicit reason.",
            {
                "type": "object",
                "properties": {
                    "source_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "reason": {"type": "string"},
                },
                "required": ["source_ids", "reason"],
                "additionalProperties": False,
            },
        ),
        (
            "finish",
            "Finish the maintenance session and atomically apply all staged operations.",
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    ))


def parse_maintenance_tool_call(
    calls: tuple[CanonicalToolCall, ...],
) -> MaintenanceToolCommand:
    if len(calls) != 1:
        raise ValueError("formal maintenance requires exactly one tool call per assistant turn")
    call = calls[0]
    if call.name not in FORMAL_MAINTENANCE_TOOL_NAMES:
        raise ValueError(f"unsupported formal maintenance tool: {call.name}")
    arguments = json.loads(call.arguments_json)
    if not isinstance(arguments, Mapping):
        raise ValueError("maintenance tool arguments must be a JSON object")
    _validate_command_shape(call.name, arguments)
    return MaintenanceToolCommand(call.call_id, call.name, dict(arguments))


def build_delete_operation(
    command: MaintenanceToolCommand,
    *,
    repository_entries: Sequence[ExperienceMemory],
) -> MaintenanceOperation:
    source_ids = _source_ids(command.arguments)
    reason = _reason(command.arguments)
    sources = _sources(source_ids, repository_entries)
    operation_id = _operation_id(
        action=MaintenanceAction.DELETE,
        source_ids=source_ids,
        target_ids=(),
        replacements=(),
        additions=(),
    )
    return MaintenanceOperation(
        operation_id=operation_id,
        action=MaintenanceAction.DELETE,
        source_ids=source_ids,
        source_tiers=tuple(source.tier.value for source in sources),
        source_preconditions={
            source.id: _source_precondition(source, source.tier.value)
            for source in sources
        },
        reason_codes=(reason,),
        remove_ids=source_ids,
    )


def build_merge_operation(
    command: MaintenanceToolCommand,
    *,
    repository_entries: Sequence[ExperienceMemory],
) -> MaintenanceOperation:
    source_ids = _source_ids(command.arguments)
    if len(source_ids) < 2:
        raise MaintenancePlanError("merge requires at least two source_ids")
    reason = _reason(command.arguments)
    sources = _sources(source_ids, repository_entries)
    anchor = sources[0]
    replacement_payload = command.arguments["replacement"]
    assert isinstance(replacement_payload, Mapping)
    content = replacement_payload["content"]
    raw_payload = replacement_payload["payload"]
    if not isinstance(content, str) or not content.strip():
        raise MaintenancePlanError("merge replacement content must be a non-empty string")
    if not isinstance(raw_payload, Mapping):
        raise MaintenancePlanError("merge replacement payload must be an object")
    payload = experience_payload_from_dict(anchor.tier, raw_payload)
    provisional = replace(
        anchor,
        content=content.strip(),
        payload=payload,
        token_count=estimate_tokens(content.strip()),
        fingerprint=content_fingerprint(content.strip()),
        created_by=ExperienceCreatedBy.MAINTENANCE,
        maintenance_operation_id="",
    )
    operation_id = _operation_id(
        action=MaintenanceAction.MERGE,
        source_ids=source_ids,
        target_ids=(anchor.id,),
        replacements=(provisional,),
        additions=(),
    )
    replacement = replace(provisional, maintenance_operation_id=operation_id)
    return MaintenanceOperation(
        operation_id=operation_id,
        action=MaintenanceAction.MERGE,
        source_ids=source_ids,
        source_tiers=tuple(source.tier.value for source in sources),
        source_preconditions={
            source.id: _source_precondition(source, source.tier.value)
            for source in sources
        },
        target_ids=(anchor.id,),
        reason_codes=(reason,),
        remove_ids=tuple(source_ids[1:]),
        replacements=(replacement,),
    )


def _validate_command_shape(name: str, arguments: Mapping[str, Any]) -> None:
    expected = {
        "lookup": ({"query"}, {"query", "tiers", "limit"}),
        "merge": ({"source_ids", "replacement", "reason"}, {"source_ids", "replacement", "reason"}),
        "delete": ({"source_ids", "reason"}, {"source_ids", "reason"}),
        "finish": ({"summary"}, {"summary"}),
    }
    required, allowed = expected[name]
    if not required.issubset(arguments) or not set(arguments).issubset(allowed):
        raise ValueError(f"{name} tool arguments do not match the formal schema")
    if name == "lookup":
        if not isinstance(arguments["query"], str):
            raise ValueError("lookup query must be a string")
        tiers = arguments.get("tiers")
        if tiers is not None and (
            not isinstance(tiers, list) or any(not isinstance(item, str) for item in tiers)
        ):
            raise ValueError("lookup tiers must be an array of strings")
        limit = arguments.get("limit")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50
        ):
            raise ValueError("lookup limit must be an integer in [1, 50]")
    elif name == "merge":
        replacement = arguments["replacement"]
        if not isinstance(replacement, Mapping) or set(replacement) != {"content", "payload"}:
            raise ValueError("merge replacement fields do not match the formal schema")
        _source_ids(arguments)
        _reason(arguments)
    elif name == "delete":
        _source_ids(arguments)
        _reason(arguments)
    else:
        if not isinstance(arguments["summary"], str):
            raise ValueError("finish summary must be a string")


def _source_ids(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    raw = arguments.get("source_ids")
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) or not item for item in raw):
        raise ValueError("source_ids must be a non-empty array of strings")
    if len(set(raw)) != len(raw):
        raise ValueError("source_ids must be unique")
    return tuple(raw)


def _reason(arguments: Mapping[str, Any]) -> str:
    reason = arguments.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("maintenance reason must be a non-empty string")
    return reason.strip()


def _sources(
    source_ids: tuple[str, ...],
    repository_entries: Sequence[ExperienceMemory],
) -> tuple[ExperienceMemory, ...]:
    by_id = {entry.id: entry for entry in repository_entries}
    missing = [source_id for source_id in source_ids if source_id not in by_id]
    if missing:
        raise MaintenancePlanError(f"maintenance source is absent: {missing[0]}")
    return tuple(by_id[source_id] for source_id in source_ids)


def _tool(name: str, description: str, parameters: Mapping[str, Any]) -> CanonicalTool:
    parameters_json = canonical_json_bytes(dict(parameters)).decode("utf-8")
    return CanonicalTool(
        name=name,
        description=description,
        parameters_json=parameters_json,
        schema_hash=canonical_sha256(dict(parameters)),
    )


__all__ = [
    "FORMAL_MAINTENANCE_TOOL_NAMES",
    "MaintenanceToolCommand",
    "build_delete_operation",
    "build_merge_operation",
    "formal_maintenance_tools",
    "parse_maintenance_tool_call",
]
