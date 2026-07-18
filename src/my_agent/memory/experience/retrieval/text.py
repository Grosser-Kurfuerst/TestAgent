"""Deterministic searchable text and lexical terms for Experience memories."""

from __future__ import annotations

import json
import re
from typing import Any

from my_agent.memory.experience.models import (
    ExperienceMemory,
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
)
from my_agent.memory.types import normalize_content

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[一-鿿]+")
_FIELD_CHAR_LIMIT = 2_000
_STEP_CHAR_LIMIT = 600
_TOTAL_CHAR_LIMIT = 12_000
_SCHEMA_MAX_DEPTH = 3
_SCHEMA_MAX_ITEMS = 64
_SCHEMA_SINGLE_CHILD_KEYS = (
    "additionalProperties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
)
_SCHEMA_SEQUENCE_CHILD_KEYS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_MAPPING_CHILD_KEYS = (
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
    "patternProperties",
)


def experience_searchable_text(memory: ExperienceMemory) -> str:
    """Build deterministic, bounded semantic text from a typed experience."""
    if not isinstance(memory, ExperienceMemory):
        raise TypeError("memory must be an ExperienceMemory")

    pieces: list[str] = [memory.content]
    payload = memory.payload
    if isinstance(payload, TrajectoryPayload):
        pieces.extend((payload.task_description, *payload.key_learnings, *payload.tags))
        for step in payload.steps:
            if step.reward is None or step.reward <= 0:
                continue
            pieces.append(
                " ".join(
                    part
                    for part in (step.observation, step.action, step.result)
                    if part
                )[:_STEP_CHAR_LIMIT]
            )
    elif isinstance(payload, TipPayload):
        pieces.extend((payload.category, payload.severity, payload.trigger))
    elif isinstance(payload, SkillPayload):
        pieces.extend((payload.category, payload.technique, *payload.preconditions, *payload.steps))
    elif isinstance(payload, ToolPayload):
        pieces.extend((
            payload.name,
            payload.language,
            payload.code,
            payload.command,
            payload.input_description,
            payload.output_description,
            *_schema_search_fragments(payload.args_schema),
            payload.repo_context,
        ))
    else:  # pragma: no cover - ExperienceMemory closes the payload union
        raise TypeError(f"unsupported experience payload: {type(payload).__name__}")

    normalized: list[str] = []
    seen: set[str] = set()
    used_chars = 0
    for piece in pieces:
        text = normalize_content(str(piece or ""))[:_FIELD_CHAR_LIMIT]
        if not text or text in seen:
            continue
        remaining = _TOTAL_CHAR_LIMIT - used_chars
        if remaining <= 0:
            break
        text = text[:remaining]
        if not text:
            break
        seen.add(text)
        normalized.append(text)
        used_chars += len(text) + 1
    return " ".join(normalized)


def experience_index_terms(memory: ExperienceMemory) -> frozenset[str]:
    """Return token and n-gram postings used for high-recall candidate lookup."""
    terms: set[str] = set()
    for token in tokenize_experience_text(experience_searchable_text(memory)):
        terms.add(token)
        if len(token) >= 2:
            terms.update(token[index:index + 2] for index in range(len(token) - 1))
        if len(token) >= 3:
            terms.update(token[index:index + 3] for index in range(len(token) - 2))
    return frozenset(terms)


def tokenize_experience_text(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(_TOKEN_RE.findall(text.casefold()))


def _schema_search_fragments(schema: dict[str, Any]) -> tuple[str, ...]:
    fragments: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str) or len(fragments) >= _SCHEMA_MAX_ITEMS:
            return
        text = value[:_FIELD_CHAR_LIMIT]
        if not text or text in seen:
            return
        seen.add(text)
        fragments.append(text)

    def visit_schema(value: Any, *, depth: int) -> None:
        if (
            depth > _SCHEMA_MAX_DEPTH
            or len(fragments) >= _SCHEMA_MAX_ITEMS
            or not isinstance(value, dict)
        ):
            return

        add(value.get("description"))

        properties = value.get("properties")
        if isinstance(properties, dict):
            for field_name in sorted(properties, key=str):
                if len(fragments) >= _SCHEMA_MAX_ITEMS:
                    return
                add(str(field_name))
                visit_schema(properties[field_name], depth=depth + 1)

        for key in _SCHEMA_SINGLE_CHILD_KEYS:
            child = value.get(key)
            if isinstance(child, dict):
                visit_schema(child, depth=depth + 1)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    visit_schema(item, depth=depth + 1)

        for key in _SCHEMA_SEQUENCE_CHILD_KEYS:
            children = value.get(key)
            if isinstance(children, (list, tuple)):
                for child in children:
                    visit_schema(child, depth=depth + 1)

        for key in _SCHEMA_MAPPING_CHILD_KEYS:
            children = value.get(key)
            if not isinstance(children, dict):
                continue
            for child_name in sorted(children, key=str):
                visit_schema(children[child_name], depth=depth + 1)

    visit_schema(schema, depth=0)
    json.dumps(fragments, ensure_ascii=False, allow_nan=False)
    return tuple(fragments)


__all__ = [
    "experience_index_terms",
    "experience_searchable_text",
    "tokenize_experience_text",
]
