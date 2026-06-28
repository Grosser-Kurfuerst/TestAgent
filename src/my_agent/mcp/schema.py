from __future__ import annotations

import json
from typing import Any


MAX_DESCRIPTION_CHARS = 1000
_REMOVED_KEYS = {"$schema", "$id", "$ref"}


def sanitize_schema(schema: object) -> dict[str, Any]:
    cleaned = _clean(schema)
    if not isinstance(cleaned, dict):
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
            "description": _truncate_description(f"Original MCP schema was non-object: {_compact_json(cleaned)}"),
        }
    if cleaned.get("type") not in (None, "object"):
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
            "description": _truncate_description(f"Original MCP schema was non-object: {_compact_json(cleaned)}"),
        }

    result = dict(cleaned)
    result["type"] = "object"
    properties = result.get("properties")
    if not isinstance(properties, dict):
        result["properties"] = {}
    else:
        result["properties"] = properties
    if "additionalProperties" not in result:
        result["additionalProperties"] = True
    description = result.get("description")
    if isinstance(description, str):
        result["description"] = _truncate_description(description)
    return result


def _clean(value: object) -> object:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        alternatives: list[str] = []
        for key, child in value.items():
            if key in _REMOVED_KEYS:
                continue
            if key in {"anyOf", "oneOf"}:
                alternatives.append(_describe_union(key, child))
                continue
            cleaned[key] = _clean(child)
        if alternatives:
            existing = cleaned.get("description")
            prefix = existing if isinstance(existing, str) and existing.strip() else ""
            suffix = "; ".join(item for item in alternatives if item)
            description = f"{prefix} ({suffix})" if prefix and suffix else suffix or prefix
            if description:
                cleaned["description"] = _truncate_description(description)
            if "type" not in cleaned:
                inferred = _infer_union_type(value)
                if inferred:
                    cleaned["type"] = inferred
        description = cleaned.get("description")
        if isinstance(description, str):
            cleaned["description"] = _truncate_description(description)
        if "properties" in cleaned and "type" not in cleaned:
            cleaned["type"] = "object"
        return cleaned
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _describe_union(keyword: str, value: object) -> str:
    if not isinstance(value, list):
        return f"{keyword} options: {_compact_json(value)}"
    options: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item_type = item.get("type")
            item_description = item.get("description")
            if isinstance(item_type, str) and isinstance(item_description, str) and item_description.strip():
                options.append(f"{item_type} ({item_description.strip()})")
            elif isinstance(item_type, str):
                options.append(item_type)
            else:
                options.append(_compact_json(item))
        else:
            options.append(_compact_json(item))
    return f"{keyword} options: {', '.join(options)}"


def _infer_union_type(schema: dict[str, object]) -> str | list[str] | None:
    for key in ("anyOf", "oneOf"):
        raw = schema.get(key)
        if not isinstance(raw, list):
            continue
        types: list[str] = []
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("type"), str):
                types.append(str(item["type"]))
        unique = list(dict.fromkeys(types))
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            return unique
    return None


def _compact_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def _truncate_description(description: str) -> str:
    if len(description) <= MAX_DESCRIPTION_CHARS:
        return description
    return description[:MAX_DESCRIPTION_CHARS] + "..."
