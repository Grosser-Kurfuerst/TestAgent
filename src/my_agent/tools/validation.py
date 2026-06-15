from __future__ import annotations

from typing import Any


def validate_arguments_schema(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("type") != "object":
        return ["tool parameters schema must be an object schema"]

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return ["tool parameters.properties must be an object"]

    required = schema.get("required", [])
    if required is None:
        required = []
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        return ["tool parameters.required must be a string array"]

    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument: {key}")

    if schema.get("additionalProperties") is False:
        for key in sorted(set(arguments) - set(properties)):
            errors.append(f"undeclared argument: {key}")

    for key, value in arguments.items():
        property_schema = properties.get(key)
        if isinstance(property_schema, dict):
            errors.extend(_validate_value(key, value, property_schema))
    return errors


def _validate_value(key: str, value: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        errors.append(f"argument {key} must be {expected_type}")
        return errors

    if "enum" in schema:
        allowed = schema["enum"]
        if isinstance(allowed, list) and value not in allowed:
            errors.append(f"argument {key} must be one of {allowed}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"argument {key} must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"argument {key} must be <= {maximum}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"argument {key} length must be >= {min_length}")
        if isinstance(max_length, int) and len(value) > max_length:
            errors.append(f"argument {key} length must be <= {max_length}")

    if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        errors.extend(f"{key}.{error}" for error in validate_arguments_schema(schema, value))
    return errors


def _matches_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True
