from __future__ import annotations

import json
from typing import Any

from my_agent.hitl.types import ApprovalRequest

SENSITIVE_KEYS = {"token", "api_key", "apikey", "password", "authorization", "secret", "credential"}


def is_wide_codepoint(cp: int) -> bool:
    return (
        0x1100 <= cp <= 0x115F
        or 0x2E80 <= cp <= 0x9FFF
        or 0xA000 <= cp <= 0xA4CF
        or 0xAC00 <= cp <= 0xD7A3
        or 0xF900 <= cp <= 0xFAFF
        or 0xFE30 <= cp <= 0xFE4F
        or 0xFF00 <= cp <= 0xFF60
        or 0xFFE0 <= cp <= 0xFFE6
        or 0x2600 <= cp <= 0x27BF
        or 0x1F300 <= cp <= 0x1FAFF
    )


def display_width(text: str) -> int:
    width = 0
    for char in text:
        cp = ord(char)
        if cp < 32 or 0x7F <= cp <= 0x9F:
            continue
        width += 2 if is_wide_codepoint(cp) else 1
    return width


def pad_right_display(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def truncate_display(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    suffix = "..."
    target = max(0, width - display_width(suffix))
    out = ""
    used = 0
    for char in text:
        char_width = display_width(char)
        if used + char_width > target:
            break
        out += char
        used += char_width
    return out + suffix


def wrap_display(text: str, width: int) -> list[str]:
    if width <= 0:
        return [""]
    lines: list[str] = []
    for raw_line in str(text).splitlines() or [""]:
        current = ""
        current_width = 0
        for char in raw_line:
            char_width = display_width(char)
            if current and current_width + char_width > width:
                lines.append(current)
                current = ""
                current_width = 0
            current += char
            current_width += char_width
        lines.append(current)
    return lines


def summarize_arguments(arguments: dict[str, Any], *, string_limit: int = 120) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in arguments.items():
        lowered = key.lower()
        if any(sensitive in lowered for sensitive in SENSITIVE_KEYS):
            summary[key] = "<redacted>"
            continue
        if isinstance(value, str):
            if len(value) > string_limit:
                summary[key] = f"{value[:string_limit]}... <chars={len(value)}>"
            else:
                summary[key] = value
            continue
        if isinstance(value, (int, float, bool)) or value is None:
            summary[key] = value
            continue
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        summary[key] = rendered if len(rendered) <= string_limit else f"{rendered[:string_limit]}... <chars={len(rendered)}>"
    return summary


def format_arguments_for_display(arguments_json: str) -> list[str]:
    try:
        payload = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return wrap_display(arguments_json, 66)
    if not isinstance(payload, dict):
        return wrap_display(arguments_json, 66)
    summary = summarize_arguments(payload)
    if not summary:
        return ["{}"]
    return [f"{key}: {value}" for key, value in summary.items()]


def format_approval_box(request: ApprovalRequest, *, width: int = 72) -> str:
    inner_width = max(20, width - 4)
    top = "+" + "-" * (inner_width + 2) + "+"
    lines = [top]
    title = f"HITL approval: {request.tool_name}"
    lines.append(f"| {pad_right_display(truncate_display(title, inner_width), inner_width)} |")
    lines.append(f"| {pad_right_display(truncate_display('Risk: ' + request.risk_level.value, inner_width), inner_width)} |")
    lines.append(f"| {pad_right_display(truncate_display(request.risk_description, inner_width), inner_width)} |")
    if request.sensitive_notice:
        for line in wrap_display("Notice: " + request.sensitive_notice, inner_width):
            lines.append(f"| {pad_right_display(truncate_display(line, inner_width), inner_width)} |")
    lines.append("| " + pad_right_display(truncate_display("Arguments:", inner_width), inner_width) + " |")
    for arg_line in format_arguments_for_display(request.arguments_json):
        for wrapped in wrap_display(str(arg_line), inner_width):
            lines.append(f"| {pad_right_display(truncate_display(wrapped, inner_width), inner_width)} |")
    options = "[Enter/y] approve  [a] approve all  [m] modify  [n] reject  [s] skip"
    lines.append(f"| {pad_right_display(truncate_display(options, inner_width), inner_width)} |")
    lines.append(top)
    return "\n".join(lines)
