from __future__ import annotations

from typing import Any


def format_mcp_status(rows: list[dict[str, Any]]) -> str:
    lines = ["MCP servers"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    lines.append("name\tstatus\ttransport\ttools\tpid\terror")
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row.get("name", "")),
                    str(row.get("status", "")),
                    str(row.get("transport", "")),
                    str(row.get("tools", 0)),
                    _optional_text(row.get("pid")),
                    _optional_text(row.get("error")),
                ]
            )
        )
    return "\n".join(lines)


def format_mcp_disabled() -> str:
    return "MCP servers\n- disabled"


def format_mcp_logs(server_name: str, logs: list[str]) -> str:
    lines = [f"MCP logs: {server_name}"]
    if not logs:
        lines.append("- none")
        return "\n".join(lines)
    lines.extend(logs)
    return "\n".join(lines)


def format_mcp_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "0 servers"
    total = len(rows)
    ready = sum(1 for row in rows if row.get("status") == "ready")
    tools = sum(_int_value(row.get("tools")) for row in rows)
    errors = sum(1 for row in rows if row.get("status") == "error")
    suffix = f", {errors} error" if errors == 1 else f", {errors} errors" if errors else ""
    return f"{ready}/{total} ready, {tools} tools{suffix}"


def _optional_text(value: object) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _int_value(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
