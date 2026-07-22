"""Protocol metrics for the four formal OPD decision roles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import json

from my_agent.memory.evolver.maintenance.formal.tools import (
    parse_maintenance_tool_call,
)
from my_agent.memory.evolver.selection.formal import parse_selection_response
from my_agent.memory.evolver.writing.formal import parse_writing_response
from my_agent.memory.evolver.writing.validation import ExperienceProposalValidator
from my_agent.policy.identity import canonical_json_bytes
from my_agent.policy.transformers_policy import parse_tool_calls
from my_agent.training.contracts import DecisionEvent
from my_agent.training.decision_log import load_decision_events
from my_agent.training.role_views import CanonicalToolCall, SelectionPublic


FORMAL_ROLE_PROTOCOL_SCHEMA_VERSION = "agentcli-formal-role-protocol-eval-v1"
FORMAL_ROLES = ("selection", "action", "writing", "maintenance")
DEFAULT_THRESHOLDS = {
    "selection.schema_valid_rate": 0.98,
    "selection.unknown_label_rate": 0.0,
    "action.decision_success_rate": 0.95,
    "action.runtime_tool_call_parse_rate": 0.95,
    "action.unknown_tool_rate": 0.0,
    "writing.json_array_rate": 0.95,
    "writing.validator_accept_rate": 0.90,
    "maintenance.runtime_tool_call_parse_rate": 0.95,
    "maintenance.unknown_memory_id_rate": 0.0,
}


def evaluate_formal_role_events(
    events: Sequence[DecisionEvent],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = [_score_event(event) for event in events]
    by_role = {
        role: [item for item in details if item["role"] == role]
        for role in FORMAL_ROLES
    }
    roles = {
        "selection": _selection_metrics(by_role["selection"]),
        "action": _action_metrics(by_role["action"]),
        "writing": _writing_metrics(by_role["writing"]),
        "maintenance": _maintenance_metrics(by_role["maintenance"]),
    }
    checks = {
        name: _threshold_check(roles, name, threshold)
        for name, threshold in DEFAULT_THRESHOLDS.items()
    }
    available_checks = [value for value in checks.values() if value is not None]
    coverage_complete = all(by_role[role] for role in FORMAL_ROLES)
    summary = {
        "schema_version": FORMAL_ROLE_PROTOCOL_SCHEMA_VERSION,
        "n_events": len(details),
        "role_coverage": {role: len(by_role[role]) for role in FORMAL_ROLES},
        "coverage_complete": coverage_complete,
        "roles": roles,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "checks": checks,
        "all_available_checks_pass": bool(available_checks) and all(available_checks),
        "acceptance_pass": coverage_complete and all(value is True for value in checks.values()),
    }
    return summary, details


def run_formal_role_protocol_evaluation(
    *,
    decision_events_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(decision_events_path)
    events = load_decision_events(source)
    summary, details = evaluate_formal_role_events(events)
    summary["decision_events_path"] = str(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "detailed_results.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "experiment_report.md").write_text(
        render_formal_role_protocol_report(summary),
        encoding="utf-8",
    )
    return summary


def render_formal_role_protocol_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Formal Role Protocol Evaluation",
        "",
        f"- Decision events: {summary['n_events']}",
        f"- Complete four-role coverage: {summary['coverage_complete']}",
        f"- Available threshold checks pass: {summary['all_available_checks_pass']}",
        f"- Acceptance pass: {summary['acceptance_pass']}",
        "",
        "## Role metrics",
        "",
        "| Role | Events | Decision success | Primary protocol metric | Unknown reference |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    roles = summary["roles"]
    for role in FORMAL_ROLES:
        metrics = roles[role]
        if role == "selection":
            primary = metrics["schema_valid_rate"]
            unknown = metrics["unknown_label_rate"]
        elif role == "writing":
            primary = metrics["validator_accept_rate"]
            unknown = None
        else:
            primary = metrics["runtime_tool_call_parse_rate"]
            unknown = metrics["unknown_tool_rate"] if role == "action" else metrics["unknown_memory_id_rate"]
        lines.append(
            f"| {role} | {metrics['n_events']} | {_format_rate(metrics['decision_success_rate'])} "
            f"| {_format_rate(primary)} | {_format_rate(unknown)} |"
        )
    lines.extend([
        "",
        "## Threshold checks",
        "",
    ])
    for name, threshold in summary["thresholds"].items():
        result = summary["checks"][name]
        lines.append(f"- `{name}` target `{threshold}`: {_format_check(result)}")
    lines.extend([
        "",
        "A missing role is reported as unavailable rather than passing. Rerun collection with "
        "enough tasks and non-empty memory candidates before accepting four-role coverage.",
        "",
    ])
    return "\n".join(lines)


def _score_event(event: DecisionEvent) -> dict[str, Any]:
    base = {
        "decision_id": event.decision_id,
        "trajectory_id": event.trajectory_id,
        "role": event.role,
        "status": event.status,
        "decision_success": event.status == "success",
        "error": _event_error(event),
    }
    if event.role == "selection":
        base.update(_score_selection(event))
    elif event.role == "action":
        base.update(_score_action(event))
    elif event.role == "writing":
        base.update(_score_writing(event))
    else:
        base.update(_score_maintenance(event))
    return base


def _score_selection(event: DecisionEvent) -> dict[str, Any]:
    schema_valid = False
    unknown_label = "unknown candidate label" in _event_error(event)
    try:
        payload = _json_value(event.raw_completion)
        public = SelectionPublic.from_dict(_user_payload(event)["public_view"])
        parse_selection_response(
            canonical_json_bytes(payload).decode("utf-8"),
            candidates=public.candidates,
        )
        schema_valid = True
    except (KeyError, TypeError, ValueError):
        pass
    return {
        "schema_valid": schema_valid,
        "unknown_label": unknown_label,
    }


def _score_action(event: DecisionEvent) -> dict[str, Any]:
    calls, parse_error = _tool_calls(event)
    tool_attempt = bool(calls) or _looks_like_tool_call(event.raw_completion) or bool(parse_error)
    known_tools = {tool.name for tool in event.canonical_tools}
    unknown_tool = any(call.name not in known_tools for call in calls)
    return {
        "tool_call_attempt": tool_attempt,
        "runtime_tool_call_parse": bool(calls) and not parse_error,
        "unknown_tool": unknown_tool,
        "tool_names": [call.name for call in calls],
        "parse_error": parse_error,
    }


def _score_writing(event: DecisionEvent) -> dict[str, Any]:
    payload: Any = None
    json_array = False
    validator_accept = False
    proposal_count = 0
    try:
        payload = _json_value(event.raw_completion)
        json_array = isinstance(payload, list)
        prompt = _user_payload(event)
        min_confidence = float(prompt.get("min_confidence", 0.70))
        max_records = int(prompt.get("max_records", 6))
        proposals = parse_writing_response(
            canonical_json_bytes(payload).decode("utf-8"),
            validator=ExperienceProposalValidator(
                min_confidence=min_confidence,
                max_records=max_records,
            ),
        )
        validator_accept = True
        proposal_count = len(proposals)
    except (KeyError, TypeError, ValueError):
        pass
    return {
        "json_array": json_array,
        "validator_accept": validator_accept,
        "proposal_count": proposal_count,
    }


def _score_maintenance(event: DecisionEvent) -> dict[str, Any]:
    calls, parse_error = _tool_calls(event)
    command = None
    if not parse_error:
        try:
            command = parse_maintenance_tool_call(calls)
        except ValueError as exc:
            parse_error = str(exc)
    snapshot_ids = _maintenance_snapshot_ids(event)
    unknown_ids: list[str] = []
    if command is not None and command.name in {"merge", "delete"}:
        source_ids = command.arguments.get("source_ids", [])
        if isinstance(source_ids, list):
            unknown_ids = [str(item) for item in source_ids if item not in snapshot_ids]
    return {
        "runtime_tool_call_parse": command is not None and not parse_error,
        "tool_name": command.name if command is not None else "",
        "unknown_memory_ids": unknown_ids,
        "has_memory_reference": command is not None and command.name in {"merge", "delete"},
        "parse_error": parse_error,
    }


def _selection_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n_events": len(details),
        "decision_success_rate": _rate(item["decision_success"] for item in details),
        "schema_valid_rate": _rate(item["schema_valid"] for item in details),
        "unknown_label_rate": _rate(item["unknown_label"] for item in details),
    }


def _action_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempts = [item for item in details if item["tool_call_attempt"]]
    parsed = [item for item in attempts if item["runtime_tool_call_parse"]]
    return {
        "n_events": len(details),
        "decision_success_rate": _rate(item["decision_success"] for item in details),
        "n_tool_call_attempts": len(attempts),
        "runtime_tool_call_parse_rate": _rate(item["runtime_tool_call_parse"] for item in attempts),
        "unknown_tool_rate": _rate(item["unknown_tool"] for item in parsed),
    }


def _writing_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n_events": len(details),
        "decision_success_rate": _rate(item["decision_success"] for item in details),
        "json_array_rate": _rate(item["json_array"] for item in details),
        "validator_accept_rate": _rate(item["validator_accept"] for item in details),
        "non_empty_proposal_rate": _rate(item["proposal_count"] > 0 for item in details),
    }


def _maintenance_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    references = [item for item in details if item["has_memory_reference"]]
    return {
        "n_events": len(details),
        "decision_success_rate": _rate(item["decision_success"] for item in details),
        "runtime_tool_call_parse_rate": _rate(item["runtime_tool_call_parse"] for item in details),
        "n_memory_reference_calls": len(references),
        "unknown_memory_id_rate": _rate(bool(item["unknown_memory_ids"]) for item in details),
    }


def _tool_calls(event: DecisionEvent) -> tuple[tuple[CanonicalToolCall, ...], str]:
    parsed = event.parsed_output
    raw_calls = parsed.get("tool_calls")
    if isinstance(raw_calls, list):
        try:
            return tuple(CanonicalToolCall.from_dict(item) for item in raw_calls), ""
        except (TypeError, ValueError) as exc:
            return (), str(exc)
    raw_call = parsed.get("tool_call")
    if isinstance(raw_call, Mapping):
        try:
            call = CanonicalToolCall(
                str(raw_call.get("call_id") or "protocol-call"),
                str(raw_call.get("name") or ""),
                canonical_json_bytes(raw_call.get("arguments", {})).decode("utf-8"),
            )
            return (call,), ""
        except (TypeError, ValueError) as exc:
            return (), str(exc)
    try:
        return parse_tool_calls(event.raw_completion), ""
    except ValueError as exc:
        return (), str(exc)


def _user_payload(event: DecisionEvent) -> Mapping[str, Any]:
    for message in reversed(event.canonical_messages):
        if message.role != "user":
            continue
        payload = json.loads(message.content)
        if isinstance(payload, Mapping):
            return payload
    raise ValueError("decision event has no JSON user payload")


def _maintenance_snapshot_ids(event: DecisionEvent) -> set[str]:
    try:
        public = _user_payload(event)["public_view"]
        snapshot = public["repository_snapshot"]
        ids = snapshot["memory_ids"]
    except (KeyError, TypeError, ValueError):
        return set()
    return {str(item) for item in ids} if isinstance(ids, list) else set()


def _json_value(text: str) -> Any:
    stripped = text.strip()
    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
    if not starts:
        raise ValueError("completion does not contain JSON")
    value, _end = json.JSONDecoder().raw_decode(stripped[min(starts):])
    return value


def _looks_like_tool_call(text: str) -> bool:
    stripped = text.strip()
    return (
        "<tool_call" in stripped
        or "<function=" in stripped
        or '"tool_calls"' in stripped
        or '"tool"' in stripped
        or '"name"' in stripped
    )


def _event_error(event: DecisionEvent) -> str:
    error = event.parsed_output.get("error")
    return str(error or "")


def _rate(values: Any) -> float | None:
    items = [1.0 if bool(value) else 0.0 for value in values]
    return sum(items) / len(items) if items else None


def _threshold_check(
    roles: Mapping[str, Mapping[str, Any]],
    name: str,
    threshold: float,
) -> bool | None:
    role, metric = name.split(".", 1)
    value = roles[role][metric]
    if value is None:
        return None
    if metric.startswith("unknown_"):
        return float(value) <= threshold
    return float(value) >= threshold


def _format_rate(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.1f}%"


def _format_check(value: Any) -> str:
    if value is None:
        return "UNAVAILABLE"
    return "PASS" if value else "FAIL"


__all__ = [
    "DEFAULT_THRESHOLDS",
    "FORMAL_ROLE_PROTOCOL_SCHEMA_VERSION",
    "evaluate_formal_role_events",
    "render_formal_role_protocol_report",
    "run_formal_role_protocol_evaluation",
]
