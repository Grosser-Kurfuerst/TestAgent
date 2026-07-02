from __future__ import annotations

from my_agent.plan import PlanState, PlanTask
from my_agent.team import ExecutionStep, TeamState
from my_agent.tools import ToolExecutionResult, ToolInvocation
from my_agent.ui.renderer import Renderer


def dispatch_repl_event(event: object, renderer: Renderer) -> dict[str, object] | None:
    event_name = getattr(event, "event", "")
    payload = getattr(event, "payload", {})
    if not isinstance(payload, dict):
        return None
    if event_name == "plan.started":
        plan = _plan_from_payload(payload)
        if plan is not None:
            renderer.plan_started(plan)
    elif event_name.startswith("plan.task."):
        task = _task_from_payload(payload)
        if task is not None:
            renderer.plan_task_updated(task, plan_id=str(payload.get("plan_id", "")))
    elif event_name in {"plan.completed", "plan.failed", "plan.cancelled", "plan.validation_failed"}:
        plan = _plan_from_payload(payload)
        if plan is not None:
            renderer.plan_completed(plan)
    elif event_name == "team.started":
        team = _team_from_payload(payload)
        if team is not None:
            renderer.team_started(team)
    elif event_name.startswith("team.step."):
        step = _team_step_from_payload(payload)
        if step is not None:
            renderer.team_step_updated(step, team_id=str(payload.get("team_id", "")))
    elif event_name in {"team.completed", "team.failed", "team.cancelled", "team.validation_failed"}:
        team = _team_from_payload(payload)
        if team is not None:
            renderer.team_completed(team)
    elif event_name == "tool.started":
        invocation = ToolInvocation(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            arguments_json=str(payload.get("arguments", "{}")),
        )
        renderer.tool_call_started(invocation)
    elif event_name == "tool.completed":
        result = ToolExecutionResult(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            ok=bool(payload.get("ok")),
            content=str(payload.get("content", "")),
            elapsed_ms=int(payload.get("elapsed_ms", 0) or 0),
            error_code=str(payload.get("error_code", "") or ""),
            retryable=bool(payload.get("retryable")),
            blocked=bool(payload.get("blocked")),
            timed_out=bool(payload.get("timed_out")),
        )
        renderer.tool_call_completed(result)
    elif event_name == "render.flush_requested":
        renderer.reset_between_iterations()
    elif event_name == "tools.schema_capped":
        omitted = int(payload.get("omitted_count", 0) or 0)
        included = int(payload.get("included_count", 0) or 0)
        renderer.status(f"tool schema budget applied: {included} exposed, {omitted} omitted")
    elif event_name == "memory.prepared":
        return dict(payload)
    elif event_name == "approval.requested":
        tool_name = str(payload.get("tool_name", ""))
        risk_level = str(payload.get("risk_level", ""))
        renderer.status(f"approval requested: {tool_name} {risk_level}")
    elif event_name == "approval.completed":
        tool_name = str(payload.get("tool_name", ""))
        decision = str(payload.get("decision", ""))
        renderer.status(f"approval completed: {tool_name} {decision}")
    elif event_name == "approval.audit_failed":
        tool_name = str(payload.get("tool_name", ""))
        error = str(payload.get("error", ""))
        renderer.status(f"approval audit failed: {tool_name} {error}")
    return None


def _plan_from_payload(payload: dict[str, object]) -> PlanState | None:
    raw = payload.get("plan")
    if not isinstance(raw, dict):
        return None
    return PlanState.from_dict(raw)


def _task_from_payload(payload: dict[str, object]) -> PlanTask | None:
    raw = payload.get("task")
    if not isinstance(raw, dict):
        return None
    return PlanTask.from_dict(raw)


def _team_from_payload(payload: dict[str, object]) -> TeamState | None:
    raw = payload.get("team")
    if not isinstance(raw, dict):
        return None
    return TeamState.from_dict(raw)


def _team_step_from_payload(payload: dict[str, object]) -> ExecutionStep | None:
    raw = payload.get("step")
    if not isinstance(raw, dict):
        return None
    return ExecutionStep.from_dict(raw)
