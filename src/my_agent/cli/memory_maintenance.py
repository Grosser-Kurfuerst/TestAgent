"""Execution and post-commit state handling for the memory maintenance CLI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from my_agent.memory.evolver import (
    MaintenanceApplyResult,
    MaintenanceApplyStatus,
    MaintenanceConfig,
    MaintenancePlan,
    build_maintenance_plan,
    load_maintenance_plan,
    load_project_attribution,
    record_post_commit_audit_error,
    write_maintenance_plan,
)
from my_agent.memory.evolver.artifacts import _resolve_maintenance_artifact_graph
from my_agent.memory.evolver.transaction import (
    _apply_maintenance_plan as apply_maintenance_plan,
)
from my_agent.memory.evolver.validation import validate_plan_semantics
from my_agent.memory.experience_store import (
    EXPERIENCE_LOCK_FILE,
    EXPERIENCE_STORAGE_FILE,
    ExperienceStore,
)
from my_agent.memory.store_errors import MemoryStoreLockTimeout, MemoryStoreRevisionConflict
from my_agent.observability.tracing import TraceWriter
from my_agent.schema import TraceEvent


def run_maintenance_command(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args)
    state = _CommandState(
        mode="apply" if bool(args.apply) else "dry_run",
        project_key=str(args.memory_project_key),
        paths=paths,
        lock_timeout_seconds=float(args.lock_timeout_seconds),
    )
    try:
        return _run_maintenance(args, state)
    except Exception as exc:
        if state.apply_result is not None and state.apply_result.mutation_committed:
            try:
                return _handle_post_commit_exception(state, exc)
            except Exception:
                try:
                    print(
                        "WARNING: memory mutation was already committed; "
                        "should_retry=false; DO NOT RETRY this plan.",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
                return 0
        return _handle_pre_commit_exception(state, exc)

@dataclass(frozen=True)
class _MaintenancePaths:
    memory_dir: Path
    attribution: Path
    output: Path
    summary: Path
    trace: Path
    history: Path
    backup_dir: Path


@dataclass
class _CommandState:
    mode: str
    project_key: str
    paths: _MaintenancePaths
    lock_timeout_seconds: float
    stage: str = "validation"
    plan: MaintenancePlan | None = None
    paths_validated: bool = False
    reuse_plan_artifact: bool = False
    apply_result: MaintenanceApplyResult | None = None


def _run_maintenance(args: argparse.Namespace, state: _CommandState) -> int:
    preplan_graph = _resolve_maintenance_artifact_graph(
        store_path=state.paths.memory_dir / EXPERIENCE_STORAGE_FILE,
        store_lock_path=state.paths.memory_dir / EXPERIENCE_LOCK_FILE,
        history_path=state.paths.history,
        backup_dir=state.paths.backup_dir,
        plan_id=None,
        memory_dir=state.paths.memory_dir,
        attribution_path=state.paths.attribution,
        plan_input_path=Path(args.plan) if args.plan else None,
        plan_output_path=state.paths.output,
        summary_path=state.paths.summary,
        trace_path=state.paths.trace,
    )
    state.reuse_plan_artifact = preplan_graph.reuse_plan_artifact
    state.paths_validated = True
    store = ExperienceStore.from_dir(
        state.paths.memory_dir,
        lock_timeout_seconds=float(args.lock_timeout_seconds),
    )
    if args.plan:
        state.stage = "strict_load"
        snapshot = store.load_strict_snapshot()
        state.stage = "plan_load"
        plan = load_maintenance_plan(args.plan)
        state.plan = plan
        state.stage = "validation"
        if plan.memory_project_key != state.project_key:
            raise ValueError("reviewed plan memory_project_key does not match CLI value")
        if plan.repository_revision != snapshot.revision:
            raise MemoryStoreRevisionConflict(
                "reviewed maintenance plan is stale: "
                f"expected {plan.repository_revision}, got {snapshot.revision}"
            )
        validate_plan_semantics(plan, repository_entries=snapshot.memories)
    else:
        state.stage = "strict_load"
        snapshot = store.load_strict_snapshot()
        state.stage = "validation"
        attribution = _load_attribution(
            state.paths,
            project_key=state.project_key,
            required=bool(args.attribution),
        )
        config = _maintenance_config_from_args(args)
        plan = build_maintenance_plan(
            entries=snapshot.memories,
            attribution=attribution,
            repository_revision=snapshot.revision,
            project_key=state.project_key,
            as_of=_parse_as_of(args.as_of),
            config=config,
        )
        state.plan = plan

    state.stage = "artifact_validation"
    state.paths_validated = False
    artifact_graph = _resolve_maintenance_artifact_graph(
        store_path=store.path,
        store_lock_path=store.lock_path,
        history_path=state.paths.history,
        backup_dir=state.paths.backup_dir,
        plan_id=plan.plan_id,
        memory_dir=state.paths.memory_dir,
        attribution_path=state.paths.attribution,
        plan_input_path=Path(args.plan) if args.plan else None,
        plan_output_path=state.paths.output,
        summary_path=state.paths.summary,
        trace_path=state.paths.trace,
    )
    state.reuse_plan_artifact = artifact_graph.reuse_plan_artifact
    state.paths_validated = True
    state.stage = "validation"
    if not state.reuse_plan_artifact:
        write_maintenance_plan(plan, state.paths.output)
    writer = TraceWriter(state.paths.trace)
    run_id = _maintenance_run_id(plan)
    writer.append(TraceEvent(
        event="memory.maintenance_started",
        payload=_started_payload(
            plan,
            mode=state.mode,
            current_revision=snapshot.revision,
            entries_total=len(snapshot.memories),
        ),
        run_id=run_id,
    ))
    writer.append(TraceEvent(
        event="memory.maintenance_proposed",
        payload=_proposed_payload(plan),
        run_id=run_id,
    ))

    if state.mode == "dry_run":
        summary = _summary_for_dry_run(state, plan)
        _write_summary(state.paths.summary, summary)
        print(_render_summary(summary))
        return 0

    result = apply_maintenance_plan(
        store=store,
        plan=plan,
        backup_dir=state.paths.backup_dir,
        history_path=state.paths.history,
        lock_timeout_seconds=float(args.lock_timeout_seconds),
        artifact_graph=artifact_graph,
    )
    state.apply_result = result
    state.stage = "post_commit" if result.mutation_committed else "validation"
    summary = _summary_for_apply(state, plan, result)
    if result.status == MaintenanceApplyStatus.PRE_COMMIT_FAILED:
        _append_trace_best_effort(
            writer,
            TraceEvent(
                event="memory.maintenance_failed",
                payload=_failed_payload(plan, result),
                run_id=run_id,
            ),
            summary,
        )
        _write_summary_best_effort(state.paths.summary, summary)
        print(_render_summary(summary))
        print(
            "Error: "
            f"status={result.status.value} "
            f"stage={_failure_phase(result.audit_error_stage)} "
            f"error={result.audit_error or 'maintenance_error'}",
            file=sys.stderr,
        )
        return 1

    return _finalize_apply_success(
        state,
        result=result,
        writer=writer,
        run_id=run_id,
        summary=summary,
    )


def _resolve_paths(args: argparse.Namespace) -> _MaintenancePaths:
    memory_dir = Path(args.memory_dir)
    output = Path(args.output) if args.output else memory_dir / "maintenance_plan.json"
    return _MaintenancePaths(
        memory_dir=memory_dir,
        attribution=(
            Path(args.attribution)
            if args.attribution
            else memory_dir / "memory_attribution.jsonl"
        ),
        output=output,
        summary=Path(str(output) + ".summary.json"),
        trace=(
            Path(args.trace_output)
            if args.trace_output
            else memory_dir / "maintenance_trace.jsonl"
        ),
        history=(
            Path(args.history_output)
            if args.history_output
            else memory_dir / "maintenance_history.jsonl"
        ),
        backup_dir=(
            Path(args.backup_dir)
            if args.backup_dir
            else memory_dir / "maintenance_backups"
        ),
    )


def _load_attribution(
    paths: _MaintenancePaths,
    *,
    project_key: str,
    required: bool,
) -> Mapping[tuple[str, str, str], Any]:
    if paths.attribution.exists():
        return load_project_attribution(
            paths.attribution,
            memory_project_key=project_key,
        )
    if required:
        raise FileNotFoundError(str(paths.attribution))
    return {}


def _maintenance_config_from_args(args: argparse.Namespace) -> MaintenanceConfig:
    payload = MaintenanceConfig().to_dict()
    overrides = {
        "delete_value_threshold": args.delete_value_threshold,
        "delete_min_confidence": args.delete_min_confidence,
        "delete_min_candidate_count": args.delete_min_candidate_count,
        "stale_after_days": args.stale_after_days,
        "merge_max_cluster_size": args.merge_max_cluster_size,
        "promote_value_threshold": args.promote_value_threshold,
        "promote_min_confidence": args.promote_min_confidence,
        "promote_min_selected_count": args.promote_min_selected_count,
        "max_promotions": args.max_promotions,
    }
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    if args.merge_threshold is not None:
        for tier in ("tip", "skill", "tool"):
            payload[f"merge_threshold_{tier}"] = args.merge_threshold
    return MaintenanceConfig.from_dict(payload)


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    normalized = value.strip()
    if not normalized:
        raise ValueError("--as-of must not be empty")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("--as-of must be a valid ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _started_payload(
    plan: MaintenancePlan,
    *,
    mode: str,
    current_revision: str,
    entries_total: int,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "policy": plan.policy,
        "scope_mode": plan.scope_mode,
        "memory_project_key": plan.memory_project_key,
        "repository_revision": plan.repository_revision,
        "current_repository_revision": current_revision,
        "entries_total": entries_total,
        "experiences_considered": _safe_int(
            plan.input_summary.get("experiences_considered")
        ),
        "as_of": plan.as_of,
    }


def _proposed_payload(plan: MaintenancePlan) -> dict[str, Any]:
    operations = [
        {
            "operation_id": operation.operation_id,
            "action": operation.action.value,
            "source_ids": list(operation.source_ids),
            "target_ids": list(operation.target_ids),
        }
        for operation in plan.operations
    ]
    return {
        "plan_id": plan.plan_id,
        "keep": _safe_int(plan.summary.get("keep")),
        "delete": _safe_int(plan.summary.get("delete")),
        "merge": _safe_int(plan.summary.get("merge")),
        "promote": _safe_int(plan.summary.get("promote")),
        "missing_attribution": _safe_int(
            plan.input_summary.get("missing_attribution")
        ),
        "redundant_pairs": sum(
            len(operation.source_ids) * (len(operation.source_ids) - 1) // 2
            for operation in plan.operations
            if operation.action.value == "merge"
        ),
        "source_entries_removed": _safe_int(
            plan.summary.get("source_entries_removed")
        ),
        "entries_added": _safe_int(plan.summary.get("entries_added")),
        "operation_summaries": operations,
    }


def _failed_payload(
    plan: MaintenancePlan,
    result: MaintenanceApplyResult,
) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "status": MaintenanceApplyStatus.PRE_COMMIT_FAILED.value,
        "stage": _failure_phase(result.audit_error_stage),
        "error": result.audit_error or "maintenance_error",
        "mutation_committed": False,
        "should_retry": result.should_retry,
    }


def _summary_for_dry_run(
    state: _CommandState,
    plan: MaintenancePlan,
) -> dict[str, Any]:
    return {
        **_base_summary(state, plan),
        "status": "dry_run",
        "mutation_committed": False,
        "audit_complete": True,
        "should_retry": False,
        "trace_complete": True,
        "summary_complete": True,
    }


def _summary_for_apply(
    state: _CommandState,
    plan: MaintenancePlan,
    result: MaintenanceApplyResult,
) -> dict[str, Any]:
    return {
        **_base_summary(state, plan),
        **result.to_dict(),
        "trace_complete": True,
        "summary_complete": True,
    }


def _base_summary(
    state: _CommandState,
    plan: MaintenancePlan,
) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "mode": state.mode,
        "plan_id": plan.plan_id,
        "policy": plan.policy,
        "scope_mode": plan.scope_mode,
        "memory_project_key": plan.memory_project_key,
        "as_of": plan.as_of,
        "repository_revision": plan.repository_revision,
        "entries_total": _safe_int(plan.input_summary.get("entries_total")),
        "experiences_considered": _safe_int(
            plan.input_summary.get("experiences_considered")
        ),
        "missing_attribution": _safe_int(
            plan.input_summary.get("missing_attribution")
        ),
        "keep": _safe_int(plan.summary.get("keep")),
        "delete": _safe_int(plan.summary.get("delete")),
        "merge": _safe_int(plan.summary.get("merge")),
        "promote": _safe_int(plan.summary.get("promote")),
        "source_entries_removed": _safe_int(
            plan.summary.get("source_entries_removed")
        ),
        "entries_added": _safe_int(plan.summary.get("entries_added")),
        "plan_path": str(state.paths.output),
        "summary_path": str(state.paths.summary),
        "trace_path": str(state.paths.trace),
        "history_path": str(state.paths.history),
        "backup_dir": str(state.paths.backup_dir),
    }


def _handle_pre_commit_exception(state: _CommandState, exc: Exception) -> int:
    phase = "lock" if isinstance(exc, MemoryStoreLockTimeout) else _failure_phase(state.stage)
    summary: dict[str, Any] = {
        "mode": state.mode,
        "status": MaintenanceApplyStatus.PRE_COMMIT_FAILED.value,
        "memory_project_key": state.project_key,
        "mutation_committed": False,
        "audit_complete": False,
        "should_retry": isinstance(exc, (MemoryStoreLockTimeout, OSError)),
        "audit_error_stage": phase,
        "audit_error": type(exc).__name__,
        "trace_complete": state.paths_validated,
        "summary_complete": state.paths_validated,
        "plan_path": str(state.paths.output),
        "summary_path": str(state.paths.summary),
        "trace_path": str(state.paths.trace),
    }
    if state.plan is not None:
        summary = {
            **_base_summary(state, state.plan),
            **summary,
        }
    if state.paths_validated:
        event = TraceEvent(
            event="memory.maintenance_failed",
            payload={
                "plan_id": state.plan.plan_id if state.plan is not None else "",
                "status": MaintenanceApplyStatus.PRE_COMMIT_FAILED.value,
                "stage": phase,
                "error": type(exc).__name__,
                "mutation_committed": False,
                "should_retry": summary["should_retry"],
            },
            run_id=(
                _maintenance_run_id(state.plan)
                if state.plan is not None
                else "maintenance-plan-load"
            ),
        )
        try:
            TraceWriter(state.paths.trace).append(event)
        except Exception as trace_exc:
            summary["trace_complete"] = False
            summary["trace_error"] = type(trace_exc).__name__
        _write_summary_best_effort(state.paths.summary, summary)
    else:
        summary["artifacts_skipped"] = True
    print(_render_summary(summary), file=sys.stderr)
    print(
        f"Error: status=pre_commit_failed stage={phase} "
        f"error={type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    return 1


def _finalize_apply_success(
    state: _CommandState,
    *,
    result: MaintenanceApplyResult,
    writer: TraceWriter,
    run_id: str,
    summary: dict[str, Any],
) -> int:
    state.stage = "summary"
    try:
        _write_summary(state.paths.summary, summary)
    except Exception as exc:
        if not result.mutation_committed:
            raise
        result = _record_post_commit_failure(
            state,
            result,
            stage="summary",
            error=exc,
        )
        summary = _post_commit_audit_error(summary, stage="summary", error=exc)
        state.stage = "render"
        rendered = _render_summary(summary)
        state.stage = "trace"
        try:
            writer.append(_completed_event(result, summary=summary, run_id=run_id))
        except Exception as trace_exc:
            result = _record_post_commit_failure(
                state,
                result,
                stage="trace",
                error=trace_exc,
            )
            summary = _post_commit_audit_error(
                summary,
                stage="trace",
                error=trace_exc,
            )
            rendered = _render_summary(summary)
        _print_stdout_best_effort(rendered)
        _print_do_not_retry_warning("summary")
        if summary.get("trace_complete") is False:
            _print_do_not_retry_warning("trace")
        return 0

    state.stage = "render"
    rendered = _render_summary(summary)
    state.stage = "trace"
    try:
        writer.append(_completed_event(result, summary=summary, run_id=run_id))
    except Exception as exc:
        if not result.mutation_committed:
            raise
        result = _record_post_commit_failure(
            state,
            result,
            stage="trace",
            error=exc,
        )
        summary = _post_commit_audit_error(summary, stage="trace", error=exc)
        _write_summary_best_effort(state.paths.summary, summary)
        state.stage = "render"
        _print_stdout_best_effort(_render_summary(summary))
        _print_do_not_retry_warning("trace")
        return 0

    _print_stdout_best_effort(rendered)
    if summary.get("status") == MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR.value:
        _print_do_not_retry_warning(
            str(summary.get("audit_error_stage") or "post_commit_audit")
        )
    return 0


def _completed_event(
    result: MaintenanceApplyResult,
    *,
    summary: Mapping[str, Any],
    run_id: str,
) -> TraceEvent:
    payload = {"mode": "apply", **result.to_dict()}
    for key in (
        "status",
        "mutation_committed",
        "audit_complete",
        "should_retry",
        "audit_error_stage",
        "audit_error",
        "trace_complete",
        "summary_complete",
    ):
        if key in summary:
            payload[key] = summary[key]
    return TraceEvent(
        event="memory.maintenance_completed",
        payload=payload,
        run_id=run_id,
    )


def _post_commit_audit_error(
    summary: Mapping[str, Any],
    *,
    stage: str,
    error: Exception,
) -> dict[str, Any]:
    updated = dict(summary)
    previous_stage = str(updated.get("audit_error_stage") or "")
    previous_error = str(updated.get("audit_error") or "")
    if previous_stage and previous_stage != stage:
        updated["previous_audit_error_stage"] = previous_stage
        updated["previous_audit_error"] = previous_error
    updated.update({
        "status": MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR.value,
        "mutation_committed": True,
        "audit_complete": False,
        "should_retry": False,
        "audit_error_stage": stage,
        "audit_error": type(error).__name__,
    })
    if stage == "trace":
        updated["trace_complete"] = False
        updated["trace_error"] = type(error).__name__
    if stage == "summary":
        updated["summary_complete"] = False
        updated["summary_error"] = type(error).__name__
    return updated


def _record_post_commit_failure(
    state: _CommandState,
    result: MaintenanceApplyResult,
    *,
    stage: str,
    error: Exception,
) -> MaintenanceApplyResult:
    plan = state.plan
    if plan is None:
        return result
    updated = record_post_commit_audit_error(
        history_path=state.paths.history,
        plan=plan,
        result=result,
        stage=stage,
        error=error,
        lock_timeout_seconds=state.lock_timeout_seconds,
    )
    state.apply_result = updated
    return updated


def _handle_post_commit_exception(state: _CommandState, exc: Exception) -> int:
    result = state.apply_result
    plan = state.plan
    if result is not None and plan is not None:
        result = _record_post_commit_failure(
            state,
            result,
            stage=state.stage or "post_commit",
            error=exc,
        )
    summary = _emergency_post_commit_summary(state, result, plan, exc)
    if state.paths_validated and result is not None and plan is not None:
        try:
            _write_summary(state.paths.summary, summary)
        except Exception as summary_exc:
            result = _record_post_commit_failure(
                state,
                result,
                stage="summary",
                error=summary_exc,
            )
            summary = _post_commit_audit_error(
                summary,
                stage="summary",
                error=summary_exc,
            )
        try:
            payload = {
                "mode": "apply",
                "plan_id": summary["plan_id"],
                "status": summary["status"],
                "mutation_committed": True,
                "audit_complete": False,
                "should_retry": False,
                "before_revision": summary["before_revision"],
                "after_revision": summary["after_revision"],
                "before_count": summary["before_count"],
                "after_count": summary["after_count"],
                "removed_ids": summary["removed_ids"],
                "updated_ids": summary["updated_ids"],
                "added_ids": summary["added_ids"],
                "backup_path": summary["backup_path"],
                "audit_error_stage": summary["audit_error_stage"],
                "audit_error": summary["audit_error"],
                "trace_complete": summary["trace_complete"],
                "summary_complete": summary["summary_complete"],
            }
            TraceWriter(state.paths.trace).append(
                TraceEvent(
                    event="memory.maintenance_completed",
                    payload=payload,
                    run_id=f"maintenance-{str(summary['plan_id'])[:24]}",
                )
            )
        except Exception as trace_exc:
            result = _record_post_commit_failure(
                state,
                result,
                stage="trace",
                error=trace_exc,
            )
            summary = _post_commit_audit_error(
                summary,
                stage="trace",
                error=trace_exc,
            )
            try:
                _write_summary(state.paths.summary, summary)
            except Exception:
                pass
    _print_emergency_summary(summary)
    return 0


def _emergency_post_commit_summary(
    state: _CommandState,
    result: MaintenanceApplyResult | None,
    plan: MaintenancePlan | None,
    error: Exception,
) -> dict[str, Any]:
    before_count = getattr(result, "before_count", 0)
    if isinstance(before_count, bool) or not isinstance(before_count, int):
        before_count = 0
    after_count = getattr(result, "after_count", 0)
    if isinstance(after_count, bool) or not isinstance(after_count, int):
        after_count = 0
    kept = getattr(result, "kept", 0)
    if isinstance(kept, bool) or not isinstance(kept, int):
        kept = 0
    deleted = getattr(result, "deleted", 0)
    if isinstance(deleted, bool) or not isinstance(deleted, int):
        deleted = 0
    merged = getattr(result, "merged", 0)
    if isinstance(merged, bool) or not isinstance(merged, int):
        merged = 0
    promoted = getattr(result, "promoted", 0)
    if isinstance(promoted, bool) or not isinstance(promoted, int):
        promoted = 0
    return {
        "schema_version": getattr(plan, "schema_version", 1),
        "mode": "apply",
        "plan_id": str(getattr(result, "plan_id", "") or getattr(plan, "plan_id", "")),
        "policy": str(getattr(plan, "policy", "")),
        "scope_mode": str(getattr(plan, "scope_mode", "single_project")),
        "memory_project_key": state.project_key,
        "status": MaintenanceApplyStatus.COMMITTED_WITH_AUDIT_ERROR.value,
        "mutation_committed": True,
        "audit_complete": False,
        "should_retry": False,
        "before_revision": str(getattr(result, "before_revision", "")),
        "after_revision": str(getattr(result, "after_revision", "")),
        "before_count": max(0, before_count),
        "after_count": max(0, after_count),
        "keep": max(0, kept),
        "delete": max(0, deleted),
        "merge": max(0, merged),
        "promote": max(0, promoted),
        "removed_ids": list(getattr(result, "removed_ids", ()) or ()),
        "updated_ids": list(getattr(result, "updated_ids", ()) or ()),
        "added_ids": list(getattr(result, "added_ids", ()) or ()),
        "backup_path": str(getattr(result, "backup_path", "")),
        "audit_error_stage": state.stage or "post_commit",
        "audit_error": type(error).__name__,
        "trace_complete": True,
        "summary_complete": True,
        "plan_path": str(state.paths.output),
        "summary_path": str(state.paths.summary),
        "trace_path": str(state.paths.trace),
    }


def _print_emergency_summary(summary: Mapping[str, Any]) -> None:
    try:
        print(
            "Maintenance completed with an audit error: "
            f"plan={summary.get('plan_id') or 'unavailable'} "
            "status=committed_with_audit_error mutation_committed=true "
            "should_retry=false",
            file=sys.stderr,
        )
        print(
            "WARNING: DO NOT RETRY this plan; the memory mutation was already committed.",
            file=sys.stderr,
        )
    except Exception:
        pass


def _append_trace_best_effort(
    writer: TraceWriter,
    event: TraceEvent,
    summary: dict[str, Any],
) -> None:
    try:
        writer.append(event)
    except Exception as exc:
        summary["trace_complete"] = False
        summary["trace_error"] = type(exc).__name__


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_summary_best_effort(path: Path, summary: Mapping[str, Any]) -> None:
    try:
        _write_summary(path, summary)
    except Exception as exc:
        try:
            print(
                f"Warning: maintenance summary write failed: {type(exc).__name__}",
                file=sys.stderr,
            )
        except Exception:
            pass


def _render_summary(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        f"Maintenance plan: {summary.get('plan_id') or 'unavailable'}",
        f"Experiences considered: {_safe_int(summary.get('experiences_considered'))}",
        (
            f"Keep: {_safe_int(summary.get('keep'))}, "
            f"delete: {_safe_int(summary.get('delete'))}, "
            f"merge: {_safe_int(summary.get('merge'))}, "
            f"promote: {_safe_int(summary.get('promote'))}"
        ),
        (
            "Source entries removed: "
            f"{_safe_int(summary.get('source_entries_removed'))}, "
            f"entries added: {_safe_int(summary.get('entries_added'))}"
        ),
        f"Mode: {summary.get('mode') or 'unknown'}",
        f"Status: {summary.get('status') or 'unknown'}",
        f"Plan: {summary.get('plan_path') or 'unavailable'}",
    ])


def _failure_phase(stage: str) -> str:
    if stage == "plan_load":
        return "plan_load"
    if stage == "lock":
        return "lock"
    if stage == "strict_load":
        return "strict_load"
    if stage == "artifact_validation":
        return "artifact_validation"
    if stage in {"history_load", "history_lock"}:
        return stage
    if stage in {"backup", "audit_intent", "persist", "verify"}:
        return stage
    return "validation"


def _maintenance_run_id(plan: MaintenancePlan) -> str:
    return f"maintenance-{plan.plan_id[:24]}"


def _print_stdout_best_effort(text: str) -> None:
    try:
        print(text)
    except Exception:
        pass


def _print_do_not_retry_warning(stage: str) -> None:
    try:
        print(
            "WARNING: memory changes were committed but post-commit audit was incomplete; "
            f"stage={stage}; should_retry=false; DO NOT RETRY this plan.",
            file=sys.stderr,
        )
    except Exception:
        pass


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = ["run_maintenance_command"]
