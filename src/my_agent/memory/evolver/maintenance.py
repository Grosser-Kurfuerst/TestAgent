"""Compatibility facade for the layered memory-maintenance implementation."""

from __future__ import annotations

from my_agent.memory.evolver.contracts import (
    AttributionKey,
    MAINTENANCE_POLICY,
    MAINTENANCE_SCHEMA_VERSION,
    MAINTENANCE_SCOPE_MODE,
    MaintenanceAction,
    MaintenanceApplyResult,
    MaintenanceApplyStatus,
    MaintenanceAttributionError,
    MaintenanceConfig,
    MaintenanceError,
    MaintenanceEvidence,
    MaintenanceLookupHit,
    MaintenanceOperation,
    MaintenancePlan,
    MaintenancePlanError,
    _operation_id,
    _plan_id,
    maintenance_plan_json,
    write_maintenance_plan,
)
from my_agent.memory.evolver.planner import (
    _validate_operation_conflicts,
    load_project_attribution,
    lookup_experiences,
    maintenance_evidence_for_entry,
    redundancy_score,
)
from my_agent.memory.evolver.service import build_maintenance_plan
from my_agent.memory.evolver.transaction import (
    _maintenance_backup_path,
    _write_backup_atomic,
    append_maintenance_history,
    apply_maintenance_plan,
    record_post_commit_audit_error,
)
from my_agent.memory.evolver.validation import load_maintenance_plan

__all__ = [
    "AttributionKey",
    "MAINTENANCE_POLICY",
    "MAINTENANCE_SCHEMA_VERSION",
    "MAINTENANCE_SCOPE_MODE",
    "MaintenanceAction",
    "MaintenanceApplyResult",
    "MaintenanceApplyStatus",
    "MaintenanceAttributionError",
    "MaintenanceConfig",
    "MaintenanceError",
    "MaintenanceEvidence",
    "MaintenanceLookupHit",
    "MaintenanceOperation",
    "MaintenancePlan",
    "MaintenancePlanError",
    "append_maintenance_history",
    "apply_maintenance_plan",
    "build_maintenance_plan",
    "load_maintenance_plan",
    "load_project_attribution",
    "lookup_experiences",
    "maintenance_evidence_for_entry",
    "maintenance_plan_json",
    "redundancy_score",
    "record_post_commit_audit_error",
    "write_maintenance_plan",
]
