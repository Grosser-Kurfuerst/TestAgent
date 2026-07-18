"""Public facade for the layered memory-maintenance implementation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from my_agent.memory.evolver.maintenance.contracts import (
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
    maintenance_plan_json,
    write_maintenance_plan,
)

_LAZY_EXPORTS = {
    "MaintenanceHistoryLockTimeout": (
        "my_agent.memory.evolver.maintenance.legacy.transaction"
    ),
    "apply_maintenance_plan": "my_agent.memory.evolver.maintenance.legacy.transaction",
    "build_maintenance_plan": "my_agent.memory.evolver.maintenance.legacy.service",
    "load_maintenance_plan": "my_agent.memory.evolver.maintenance.legacy.validation",
    "load_project_attribution": "my_agent.memory.evolver.maintenance.legacy.planner",
    "lookup_experiences": "my_agent.memory.evolver.maintenance.legacy.planner",
    "maintenance_evidence_for_entry": (
        "my_agent.memory.evolver.maintenance.legacy.planner"
    ),
    "record_post_commit_audit_error": (
        "my_agent.memory.evolver.maintenance.legacy.transaction"
    ),
    "redundancy_score": "my_agent.memory.evolver.maintenance.legacy.planner",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

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
    "MaintenanceHistoryLockTimeout",
    "MaintenanceLookupHit",
    "MaintenanceOperation",
    "MaintenancePlan",
    "MaintenancePlanError",
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
