"""Orchestration boundary for validated maintenance planning."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from my_agent.memory.evolver.attribution import MemoryAttributionRecord
from my_agent.memory.evolver.contracts import (
    AttributionKey,
    MaintenanceConfig,
    MaintenancePlan,
)
from my_agent.memory.evolver.planner import _build_maintenance_plan
from my_agent.memory.evolver.validation import validate_plan_semantics
from my_agent.memory.types import MemoryEntry


def build_maintenance_plan(
    *,
    entries: Sequence[MemoryEntry],
    attribution: Mapping[AttributionKey, MemoryAttributionRecord],
    repository_revision: str,
    project_key: str,
    as_of: datetime,
    config: MaintenanceConfig | None = None,
) -> MaintenancePlan:
    plan = _build_maintenance_plan(
        entries=entries,
        attribution=attribution,
        repository_revision=repository_revision,
        project_key=project_key,
        as_of=as_of,
        config=config,
    )
    validate_plan_semantics(plan, repository_entries=entries)
    return plan


__all__ = ["build_maintenance_plan"]
