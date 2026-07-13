"""Shared reducer for maintenance observability events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class MaintenanceEventCounters:
    runs: int = 0
    applied_runs: int = 0
    keep: int = 0
    delete: int = 0
    merge: int = 0
    promote: int = 0
    removed_entries: int = 0
    added_entries: int = 0
    failures: int = 0
    committed_with_audit_error: int = 0

    def observe(self, event_name: object, payload: Mapping[str, Any]) -> bool:
        """Consume one maintenance event and report whether it was recognized."""
        if event_name == "memory.maintenance_started":
            self.runs += 1
            return True
        if event_name == "memory.maintenance_proposed":
            self.keep += _nonnegative_int(payload.get("keep"))
            self.delete += _nonnegative_int(payload.get("delete"))
            self.merge += _nonnegative_int(payload.get("merge"))
            self.promote += _nonnegative_int(payload.get("promote"))
            self.removed_entries += _nonnegative_int(
                payload.get("source_entries_removed")
            )
            self.added_entries += _nonnegative_int(payload.get("entries_added"))
            return True
        if event_name == "memory.maintenance_completed":
            self.applied_runs += 1
            if payload.get("status") == "committed_with_audit_error":
                self.committed_with_audit_error += 1
            return True
        if event_name == "memory.maintenance_failed":
            self.failures += 1
            return True
        return False


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0


__all__ = ["MaintenanceEventCounters"]
