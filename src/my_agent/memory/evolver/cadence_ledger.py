"""Compatibility module alias for the maintenance cadence ledger."""

from __future__ import annotations

import sys

from my_agent.memory.evolver.maintenance.cadence import ledger as _ledger

sys.modules[__name__] = _ledger
